"""A segment: an immutable piece of an index, written once to one file and read through ``mmap``.

Layout of the file (all numbers little endian)::

    magic
    keys                    the external id of every document (varint length + bytes)
    stored, stored index    the original text of the stored fields, 16 documents per zlib block
    per field:
      lengths               the number of tokens of the field in every document (uint32 array)
      terms                 the sorted terms in blocks of 32 (front coded), each with df, total tf and the offsets of its postings
      term index            the first term of every block (a binary search over it finds the block)
      postings              per term: a skip table (when there are more than 128 documents) and blocks of 128 document gaps and
                            term frequencies, bit packed
      positions             per term: the positions of the term in each document, as gaps (varint)
    footer (JSON)           counts and the offset/length of every section
    footer length, crc32 of everything before the footer, magic

A term is looked up with a binary search over the term index and a scan of one block.  The postings of a term are read block by
block; the skip table (the last document, the offsets and the largest term frequency of each block) lets a reader jump over blocks.
"""
from __future__ import annotations

import json
import mmap
import os
import struct
import sys
import zlib
from array import array
from bisect import bisect_left, bisect_right
from typing import Dict, Iterable, Iterator, List, NamedTuple, Optional, Sequence, Tuple

from .codec import BLOCK, pack_block, read_varint, unpack_block, write_varint
from .errors import CorruptIndexError
from .schema import Schema

MAGIC = b"KARAMA01"
NO_MORE = 0x7FFFFFFF
TERM_BLOCK = 32
STORED_BLOCK = 16
_SKIP = struct.Struct("<IIII")


class TermInfo(NamedTuple):
    df: int  # the number of documents with the term
    ttf: int  # the number of times it occurs in all of them
    post_off: int
    pos_off: int


# ---------------------------------------------------------------------------------------------------------------- building

class SegmentBuilder:
    """The documents that are waiting in memory to become a segment."""

    def __init__(self, schema: Schema) -> None:
        self.schema = schema
        self.keys: List[str] = []
        self.stored: List[dict] = []
        self.lens: List[array] = [array("I") for _ in schema.fields]
        # per field: term -> (documents, term frequencies, positions of all the documents one after the other)
        self.terms: List[Dict[str, Tuple[List[int], List[int], List[int]]]] = [{} for _ in schema.fields]
        self.dead: set = set()
        self.postings = 0

    def __len__(self) -> int:
        return len(self.keys)

    def add(self, key: str, doc: dict) -> int:
        d = len(self.keys)
        self.keys.append(key)
        stored = {}
        for fi, field in enumerate(self.schema.fields):
            value = doc.get(field.name)
            if value is None:
                self.lens[fi].append(0)
                continue
            text = value if isinstance(value, str) else str(value)
            if field.stored:
                stored[field.name] = text
            tokens = field.analyzer.tokens(text)
            self.lens[fi].append(len(tokens))
            if not tokens:
                continue
            local: Dict[str, List[int]] = {}
            for t in tokens:
                try:
                    local[t.term].append(t.pos)
                except KeyError:
                    local[t.term] = [t.pos]
            table = self.terms[fi]
            for term, plist in local.items():
                entry = table.get(term)
                if entry is None:
                    entry = table[term] = ([], [], [])
                entry[0].append(d)
                entry[1].append(len(plist))
                entry[2].extend(plist)
            self.postings += len(tokens)
        self.stored.append(stored)
        return d

    def streams(self) -> List[Iterator[Tuple[str, List[int], List[int], List[int]]]]:
        def one(table):
            for t in sorted(table):
                docs, freqs, positions = table[t]
                yield t, docs, freqs, positions

        return [one(table) for table in self.terms]


def _encode_postings(post: bytearray, pos: bytearray, docs: List[int], freqs: List[int], positions: List[int]) -> None:
    df = len(docs)
    nblocks = (df + BLOCK - 1) // BLOCK
    skip = bytearray()
    blocks = bytearray()
    pos_base = len(pos)
    prev = -1
    pi = 0
    for b in range(nblocks):
        lo = b * BLOCK
        hi = min(df, lo + BLOCK)
        dd = docs[lo:hi]
        deltas = [dd[0] - prev]
        deltas.extend(dd[k] - dd[k - 1] for k in range(1, len(dd)))
        ff = freqs[lo:hi]
        data_off = len(blocks)
        blocks += pack_block(deltas)
        blocks += pack_block([f - 1 for f in ff])
        pos_off = len(pos) - pos_base
        for n in ff:
            last = 0
            for p in positions[pi : pi + n]:
                write_varint(pos, p - last)
                last = p
            pi += n
        skip += _SKIP.pack(dd[-1], data_off, pos_off, max(ff))
        prev = dd[-1]
    if df > BLOCK:
        post += skip
    post += blocks


def write_segment(
    path: str,
    schema: Schema,
    keys: Sequence[str],
    stored: Sequence[dict],
    lens: Sequence[array],
    streams: Sequence[Iterable[Tuple[str, List[int], List[int], List[int]]]],
) -> dict:
    """Writes a segment file (to ``path`` + ``.tmp``, then renamed) and returns the numbers that describe it."""
    tmp = path + ".tmp"
    footer: dict = {"version": 1, "numdocs": len(keys), "fields": []}
    crc = 0
    offset = 0
    with open(tmp, "wb") as fh:

        def emit(data: bytes) -> Tuple[int, int]:
            nonlocal crc, offset
            start = offset
            fh.write(data)
            crc = zlib.crc32(data, crc)
            offset += len(data)
            return start, len(data)

        emit(MAGIC)
        buf = bytearray()
        for key in keys:
            raw = key.encode("utf-8", "surrogatepass")
            write_varint(buf, len(raw))
            buf += raw
        footer["keys"] = emit(bytes(buf))

        blocks = bytearray()
        index = array("Q", [0])
        for lo in range(0, len(stored), STORED_BLOCK):
            raw = json.dumps(list(stored[lo : lo + STORED_BLOCK]), ensure_ascii=False, separators=(",", ":")).encode("utf-8", "surrogatepass")
            blocks += zlib.compress(raw, 6)
            index.append(len(blocks))
        footer["stored"] = emit(bytes(blocks))
        if sys.byteorder == "big":
            index.byteswap()
        footer["stored_index"] = emit(index.tobytes())

        for fi, field in enumerate(schema.fields):
            fl = array("I", lens[fi])
            sum_len = sum(fl)
            if sys.byteorder == "big":
                fl.byteswap()
            info: dict = {"name": field.name, "sum_len": sum_len, "lens": emit(fl.tobytes())}
            post = bytearray()
            pos = bytearray()
            terms = bytearray()
            tindex = bytearray()
            block: List[Tuple[bytes, int, int, int, int]] = []
            nterms = 0
            nblocks = 0
            ndocs_postings = 0

            def flush_block() -> None:
                nonlocal nblocks
                if not block:
                    return
                write_varint(tindex, len(block[0][0]))
                tindex.extend(block[0][0])
                write_varint(tindex, len(terms))
                prev_term = b""
                prev_post = prev_pos = 0
                for k, (term, df, ttf, post_off, pos_off) in enumerate(block):
                    if k == 0:
                        write_varint(terms, len(term))
                        terms.extend(term)
                        write_varint(terms, df)
                        write_varint(terms, ttf)
                        write_varint(terms, post_off)
                        write_varint(terms, pos_off)
                    else:
                        shared = 0
                        limit = min(len(term), len(prev_term))
                        while shared < limit and term[shared] == prev_term[shared]:
                            shared += 1
                        write_varint(terms, shared)
                        write_varint(terms, len(term) - shared)
                        terms.extend(term[shared:])
                        write_varint(terms, df)
                        write_varint(terms, ttf)
                        write_varint(terms, post_off - prev_post)
                        write_varint(terms, pos_off - prev_pos)
                    prev_term, prev_post, prev_pos = term, post_off, pos_off
                nblocks += 1
                block.clear()

            for term, docs, freqs, positions in streams[fi]:
                post_off = len(post)
                pos_off = len(pos)
                _encode_postings(post, pos, docs, freqs, positions)
                block.append((term.encode("utf-8", "surrogatepass"), len(docs), sum(freqs), post_off, pos_off))
                nterms += 1
                ndocs_postings += len(docs)
                if len(block) == TERM_BLOCK:
                    flush_block()
            flush_block()
            info["terms"] = emit(bytes(terms))
            info["term_index"] = emit(bytes(tindex))
            info["postings"] = emit(bytes(post))
            info["positions"] = emit(bytes(pos))
            info["numterms"] = nterms
            info["nblocks"] = nblocks
            info["numpostings"] = ndocs_postings
            footer["fields"].append(info)

        body = json.dumps(footer, separators=(",", ":")).encode("utf-8")
        fh.write(body)
        fh.write(struct.pack("<II", len(body), crc & 0xFFFFFFFF))
        fh.write(MAGIC)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return {"numdocs": len(keys), "fields": [{"name": f["name"], "sum_len": f["sum_len"], "numterms": f["numterms"]} for f in footer["fields"]]}


# ---------------------------------------------------------------------------------------------------------------- reading

class _FieldData:
    __slots__ = ("name", "sum_len", "numterms", "nblocks", "lens", "terms", "term_index", "postings", "positions", "first_terms", "block_offs", "cache")


class SegmentReader:
    """Opens a segment file and gives access to its terms, postings, lengths, keys and stored fields."""

    def __init__(self, path: str, schema: Schema) -> None:
        self.path = path
        self.schema = schema
        with open(path, "rb") as fh:
            size = os.fstat(fh.fileno()).st_size
            if size < len(MAGIC) * 2 + 8:
                raise CorruptIndexError("%s is too short to be a segment" % path)
            self._mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        mm = self._mm
        if mm[: len(MAGIC)] != MAGIC or mm[size - len(MAGIC) :] != MAGIC:
            raise CorruptIndexError("%s: the header or the trailer is wrong" % path)
        flen, self._crc = struct.unpack_from("<II", mm, size - len(MAGIC) - 8)
        fstart = size - len(MAGIC) - 8 - flen
        if fstart < len(MAGIC):
            raise CorruptIndexError("%s: the footer does not fit" % path)
        self._footer_start = fstart
        try:
            footer = json.loads(mm[fstart : fstart + flen])
        except ValueError as e:
            raise CorruptIndexError("%s: the footer is not valid (%s)" % (path, e)) from None
        self._footer = footer
        if footer.get("version") != 1 or len(footer["fields"]) != len(schema.fields):
            raise CorruptIndexError("%s does not match the schema of the index" % path)
        self.numdocs: int = footer["numdocs"]
        self._fields: List[_FieldData] = []
        for fi, info in enumerate(footer["fields"]):
            fd = _FieldData()
            fd.name = info["name"]
            if fd.name != schema.fields[fi].name:
                raise CorruptIndexError("%s: the fields differ from the schema" % path)
            fd.sum_len = info["sum_len"]
            fd.numterms = info["numterms"]
            fd.nblocks = info["nblocks"]
            fd.lens = None
            fd.terms = info["terms"][0]
            fd.term_index = info["term_index"]
            fd.postings = info["postings"][0]
            fd.positions = info["positions"][0]
            fd.cache = {}
            fd.first_terms = None
            fd.block_offs = None
            self._fields.append(fd)
        self._lens_loc = [info["lens"] for info in footer["fields"]]
        self._keys: Optional[List[str]] = None
        self._stored_cache: Dict[int, list] = {}
        self._doc_len: Optional[array] = None

    # -- housekeeping

    def close(self) -> None:
        try:
            self._mm.close()
        except (BufferError, ValueError):
            pass

    def verify(self) -> None:
        """Reads the whole file and compares the checksum (``open`` only looks at the header and the footer)."""
        if (zlib.crc32(self._mm[: self._footer_start]) & 0xFFFFFFFF) != self._crc:
            raise CorruptIndexError("%s: the checksum does not match" % self.path)

    # -- documents

    def keys(self) -> List[str]:
        if self._keys is None:
            off, length = self._footer["keys"]
            buf = self._mm
            pos = off
            end = off + length
            out = []
            while pos < end:
                n, pos = read_varint(buf, pos)
                out.append(bytes(buf[pos : pos + n]).decode("utf-8", "surrogatepass"))
                pos += n
            if len(out) != self.numdocs:
                raise CorruptIndexError("%s: %d keys for %d documents" % (self.path, len(out), self.numdocs))
            self._keys = out
        return self._keys

    def stored(self, doc: int) -> dict:
        b = doc // STORED_BLOCK
        block = self._stored_cache.get(b)
        if block is None:
            ioff, ilen = self._footer["stored_index"]
            lo, hi = struct.unpack_from("<QQ", self._mm, ioff + 8 * b)
            soff, _ = self._footer["stored"]
            raw = zlib.decompress(self._mm[soff + lo : soff + hi])
            block = json.loads(raw.decode("utf-8", "surrogatepass"))
            if len(self._stored_cache) > 64:
                self._stored_cache.clear()
            self._stored_cache[b] = block
        return block[doc % STORED_BLOCK]

    def field_lengths(self, fi: int) -> array:
        fd = self._fields[fi]
        if fd.lens is None:
            off, length = self._lens_loc[fi]
            a = array("I")
            a.frombytes(self._mm[off : off + length])
            if sys.byteorder == "big":
                a.byteswap()
            fd.lens = a
        return fd.lens

    def doc_lengths(self) -> array:
        """The number of tokens of every document in all the fields together."""
        if self._doc_len is None:
            total = array("I", [0]) * self.numdocs if self.numdocs else array("I")
            for fi in range(len(self._fields)):
                fl = self.field_lengths(fi)
                for d in range(self.numdocs):
                    total[d] += fl[d]
            self._doc_len = total
        return self._doc_len

    def sum_length(self, fi: int) -> int:
        return self._fields[fi].sum_len

    # -- terms

    def _load_index(self, fd: _FieldData) -> None:
        off, length = fd.term_index
        buf = self._mm
        pos = off
        end = off + length
        firsts = []
        offs = []
        while pos < end:
            n, pos = read_varint(buf, pos)
            firsts.append(bytes(buf[pos : pos + n]).decode("utf-8", "surrogatepass"))
            pos += n
            o, pos = read_varint(buf, pos)
            offs.append(o)
        fd.first_terms = firsts
        fd.block_offs = offs

    def _block(self, fi: int, b: int) -> Tuple[List[str], List[TermInfo]]:
        fd = self._fields[fi]
        hit = fd.cache.get(b)
        if hit is not None:
            return hit
        buf = self._mm
        pos = fd.terms + fd.block_offs[b]
        count = min(TERM_BLOCK, fd.numterms - b * TERM_BLOCK)
        names: List[str] = []
        infos: List[TermInfo] = []
        prev = b""
        post = posn = 0
        for k in range(count):
            if k == 0:
                n, pos = read_varint(buf, pos)
                term = bytes(buf[pos : pos + n])
                pos += n
            else:
                shared, pos = read_varint(buf, pos)
                n, pos = read_varint(buf, pos)
                term = prev[:shared] + bytes(buf[pos : pos + n])
                pos += n
            df, pos = read_varint(buf, pos)
            ttf, pos = read_varint(buf, pos)
            dp, pos = read_varint(buf, pos)
            dq, pos = read_varint(buf, pos)
            if k == 0:
                post, posn = dp, dq
            else:
                post += dp
                posn += dq
            names.append(term.decode("utf-8", "surrogatepass"))
            infos.append(TermInfo(df, ttf, post, posn))
            prev = term
        if len(fd.cache) > 512:
            fd.cache.clear()
        fd.cache[b] = (names, infos)
        return names, infos

    def term_info(self, fi: int, term: str) -> Optional[TermInfo]:
        fd = self._fields[fi]
        if fd.first_terms is None:
            self._load_index(fd)
        b = bisect_right(fd.first_terms, term) - 1
        if b < 0:
            return None
        names, infos = self._block(fi, b)
        i = bisect_left(names, term)
        if i < len(names) and names[i] == term:
            return infos[i]
        return None

    def terms(self, fi: int, prefix: str = "") -> Iterator[Tuple[str, TermInfo]]:
        """The terms of a field that start with ``prefix`` (all of them for ""), in order."""
        fd = self._fields[fi]
        if fd.first_terms is None:
            self._load_index(fd)
        b = max(bisect_right(fd.first_terms, prefix) - 1, 0)
        while b < fd.nblocks:
            names, infos = self._block(fi, b)
            i = bisect_left(names, prefix) if prefix else 0
            while i < len(names):
                name = names[i]
                if not name.startswith(prefix):
                    return
                yield name, infos[i]
                i += 1
            b += 1

    def num_terms(self, fi: int) -> int:
        return self._fields[fi].numterms

    def postings(self, fi: int, info: TermInfo) -> "PostingsIter":
        fd = self._fields[fi]
        return PostingsIter(self._mm, fd.postings + info.post_off, fd.positions + info.pos_off, info.df)


class PostingsIter:
    """Walks the documents of one term of one field in order.  ``doc`` is the current document (``NO_MORE`` at the end), ``freq`` its
    term frequency; ``positions()`` decodes where the term occurs in it."""

    __slots__ = ("buf", "post_start", "pos_base", "df", "nblocks", "last_docs", "data_offs", "pos_offs", "max_tfs", "data_start",
                 "blk", "docs", "freqs", "i", "doc", "freq", "_pk", "_pp", "_pcache_doc", "_pcache")

    def __init__(self, buf, post_start: int, pos_base: int, df: int) -> None:
        self.buf = buf
        self.pos_base = pos_base
        self.df = df
        self.nblocks = (df + BLOCK - 1) // BLOCK
        if df > BLOCK:
            raw = struct.unpack_from("<%dI" % (4 * self.nblocks), buf, post_start)
            self.last_docs = raw[0::4]
            self.data_offs = raw[1::4]
            self.pos_offs = raw[2::4]
            self.max_tfs = raw[3::4]
            self.data_start = post_start + 16 * self.nblocks
        else:
            self.last_docs = None
            self.data_offs = (0,)
            self.pos_offs = (0,)
            self.max_tfs = None
            self.data_start = post_start
        self.blk = -1
        self.docs: List[int] = []
        self.freqs: List[int] = []
        self.i = -1
        self.doc = -1
        self.freq = 0
        self._pk = 0
        self._pp = 0
        self._pcache_doc = -1
        self._pcache: List[int] = []

    def _load(self, blk: int, base: int) -> None:
        n = BLOCK if blk < self.nblocks - 1 else self.df - blk * BLOCK
        buf = self.buf
        pos = self.data_start + self.data_offs[blk]
        deltas, pos = unpack_block(buf, pos, n)
        fq, pos = unpack_block(buf, pos, n)
        docs = []
        d = base
        for x in deltas:
            d += x
            docs.append(d)
        self.docs = docs
        self.freqs = [f + 1 for f in fq]
        self.blk = blk
        self._pk = 0
        self._pp = self.pos_base + self.pos_offs[blk]
        self._pcache_doc = -1

    def _base(self, blk: int) -> int:
        return self.last_docs[blk - 1] if blk > 0 else -1

    def next(self) -> int:
        i = self.i + 1
        if i >= len(self.docs):
            blk = self.blk + 1
            if blk >= self.nblocks:
                self.i = len(self.docs)
                self.doc = NO_MORE
                return NO_MORE
            self._load(blk, self.docs[-1] if self.docs else -1)
            i = 0
        self.i = i
        self.doc = self.docs[i]
        self.freq = self.freqs[i]
        return self.doc

    def advance(self, target: int) -> int:
        """Moves to the first document >= target (it never moves backwards)."""
        if target < 0:
            target = 0
        if self.doc >= target:
            return self.doc
        docs = self.docs
        if self.blk < 0 or docs[-1] < target:
            if self.last_docs is None:
                if self.blk >= 0:
                    return self._end()
                self._load(0, -1)
            else:
                blk = bisect_left(self.last_docs, target)
                if blk >= self.nblocks:
                    return self._end()
                self._load(blk, self._base(blk))
            docs = self.docs
            if docs[-1] < target:
                return self._end()
            self.i = 0
        i = bisect_left(docs, target, max(self.i, 0))
        self.i = i
        self.doc = docs[i]
        self.freq = self.freqs[i]
        return self.doc

    def _end(self) -> int:
        self.i = len(self.docs)
        self.doc = NO_MORE
        return NO_MORE

    def positions(self) -> List[int]:
        """The positions of the term in the current document."""
        i = self.i
        if self._pcache_doc == self.doc:
            return self._pcache
        buf = self.buf
        p = self._pp
        k = self._pk
        freqs = self.freqs
        while k < i:
            n = freqs[k]
            while n:
                if buf[p] < 0x80:
                    n -= 1
                p += 1
            k += 1
        self._pk = k
        self._pp = p
        out = []
        acc = 0
        for _ in range(freqs[i]):
            b = buf[p]
            p += 1
            if b < 0x80:
                acc += b
            else:
                v = b & 0x7F
                shift = 7
                while True:
                    b = buf[p]
                    p += 1
                    v |= (b & 0x7F) << shift
                    if b < 0x80:
                        break
                    shift += 7
                acc += v
            out.append(acc)
        self._pcache_doc = self.doc
        self._pcache = out
        return out

    def block_max_tf(self) -> int:
        """The largest term frequency in the block of the current document (all documents of a small term count as one block)."""
        if self.max_tfs is not None:
            return self.max_tfs[self.blk]
        return max(self.freqs) if self.freqs else 0
