"""An index on disk: a directory of immutable segments and a manifest that says which ones are current.

* ``manifest.json`` is replaced atomically (write a temp file, fsync, rename, fsync the directory), so a reader or a crash sees either
  the old list of segments or the new one, never a mix.  Everything else in the directory is written once and never modified.
* A deleted document is only marked: the segment keeps it and a small ``.del`` file lists the ids that are gone.  The space comes back
  when segments are merged.
* One writer at a time (a ``write.lock`` file with an exclusive lock); any number of readers, each looking at the snapshot it opened.
* The documents that are added wait in memory and become a segment when the buffer is full or at ``commit()``; what is not committed
  is lost on a crash.
"""
from __future__ import annotations

import heapq
import json
import os
import struct
import zlib
from array import array
from typing import Dict, List, Optional, Sequence, Tuple

from .errors import CorruptIndexError, LockError, SchemaError, SearchError
from .schema import Schema
from .segment import NO_MORE, SegmentBuilder, SegmentReader, write_segment

try:  # POSIX
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore

MANIFEST = "manifest.json"
FORMAT = 1
_DEL_MAGIC = b"KDEL0001"


# ---------------------------------------------------------------------------------------------------------------- files

def _fsync_dir(path: str) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:  # pragma: no cover
        return
    try:
        os.fsync(fd)
    except OSError:  # pragma: no cover
        pass
    finally:
        os.close(fd)


def write_deleted(path: str, ids: Sequence[int]) -> None:
    a = array("I", sorted(ids))
    body = a.tobytes()
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(_DEL_MAGIC)
        fh.write(struct.pack("<I", len(a)))
        fh.write(body)
        fh.write(struct.pack("<I", zlib.crc32(body) & 0xFFFFFFFF))
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def read_deleted(path: str) -> frozenset:
    with open(path, "rb") as fh:
        data = fh.read()
    if data[:8] != _DEL_MAGIC or len(data) < 16:
        raise CorruptIndexError("%s is not a list of deleted documents" % path)
    (n,) = struct.unpack_from("<I", data, 8)
    body = data[12 : 12 + 4 * n]
    if len(body) != 4 * n or len(data) != 16 + 4 * n or struct.unpack_from("<I", data, 12 + 4 * n)[0] != zlib.crc32(body) & 0xFFFFFFFF:
        raise CorruptIndexError("%s: the checksum does not match" % path)
    a = array("I")
    a.frombytes(body)
    return frozenset(a)


def read_manifest(path: str) -> dict:
    mpath = os.path.join(path, MANIFEST)
    try:
        with open(mpath, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        raise SearchError("%s is not an index (there is no %s)" % (path, MANIFEST)) from None
    except ValueError as e:
        raise CorruptIndexError("%s is damaged: %s" % (mpath, e)) from None
    if data.get("format") != FORMAT:
        raise CorruptIndexError("%s: unknown format %r" % (mpath, data.get("format")))
    return data


def write_manifest(path: str, data: dict) -> None:
    mpath = os.path.join(path, MANIFEST)
    tmp = mpath + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, mpath)
    _fsync_dir(path)


# ---------------------------------------------------------------------------------------------------------------- merging

def merge_segments(out_path: str, schema: Schema, readers: Sequence[SegmentReader], deleted: Sequence[frozenset]) -> Tuple[dict, List[List[int]]]:
    """Writes the live documents of several segments as one segment.  Returns the description of the new segment and, for every
    input, the new id of each of its documents (-1 for a deleted one)."""
    docmaps: List[List[int]] = []
    keys: List[str] = []
    stored: List[dict] = []
    lens: List[array] = [array("I") for _ in schema.fields]
    for reader, dead in zip(readers, deleted):
        m = [-1] * reader.numdocs
        rkeys = reader.keys()
        flens = [reader.field_lengths(fi) for fi in range(len(schema.fields))]
        for d in range(reader.numdocs):
            if d in dead:
                continue
            m[d] = len(keys)
            keys.append(rkeys[d])
            stored.append(reader.stored(d))
            for fi in range(len(schema.fields)):
                lens[fi].append(flens[fi][d])
        docmaps.append(m)

    def tagged(fi: int, ri: int):
        for term, info in readers[ri].terms(fi):
            yield term, ri, info

    def stream(fi: int):
        iters = [tagged(fi, ri) for ri in range(len(readers))]
        current = None
        docs: List[int] = []
        freqs: List[int] = []
        positions: List[int] = []
        for term, ri, info in heapq.merge(*iters):
            if term != current:
                if docs:
                    yield current, docs, freqs, positions
                current, docs, freqs, positions = term, [], [], []
            it = readers[ri].postings(fi, info)
            m = docmaps[ri]
            while it.next() != NO_MORE:
                nd = m[it.doc]
                if nd < 0:
                    continue
                docs.append(nd)
                freqs.append(it.freq)
                positions.extend(it.positions())
        if docs:
            yield current, docs, freqs, positions

    meta = write_segment(out_path, schema, keys, stored, lens, [stream(fi) for fi in range(len(schema.fields))])
    return meta, docmaps


class MergePolicy:
    """Tiered merging: segments of a similar size (a power of ``factor`` of live documents) are merged when there are ``factor`` of
    them, and a segment where more than ``max_deleted`` of the documents are gone is rewritten alone."""

    def __init__(self, factor: int = 10, base: int = 1000, max_deleted: float = 0.3) -> None:
        if factor < 2:
            raise ValueError("the merge factor must be at least 2")
        self.factor = factor
        self.base = base
        self.max_deleted = max_deleted

    def level(self, live: int) -> int:
        level = 0
        size = self.base
        while live > size:
            size *= self.factor
            level += 1
        return level

    def select(self, infos: Sequence[Tuple[str, int, int]]) -> List[List[str]]:
        """``infos``: (name, documents, deleted).  Returns groups of names to merge."""
        groups: List[List[str]] = []
        used = set()
        by_level: Dict[int, List[str]] = {}
        for name, n, dead in infos:
            live = n - dead
            if n and dead / n > self.max_deleted:
                groups.append([name])
                used.add(name)
                continue
            by_level.setdefault(self.level(live), []).append(name)
        for level in sorted(by_level):
            names = [x for x in by_level[level] if x not in used]
            while len(names) >= self.factor:
                groups.append(names[: self.factor])
                names = names[self.factor :]
        return groups


# ---------------------------------------------------------------------------------------------------------------- the index

class Index:
    """A directory with an index.  ``Index.create(path, schema)`` makes a new one, ``Index.open(path)`` opens it."""

    def __init__(self, path: str) -> None:
        self.path = path
        data = read_manifest(path)
        self.schema = Schema.from_json(data["schema"])

    @classmethod
    def create(cls, path: str, schema: Optional[Schema] = None, exist_ok: bool = False) -> "Index":
        os.makedirs(path, exist_ok=True)
        if os.path.exists(os.path.join(path, MANIFEST)):
            if not exist_ok:
                raise SearchError("%s already has an index" % path)
            return cls(path)
        schema = schema if schema is not None else Schema()
        write_manifest(path, {"format": FORMAT, "generation": 0, "schema": schema.to_json(), "next_segment": 0, "segments": []})
        return cls(path)

    @classmethod
    def open(cls, path: str) -> "Index":
        return cls(path)

    @staticmethod
    def exists(path: str) -> bool:
        return os.path.exists(os.path.join(path, MANIFEST))

    def manifest(self) -> dict:
        return read_manifest(self.path)

    def writer(self, **options) -> "IndexWriter":
        return IndexWriter(self, **options)

    def searcher(self):
        from .searcher import Searcher

        return Searcher.open(self)

    def check(self) -> List[str]:
        """Reads every file and returns the problems found (an empty list for a healthy index)."""
        problems: List[str] = []
        try:
            data = read_manifest(self.path)
        except SearchError as e:
            return [str(e)]
        referenced = {MANIFEST, "write.lock"}
        for seg in data["segments"]:
            fpath = os.path.join(self.path, seg["file"])
            referenced.add(seg["file"])
            try:
                reader = SegmentReader(fpath, self.schema)
            except (OSError, SearchError) as e:
                problems.append("%s: %s" % (seg["file"], e))
                continue
            try:
                reader.verify()
                if reader.numdocs != seg["numdocs"]:
                    problems.append("%s: %d documents, the manifest says %d" % (seg["file"], reader.numdocs, seg["numdocs"]))
                if len(reader.keys()) != reader.numdocs:
                    problems.append("%s: wrong number of keys" % seg["file"])
                for fi in range(len(self.schema)):
                    prev = None
                    count = 0
                    for term, info in reader.terms(fi):
                        if prev is not None and term <= prev:
                            problems.append("%s: the terms of %s are not sorted" % (seg["file"], self.schema.fields[fi].name))
                            break
                        prev = term
                        count += 1
                    else:
                        if count != reader.num_terms(fi):
                            problems.append("%s: %d terms found, %d expected" % (seg["file"], count, reader.num_terms(fi)))
            except SearchError as e:
                problems.append("%s: %s" % (seg["file"], e))
            finally:
                reader.close()
            dele = seg.get("deleted")
            if dele:
                referenced.add(dele["file"])
                try:
                    ids = read_deleted(os.path.join(self.path, dele["file"]))
                    if len(ids) != dele["count"] or (ids and max(ids) >= seg["numdocs"]):
                        problems.append("%s: the list of deleted documents does not fit the segment" % dele["file"])
                except (OSError, SearchError) as e:
                    problems.append("%s: %s" % (dele["file"], e))
        return problems


class _Seg:
    """A segment as the writer knows it."""

    def __init__(self, name: str, numdocs: int, deleted: set, deleted_file: Optional[str]) -> None:
        self.name = name
        self.numdocs = numdocs
        self.deleted = deleted
        self.deleted_file = deleted_file
        self.dirty = False
        self.reader: Optional[SegmentReader] = None

    @property
    def file(self) -> str:
        return self.name + ".kseg"

    def live(self) -> int:
        return self.numdocs - len(self.deleted)


class IndexWriter:
    def __init__(self, index: Index, max_buffered_docs: int = 10000, merge_policy: Optional[MergePolicy] = None, auto_merge: bool = True) -> None:
        self.index = index
        self.path = index.path
        self.schema = index.schema
        self.max_buffered_docs = max_buffered_docs
        self.merge_policy = merge_policy or MergePolicy()
        self.auto_merge = auto_merge
        self._lock_fd = self._acquire()
        self._closed = False
        try:
            data = read_manifest(self.path)
            self._generation = data["generation"]
            self._next_segment = data["next_segment"]
            self._segs: List[_Seg] = []
            for rec in data["segments"]:
                dele = rec.get("deleted")
                deleted = set(read_deleted(os.path.join(self.path, dele["file"]))) if dele else set()
                self._segs.append(_Seg(rec["name"], rec["numdocs"], deleted, dele["file"] if dele else None))
            self._builder = SegmentBuilder(self.schema)
            self._buf_keys: Dict[str, int] = {}
            self._key_map: Optional[Dict[str, Tuple[_Seg, int]]] = None
            self._obsolete: List[_Seg] = []  # merged away: their files are removed when the commit that drops them is done
            self._remove_orphans(data)
        except BaseException:
            self._release()
            raise

    # -- locking

    def _acquire(self):
        fd = os.open(os.path.join(self.path, "write.lock"), os.O_RDWR | os.O_CREAT, 0o644)
        if fcntl is not None:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                os.close(fd)
                raise LockError("another writer has %s open" % self.path) from None
        return fd

    def _release(self) -> None:
        if self._lock_fd is not None:
            try:
                if fcntl is not None:
                    fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(self._lock_fd)
                self._lock_fd = None

    def _remove_orphans(self, data: dict) -> None:
        """Files left by a writer that crashed before its commit: segments and temp files that the manifest does not mention."""
        keep = {MANIFEST, "write.lock"}
        for rec in data["segments"]:
            keep.add(rec["name"] + ".kseg")
            if rec.get("deleted"):
                keep.add(rec["deleted"]["file"])
        for name in os.listdir(self.path):
            if name in keep:
                continue
            if name.endswith((".kseg", ".tmp")) or ".del." in name:
                try:
                    os.remove(os.path.join(self.path, name))
                except OSError:  # pragma: no cover
                    pass

    # -- documents

    def _reader(self, seg: _Seg) -> SegmentReader:
        if seg.reader is None:
            seg.reader = SegmentReader(os.path.join(self.path, seg.file), self.schema)
        return seg.reader

    def _keys(self) -> Dict[str, Tuple[_Seg, int]]:
        if self._key_map is None:
            km: Dict[str, Tuple[_Seg, int]] = {}
            for seg in self._segs:
                for d, key in enumerate(self._reader(seg).keys()):
                    if d not in seg.deleted:
                        km[key] = (seg, d)
            self._key_map = km
        return self._key_map

    def _check(self, doc: dict) -> None:
        if not isinstance(doc, dict):
            raise SchemaError("a document is a dict of field names and texts")
        for name in doc:
            if name not in self.schema.index:
                raise SchemaError("the document has a field that the schema does not have: %s" % name)

    def has(self, key: str) -> bool:
        key = str(key)
        return key in self._buf_keys or key in self._keys()

    def __len__(self) -> int:
        return len(self._buf_keys) + sum(s.live() for s in self._segs)

    def add(self, key: str, doc: dict) -> None:
        """Adds a document; a document with the same key is replaced."""
        key = str(key)
        self._check(doc)
        self._remove(key)
        self._buf_keys[key] = self._builder.add(key, doc)
        if len(self._builder) >= self.max_buffered_docs or self._builder.postings >= 2_000_000:
            self._flush()

    def delete(self, key: str) -> bool:
        return self._remove(str(key))

    def _remove(self, key: str) -> bool:
        d = self._buf_keys.pop(key, None)
        if d is not None:
            self._builder.dead.add(d)
            return True
        km = self._keys()
        loc = km.pop(key, None)
        if loc is None:
            return False
        seg, d = loc
        seg.deleted.add(d)
        seg.dirty = True
        return True

    # -- segments

    def _new_name(self) -> str:
        name = "s%06d" % self._next_segment
        self._next_segment += 1
        return name

    def _flush(self) -> None:
        b = self._builder
        if not len(b):
            return
        name = self._new_name()
        write_segment(os.path.join(self.path, name + ".kseg"), self.schema, b.keys, b.stored, b.lens, b.streams())
        seg = _Seg(name, len(b), set(b.dead), None)
        seg.dirty = bool(seg.deleted)
        self._segs.append(seg)
        if self._key_map is not None:
            for key, d in self._buf_keys.items():
                self._key_map[key] = (seg, d)
        self._builder = SegmentBuilder(self.schema)
        self._buf_keys = {}

    def _merge(self, group: List[_Seg]) -> None:
        for seg in group:
            self._reader(seg)
        name = self._new_name()
        out_path = os.path.join(self.path, name + ".kseg")
        meta, _ = merge_segments(out_path, self.schema, [s.reader for s in group], [frozenset(s.deleted) for s in group])
        position = min(self._segs.index(s) for s in group)
        for seg in group:
            self._segs.remove(seg)
        if meta["numdocs"]:
            self._segs.insert(min(position, len(self._segs)), _Seg(name, meta["numdocs"], set(), None))
        else:  # every document of the group was deleted
            os.remove(out_path)
        self._key_map = None
        self._obsolete.extend(group)

    def merge(self, max_segments: int = 1) -> None:
        """Merges segments until at most ``max_segments`` are left (``merge(1)`` is the "optimize" of other engines)."""
        self._flush()
        while len(self._segs) > max_segments:
            segs = sorted(self._segs, key=lambda s: s.live())
            take = max(2, len(segs) - max_segments + 1)
            self._merge(segs[:take])
        if max_segments == 1 and self._segs and self._segs[0].deleted:
            self._merge([self._segs[0]])

    def commit(self) -> None:
        self._flush()
        if self.auto_merge:
            while True:
                infos = [(s.name, s.numdocs, len(s.deleted)) for s in self._segs]
                groups = self.merge_policy.select(infos)
                if not groups:
                    break
                by_name = {s.name: s for s in self._segs}
                self._merge([by_name[n] for n in groups[0]])
        self._generation += 1
        records = []
        stale = []
        for seg in self._segs:
            rec = {"name": seg.name, "file": seg.file, "numdocs": seg.numdocs, "deleted": None}
            if seg.deleted:
                if seg.dirty or seg.deleted_file is None:
                    fname = "%s.del.%d" % (seg.name, self._generation)
                    write_deleted(os.path.join(self.path, fname), seg.deleted)
                    if seg.deleted_file:
                        stale.append(seg.deleted_file)
                    seg.deleted_file = fname
                    seg.dirty = False
                rec["deleted"] = {"file": seg.deleted_file, "count": len(seg.deleted)}
            records.append(rec)
        _fsync_dir(self.path)
        write_manifest(
            self.path,
            {"format": FORMAT, "generation": self._generation, "schema": self.schema.to_json(), "next_segment": self._next_segment, "segments": records},
        )
        for seg in self._obsolete:
            if seg.reader is not None:
                seg.reader.close()
                seg.reader = None
            for fname in (seg.file, seg.deleted_file):
                if fname:
                    try:
                        os.remove(os.path.join(self.path, fname))
                    except OSError:  # pragma: no cover
                        pass
        self._obsolete = []
        for fname in stale:
            try:
                os.remove(os.path.join(self.path, fname))
            except OSError:  # pragma: no cover
                pass

    def segments(self) -> List[Tuple[str, int, int]]:
        return [(s.name, s.numdocs, len(s.deleted)) for s in self._segs]

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for seg in self._segs:
            if seg.reader is not None:
                seg.reader.close()
        self._release()

    def __enter__(self) -> "IndexWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            if exc_type is None:
                self.commit()
        finally:
            self.close()
