import os
import random

import pytest

import fuzzgen
from kendi_arama import CorruptIndexError, Field, Schema
from kendi_arama.segment import NO_MORE, SegmentBuilder, SegmentReader, write_segment


def build(tmp_path, docs, schema=None, name="s.kseg"):
    schema = schema or Schema([Field("title"), Field("body")])
    b = SegmentBuilder(schema)
    for i, d in enumerate(docs):
        b.add("k%d" % i, d)
    path = str(tmp_path / name)
    meta = write_segment(path, schema, b.keys, b.stored, b.lens, b.streams())
    return b, SegmentReader(path, schema), meta, path


def test_postings_roundtrip_for_terms_of_every_size(tmp_path):
    rng = random.Random(3)
    docs = []
    for i in range(1500):
        words = ["common"]  # in every document
        if i % 2 == 0:
            words.append("half")
        if i % 7 == 0:
            words += ["seventh"] * rng.randint(1, 4)
        if i in (3, 400, 1499):
            words.append("rare")
        words += ["w%d" % rng.randint(0, 40) for _ in range(rng.randint(0, 20))]
        rng.shuffle(words)
        docs.append({"title": " ".join(words[:3]), "body": " ".join(words)})
    b, r, meta, _ = build(tmp_path, docs)
    r.verify()
    assert meta["numdocs"] == 1500 and r.numdocs == 1500
    for fi in (0, 1):
        for term, (dd, ff, pp) in b.terms[fi].items():
            info = r.term_info(fi, term)
            assert (info.df, info.ttf) == (len(dd), sum(ff))
            it = r.postings(fi, info)
            docs_got, freqs_got, pos_got = [], [], []
            while it.next() != NO_MORE:
                docs_got.append(it.doc)
                freqs_got.append(it.freq)
                pos_got.extend(it.positions())
            assert (docs_got, freqs_got, pos_got) == (dd, ff, pp), term
            assert it.next() == NO_MORE  # stays at the end
    assert r.term_info(1, "nosuchterm") is None and r.term_info(1, "zzzzzz") is None and r.term_info(1, "") is None


def test_advance_lands_on_the_first_document_at_or_after_the_target(tmp_path):
    rng = random.Random(4)
    docs = [{"body": "common " + ("half " if i % 2 == 0 else "") + ("tenth " if i % 10 == 0 else "")} for i in range(2000)]
    b, r, _, _ = build(tmp_path, docs, Schema([Field("body")]))
    for term in ("common", "half", "tenth"):
        dd, ff, pp = b.terms[0][term]
        info = r.term_info(0, term)
        for _ in range(60):
            it = r.postings(0, info)
            current = -1
            for t in sorted(rng.sample(range(-3, 2100), rng.randint(1, 12))):
                got = it.advance(t)
                want = next((d for d in dd if d >= max(t, current, 0)), NO_MORE)
                assert got == want
                if got == NO_MORE:
                    break
                current = got
                assert it.freq == ff[dd.index(got)]
                assert it.positions() == pp[sum(ff[: dd.index(got)]) : sum(ff[: dd.index(got)]) + it.freq]
        # mixing next() and advance()
        it = r.postings(0, info)
        seen = []
        d = it.next()
        while d != NO_MORE:
            seen.append(d)
            d = it.advance(d + 3) if len(seen) % 3 == 0 else it.next()
        assert seen == sorted(seen) and set(seen) <= set(dd)


def test_advance_on_a_small_term(tmp_path):
    b, r, _, _ = build(tmp_path, [{"body": "a"}, {"body": "b"}, {"body": "a b"}, {"body": "c"}, {"body": "a"}], Schema([Field("body")]))
    it = r.postings(0, r.term_info(0, "a"))
    assert it.advance(0) == 0 and it.advance(1) == 2 and it.advance(2) == 2 and it.advance(3) == 4 and it.advance(5) == NO_MORE
    it = r.postings(0, r.term_info(0, "a"))
    assert it.advance(5) == NO_MORE and it.next() == NO_MORE


def test_terms_are_sorted_and_prefix_scans_are_exact(tmp_path):
    rng = random.Random(5)
    words = sorted({"".join(rng.choice("abcde") for _ in range(rng.randint(1, 6))) for _ in range(900)})
    docs = [{"body": " ".join(words[i : i + 5])} for i in range(0, len(words), 5)]
    b, r, _, _ = build(tmp_path, docs, Schema([Field("body")]))
    assert [t for t, _ in r.terms(0)] == words and r.num_terms(0) == len(words)
    for prefix in ["", "a", "ab", "abc", "e", "ee", "eeeeee", "zzz", "b", "dd", "cab"]:
        assert [t for t, _ in r.terms(0, prefix)] == [w for w in words if w.startswith(prefix)], prefix
    for w in rng.sample(words, 50):
        assert r.term_info(0, w) is not None


def test_stored_fields_keys_and_lengths(tmp_path):
    docs = [{"title": "T%d é" % i, "body": "b%d " % i * (i % 5)} for i in range(70)] + [{}, {"title": ""}]
    schema = Schema([Field("title"), Field("body"), Field("silent", stored=False)])
    b, r, _, _ = build(tmp_path, [dict(d, silent="x y z") for d in docs], schema)
    assert r.keys() == ["k%d" % i for i in range(72)]
    for i in (0, 15, 16, 17, 31, 32, 69, 70, 71):
        stored = r.stored(i)
        assert stored.get("title", None) == docs[i].get("title", None) or docs[i].get("title") == "" and stored.get("title") == ""
        assert "silent" not in stored
    assert r.stored(3) == {"title": "T3 é", "body": "b3 " * 3}
    assert list(r.field_lengths(0))[:3] == [2, 2, 2] and list(r.field_lengths(1))[:6] == [0, 1, 2, 3, 4, 0]
    assert list(r.field_lengths(2))[:2] == [3, 3]
    assert r.sum_length(2) == 3 * 70 + 3 * 2 - 0 or r.sum_length(2) == sum(r.field_lengths(2))
    assert list(r.doc_lengths())[:3] == [2 + 0 + 3, 2 + 1 + 3, 2 + 2 + 3]


def test_empty_segment(tmp_path):
    b, r, meta, _ = build(tmp_path, [])
    r.verify()
    assert r.numdocs == 0 and list(r.terms(0)) == [] and r.keys() == []


def test_unusual_text(tmp_path):
    docs = [{"body": "x" * 5000 + " " + "ünï " * 200}, {"body": "\U0001F600 \u0000 \ud800 a"}]
    b, r, _, _ = build(tmp_path, docs, Schema([Field("body")]))
    assert r.term_info(0, "x" * 5000) is not None
    assert r.term_info(0, "uni").df == 1
    assert r.stored(0)["body"] == docs[0]["body"]


def flip(path, offset):
    with open(path, "r+b") as fh:
        fh.seek(offset)
        b = fh.read(1)
        fh.seek(offset)
        fh.write(bytes([b[0] ^ 0x40]))


def test_a_damaged_file_is_detected(tmp_path):
    docs = [{"title": "alpha beta", "body": "gamma delta " * 30} for _ in range(40)]
    _, r, _, path = build(tmp_path, docs)
    size = os.path.getsize(path)
    r.close()
    for offset in (10, size // 3, size // 2, size - 40):
        copy = str(tmp_path / ("bad%d.kseg" % offset))
        with open(path, "rb") as a, open(copy, "wb") as b2:
            b2.write(a.read())
        flip(copy, offset)
        schema = Schema([Field("title"), Field("body")])
        with pytest.raises(CorruptIndexError):
            SegmentReader(copy, schema).verify()
    # a file that is too short, with another magic, or whose footer is destroyed
    short = str(tmp_path / "short.kseg")
    with open(short, "wb") as fh:
        fh.write(b"KARAMA01")
    with pytest.raises(CorruptIndexError):
        SegmentReader(short, Schema([Field("title"), Field("body")]))
    other = str(tmp_path / "other.kseg")
    with open(path, "rb") as a, open(other, "wb") as b2:
        data = bytearray(a.read())
        data[:8] = b"NOTMINE!"
        b2.write(data)
    with pytest.raises(CorruptIndexError):
        SegmentReader(other, Schema([Field("title"), Field("body")]))
    cut = str(tmp_path / "cut.kseg")
    with open(path, "rb") as a, open(cut, "wb") as b2:
        b2.write(a.read()[:-30])
    with pytest.raises(CorruptIndexError):
        SegmentReader(cut, Schema([Field("title"), Field("body")]))


def test_a_segment_does_not_open_with_another_schema(tmp_path):
    _, r, _, path = build(tmp_path, [{"title": "a"}])
    r.close()
    with pytest.raises(CorruptIndexError):
        SegmentReader(path, Schema([Field("title")]))
    with pytest.raises(CorruptIndexError):
        SegmentReader(path, Schema([Field("title"), Field("other")]))


def test_random_documents_roundtrip(tmp_path):
    for seed in range(5):
        rng = random.Random(seed)
        vocab = fuzzgen.make_vocab(rng, 50)
        docs = [fuzzgen.make_doc(rng, vocab, ["title", "body"]) for _ in range(rng.randint(1, 300))]
        b, r, _, _ = build(tmp_path, docs, name="r%d.kseg" % seed)
        r.verify()
        for fi in (0, 1):
            for term, (dd, ff, pp) in b.terms[fi].items():
                it = r.postings(fi, r.term_info(fi, term))
                got = []
                while it.next() != NO_MORE:
                    got.append((it.doc, it.freq, it.positions()))
                pos = 0
                want = []
                for d, f in zip(dd, ff):
                    want.append((d, f, pp[pos : pos + f]))
                    pos += f
                assert got == want
