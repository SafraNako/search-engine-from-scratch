import json
import os
import random
import subprocess
import sys
import textwrap

import pytest

import fuzzgen
from kendi_arama import BM25, CorruptIndexError, Field, Index, LockError, MergePolicy, QuerySyntaxError, Schema, SchemaError, SearchError
from kendi_arama.index import read_deleted, write_deleted

SRC = os.path.join(os.path.dirname(__file__), "..", "src")


def make(path, fields=("title", "body"), **kw):
    return Index.create(path, Schema([Field(f) for f in fields]), **kw)


def keys(searcher, query):
    return sorted(h.key for h in searcher.search(query, limit=None))


def files(path):
    return sorted(os.listdir(path))


def test_create_open_and_the_schema_is_kept(index_dir):
    schema = Schema([Field("title", "porter", weight=3.0), Field("body", "english", stored=False), Field("tag", "trigram")])
    Index.create(index_dir, schema)
    again = Index.open(index_dir)
    assert again.schema == schema
    assert again.schema.field("title").weight == 3.0 and not again.schema.field("body").stored
    with pytest.raises(SearchError):
        Index.create(index_dir, schema)
    assert Index.create(index_dir, schema, exist_ok=True).schema == schema
    with pytest.raises(SearchError):
        Index.open(index_dir + "-missing")
    assert Index.exists(index_dir) and not Index.exists(index_dir + "-missing")


def test_schema_errors():
    with pytest.raises(SchemaError):
        Schema([])
    with pytest.raises(SchemaError):
        Schema([Field("a"), Field("a")])
    with pytest.raises(SchemaError):
        Field("not valid")
    with pytest.raises(SchemaError):
        Field("1abc")
    assert Schema(["a", "b"]).names() == ["a", "b"]


def test_add_search_update_delete(index_dir):
    idx = make(index_dir)
    with idx.writer() as w:
        w.add("1", {"title": "red fox", "body": "the quick red fox jumps"})
        w.add("2", {"title": "lazy dog", "body": "a dog sleeps"})
        w.add(3, {"title": "cats", "body": "cats and dogs"})  # keys are strings, whatever is given
        assert len(w) == 3 and w.has("1") and w.has(3) and not w.has("9")
    with idx.searcher() as s:
        assert len(s) == 3 and keys(s, "fox") == ["1"] and keys(s, "dog") == ["2"] and s.get("3") == {"title": "cats", "body": "cats and dogs"}
    with idx.writer() as w:
        w.add("1", {"title": "blue whale", "body": "big"})  # replaces
        assert w.delete("2") is True and w.delete("2") is False and w.delete("nope") is False
    with idx.searcher() as s:
        assert len(s) == 2 and keys(s, "fox") == [] and keys(s, "whale") == ["1"] and keys(s, "dog") == [] and s.get("2") is None
        assert s.get("1")["title"] == "blue whale"


def test_the_last_version_in_a_batch_wins_and_values_become_text(index_dir):
    idx = make(index_dir)
    with idx.writer() as w:
        w.add("k", {"title": "first"})
        w.add("k", {"title": "second"})
        w.add("k", {"title": 12345, "body": None})
    with idx.searcher() as s:
        assert len(s) == 1 and s.get("k") == {"title": "12345"} and keys(s, "12345") == ["k"] and keys(s, "first") == []


def test_unknown_fields_are_refused(index_dir):
    idx = make(index_dir)
    with idx.writer() as w:
        with pytest.raises(SchemaError):
            w.add("1", {"nope": "x"})
        with pytest.raises(SchemaError):
            w.add("1", "not a dict")


def test_uncommitted_work_is_not_visible_and_not_kept(index_dir):
    idx = make(index_dir)
    w = idx.writer()
    w.add("1", {"title": "x"})
    with idx.searcher() as s:
        assert len(s) == 0
    w.close()  # no commit
    with idx.searcher() as s:
        assert len(s) == 0
    with pytest.raises(RuntimeError):
        with idx.writer() as w:
            w.add("2", {"title": "y"})
            raise RuntimeError("boom")  # the context manager commits only when the block ends well
    with idx.searcher() as s:
        assert len(s) == 0


def test_a_searcher_is_a_snapshot(index_dir):
    idx = make(index_dir)
    with idx.writer() as w:
        w.add("1", {"title": "alpha"})
    old = idx.searcher()
    with idx.writer() as w:
        w.add("2", {"title": "alpha beta"})
        w.delete("1")
        w.merge(1)
    new = idx.searcher()
    try:
        assert keys(old, "alpha") == ["1"] and len(old) == 1  # still reads the files that a merge has deleted
        assert keys(new, "alpha") == ["2"]
        assert old.generation < new.generation
    finally:
        old.close()
        new.close()


def test_merging_keeps_the_results_and_removes_the_files(index_dir):
    rng = random.Random(1)
    vocab = fuzzgen.make_vocab(rng, 40)
    idx = make(index_dir)
    live = {}
    with idx.writer(max_buffered_docs=5, auto_merge=False) as w:
        for i in range(60):
            doc = fuzzgen.make_doc(rng, vocab, ["title", "body"])
            w.add(str(i), doc)
            live[str(i)] = doc
        for i in range(0, 60, 4):
            w.delete(str(i))
            del live[str(i)]
        w.commit()
        assert len(w.segments()) == 12
    queries = ["the", "cat OR dog", '"quick" ', "run*", "title:the NOT body:of", "NEAR(the of, 3)"]
    with idx.searcher() as s:
        before = {q: [(h.key, round(h.score, 12)) for h in s.search(q, limit=None)] for q in queries}
        assert len(s) == len(live)
    for target in (6, 3, 1):
        with idx.writer(auto_merge=False) as w:
            w.merge(target)
            assert len(w.segments()) == target
        with idx.searcher() as s:
            assert len(s.segs) == target and len(s) == len(live)
            # the score does not depend on how many segments there are
            after = {q: [(h.key, round(h.score, 12)) for h in s.search(q, limit=None)] for q in queries}
            assert {q: sorted(v) for q, v in after.items()} == {q: sorted(v) for q, v in before.items()}
        names = files(index_dir)
        assert len([n for n in names if n.endswith(".kseg")]) == target
    assert not [n for n in names if ".del." in n or n.endswith(".tmp")]  # nothing deleted is left after a full merge
    assert Index.open(index_dir).check() == []


def test_the_merge_policy():
    p = MergePolicy(factor=3, base=10, max_deleted=0.5)
    assert p.select([("a", 5, 0), ("b", 6, 0)]) == []
    assert p.select([("a", 5, 0), ("b", 6, 0), ("c", 7, 0)]) == [["a", "b", "c"]]
    assert p.select([("a", 5, 0), ("b", 6, 0), ("c", 100, 0)]) == []
    assert p.select([("a", 5, 0), ("b", 6, 0), ("c", 40, 0), ("d", 50, 0), ("e", 60, 0), ("f", 7, 0)]) == [["a", "b", "f"], ["c", "d", "e"]]
    assert p.select([("a", 100, 60), ("b", 100, 10)]) == [["a"]]  # a segment that is mostly deleted is rewritten
    assert p.level(10) == 0 and p.level(11) == 1 and p.level(30) == 1 and p.level(31) == 2
    with pytest.raises(ValueError):
        MergePolicy(factor=1)


def test_commit_merges_by_the_policy(index_dir):
    idx = make(index_dir)
    with idx.writer(max_buffered_docs=2, merge_policy=MergePolicy(factor=3, base=4)) as w:
        for i in range(30):
            w.add(str(i), {"title": "word%d common" % i})
        w.commit()
        assert len(w.segments()) < 15
        assert len(w) == 30
    with idx.searcher() as s:
        assert len(s) == 30 and keys(s, "common") == sorted(str(i) for i in range(30))


def test_one_writer_at_a_time(index_dir):
    idx = make(index_dir)
    w = idx.writer()
    with pytest.raises(LockError):
        idx.writer()
    # another process cannot get the lock either
    code = textwrap.dedent(
        """
        import sys
        sys.path.insert(0, %r)
        from kendi_arama import Index, LockError
        try:
            Index.open(%r).writer()
        except LockError:
            print("locked")
        """
    ) % (SRC, index_dir)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert out.stdout.strip() == "locked", out
    w.close()
    idx.writer().close()  # free again
    # readers never wait for a writer
    w = idx.writer()
    with idx.searcher() as s:
        assert len(s) == 0
    w.close()


CRASH = textwrap.dedent(
    """
    import os, sys
    sys.path.insert(0, %(src)r)
    from kendi_arama import Index, MergePolicy
    idx = Index.open(%(path)r)
    w = idx.writer(max_buffered_docs=3, merge_policy=MergePolicy(factor=2, base=2))
    for i in range(100, 140):
        w.add("new%%d" %% i, {"title": "crash words %%d" %% i})
    w.delete("old1")
    %(action)s
    os._exit(7)
    """
)


@pytest.mark.parametrize(
    "action",
    [
        "pass",  # killed before the commit: segments were flushed, nothing was published
        "w._flush()",
        "w.merge(1)",  # killed after merging, before the manifest is replaced
        "import kendi_arama.index as ix; ix.write_manifest = lambda *a: os._exit(9); w.commit()",  # killed at the very moment of the commit
    ],
)
def test_a_crash_leaves_the_last_commit_intact(index_dir, action):
    idx = make(index_dir)
    with idx.writer() as w:
        for i in range(10):
            w.add("old%d" % i, {"title": "old words %d" % i})
    snapshot_before = None
    with idx.searcher() as s:
        snapshot_before = keys(s, "old")
    code = CRASH % {"src": SRC, "path": index_dir, "action": action}
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert proc.returncode in (7, 9), proc
    # the index is exactly as it was, and can be used and written again
    idx = Index.open(index_dir)
    with idx.searcher() as s:
        assert keys(s, "old") == snapshot_before == sorted("old%d" % i for i in range(10))
        assert keys(s, "crash") == []
    w = idx.writer()  # opening a writer removes what the crashed one left
    names = files(index_dir)
    assert not [n for n in names if n.endswith(".tmp")]
    manifest = json.load(open(os.path.join(index_dir, "manifest.json")))
    referenced = {r["file"] for r in manifest["segments"]} | {r["deleted"]["file"] for r in manifest["segments"] if r["deleted"]}
    assert {n for n in names if n.endswith(".kseg") or ".del." in n} == referenced
    w.add("fresh", {"title": "fresh start"})
    w.commit()
    w.close()
    assert idx.check() == []
    with idx.searcher() as s:
        assert keys(s, "fresh") == ["fresh"] and len(s) == 11


def test_a_crash_after_the_manifest_is_the_new_state(index_dir):
    idx = make(index_dir)
    with idx.writer() as w:
        w.add("a", {"title": "one"})
    code = CRASH.replace("%(src)r", repr(SRC)).replace("%(path)r", repr(index_dir)).replace("%%", "%").replace("%(action)s", "w.commit()")
    code = code.replace('w.delete("old1")', 'w.delete("a")')
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 7, proc
    with Index.open(index_dir).searcher() as s:
        assert len(s) == 40 and s.get("a") is None and keys(s, "crash")[:2] == ["new100", "new101"]


def test_check_finds_damage(index_dir):
    idx = make(index_dir)
    with idx.writer() as w:
        for i in range(20):
            w.add(str(i), {"title": "some words here %d" % i, "body": "more words " * 20})
        w.delete("3")
    assert idx.check() == []
    seg = [n for n in files(index_dir) if n.endswith(".kseg")][0]
    path = os.path.join(index_dir, seg)
    with open(path, "r+b") as fh:
        fh.seek(os.path.getsize(path) // 2)
        b = fh.read(1)
        fh.seek(-1, 1)
        fh.write(bytes([b[0] ^ 1]))
    problems = idx.check()
    assert problems and seg in problems[0]
    os.remove(path)
    assert any("No such file" in p or "no such file" in p.lower() for p in idx.check())


def test_a_damaged_manifest_or_deletion_list_is_reported(index_dir, tmp_path):
    idx = make(index_dir)
    with idx.writer() as w:
        for i in range(1, 11):
            w.add(str(i), {"title": "x%d" % i})
        w.delete("1")
    dele = [n for n in files(index_dir) if ".del." in n][0]
    path = os.path.join(index_dir, dele)
    assert sorted(read_deleted(path)) == [0]
    data = bytearray(open(path, "rb").read())
    data[13] ^= 1
    open(path, "wb").write(data)
    assert any(dele in p for p in idx.check())
    with pytest.raises(CorruptIndexError):
        idx.searcher()
    write_deleted(path, [0])
    assert idx.check() == []
    with open(os.path.join(index_dir, "manifest.json"), "w") as fh:
        fh.write("{ not json")
    with pytest.raises(CorruptIndexError):
        Index.open(index_dir)
    with open(os.path.join(index_dir, "manifest.json"), "w") as fh:
        fh.write(json.dumps({"format": 99}))
    with pytest.raises(CorruptIndexError):
        Index.open(index_dir)


def test_deleted_documents_do_not_count_in_the_statistics(index_dir):
    idx = make(index_dir, ("body",))
    with idx.writer(auto_merge=False) as w:
        for i in range(10):
            w.add(str(i), {"body": "common " * 3 + ("rare" if i < 2 else "")})
    with idx.searcher() as s:
        total_before = s.stats.total_length
        n_before = s.num_docs
    with idx.writer(auto_merge=False) as w:
        w.delete("0")
        w.delete("5")
    with idx.searcher() as s:
        assert s.num_docs == n_before - 2
        assert s.stats.total_length == total_before - 3 - 4  # doc 0 has 3 + 1 words, doc 5 has 3
        assert s.terms("body", "r") == [("rare", 1)]
        assert s.terms("body") == [("common", 8), ("rare", 1)]
    # the same statistics as an index that never had those documents
    other = Index.create(index_dir + "2", Schema([Field("body")]))
    with other.writer() as w:
        for i in range(10):
            if i not in (0, 5):
                w.add(str(i), {"body": "common " * 3 + ("rare" if i < 2 else "")})
    with idx.searcher() as a, other.searcher() as b:
        for q in ("common", "rare", "common OR rare", '"common common"'):
            got = [(h.key, h.score) for h in a.search(q, scorer=BM25(), limit=None)]
            want = [(h.key, h.score) for h in b.search(q, scorer=BM25(), limit=None)]
            assert got == want


def test_more_documents_than_the_buffer_and_documents_without_words(index_dir):
    idx = make(index_dir)
    with idx.writer(max_buffered_docs=4) as w:
        for i in range(25):
            w.add(str(i), {"title": "" if i % 3 == 0 else "word%d" % i})
        w.add("empty", {})
    with idx.searcher() as s:
        assert len(s) == 26 and s.search("word1").total == 1 and s.stats.total_length == 16
        with pytest.raises(QuerySyntaxError):
            s.search("NOT word1")  # a query cannot start with NOT
