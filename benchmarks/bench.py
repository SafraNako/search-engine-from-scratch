"""Speed of this engine against SQLite FTS5 on a real text: the docstrings of the standard library of the running Python.

    python benchmarks/bench.py 5000        # 5000 documents (the default is 5000)

Both engines index the same texts and answer the same queries with the best ten documents by BM25.  FTS5 is C code and is much
faster; the point is to see how far a pure Python engine is, and where the time goes.
"""
from __future__ import annotations

import ast
import os
import shutil
import sqlite3
import statistics
import sys
import sysconfig
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kendi_arama import BM25, Field, Index, Schema  # noqa: E402


def corpus(limit):
    root = sysconfig.get_paths()["stdlib"]
    docs = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in ("test", "tests", "site-packages", "idlelib", "turtledemo", "__pycache__") and not d.startswith("lib2to3"))
        for name in sorted(filenames):
            if not name.endswith(".py"):
                continue
            try:
                tree = ast.parse(open(os.path.join(dirpath, name), encoding="utf-8").read())
            except (SyntaxError, UnicodeDecodeError, ValueError):
                continue
            mod = os.path.relpath(os.path.join(dirpath, name), root)
            for node in ast.walk(tree):
                if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    text = ast.get_docstring(node)
                    if text and len(text.split()) >= 8:
                        docs.append(("%s:%s:%d" % (mod, getattr(node, "name", "module"), len(docs)), getattr(node, "name", "module"), text))
                        if len(docs) >= limit:
                            return docs
    return docs


QUERIES = [
    ("common word", "the"),
    ("rare word", "tokenizer"),
    ("two words (AND)", "file system"),
    ("three words (OR)", "socket OR thread OR signal"),
    ("phrase", '"returns the"'),
    ("prefix", "inter*"),
    ("NEAR", "NEAR(open file, 5)"),
    ("NOT", "string NOT bytes"),
    ("field", "title:path"),
]


def timeit(fn, repeat):
    times = []
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t)
    return statistics.median(times)


def main(n=5000):
    docs = corpus(n)
    n = len(docs)
    words = sum(len(t.split()) for _, _, t in docs)
    print("%d documents, %d words, from the standard library of Python %s, SQLite %s" % (n, words, sys.version.split()[0], sqlite3.sqlite_version))
    work = tempfile.mkdtemp(prefix="bench-")
    try:
        t = time.perf_counter()
        index = Index.create(os.path.join(work, "idx"), Schema([Field("title"), Field("body")]))
        with index.writer(max_buffered_docs=max(n // 6, 1), auto_merge=False) as w:  # six segments, to have something to merge
            for key, title, text in docs:
                w.add(key, {"title": title, "body": text})
        ours_index = time.perf_counter() - t
        with index.searcher() as s0:
            segments = len(s0.segs)
        t = time.perf_counter()
        with index.writer() as w:
            w.merge(1)
        merge = time.perf_counter() - t
        size = sum(os.path.getsize(os.path.join(work, "idx", f)) for f in os.listdir(os.path.join(work, "idx")))

        db = sqlite3.connect(os.path.join(work, "fts.db"))
        db.execute("create virtual table t using fts5(title, body)")
        t = time.perf_counter()
        db.executemany("insert into t(rowid, title, body) values (?, ?, ?)", [(i, title, text) for i, (_, title, text) in enumerate(docs)])
        db.commit()
        fts_index = time.perf_counter() - t
        fts_size = os.path.getsize(os.path.join(work, "fts.db"))

        print("\n%-24s %12s %12s %8s" % ("", "this engine", "FTS5", "ratio"))
        print("%-24s %10.0f ms %10.0f ms %7.0fx" % ("indexing", ours_index * 1000, fts_index * 1000, ours_index / fts_index))
        print("%-24s %10.0f KiB %9.0f KiB %7.2fx" % ("size on disk", size / 1024, fts_size / 1024, size / fts_size))
        print("%-24s %10.0f ms   (%d segments -> 1)" % ("merge", merge * 1000, segments))
        print()
        scorer = BM25(idf="fts5")
        with index.searcher() as s:
            for label, q in QUERIES:
                total = s.search(q, limit=10, scorer=scorer).total
                a = timeit(lambda: s.search(q, limit=10, scorer=scorer), 5)
                b = timeit(lambda: db.execute("select rowid, bm25(t) from t where t match ? order by rank limit 10", (q,)).fetchall(), 5)
                print("%-24s %10.2f ms %10.2f ms %7.0fx   (%d matches)" % (label, a * 1000, b * 1000, a / b if b else float("inf"), total))
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 5000)
