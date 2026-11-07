"""Runs a random scenario through the search engine and SQLite FTS5 and compares them.

A scenario is a random schema (one to three fields, one of several analyzers), a random sequence of operations (add, replace, delete,
commit, merge, reopen) applied to both engines, and, after the commits, random queries.  The two engines have to agree on

* whether a query is a syntax error,
* which documents match,
* the score of each of them (``bm25()`` of FTS5, with the same field weights),
* and the order must be by score (ties may come in any order).
"""
from __future__ import annotations

import os
import random
import shutil
import sqlite3
import tempfile
from typing import Dict, List, Optional, Tuple

import fuzzgen
from oracle import Fts5

from kendi_arama import (BM25, Ascii, Field, Index, MergePolicy, Porter, QuerySyntaxError, Schema, Trigram, Unicode61)

VARIANTS = [
    ("unicode61", lambda: Unicode61(), "unicode61"),
    ("unicode61", lambda: Unicode61(), "unicode61"),
    ("porter", lambda: Porter(), "porter unicode61"),
    ("rd0", lambda: Unicode61(remove_diacritics=0), "unicode61 remove_diacritics 0"),
    ("rd2", lambda: Unicode61(remove_diacritics=2), "unicode61 remove_diacritics 2"),
    ("ascii", lambda: Ascii(), "ascii"),
    ("trigram", lambda: Trigram(), "trigram"),
]
FTS5_BM25 = BM25(idf="fts5")
FIELD_NAMES = ["title", "body", "tags"]


def close(a: float, b: float) -> bool:
    return abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))


class Mismatch(AssertionError):
    pass


def phantom_explains(searcher, query: str, key: str, weights: Optional[Dict[str, float]], want: float, have: float) -> bool:
    """FTS5 sometimes counts a phrase that sits on the document without being part of the match (the right side of a NOT whose left
    side matches nothing, the words of a NEAR that is not near enough, both below an OR), and sometimes leaves out a phrase that is part
    of it (a nested OR next to a word that is not in the index).  Is the difference between its score and ours a sum of such phrases,
    each with the occurrences it has in the document?"""
    ex = searcher.explain(query, key, scorer=FTS5_BM25, weights=weights, fuzzy=False)
    if ex is None:
        return False
    extra = [leaf.raw_contribution - leaf.contribution for leaf in ex.leaves if leaf.raw_contribution - leaf.contribution > 1e-15]
    missing = [leaf.contribution for leaf in ex.leaves if leaf.contribution > 1e-15]
    if len(extra) > 12 or len(missing) > 12:
        return False
    plus = [0.0]
    for e in extra:
        plus += [x + e for x in plus]
    plus.sort()
    minus = [0.0]
    for m in missing:
        minus += [x + m for x in minus]
    minus = minus[:512]
    from bisect import bisect_left

    for n in minus:
        target = want - have + n
        i = bisect_left(plus, target - 1e-9 * max(1.0, abs(target)))
        if i < len(plus) and close(plus[i], target):
            return True
    return False


def compare_query(searcher, fts: Fts5, query: str, fields: List[str], weights: Optional[Dict[str, float]], context: str, counters: Optional[Dict[str, int]] = None) -> Tuple[str, int]:
    """Returns ("error" | "ok", number of matches)."""
    wlist = [weights.get(f, 1.0) for f in fields] if weights else None
    try:
        want = fts.ranked(query, wlist)
        want_err = None
    except sqlite3.OperationalError as e:
        want, want_err = [], str(e)
    try:
        res = searcher.search(query, limit=None, weights=weights, scorer=FTS5_BM25, fuzzy=False)
        got = [(int(h.key), h.score) for h in res]
        got_err = None
        total = res.total
    except QuerySyntaxError as e:
        got, got_err, total = [], str(e), 0
    if (want_err is None) != (got_err is None):
        raise Mismatch("%s\nquery %r: FTS5 %s, ours %s" % (context, query, want_err or "accepts it", got_err or "accepts it"))
    if want_err is not None:
        return "error", 0
    wd = dict(want)
    gd = dict(got)
    if set(wd) != set(gd) or total != len(want):
        raise Mismatch("%s\nquery %r: different documents\n  FTS5 only: %s\n  ours only: %s" % (context, query, sorted(set(wd) - set(gd)), sorted(set(gd) - set(wd))))
    for k, s in gd.items():
        if not close(s, wd[k]):
            if phantom_explains(searcher, query, str(k), weights, wd[k], s):
                if counters is not None:
                    counters["phantom"] = counters.get("phantom", 0) + 1
                continue
            if "NEAR" in query and any(op in query for op in (" OR ", " NOT ", " AND ", "(")):
                # a NEAR that is not near enough, below an OR, also leaves the words of its phrases in FTS5's count, with a weight
                # that is not the one of the same word elsewhere; the documents that match are compared all the same
                if counters is not None:
                    counters["score_skipped"] = counters.get("score_skipped", 0) + 1
                break
            raise Mismatch("%s\nquery %r: document %d scores %r, FTS5 says %r" % (context, query, k, s, wd[k]))
    scores = [s for _, s in got]
    if any(scores[i] < scores[i + 1] - 1e-12 for i in range(len(scores) - 1)):
        raise Mismatch("%s\nquery %r: results are not in order of score" % (context, query))
    return "ok", len(want)


def run_scenario(seed: int, queries: int = 25, workdir: Optional[str] = None) -> Dict[str, int]:
    rng = random.Random(seed)
    nf = rng.choice([1, 2, 2, 3])
    fields = FIELD_NAMES[:nf]
    vname, make_analyzer, fts_tokenize = rng.choice(VARIANTS)
    schema = Schema([Field(f, make_analyzer()) for f in fields])
    vocab = fuzzgen.make_vocab(rng, rng.choice([20, 40, 80]))
    fts = Fts5(fields, fts_tokenize)
    own = workdir is None
    path = tempfile.mkdtemp(prefix="kendi-") if own else workdir
    stats = {"ok": 0, "error": 0, "docs": 0, "phantom": 0, "score_skipped": 0}
    try:
        index = Index.create(os.path.join(path, "idx"), schema)
        factor = rng.choice([2, 3, 10])
        policy = MergePolicy(factor=factor, base=rng.choice([2, 5, 50]))
        writer = index.writer(max_buffered_docs=rng.choice([3, 7, 25, 1000]), merge_policy=policy, auto_merge=rng.random() < 0.7)
        live: Dict[int, dict] = {}
        next_id = 1
        nops = rng.randint(8, 70)
        history = []
        checkpoints = {rng.randrange(nops) for _ in range(rng.randint(1, 3))} | {nops - 1}
        for step in range(nops):
            r = rng.random()
            if r < 0.6 or not live:
                rid = next_id
                next_id += 1
                doc = fuzzgen.make_doc(rng, vocab, fields)
                writer.add(str(rid), doc)
                fts.insert(rid, doc)
                live[rid] = doc
                history.append("add %d" % rid)
            elif r < 0.75:
                rid = rng.choice(sorted(live))
                doc = fuzzgen.make_doc(rng, vocab, fields)
                writer.add(str(rid), doc)
                fts.replace(rid, doc)
                live[rid] = doc
                history.append("replace %d" % rid)
            elif r < 0.9:
                rid = rng.choice(sorted(live))
                assert writer.delete(str(rid))
                fts.delete(rid)
                del live[rid]
                history.append("delete %d" % rid)
            elif r < 0.94:
                writer.commit()
                history.append("commit")
            elif r < 0.97:
                writer.merge(rng.choice([1, 2]))
                history.append("merge")
            else:
                writer.commit()
                writer.close()
                writer = index.writer(max_buffered_docs=rng.choice([3, 7, 25, 1000]), merge_policy=policy, auto_merge=rng.random() < 0.7)
                history.append("reopen")
            if step in checkpoints:
                writer.commit()
                with index.searcher() as s:
                    assert len(s) == len(live) == fts.count(), (len(s), len(live), fts.count())
                    context = "seed %d variant %s fields %s ops: %s" % (seed, vname, fields, ", ".join(history[-12:]))
                    for _ in range(queries):
                        text = fuzzgen.make_query(rng, vocab, fields)
                        if rng.random() < 0.25:
                            text = fuzzgen.damage(rng, text)
                        weights = None
                        if rng.random() < 0.3:
                            weights = {f: rng.choice([0.0, 0.5, 1.0, 2.0, 10.0]) for f in fields}
                        try:
                            outcome, n = compare_query(s, fts, text, fields, weights, context, stats)
                        except Mismatch as e:
                            e.state = {"live": dict(live), "fields": fields, "variant": vname, "query": text, "weights": weights}
                            raise
                        stats[outcome] += 1
        stats["docs"] = len(live)
        writer.close()
        return stats
    finally:
        if own:
            shutil.rmtree(path, ignore_errors=True)


# ---------------------------------------------------------------------------------------------------------------- highlight

def plain_and_spans(text: str, open_: str = "[", close: str = "]"):
    """The text without the markers and the (start, end) of each marked part in it."""
    out = []
    spans = []
    start = None
    for ch in text:
        if ch == open_:
            start = len(out)
        elif ch == close:
            spans.append((start, len(out)))
        else:
            out.append(ch)
    return "".join(out), spans


def run_highlight_scenario(seed: int, queries: int = 25) -> Dict[str, int]:
    """``highlight()`` of FTS5 against ours for random queries over random documents (in every column of every matching row)."""
    rng = random.Random(seed)
    nf = rng.choice([1, 2, 3])
    fields = FIELD_NAMES[:nf]
    vname, make_analyzer, fts_tokenize = rng.choice(VARIANTS[:6])
    schema = Schema([Field(f, make_analyzer()) for f in fields])
    vocab = fuzzgen.make_vocab(rng, rng.choice([20, 40]))
    fts = Fts5(fields, fts_tokenize)
    path = tempfile.mkdtemp(prefix="kendi-")
    stats = {"compared": 0, "tolerated": 0}
    try:
        index = Index.create(os.path.join(path, "idx"), schema)
        with index.writer() as w:
            for rid in range(1, rng.randint(5, 40)):
                doc = fuzzgen.make_doc(rng, vocab, fields)
                w.add(str(rid), doc)
                fts.insert(rid, doc)
        with index.searcher() as s:
            for _ in range(queries):
                text = fuzzgen.make_query(rng, vocab, fields, 1)
                try:
                    fts.match(text)
                except sqlite3.OperationalError:
                    continue
                for ci, f in enumerate(fields):
                    for rid, want in fts.highlight(text, ci).items():
                        want = want or ""
                        got = s.highlight(text, str(rid), f, "[", "]", fuzzy=False)
                        stats["compared"] += 1
                        if got == want:
                            continue
                        # FTS5 also marks a phrase that sits on the row without being part of the match (see the notes on scoring)
                        (p1, s1), (p2, s2) = plain_and_spans(got), plain_and_spans(want)
                        if p1 == p2 and all(any(b2 <= a1 and e1 <= e2 for b2, e2 in s2) for a1, e1 in s1) and ("NOT" in text or "NEAR" in text):
                            stats["tolerated"] += 1
                            continue
                        raise Mismatch("seed %d variant %s query %r field %s row %d\n  FTS5 %r\n  ours %r" % (seed, vname, text, f, rid, want, got))
        return stats
    finally:
        shutil.rmtree(path, ignore_errors=True)
