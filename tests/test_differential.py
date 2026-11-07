"""The comparison with SQLite FTS5: random schemas, documents, edits and queries run through both engines.

``KENDI_FUZZ=800 pytest tests/test_differential.py`` runs 800 scenarios (the default is a quick 120); a failing seed is repeated with
``KENDI_FUZZ_START=<seed> KENDI_FUZZ=1``.
"""
import random
import sqlite3

import pytest

import fuzzgen
import harness
from conftest import fuzz_count, fuzz_start, needs_fts5
from oracle import Fts5

from kendi_arama import Field, QuerySyntaxError, Schema
from kendi_arama.query import parse_query

BATCHES = 6


def seeds(batch):
    total = fuzz_count(120)
    start = fuzz_start()
    return range(start + batch, start + total, BATCHES)


@needs_fts5
@pytest.mark.parametrize("batch", range(BATCHES))
def test_random_scenarios_agree_with_fts5(batch):
    totals = {"ok": 0, "error": 0, "docs": 0, "phantom": 0, "score_skipped": 0}
    for seed in seeds(batch):
        for k, v in harness.run_scenario(seed).items():
            totals[k] += v
    if totals["ok"] > 200:
        # the explained and the skipped differences are exceptions, they must stay rare
        assert totals["phantom"] <= 0.03 * totals["ok"], totals
        assert totals["score_skipped"] <= 0.03 * totals["ok"], totals


@needs_fts5
@pytest.mark.parametrize("batch", range(BATCHES))
def test_highlight_agrees_with_fts5(batch):
    compared = tolerated = 0
    for seed in seeds(batch):
        st = harness.run_highlight_scenario(seed + 1_000_000)
        compared += st["compared"]
        tolerated += st["tolerated"]
    if compared > 500:
        assert tolerated <= 0.02 * compared


@needs_fts5
def test_the_syntax_that_fts5_accepts_is_the_syntax_we_accept():
    rng = random.Random(fuzz_start() + 77)
    fields = ["title", "body", "tags"]
    schema = Schema([Field(f) for f in fields])
    fts = Fts5(fields)
    vocab = fuzzgen.make_vocab(rng, 30)
    accepted = rejected = 0
    for i in range(max(600, fuzz_count(120) * 10)):
        text = fuzzgen.make_query(rng, vocab, fields)
        if i % 2:
            text = fuzzgen.damage(rng, text)
        try:
            fts.match(text)
            want = True
        except sqlite3.OperationalError:
            want = False
        try:
            parse_query(text, schema, fuzzy=False)
            got = True
        except QuerySyntaxError:
            got = False
        assert got == want, "%r: FTS5 %s, we %s" % (text, "accepts" if want else "rejects", "accept" if got else "reject")
        accepted += want
        rejected += not want
    assert accepted > 100 and rejected > 100  # both kinds were really tried


@needs_fts5
def test_hand_picked_queries_agree_with_fts5():
    """Queries that explore the corners of the language, with the answers of FTS5 on a small fixed table."""
    fields = ["title", "body"]
    docs = [
        ("a b", "x"), ("a x b", ""), ("a x y b", ""), ("a x y z b", ""), ("b a", ""), ("a a b", ""), ("a b a b", ""), ("a b c", ""), ("c b a", ""), ("b c a", ""),
        ("a x x x x x x x x x x b", ""), ("a x x x x x x x x x b", ""), ("a b c d", ""), ("a x b y c", ""), ("x a b y c", ""), ("one two three four", "a b"),
        ("a", "b"), ("b", "a"), ("p q", "a r"), ("running runs ran", "foxes fox"),
    ]
    fts = Fts5(fields)
    import os, tempfile
    from kendi_arama import Index

    d = tempfile.mkdtemp()
    idx = Index.create(os.path.join(d, "i"), Schema([Field(f) for f in fields]))
    with idx.writer() as w:
        for i, (t, b) in enumerate(docs, 1):
            fts.insert(i, {"title": t, "body": b})
            w.add(str(i), {"title": t, "body": b})
    queries = [
        'NEAR(a b)', 'NEAR(a b, 0)', 'NEAR(a b, 1)', 'NEAR(a b, 2)', 'NEAR(b a, 0)', 'NEAR(a b c, 1)', 'NEAR(a b c)', 'NEAR(a a, 0)', 'NEAR("a x" b, 1)',
        'NEAR(a "x y", 0)', 'NEAR(a)', 'NEAR(a b c, 2)', 'NEAR("a b" b)', 'NEAR(a* b)', 'NEAR(a b, 0) c', '"a b" + c', 'a + b', 'a + b + c', 'a b OR c',
        'a OR b c', 'a NOT b', 'a NOT b NOT c', 'a NOT (b c)', '^a', '^"a x"', '^a b', 'a ^b', 'a*', '"a b"*', 'a b*', 'title:a', 'body:a', 'title: a b',
        'title:(a b)', '{title body}:a', '-title:a', 'title:NEAR(a b)', 'title:^a', 'a AND b OR c AND d', 'a AND (b OR c)', 'run*', 'running', 'fox*', 'a "" b',
        'a ""', '"" a', 'a AND ""', '"" AND a', 'a OR ""', 'a NOT ""', '"" NOT a', '"!!"', 'a "!!" b', 'TITLE:a', '{TITLE Body}:a', 'a AND NEAR("!!")', 'NEAR', 'NEAR a', 'NEAR (a b)', 'NEAR"a b"', 'a NEAR', 'NEAR NEAR b', 'title:NEAR"a b"',
    ]
    with idx.searcher() as s:
        for q in queries:
            want = fts.match(q)
            got = sorted(int(k) for k in s.search(q, limit=None, fuzzy=False).keys())
            assert got == want, q
