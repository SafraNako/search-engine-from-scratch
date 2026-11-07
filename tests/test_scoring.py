import math
import random

import pytest

import fuzzgen
from kendi_arama import BM25, Field, Index, Schema, SearchError, TfIdf, Unicode61, scorer_from_name
from kendi_arama.scoring import Stats


def searcher(path, docs, fields=("text",), **kw):
    idx = Index.create(path, Schema([Field(f) for f in fields]))
    with idx.writer(**kw) as w:
        for key, d in docs.items():
            w.add(key, d if isinstance(d, dict) else {fields[0]: d})
    return idx.searcher()


def test_bm25_by_hand(index_dir):
    docs = {"a": "fox fox dog", "b": "fox cat cat cat cat", "c": "dog", "d": "bird", "e": "bird bird"}
    with searcher(index_dir, docs) as s:
        N, avgdl = 5, (3 + 5 + 1 + 1 + 2) / 5
        df = 2  # fox
        for idf_kind, idf in (("lucene", math.log(1 + (N - df + 0.5) / (df + 0.5))), ("fts5", math.log((N - df + 0.5) / (df + 0.5)))):
            for k1, b in ((1.2, 0.75), (2.0, 0.3), (0.5, 1.0), (1.2, 0.0)):
                res = s.search("fox", limit=None, scorer=BM25(k1, b, idf_kind))
                got = {h.key: h.score for h in res}
                for key, tf, dl in (("a", 2, 3), ("b", 1, 5)):
                    want = idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))
                    assert got[key] == pytest.approx(want, rel=1e-12)


def test_bm25_properties(index_dir):
    docs = {"one": "x " + "pad " * 20, "two": "x x " + "pad " * 20, "five": "x x x x x " + "pad " * 20, "short": "x", "other": "y"}
    with searcher(index_dir, docs) as s:
        def score(query, key, **kw):
            return {h.key: h.score for h in s.search(query, limit=None, scorer=BM25(**kw))}[key]

        # more occurrences count more, but less and less (saturation: never more than k1 + 1 times the idf)
        a, b, c = score("x", "one"), score("x", "two"), score("x", "five")
        assert a < b < c and (c - b) < (b - a) * 3
        idf = BM25().weight(4, Stats(5, 1))
        assert score("x", "five", k1=1.2, b=0.0) < idf * 2.2
        assert score("x", "five", k1=100, b=0.0) > idf * 5 * 101 / 105 * 0.99  # a big k1 makes the frequency count almost linearly
        # a short document is worth more than a long one with the same count; with b = 0 the length does not matter
        assert score("x", "short") > score("x", "one")
        assert score("x", "short", b=0.0) == pytest.approx(score("x", "one", b=0.0))
        # k1 = 0: only whether the word is there counts
        assert score("x", "five", k1=0.0, b=0.0) == pytest.approx(score("x", "one", k1=0.0, b=0.0))
        # a rarer word is worth more
        assert BM25().weight(1, Stats(100, 1000)) > BM25().weight(50, Stats(100, 1000)) > 0
        # fts5: below zero is lifted to 1e-6
        assert BM25(idf="fts5").weight(60, Stats(100, 1000)) == 1e-6 and BM25().weight(60, Stats(100, 1000)) > 0
    with pytest.raises(SearchError):
        BM25(idf="what")


def test_a_phrase_counts_as_one_unit(index_dir):
    docs = {"1": "new york is big", "2": "york new new york new york", "3": "new and york", "4": "something else", "5": "more filler words", "6": "even more"}
    with searcher(index_dir, docs) as s:
        ex = s.explain('"new york"', "2")
        assert [(l.text, l.df, l.freq) for l in ex.leaves] == [("new york", 2, 2)]  # the phrase has its own frequency and document count
        two = {h.key: h.score for h in s.search('"new york"', limit=None)}
        assert two["2"] > two["1"]


def textbook(path, scorer, query):
    """Three novels of the book of Manning, Raghavan and Schütze (Introduction to Information Retrieval, figure 6.9)."""
    counts = {"SaS": (115, 10, 2), "PaP": (58, 7, 0), "WH": (20, 11, 6)}  # affection, jealous, gossip
    docs = {k: " ".join(["affection"] * a + ["jealous"] * j + ["gossip"] * g) for k, (a, j, g) in counts.items()}
    with searcher(path, docs) as s:
        return {h.key: h.score for h in s.search(query, limit=None, scorer=scorer)}


def test_tfidf_cosine_of_the_textbook(index_dir):
    scorer = TfIdf(tf="natural", idf="none", norm="cosine", doc_idf=False, normalize_query=True)
    # figure 6.13: the length-normalised term frequencies of the three novels
    got = textbook(index_dir, scorer, "affection")
    assert {k: round(v, 3) for k, v in got.items()} == {"SaS": 0.996, "PaP": 0.993, "WH": 0.847}
    got = textbook(index_dir + "1", scorer, "jealous")
    assert {k: round(v, 3) for k, v in got.items()} == {"SaS": 0.087, "PaP": 0.120, "WH": 0.466}
    got = textbook(index_dir + "2", scorer, "gossip")
    assert {k: round(v, 3) for k, v in got.items()} == {"SaS": 0.017, "WH": 0.254}
    # the query "jealous gossip": a vector (1, 1) / sqrt 2 against these three, 0.074 0.085 0.509 in the book
    got = textbook(index_dir + "3", scorer, "jealous OR gossip")
    assert got == {"SaS": pytest.approx(0.074, abs=6e-4), "PaP": pytest.approx(0.085, abs=6e-4), "WH": pytest.approx(0.509, abs=6e-4)}  # the book rounds in between
    ranked = sorted(got, key=got.get, reverse=True)
    assert ranked == ["WH", "PaP", "SaS"]


def test_tfidf_with_idf_by_hand(index_dir):
    docs = {"a": "red red blue", "b": "red green", "c": "blue blue blue green", "d": "yellow"}
    N = 4
    with searcher(index_dir, docs) as s:
        scorer = TfIdf(tf="log", idf="classic", norm="cosine", doc_idf=False, normalize_query=True)
        got = {h.key: h.score for h in s.search("red OR blue", limit=None, scorer=scorer)}
        idf = {"red": math.log(N / 2), "blue": math.log(N / 2)}
        qnorm = math.sqrt(sum(v * v for v in idf.values()))
        for key, tf in (("a", {"red": 2, "blue": 1}), ("b", {"red": 1, "green": 1}), ("c", {"blue": 3, "green": 1})):
            w = {t: 1 + math.log(f) for t, f in tf.items()}
            dnorm = math.sqrt(sum(x * x for x in w.values()))
            want = sum(idf[t] * w[t] for t in idf if t in w) / dnorm / qnorm
            assert got[key] == pytest.approx(want, rel=1e-12)
        # the same word with doc_idf: idf on both sides
        scorer = TfIdf(tf="natural", idf="smooth", norm="none", doc_idf=True, normalize_query=False)
        got = {h.key: h.score for h in s.search("red", limit=None, scorer=scorer)}
        i = math.log((1 + N) / (1 + 2)) + 1
        assert got == {"a": pytest.approx(i * i * 2), "b": pytest.approx(i * i)}


def test_tfidf_options(index_dir):
    docs = {"a": "x x x x", "b": "x y", "c": "z z"}
    with searcher(index_dir, docs) as s:
        def one(**kw):
            return {h.key: h.score for h in s.search("x", limit=None, scorer=TfIdf(normalize_query=False, norm="none", idf="none", **kw))}

        assert one(tf="natural") == {"a": 4, "b": 1}
        assert one(tf="boolean") == {"a": 1, "b": 1}
        assert one(tf="sqrt") == {"a": pytest.approx(2.0), "b": 1}
        assert one(tf="log") == {"a": pytest.approx(1 + math.log(4)), "b": 1}
        by_length = {h.key: h.score for h in s.search("x", limit=None, scorer=TfIdf(tf="natural", idf="none", norm="length", normalize_query=False))}
        assert by_length == {"a": pytest.approx(4 / 2), "b": pytest.approx(1 / math.sqrt(2))}
        for bad in (dict(tf="x"), dict(idf="x"), dict(norm="x")):
            with pytest.raises(SearchError):
                TfIdf(**bad)
    assert isinstance(scorer_from_name("bm25"), BM25) and scorer_from_name("bm25-fts5").idf == "fts5" and scorer_from_name("tfidf-sklearn").doc_idf
    with pytest.raises(SearchError):
        scorer_from_name("magic")


def test_tfidf_norms_do_not_depend_on_the_segments(index_dir):
    rng = random.Random(7)
    vocab = fuzzgen.make_vocab(rng, 30)
    docs = {str(i): fuzzgen.make_text(rng, vocab, 30) for i in range(60)}
    for i in (3, 9, 33):
        docs[str(i)] = ""
    one = searcher(index_dir + "a", docs)
    many = searcher(index_dir + "b", docs, max_buffered_docs=7, auto_merge=False)
    with one, many:
        for scorer in (TfIdf(), TfIdf(tf="log", idf="smooth", doc_idf=True), TfIdf(tf="sqrt", idf="lucene", norm="length", doc_idf=True)):
            for q in ("the OR cat OR run", "fox"):
                a = {h.key: h.score for h in one.search(q, limit=None, scorer=scorer)}
                b = {h.key: h.score for h in many.search(q, limit=None, scorer=scorer)}
                assert a.keys() == b.keys() and all(a[k] == pytest.approx(b[k], rel=1e-12) for k in a)


# ---------------------------------------------------------------------------------------------------------- scikit-learn


@pytest.mark.parametrize(
    "sublinear,norm_query,ours",
    [
        (True, True, TfIdf(tf="log", idf="smooth", norm="cosine", doc_idf=True, normalize_query=True)),
        (False, True, TfIdf(tf="natural", idf="smooth", norm="cosine", doc_idf=True, normalize_query=True)),
    ],
)
def test_tfidf_cosine_equals_scikit_learn(index_dir, sublinear, norm_query, ours):
    sklearn_text = pytest.importorskip("sklearn.feature_extraction.text")
    rng = random.Random(5)
    vocab = fuzzgen.make_vocab(rng, 60)
    analyzer = Unicode61()
    docs = [fuzzgen.make_text(rng, vocab, 40) for _ in range(250)]
    idx = Index.create(index_dir, Schema([Field("text", analyzer)]))
    with idx.writer(max_buffered_docs=60) as w:  # several segments
        for i, t in enumerate(docs):
            w.add(str(i), {"text": t})
        w.delete("7")
        w.delete("100")
    kept = [i for i in range(len(docs)) if i not in (7, 100)]
    vec = sklearn_text.TfidfVectorizer(tokenizer=analyzer.terms, lowercase=False, token_pattern=None, smooth_idf=True, sublinear_tf=sublinear, norm="l2")
    X = vec.fit_transform([docs[i] for i in kept])
    compared = 0
    with idx.searcher() as s:
        for _ in range(40):
            terms = list(dict.fromkeys(t for x in (fuzzgen.zipf_word(rng, vocab) for _ in range(rng.randint(1, 4))) for t in analyzer.terms(x)))
            if not terms:
                continue
            sims = (X @ vec.transform([" ".join(terms)]).T).toarray().ravel()
            got = {int(h.key): h.score for h in s.search(" OR ".join(terms), limit=None, scorer=ours)}
            for row, i in enumerate(kept):
                if sims[row] > 1e-12:
                    assert got[i] == pytest.approx(sims[row], abs=1e-12), (terms, i)
                    compared += 1
                else:
                    assert got.get(i, 0.0) < 1e-12
    assert compared > 500


def test_tfidf_without_normalisation_equals_scikit_learn(index_dir):
    sklearn_text = pytest.importorskip("sklearn.feature_extraction.text")
    rng = random.Random(8)
    vocab = fuzzgen.make_vocab(rng, 40)
    analyzer = Unicode61()
    docs = [fuzzgen.make_text(rng, vocab, 30) for _ in range(120)]
    idx = Index.create(index_dir, Schema([Field("text", analyzer)]))
    with idx.writer(max_buffered_docs=50) as w:
        for i, t in enumerate(docs):
            w.add(str(i), {"text": t})
    vec = sklearn_text.TfidfVectorizer(tokenizer=analyzer.terms, lowercase=False, token_pattern=None, smooth_idf=True, sublinear_tf=True, norm=None)
    X = vec.fit_transform(docs)
    ours = TfIdf(tf="log", idf="smooth", norm="none", doc_idf=True, normalize_query=False)
    with idx.searcher() as s:
        for _ in range(30):
            terms = list(dict.fromkeys(t for x in (fuzzgen.zipf_word(rng, vocab) for _ in range(rng.randint(1, 3))) for t in analyzer.terms(x)))
            if not terms:
                continue
            sims = (X @ vec.transform([" ".join(terms)]).T).toarray().ravel()
            got = {int(h.key): h.score for h in s.search(" OR ".join(terms), limit=None, scorer=ours)}
            for i, sc in enumerate(sims):
                if sc > 1e-12:
                    assert got[i] == pytest.approx(sc, rel=1e-9)
