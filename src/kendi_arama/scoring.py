"""Ranking functions: how much a document is worth for a query.

A query is a list of phrases.  For each phrase the scorer gets its *weight* from the number of documents that contain it (rare phrases
weigh more, ``idf``), and for each document the *frequency* of the phrase in it (several occurrences count more, but less and less)
and the length of the document (a hit in a short text means more than one in a long text).

``BM25`` follows the function of SQLite FTS5 exactly (``bm25()``), so that the tests can compare the numbers.  ``TfIdf`` is the
classic vector space model: the query and the document are weighted vectors of terms and the score is their cosine.
"""
from __future__ import annotations

import math
from typing import Callable, Sequence

from .errors import SearchError


class Stats:
    """The numbers of the whole index (live documents only) that a scorer needs."""

    def __init__(self, num_docs: int, total_length: int) -> None:
        self.num_docs = num_docs
        self.total_length = total_length
        self.avg_length = total_length / num_docs if num_docs else 0.0


class Scorer:
    name = "scorer"
    needs_norms = False

    def weight(self, df: int, stats: Stats) -> float:
        """The importance of a phrase that is in ``df`` documents."""
        raise NotImplementedError

    def bind(self, weights: Sequence[float], stats: Stats) -> Callable[[Sequence[float], int, float], float]:
        """A function ``score(freqs, length, norm)`` for one search: the frequency of every phrase in a document, the length of the
        document in words and its vector norm (only for scorers with ``needs_norms``)."""
        raise NotImplementedError


class BM25(Scorer):
    """Okapi BM25.  ``idf="lucene"`` (the default) is ``log(1 + (N - n + 0.5) / (n + 0.5))``, always positive; ``idf="fts5"`` is the
    variant of SQLite (``log((N - n + 0.5) / (n + 0.5))``, never below 1e-6, so a phrase in more than half of the documents hardly
    counts), which makes the scores equal to the ``bm25()`` of FTS5."""

    name = "bm25"

    def __init__(self, k1: float = 1.2, b: float = 0.75, idf: str = "lucene") -> None:
        if idf not in ("fts5", "lucene"):
            raise SearchError("idf must be 'fts5' or 'lucene'")
        self.k1 = k1
        self.b = b
        self.idf = idf

    def weight(self, df: int, stats: Stats) -> float:
        n = stats.num_docs
        if self.idf == "lucene":
            return math.log(1.0 + (n - df + 0.5) / (df + 0.5))
        w = math.log((n - df + 0.5) / (df + 0.5))
        return w if w > 0 else 1e-6

    def bind(self, weights: Sequence[float], stats: Stats) -> Callable[[Sequence[float], int, float], float]:
        k1p = self.k1 + 1.0
        base = self.k1 * (1.0 - self.b)
        per_word = self.k1 * self.b / stats.avg_length if stats.avg_length else 0.0
        ws = list(weights)

        def score(freqs: Sequence[float], length: int, norm: float) -> float:
            add = base + per_word * length
            total = 0.0
            for w, f in zip(ws, freqs):
                if f:
                    total += w * (f * k1p) / (f + add)
            return total

        return score

    def __repr__(self) -> str:
        return "BM25(k1=%r, b=%r, idf=%r)" % (self.k1, self.b, self.idf)


_TF = {
    "natural": lambda f: f,
    "log": lambda f: 1.0 + math.log(f),
    "sqrt": math.sqrt,
    "boolean": lambda f: 1.0,
}
_IDF = {
    "classic": lambda n, df: math.log(n / df) if df else 0.0,
    "smooth": lambda n, df: math.log((1.0 + n) / (1.0 + df)) + 1.0,  # scikit-learn
    "lucene": lambda n, df: 1.0 + math.log(n / (df + 1.0)),  # the classic similarity of Lucene
    "prob": lambda n, df: max(0.0, math.log((n - df) / df)) if df and n > df else 0.0,
    "none": lambda n, df: 1.0,
}


class TfIdf(Scorer):
    """The vector space model, in the notation of the book by Manning, Raghavan and Schütze (SMART): the frequency of a term is
    transformed (``tf``), multiplied by its rarity (``idf``) and the vector of the document is divided by its length (``norm``).

    * ``tf``: ``"natural"`` (the count), ``"log"`` (1 + ln count), ``"sqrt"``, ``"boolean"``
    * ``idf``: ``"classic"`` (ln N/df), ``"smooth"`` (ln((1+N)/(1+df)) + 1, scikit-learn), ``"lucene"``, ``"prob"``, ``"none"``
    * ``doc_idf``: weight the terms of the document by idf as well (scikit-learn does); the default weights only the query, which is
      the ``lnc.ltc`` scheme of the book and lets the norm of a document be computed without knowing the other documents
    * ``norm``: ``"cosine"`` divides by the length of the document vector, ``"length"`` by the square root of the number of words
      (what Lucene's classic similarity does), ``"none"`` leaves the score as it is
    * ``normalize_query``: divide by the length of the query vector too, so that a score is a cosine between 0 and 1
    """

    name = "tfidf"

    def __init__(self, tf: str = "log", idf: str = "classic", norm: str = "cosine", doc_idf: bool = False, normalize_query: bool = True) -> None:
        if tf not in _TF or idf not in _IDF or norm not in ("cosine", "length", "none"):
            raise SearchError("unknown tf, idf or norm")
        self.tf, self.idf, self.norm, self.doc_idf, self.normalize_query = tf, idf, norm, doc_idf, normalize_query
        self._tf = _TF[tf]
        self._idf = _IDF[idf]
        self.needs_norms = norm == "cosine"

    def weight(self, df: int, stats: Stats) -> float:
        return self._idf(stats.num_docs, df) if df else 0.0

    def term_weight(self, f: float, w: float) -> float:
        """The weight of a term with frequency ``f`` in a document (``w`` is its idf)."""
        return self._tf(f) * (w if self.doc_idf else 1.0)

    def bind(self, weights: Sequence[float], stats: Stats) -> Callable[[Sequence[float], int, float], float]:
        ws = list(weights)
        qnorm = 1.0
        if self.normalize_query:
            qn = math.sqrt(sum(w * w for w in ws))
            qnorm = qn if qn > 0 else 1.0
        tf = self._tf
        di = self.doc_idf
        mode = self.norm

        def score(freqs: Sequence[float], length: int, norm: float) -> float:
            total = 0.0
            for w, f in zip(ws, freqs):
                if f and w:
                    total += w * tf(f) * (w if di else 1.0)
            if not total:
                return 0.0
            if mode == "cosine":
                total /= norm if norm else 1.0
            elif mode == "length":
                total /= math.sqrt(length) if length else 1.0
            return total / qnorm

        return score

    def __repr__(self) -> str:
        return "TfIdf(tf=%r, idf=%r, norm=%r, doc_idf=%r, normalize_query=%r)" % (self.tf, self.idf, self.norm, self.doc_idf, self.normalize_query)


def scorer_from_name(name: str) -> Scorer:
    """``"bm25"``, ``"bm25-fts5"``, ``"tfidf"`` (classic idf, cosine), ``"tfidf-sklearn"``, ``"tfidf-lucene"``."""
    if name == "bm25":
        return BM25()
    if name == "bm25-fts5":
        return BM25(idf="fts5")
    if name == "tfidf":
        return TfIdf()
    if name == "tfidf-sklearn":
        return TfIdf(tf="log", idf="smooth", norm="cosine", doc_idf=True)
    if name == "tfidf-lucene":
        return TfIdf(tf="sqrt", idf="lucene", norm="length", doc_idf=True, normalize_query=False)
    raise SearchError("unknown scorer %r" % name)
