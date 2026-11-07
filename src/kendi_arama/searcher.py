"""Searching a snapshot of an index.

``Index.searcher()`` opens the segments that are current at that moment; what happens to the index afterwards (new documents, merges)
does not change what this searcher sees.  The searcher counts what a ranking function needs over *all* the segments (the number of
documents, the average length, the number of documents with a phrase), skipping deleted documents, so a document gets the same score
whether the index has one segment or twenty.
"""
from __future__ import annotations

import heapq
import math
import os
from collections import defaultdict
from typing import Dict, Iterator, List, Optional, Tuple, Union

from . import highlight as hl
from . import matchers as mt
from . import query as q
from .fuzzy import edit_distance
from .index import Index, read_deleted, read_manifest
from .schema import Schema
from .scoring import BM25, Scorer, Stats, TfIdf
from .segment import NO_MORE, SegmentReader


class SegView:
    def __init__(self, name: str, reader: SegmentReader, deleted: frozenset) -> None:
        self.name = name
        self.reader = reader
        self.deleted = deleted

    @property
    def live(self) -> int:
        return self.reader.numdocs - len(self.deleted)


class Hit:
    """One result: the key of the document, its score, and the stored text of its fields (``hit["title"]``)."""

    __slots__ = ("key", "score", "_seg", "_doc", "_fields")

    def __init__(self, key: str, score: float, seg: SegView, doc: int) -> None:
        self.key = key
        self.score = score
        self._seg = seg
        self._doc = doc
        self._fields: Optional[dict] = None

    @property
    def fields(self) -> dict:
        if self._fields is None:
            self._fields = self._seg.reader.stored(self._doc)
        return self._fields

    def __getitem__(self, name: str) -> str:
        return self.fields[name]

    def get(self, name: str, default=None):
        return self.fields.get(name, default)

    def __repr__(self) -> str:
        return "Hit(%r, score=%.4f)" % (self.key, self.score)


class Results:
    """``hits`` (the requested page, best first) and ``total``: the number of documents that match, all of them."""

    def __init__(self, hits: List[Hit], total: int) -> None:
        self.hits = hits
        self.total = total

    def __iter__(self) -> Iterator[Hit]:
        return iter(self.hits)

    def __len__(self) -> int:
        return len(self.hits)

    def __getitem__(self, i):
        return self.hits[i]

    def keys(self) -> List[str]:
        return [h.key for h in self.hits]

    def __repr__(self) -> str:
        return "Results(total=%d, hits=%r)" % (self.total, self.hits)


class LeafScore:
    """How one phrase of the query counts for one document."""

    __slots__ = ("text", "df", "weight", "freq", "contribution", "counted", "raw_freq", "raw_contribution")

    def __init__(self, text: str, df: int, weight: float, freq: float, contribution: float, counted: bool, raw_freq: float, raw_contribution: float) -> None:
        self.text = text
        self.df = df  # documents with the phrase
        self.weight = weight  # its idf
        self.freq = freq  # weighted occurrences that took part in the match
        self.contribution = contribution
        self.counted = counted
        self.raw_freq = raw_freq  # occurrences of the phrase in the document, whether or not they took part
        self.raw_contribution = raw_contribution

    def __repr__(self) -> str:
        return "LeafScore(%r, df=%d, weight=%.4f, freq=%g, contribution=%.6f)" % (self.text, self.df, self.weight, self.freq, self.contribution)


class Explanation:
    def __init__(self, key: str, score: float, leaves: List[LeafScore]) -> None:
        self.key = key
        self.score = score
        self.leaves = leaves

    def __repr__(self) -> str:
        return "Explanation(%r, score=%.6f, leaves=%r)" % (self.key, self.score, self.leaves)


class Searcher:
    def __init__(self, schema: Schema, segs: List[SegView], generation: int = 0) -> None:
        self.schema = schema
        self.segs = segs
        self.generation = generation
        self.num_docs = sum(s.live for s in segs)
        total = 0
        for s in segs:
            t = sum(s.reader.sum_length(fi) for fi in range(len(schema)))
            if s.deleted:
                dl = s.reader.doc_lengths()
                t -= sum(dl[d] for d in s.deleted)
            total += t
        self.stats = Stats(self.num_docs, total)
        self._df_cache: Dict[tuple, int] = {}
        self._norm_cache: Dict[tuple, List[List[float]]] = {}
        self._key_map: Optional[Dict[str, Tuple[SegView, int]]] = None

    @classmethod
    def open(cls, index: Index) -> "Searcher":
        data = read_manifest(index.path)
        segs: List[SegView] = []
        try:
            for rec in data["segments"]:
                reader = SegmentReader(os.path.join(index.path, rec["file"]), index.schema)
                dele = rec.get("deleted")
                deleted = read_deleted(os.path.join(index.path, dele["file"])) if dele else frozenset()
                segs.append(SegView(rec["name"], reader, deleted))
        except BaseException:
            for s in segs:
                s.reader.close()
            raise
        return cls(index.schema, segs, data["generation"])

    def close(self) -> None:
        for s in self.segs:
            s.reader.close()

    def __enter__(self) -> "Searcher":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- queries

    def parse(self, query: Union[str, q.Query], fuzzy: bool = True):
        if isinstance(query, str):
            query = q.parse_query(query, self.schema, fuzzy)
        return mt.plan(query, self.schema)

    def _leaf_df(self, leaf: mt.PLeaf) -> int:
        key = leaf.key
        cached = self._df_cache.get(key)
        if cached is not None:
            return cached
        total = 0
        single = len(leaf.per_field) == 1 and not leaf.first
        for seg in self.segs:
            if single and not seg.deleted:
                (fi, terms), = leaf.per_field.items()
                if len(terms) == 1 and terms[0].kind == "exact":
                    info = seg.reader.term_info(fi, terms[0].text)
                    total += info.df if info else 0
                    continue
            m = mt.build_leaf(leaf, seg.reader)
            if isinstance(m, mt.NoMatcher):
                continue
            dead = seg.deleted
            d = m.next()
            while d != NO_MORE:
                if d not in dead:
                    total += 1
                d = m.next()
        self._df_cache[key] = total
        return total

    def matches(self, query: Union[str, q.Query], fuzzy: bool = True) -> List[str]:
        """The keys of the documents that match, in the order of the index (no ranking)."""
        p = self.parse(query, fuzzy)
        out: List[str] = []
        if p is None:
            return out
        for seg in self.segs:
            m = mt.build_matcher(p, seg.reader)
            if isinstance(m, mt.NoMatcher):
                continue
            keys = seg.reader.keys()
            d = m.next()
            while d != NO_MORE:
                if d not in seg.deleted:
                    out.append(keys[d])
                d = m.next()
        return out

    def count(self, query: Union[str, q.Query], fuzzy: bool = True) -> int:
        return len(self.matches(query, fuzzy))

    def search(
        self,
        query: Union[str, q.Query],
        limit: Optional[int] = 10,
        offset: int = 0,
        scorer: Optional[Scorer] = None,
        weights: Optional[Dict[str, float]] = None,
        fuzzy: bool = True,
    ) -> Results:
        """The documents that match the query, best first.  ``weights`` changes the importance of fields for this search
        (``{"title": 5}``); the default is the weight of each field in the schema."""
        scorer = scorer or BM25()
        p = self.parse(query, fuzzy)
        if p is None:
            return Results([], 0)
        leaves = mt.plan_leaves(p)
        dfs = [self._leaf_df(leaf) for leaf in leaves]
        lw = [scorer.weight(df, self.stats) for df in dfs]
        score = scorer.bind(lw, self.stats)
        fw = [f.weight for f in self.schema.fields]
        if weights:
            for name, w in weights.items():
                fw[self.schema.index[name]] = float(w)
        uniform = all(w == 1.0 for w in fw)
        want = None if limit is None else offset + limit
        heap: List[Tuple[float, int, SegView, int]] = []
        total = 0
        norms_all = self._cosine_norms(scorer) if scorer.needs_norms else None
        for si, seg in enumerate(self.segs):
            m = mt.build_matcher(p, seg.reader)
            if isinstance(m, mt.NoMatcher):
                continue
            dl = seg.reader.doc_lengths()
            norms = norms_all[si] if norms_all is not None else None
            dead = seg.deleted
            while True:
                d = m.next()
                if d == NO_MORE:
                    break
                if d in dead:
                    continue
                total += 1
                if want == 0:
                    continue
                out: List[List[int]] = []
                m.collect(out, d)
                if uniform:
                    freqs = [len(x) for x in out]
                else:
                    freqs = [sum(fw[x >> 32] for x in lst) for lst in out]
                s = score(freqs, dl[d], norms[d] if norms is not None else 1.0)
                item = (s, -((si << 32) | d), seg, d)
                if want is None or len(heap) < want:
                    heapq.heappush(heap, item)
                elif item[:2] > heap[0][:2]:
                    heapq.heapreplace(heap, item)
        ranked = sorted(heap, key=lambda t: (-t[0], -t[1]))
        page = ranked[offset:] if limit is None else ranked[offset : offset + limit]
        hits = [Hit(seg.reader.keys()[d], s, seg, d) for s, _, seg, d in page]
        return Results(hits, total)

    def explain(self, query: Union[str, q.Query], key: str, scorer: Optional[Scorer] = None, weights: Optional[Dict[str, float]] = None, fuzzy: bool = True) -> Optional[Explanation]:
        """The score of one document, phrase by phrase.  ``None`` if the document does not exist or does not match."""
        scorer = scorer or BM25()
        parsed = q.parse_query(query, self.schema, fuzzy) if isinstance(query, str) else query
        dropped: List[mt.PLeaf] = []
        p = mt.plan(parsed, self.schema, None, dropped)
        if p is None:
            return None
        self.get(key)  # builds the key map
        loc = self._key_map.get(str(key))  # type: ignore[union-attr]
        if loc is None:
            return None
        seg, d = loc
        si = self.segs.index(seg)
        leaves = mt.plan_leaves(p)
        dfs = [self._leaf_df(leaf) for leaf in leaves]
        lw = [scorer.weight(df, self.stats) for df in dfs]
        score = scorer.bind(lw, self.stats)
        fw = [f.weight for f in self.schema.fields]
        if weights:
            for name, w in weights.items():
                fw[self.schema.index[name]] = float(w)
        m = mt.build_matcher(p, seg.reader)
        if m.advance(d) != d:
            return None
        out: List[List[int]] = []
        m.collect(out, d)
        dl = seg.reader.doc_lengths()[d]
        norm = self._cosine_norms(scorer)[si][d] if scorer.needs_norms else 1.0

        def freq_of(lst: List[int]) -> float:
            return sum(fw[x >> 32] for x in lst)

        result = []
        for i, leaf in enumerate(leaves):
            counted = freq_of(out[i])
            lm = mt.build_leaf(leaf, seg.reader)
            raw = 0.0
            if lm.advance(d) == d:
                raw = freq_of(lm.poslist())  # type: ignore[attr-defined]
            one = [0.0] * len(leaves)
            one[i] = counted
            raw_one = [0.0] * len(leaves)
            raw_one[i] = raw
            result.append(LeafScore(mt.leaf_text(leaf, self.schema), dfs[i], lw[i], counted, score(one, dl, norm), bool(out[i]), raw, score(raw_one, dl, norm)))
        total = score([freq_of(x) for x in out], dl, norm)
        for leaf in dropped:  # phrases of the query that cannot take part in any match: they are reported with what they would have counted
            lm = mt.build_leaf(leaf, seg.reader)
            raw = freq_of(lm.poslist()) if lm.advance(d) == d else 0.0  # type: ignore[attr-defined]
            df = self._leaf_df(leaf)
            w = scorer.weight(df, self.stats)
            raw_one = [0.0] * 1
            raw_one[0] = raw
            one = scorer.bind([w], self.stats)([raw], dl, norm)
            result.append(LeafScore(mt.leaf_text(leaf, self.schema), df, w, 0.0, 0.0, False, raw, one))
        return Explanation(str(key), total, result)

    def terms(self, field: str, prefix: str = "") -> List[Tuple[str, int]]:
        """The terms of a field that start with ``prefix``, with the number of documents that have each."""
        fi = self.schema.index[field]
        counts: Dict[str, int] = {}
        for seg in self.segs:
            for term, info in seg.reader.terms(fi, prefix):
                if not seg.deleted:
                    counts[term] = counts.get(term, 0) + info.df
                else:
                    it = seg.reader.postings(fi, info)
                    n = 0
                    while it.next() != NO_MORE:
                        if it.doc not in seg.deleted:
                            n += 1
                    if n:
                        counts[term] = counts.get(term, 0) + n
        return sorted(counts.items())

    def suggest(self, word: str, field: Optional[str] = None, max_distance: int = 2, limit: int = 5, prefix_length: int = 0) -> List[Tuple[str, int, int]]:
        """Terms of the index that are close to ``word`` (a "did you mean"): ``(term, edit distance, documents)``, the closest and then
        the most common first.  The word is analysed like the text of the field."""
        fi = self.schema.index[field] if field else 0
        analyzer = self.schema.fields[fi].analyzer
        toks = analyzer.terms(word)
        if len(toks) != 1:
            return []
        target = toks[0]
        df: Dict[str, int] = {}
        for term, n in self.terms(self.schema.fields[fi].name):
            if abs(len(term) - len(target)) <= max_distance and (not prefix_length or term[:prefix_length] == target[:prefix_length]):
                d = edit_distance(target, term, max_distance)
                if d <= max_distance:
                    df[term] = (d, n)  # type: ignore[assignment]
        found = [(t, d, n) for t, (d, n) in df.items()]  # type: ignore[misc]
        found.sort(key=lambda x: (x[1], -x[2], x[0]))
        return found[:limit]

    # -- highlighting

    def _instances(self, query, key: str, field: str, fuzzy: bool):
        """(text, tokens, instances) of one stored field of a document for a query; ``None`` if the document does not match."""
        fi = self.schema.index[field]
        p = self.parse(query, fuzzy)
        if p is None:
            return None
        self.get(key)
        loc = self._key_map.get(str(key))  # type: ignore[union-attr]
        if loc is None:
            return None
        seg, d = loc
        m = mt.build_matcher(p, seg.reader)
        if m.advance(d) != d:
            return None
        out: List[List[int]] = []
        m.collect(out, d)
        leaves = mt.plan_leaves(p)
        instances = []
        for leaf, lst in zip(leaves, out):
            span = leaf.span(fi)
            for x in lst:
                if x >> 32 == fi:
                    pos = x & 0xFFFFFFFF
                    instances.append((pos, pos + span - 1))
        text = seg.reader.stored(d).get(field, "")
        return text, self.schema.fields[fi].analyzer.tokens(text), instances

    def highlight(self, query: Union[str, q.Query], key: str, field: str, open_: str = "<b>", close: str = "</b>", fuzzy: bool = True) -> Optional[str]:
        """The stored text of ``field`` with the words that matched wrapped in ``open_`` and ``close``."""
        r = self._instances(query, key, field, fuzzy)
        if r is None:
            return None
        text, tokens, instances = r
        return hl.highlight(text, tokens, instances, open_, close)

    def snippet(self, query: Union[str, q.Query], key: str, field: str, open_: str = "<b>", close: str = "</b>", ellipsis: str = "...", max_tokens: int = 16, fuzzy: bool = True) -> Optional[str]:
        """A short extract of ``field`` around the matches."""
        r = self._instances(query, key, field, fuzzy)
        if r is None:
            return None
        text, tokens, instances = r
        return hl.snippet(text, tokens, instances, open_, close, ellipsis, max_tokens)

    # -- cosine norms

    def _term_groups(self, seg: SegView):
        """(term, [(field, info) ...]) for every term of the segment, the same word in different fields grouped."""
        nf = len(self.schema)
        if nf == 1:
            for term, info in seg.reader.terms(0):
                yield term, [(0, info)]
            return
        def tagged(fi: int):
            for term, info in seg.reader.terms(fi):
                yield term, fi, info

        iters = [tagged(fi) for fi in range(nf)]
        current = None
        group: List[Tuple[int, object]] = []
        for term, fi, info in heapq.merge(*iters):
            if term != current:
                if group:
                    yield current, group
                current, group = term, []
            group.append((fi, info))
        if group:
            yield current, group

    def _term_docs(self, seg: SegView, group) -> Dict[int, int]:
        """document -> frequency of the word in all the fields, for the live documents of a segment."""
        acc: Dict[int, int] = {}
        dead = seg.deleted
        for fi, info in group:
            it = seg.reader.postings(fi, info)
            while it.next() != NO_MORE:
                d = it.doc
                if d in dead:
                    continue
                acc[d] = acc.get(d, 0) + it.freq
        return acc

    def _cosine_norms(self, scorer: TfIdf) -> List[List[float]]:
        key = (scorer.tf, scorer.idf, scorer.doc_idf)
        cached = self._norm_cache.get(key)
        if cached is not None:
            return cached
        gdf: Dict[str, int] = defaultdict(int)
        if scorer.doc_idf:
            for seg in self.segs:
                for term, group in self._term_groups(seg):
                    if len(group) == 1 and not seg.deleted:
                        gdf[term] += group[0][1].df
                    else:
                        gdf[term] += len(self._term_docs(seg, group))
        out: List[List[float]] = []
        for seg in self.segs:
            acc = [0.0] * seg.reader.numdocs
            for term, group in self._term_groups(seg):
                w = scorer.weight(gdf[term], self.stats) if scorer.doc_idf else 1.0
                for d, f in self._term_docs(seg, group).items():
                    x = scorer.term_weight(f, w)
                    acc[d] += x * x
            out.append([math.sqrt(x) for x in acc])
        self._norm_cache[key] = out
        return out

    # -- documents

    def get(self, key: str) -> Optional[dict]:
        """The stored fields of the document with that key."""
        if self._key_map is None:
            km: Dict[str, Tuple[SegView, int]] = {}
            for seg in self.segs:
                for d, k in enumerate(seg.reader.keys()):
                    if d not in seg.deleted:
                        km[k] = (seg, d)
            self._key_map = km
        loc = self._key_map.get(str(key))
        if loc is None:
            return None
        return loc[0].reader.stored(loc[1])

    def __len__(self) -> int:
        return self.num_docs
