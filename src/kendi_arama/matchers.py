"""Turning a query into something that walks the documents of a segment.

A query is first *planned* against the schema: every phrase is analysed with the analyzer of each field it may match in (the words of a
query go through the same analysis as the text), and phrases that are empty after that (``"!!"``) disappear, like in FTS5.

For one segment the plan becomes a tree of *matchers*.  A matcher is a cursor over the documents that match, in increasing order:
``doc`` is the current one (``NO_MORE`` at the end), ``next()`` and ``advance(target)`` move it.  After a move, ``collect(out, doc)``
appends for every phrase of the sub-tree the positions where the phrase matched in document ``doc`` if the phrase is part of the match
(its cursor is on that document), and an empty list if not: these are what a scorer counts.  The phrases on the right of a ``NOT`` are
never part of a match.  A position is ``field << 32 | word number``.
"""
from __future__ import annotations

import heapq
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

from . import query as q
from .fuzzy import close_terms
from .segment import NO_MORE, PostingsIter, SegmentReader
from .schema import Schema

# ---------------------------------------------------------------------------------------------------------------- planning


class Term(NamedTuple):
    text: str
    offset: int  # where the word is in the phrase (0 for the first one; a gap is left where a stop word was dropped)
    kind: str  # "exact", "prefix" or "fuzzy"
    arg: int = 0  # the edit distance of a fuzzy term


class PLeaf:
    """A phrase, as it will be looked for in each field where it may match."""

    __slots__ = ("per_field", "first", "key")

    def __init__(self, per_field: Dict[int, Tuple[Term, ...]], first: bool) -> None:
        self.per_field = per_field
        self.first = first
        self.key = (tuple(sorted(per_field.items())), first)

    def span(self, fi: int) -> int:
        terms = self.per_field.get(fi)
        return terms[-1].offset + 1 if terms else 0

    def max_span(self) -> int:
        return max((t[-1].offset + 1 for t in self.per_field.values()), default=0)


class PNear:
    __slots__ = ("leaves", "distance")

    def __init__(self, leaves: Sequence[PLeaf], distance: int) -> None:
        self.leaves = tuple(leaves)
        self.distance = distance


class PBool:
    __slots__ = ("op", "left", "right")

    def __init__(self, op: str, left, right) -> None:
        self.op = op
        self.left = left
        self.right = right


def _plan_phrase(p: q.Phrase, schema: Schema, colset: Optional[frozenset]) -> Optional[PLeaf]:
    allowed = p.colset if p.colset is not None else colset
    if p.colset is not None and colset is not None:
        allowed = p.colset & colset
    fields = range(len(schema)) if allowed is None else sorted(allowed)
    per_field: Dict[int, Tuple[Term, ...]] = {}
    for fi in fields:
        analyzer = schema.fields[fi].analyzer
        terms: List[Term] = []
        base = 0
        for part in p.parts:
            toks = analyzer.tokens(part.text)
            if not toks:
                continue
            first_pos = toks[0].pos
            for k, t in enumerate(toks):
                last = k == len(toks) - 1
                if part.fuzzy:
                    terms.append(Term(t.term, base + t.pos - first_pos, "fuzzy", part.fuzzy))
                elif part.prefix and last:
                    terms.append(Term(t.term, base + t.pos - first_pos, "prefix"))
                else:
                    terms.append(Term(t.term, base + t.pos - first_pos, "exact"))
            base = terms[-1].offset + 1
        if terms:
            per_field[fi] = tuple(terms)
    if not per_field:
        # nothing to look for in the allowed fields: a phrase that never matches (it is dropped only when the text has no words at all)
        for f in schema.fields:
            if any(f.analyzer.tokens(part.text) for part in p.parts):
                return PLeaf({}, p.first)
        return None
    return PLeaf(per_field, p.first)


def plan(node: q.Query, schema: Schema, colset: Optional[frozenset] = None, dropped: Optional[List[PLeaf]] = None):
    """The plan of a parsed query; ``None`` when nothing is left of it (a query of empty phrases matches nothing).  The phrases of
    the parts that could never match (the other side of an AND with an empty phrase ...) are added to ``dropped`` if it is given."""
    if isinstance(node, q.Phrase):
        return _plan_phrase(node, schema, colset)
    if isinstance(node, q.Near):
        allowed = node.colset if node.colset is not None else colset
        leaves = [x for x in (_plan_phrase(ph, schema, allowed) for ph in node.phrases) if x is not None]
        return PNear(leaves, node.distance) if leaves else None
    left = plan(node.left, schema, colset, dropped)  # type: ignore[attr-defined]
    right = plan(node.right, schema, colset, dropped)  # type: ignore[attr-defined]

    def drop(p) -> None:
        if dropped is not None and p is not None:
            dropped.extend(plan_leaves(p))

    # None is a phrase without words ("!!"): in a list of phrases or around OR and NOT it just disappears, but a document can never
    # match an explicit AND with it
    if isinstance(node, q.Not):
        if left is None:
            drop(right)
            return None
        return left if right is None else PBool("not", left, right)
    if isinstance(node, q.And) and not node.implicit:
        if left is None or right is None:
            drop(left)
            drop(right)
            return None
        return PBool("and", left, right)
    if left is None:
        return right
    if right is None:
        return left
    return PBool("and" if isinstance(node, q.And) else "or", left, right)


def leaf_text(leaf: PLeaf, schema: Schema) -> str:
    """A readable form of a phrase: the fields (when they are not all) and the words."""
    parts = []
    for fi, terms in leaf.per_field.items():
        words = " ".join(t.text + ("*" if t.kind == "prefix" else "~%d" % t.arg if t.kind == "fuzzy" else "") for t in terms)
        parts.append("%s:%s" % (schema.fields[fi].name, words) if len(schema) > 1 else words)
    return ("^" if leaf.first else "") + (" | ".join(parts) if parts else "<nothing>")


def plan_leaves(p) -> List[PLeaf]:
    """The phrases of a plan, left to right (the order in which a scorer sees them)."""
    if isinstance(p, PLeaf):
        return [p]
    if isinstance(p, PNear):
        return list(p.leaves)
    return plan_leaves(p.left) + plan_leaves(p.right)


# ---------------------------------------------------------------------------------------------------------------- cursors on terms

class UnionPostings:
    """Several terms as one (the words that a prefix or a fuzzy term stands for)."""

    def __init__(self, iters: List[PostingsIter]) -> None:
        self.iters = iters
        self.heap: List[Tuple[int, int]] = []
        self.doc = -1
        self.freq = 0
        self.at: List[int] = []
        self.df = sum(it.df for it in iters)
        for k, it in enumerate(iters):
            if it.next() != NO_MORE:
                self.heap.append((it.doc, k))
        heapq.heapify(self.heap)
        self._settle()

    def _settle(self) -> None:
        heap = self.heap
        if not heap:
            self.doc = NO_MORE
            self.at = []
            self.freq = 0
            return
        d = heap[0][0]
        at = []
        while heap and heap[0][0] == d:
            at.append(heapq.heappop(heap)[1])
        for k in at:
            heapq.heappush(heap, (d, k))
        self.doc = d
        self.at = at
        self.freq = sum(self.iters[k].freq for k in at)

    def _move(self, target: int) -> None:
        heap = self.heap
        iters = self.iters
        while heap and heap[0][0] < target:
            d, k = heapq.heappop(heap)
            it = iters[k]
            nxt = it.advance(target) if target > d + 1 else it.next()
            if nxt != NO_MORE:
                heapq.heappush(heap, (nxt, k))
        self._settle()

    def next(self) -> int:
        if self.doc == NO_MORE:
            return NO_MORE
        self._move(self.doc + 1)
        return self.doc

    def advance(self, target: int) -> int:
        if target < 0:
            target = 0
        if self.doc >= target:
            return self.doc
        self._move(target)
        return self.doc

    def positions(self) -> List[int]:
        out: List[int] = []
        for k in self.at:
            out.extend(self.iters[k].positions())
        out.sort()
        return out


def _term_iterator(reader: SegmentReader, fi: int, term: Term):
    if term.kind == "exact":
        info = reader.term_info(fi, term.text)
        return reader.postings(fi, info) if info is not None else None
    if term.kind == "prefix":
        infos = [info for _, info in reader.terms(fi, term.text)]
    else:
        infos = [info for _, info in close_terms(term.text, reader.terms(fi), term.arg)]
    if not infos:
        return None
    its = [reader.postings(fi, info) for info in infos]
    return its[0] if len(its) == 1 else UnionPostings(its)


def _align(iters, d: int) -> int:
    """The smallest document >= d that all the cursors contain."""
    while True:
        for it in iters:
            x = it.advance(d)
            if x != d:
                if x == NO_MORE:
                    return NO_MORE
                d = x
                break
        else:
            return d


# ---------------------------------------------------------------------------------------------------------------- matchers

class Matcher:
    doc = -1
    nleaves = 1

    def next(self) -> int:
        raise NotImplementedError

    def advance(self, target: int) -> int:
        raise NotImplementedError

    def collect(self, out: List[List[int]], root: int) -> None:
        raise NotImplementedError

    def set_eof(self) -> None:
        self.doc = NO_MORE

    def cost(self) -> int:
        return 1


class NoMatcher(Matcher):
    """Never matches (a term that is not in the segment).  Remembers how many phrases it stands for, so that ``collect`` fills them in."""

    def __init__(self, nleaves: int = 1) -> None:
        self.doc = NO_MORE
        self.nleaves = nleaves

    def next(self) -> int:
        return NO_MORE

    def advance(self, target: int) -> int:
        return NO_MORE

    def collect(self, out: List[List[int]], root: int) -> None:
        for _ in range(self.nleaves):
            out.append([])

    def cost(self) -> int:
        return 0


class _FieldPhrase:
    """A phrase in one field of one segment."""

    def __init__(self, fi: int, iters: list, terms: Tuple[Term, ...], first: bool) -> None:
        self.fi = fi
        self.iters = iters
        self.offsets = [t.offset for t in terms]
        self.first = first
        self.order = sorted(iters, key=lambda it: it.df)
        self.simple = len(iters) == 1 and not first
        self.doc = -1
        self.hits: List[int] = []

    def cost(self) -> int:
        return self.order[0].df

    def _check(self, d: int) -> bool:
        iters = self.iters
        if len(iters) == 1:
            ps = iters[0].positions()
            if self.first:
                self.hits = [0] if ps and ps[0] == 0 else []
                return bool(self.hits)
            self.hits = ps
            return True
        offsets = self.offsets
        starts = {p - offsets[0] for p in iters[0].positions()}
        for k in range(1, len(iters)):
            if not starts:
                break
            off = offsets[k]
            starts &= {p - off for p in iters[k].positions()}
        if self.first:
            starts &= {0}
        if starts:
            self.hits = sorted(starts)
            return True
        return False

    def _find(self, target: int) -> int:
        while True:
            d = _align(self.order, target) if len(self.order) > 1 else self.order[0].advance(target)
            if d == NO_MORE:
                self.doc = NO_MORE
                return NO_MORE
            if self.simple:
                self.doc = d
                return d
            if self._check(d):
                self.doc = d
                return d
            target = d + 1

    def next(self) -> int:
        return self._find(self.doc + 1)

    def advance(self, target: int) -> int:
        if self.doc >= target:
            return self.doc
        return self._find(target)

    def positions(self) -> List[int]:
        if self.simple:
            return self.iters[0].positions()
        return self.hits


class LeafMatcher(Matcher):
    """A phrase over all the fields it may match in; positions are reported as ``field << 32 | position``."""

    def __init__(self, parts: List[_FieldPhrase], span: int = 1) -> None:
        self.parts = parts
        self.span = span
        self.doc = -1

    def cost(self) -> int:
        return sum(p.cost() for p in self.parts)

    def _settle(self) -> int:
        self.doc = min(p.doc for p in self.parts)
        return self.doc

    def next(self) -> int:
        if self.doc == NO_MORE:
            return NO_MORE
        if self.doc < 0:
            for p in self.parts:
                p.next()
        else:
            d = self.doc
            for p in self.parts:
                if p.doc == d:
                    p.next()
        return self._settle()

    def advance(self, target: int) -> int:
        if self.doc >= target:
            return self.doc
        for p in self.parts:
            if p.doc < target:
                p.advance(target)
        return self._settle()

    def poslist(self) -> List[int]:
        d = self.doc
        out: List[int] = []
        for p in self.parts:
            if p.doc == d:
                base = p.fi << 32
                if base:
                    out.extend(base + x for x in p.positions())
                else:
                    out.extend(p.positions())
        return out

    def collect(self, out: List[List[int]], root: int) -> None:
        out.append(self.poslist() if self.doc == root else [])


def near_positions(lists: List[List[int]], spans: List[int], distance: int) -> Optional[List[List[int]]]:
    """The proximity test of FTS5: ``lists`` are the positions of each phrase in the document, ``spans`` the number of words of each.
    Returns the positions that take part in a match (or ``None`` when the phrases are never close enough)."""
    n = len(lists)
    EOF = 1 << 62
    idx = [0] * n
    outs: List[List[int]] = [[] for _ in range(n)]

    def pos(i: int) -> int:
        return lists[i][idx[i]] if idx[i] < len(lists[i]) else EOF

    def look(i: int) -> int:
        return lists[i][idx[i] + 1] if idx[i] + 1 < len(lists[i]) else EOF

    while True:
        imax = pos(0)
        while True:
            matched = True
            for i in range(n):
                imin = imax - spans[i] - distance
                p = pos(i)
                if p < imin or p > imax:
                    matched = False
                    while pos(i) < imin:
                        idx[i] += 1
                        if pos(i) == EOF:
                            return outs if outs[0] else None
                    if pos(i) > imax:
                        imax = pos(i)
            if matched:
                break
        for i in range(n):
            p = pos(i)
            if not outs[i] or outs[i][-1] != p:
                outs[i].append(p)
        adv = 0
        imin = look(0)
        for i in range(n):
            if look(i) < imin:
                imin = look(i)
                adv = i
        idx[adv] += 1
        if pos(adv) == EOF:
            return outs if outs[0] else None


class NearMatcher(Matcher):
    def __init__(self, leaves: List[LeafMatcher], distance: int) -> None:
        self.leaves = leaves
        self.distance = distance
        self.nleaves = len(leaves)
        self.spans = [lf.span for lf in leaves]
        self.order = sorted(leaves, key=lambda lf: lf.cost())
        self.doc = -1
        self.trimmed: List[List[int]] = []

    def cost(self) -> int:
        return self.order[0].cost()

    def _find(self, target: int) -> int:
        while True:
            d = _align(self.order, target)
            if d == NO_MORE:
                self.doc = NO_MORE
                return NO_MORE
            lists = [lf.poslist() for lf in self.leaves]
            res = near_positions(lists, self.spans, self.distance) if len(lists) > 1 else lists
            if res is not None:
                self.doc = d
                self.trimmed = res
                return d
            target = d + 1

    def next(self) -> int:
        return self._find(self.doc + 1)

    def advance(self, target: int) -> int:
        if self.doc >= target:
            return self.doc
        return self._find(target)

    def collect(self, out: List[List[int]], root: int) -> None:
        if self.doc == root:
            out.extend(self.trimmed)
        else:
            out.extend([] for _ in self.leaves)

    def set_eof(self) -> None:
        self.doc = NO_MORE
        for lf in self.leaves:
            lf.doc = NO_MORE


class AndMatcher(Matcher):
    def __init__(self, left: Matcher, right: Matcher) -> None:
        self.children = [left, right]
        self.order = sorted(self.children, key=lambda m: m.cost())
        self.nleaves = left.nleaves + right.nleaves
        self.doc = -1

    def cost(self) -> int:
        return self.order[0].cost()

    def next(self) -> int:
        return self.advance_to(self.doc + 1)

    def advance_to(self, target: int) -> int:
        self.doc = _align(self.order, target)
        if self.doc == NO_MORE:
            self.set_eof()
        return self.doc

    def advance(self, target: int) -> int:
        if self.doc >= target:
            return self.doc
        return self.advance_to(target)

    def set_eof(self) -> None:
        self.doc = NO_MORE
        for c in self.children:
            c.set_eof()

    def collect(self, out: List[List[int]], root: int) -> None:
        for c in self.children:
            c.collect(out, root)


class OrMatcher(Matcher):
    def __init__(self, left: Matcher, right: Matcher) -> None:
        self.children = [left, right]
        self.nleaves = left.nleaves + right.nleaves
        self.doc = -1

    def cost(self) -> int:
        return sum(c.cost() for c in self.children)

    def next(self) -> int:
        if self.doc == NO_MORE:
            return NO_MORE
        d = self.doc
        if d < 0:
            for c in self.children:
                c.next()
        else:
            for c in self.children:
                if c.doc == d:
                    c.next()
        self.doc = min(c.doc for c in self.children)
        return self.doc

    def advance(self, target: int) -> int:
        if self.doc >= target:
            return self.doc
        for c in self.children:
            if c.doc < target:
                c.advance(target)
        self.doc = min(c.doc for c in self.children)
        return self.doc

    def set_eof(self) -> None:
        self.doc = NO_MORE
        for c in self.children:
            c.set_eof()

    def collect(self, out: List[List[int]], root: int) -> None:
        for c in self.children:
            c.collect(out, root)


class NotMatcher(Matcher):
    def __init__(self, left: Matcher, right: Matcher) -> None:
        self.left = left
        self.right = right
        self.nleaves = left.nleaves + right.nleaves
        self.doc = -1

    def cost(self) -> int:
        return self.left.cost()

    def set_eof(self) -> None:
        self.doc = NO_MORE
        self.left.set_eof()
        self.right.set_eof()

    def _find(self, target: int) -> int:
        left, right = self.left, self.right
        d = left.advance(target)
        while d != NO_MORE:
            if right.advance(d) != d:
                self.doc = d
                return d
            d = left.advance(d + 1)
        self.doc = NO_MORE
        return NO_MORE

    def next(self) -> int:
        return self._find(self.doc + 1)

    def advance(self, target: int) -> int:
        if self.doc >= target:
            return self.doc
        return self._find(target)

    def collect(self, out: List[List[int]], root: int) -> None:
        self.left.collect(out, root)
        out.extend([] for _ in range(self.right.nleaves))  # the right side of a NOT never takes part in a match


def build_leaf(p: PLeaf, reader: SegmentReader) -> Matcher:
    parts: List[_FieldPhrase] = []
    for fi, terms in p.per_field.items():
        iters = []
        for t in terms:
            it = _term_iterator(reader, fi, t)
            if it is None:
                break
            iters.append(it)
        else:
            parts.append(_FieldPhrase(fi, iters, terms, p.first))
    if not parts:
        return NoMatcher(1)
    return LeafMatcher(parts, p.max_span())


def build_matcher(p, reader: SegmentReader) -> Matcher:
    if isinstance(p, PLeaf):
        return build_leaf(p, reader)
    if isinstance(p, PNear):
        leaves = [build_leaf(x, reader) for x in p.leaves]
        if any(isinstance(x, NoMatcher) for x in leaves):
            return NoMatcher(len(leaves))
        if len(leaves) == 1:
            return leaves[0]
        return NearMatcher(leaves, p.distance)  # type: ignore[arg-type]
    left = build_matcher(p.left, reader)
    right = build_matcher(p.right, reader)
    if p.op == "and":
        if isinstance(left, NoMatcher) or isinstance(right, NoMatcher):
            return NoMatcher(left.nleaves + right.nleaves)
        return AndMatcher(left, right)
    if p.op == "or":
        if isinstance(left, NoMatcher) and isinstance(right, NoMatcher):
            return NoMatcher(left.nleaves + right.nleaves)
        return OrMatcher(left, right)
    if isinstance(right, NoMatcher):
        return _WithEmpty(left, right.nleaves)
    return NotMatcher(left, right)


class _WithEmpty(Matcher):
    """``left`` followed by phrases that never match (the right side of a NOT that is not in the segment)."""

    def __init__(self, left: Matcher, extra: int) -> None:
        self.left = left
        self.extra = extra
        self.nleaves = left.nleaves + extra

    @property
    def doc(self) -> int:  # type: ignore[override]
        return self.left.doc

    def cost(self) -> int:
        return self.left.cost()

    def next(self) -> int:
        return self.left.next()

    def advance(self, target: int) -> int:
        return self.left.advance(target)

    def set_eof(self) -> None:
        self.left.set_eof()

    def collect(self, out: List[List[int]], root: int) -> None:
        self.left.collect(out, root)
        out.extend([] for _ in range(self.extra))
