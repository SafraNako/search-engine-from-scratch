"""Edit distance and the terms of a dictionary that are close to a word (for ``colour~1``).

The distance is the Damerau-Levenshtein distance in its "optimal string alignment" form: the number of insertions, deletions,
replacements and swaps of two neighbouring letters that turn one word into the other, where no part of the word is edited twice.
"""
from __future__ import annotations

from typing import Iterable, Iterator, List, Optional, Tuple


def edit_distance(a: str, b: str, limit: Optional[int] = None) -> int:
    """The distance of ``a`` and ``b``; with a ``limit`` it stops early and returns ``limit + 1`` when the distance is larger."""
    la, lb = len(a), len(b)
    if limit is not None and abs(la - lb) > limit:
        return limit + 1
    if la == 0 or lb == 0:
        return la or lb
    prev2: List[int] = []
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        ca = a[i - 1]
        best = cur[0]
        for j in range(1, lb + 1):
            cost = 0 if ca == b[j - 1] else 1
            v = prev[j - 1] + cost
            t = prev[j] + 1
            if t < v:
                v = t
            t = cur[j - 1] + 1
            if t < v:
                v = t
            if i > 1 and j > 1 and ca == b[j - 2] and a[i - 2] == b[j - 1]:
                t = prev2[j - 2] + 1
                if t < v:
                    v = t
            cur[j] = v
            if v < best:
                best = v
        if limit is not None and best > limit:
            return limit + 1
        prev2, prev = prev, cur
    return prev[lb] if limit is None else min(prev[lb], limit + 1)


def close_terms(word: str, terms: Iterable[Tuple[str, object]], max_distance: int, prefix_length: int = 0) -> Iterator[Tuple[str, object]]:
    """The ``(term, info)`` pairs whose term is within ``max_distance`` edits of ``word``.  The first ``prefix_length`` letters have to
    match exactly (which also makes the scan cheaper: a dictionary can be read from that prefix only)."""
    lw = len(word)
    for term, info in terms:
        if abs(len(term) - lw) > max_distance:
            continue
        if prefix_length and term[:prefix_length] != word[:prefix_length]:
            continue
        if edit_distance(word, term, max_distance) <= max_distance:
            yield term, info
