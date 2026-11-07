"""Showing where a query matched: the text with the matching words wrapped in markers, and short extracts.

The positions come from the matchers (the words of the stored text are analysed again to turn a position into a place in the text),
so what is marked is exactly what the query matched, with the same analysis of case, accents and stemming.
"""
from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

from .analysis import Token

Instance = Tuple[int, int]  # first and last word position of a match


def _clamp(instances: Iterable[Instance], tokens: Sequence[Token]) -> List[Instance]:
    """Instances moved onto words that exist (a stop word that was dropped leaves a gap in the positions)."""
    positions = [t.pos for t in tokens]
    if not positions:
        return []
    lo, hi = positions[0], positions[-1]
    if positions == list(range(lo, hi + 1)):
        return sorted({(max(s, lo), min(e, hi)) for s, e in instances if s <= hi and e >= lo})
    out = set()
    for s, e in instances:
        inside = [p for p in positions if s <= p <= e]
        if inside:
            out.add((inside[0], inside[-1]))
    return sorted(out)


def highlight(text: str, tokens: Sequence[Token], instances: Iterable[Instance], open_: str = "<b>", close: str = "</b>") -> str:
    """``text`` with every instance wrapped in ``open_`` and ``close``.  Instances that overlap are one (the words of a phrase that
    occurs twice in a row, or a word and a phrase that contains it); instances that only touch are marked one by one."""
    merged: List[List[int]] = []
    for s, e in _clamp(instances, tokens):
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    out: List[str] = []
    off = 0
    by_pos = {t.pos: t for t in tokens}
    for s, e in merged:
        start = by_pos[s].start
        end = by_pos[e].end
        out.append(text[off:start])
        out.append(open_)
        out.append(text[start:end])
        out.append(close)
        off = end
    out.append(text[off:])
    return "".join(out)


def snippet(
    text: str,
    tokens: Sequence[Token],
    instances: Iterable[Instance],
    open_: str = "<b>",
    close: str = "</b>",
    ellipsis: str = "...",
    max_tokens: int = 16,
) -> str:
    """A short extract of at most ``max_tokens`` words that contains as many matches as possible (the first such window), with the
    matches marked, and ``ellipsis`` where the text was cut."""
    toks = list(tokens)
    insts = _clamp(instances, toks)
    if len(toks) <= max_tokens or not insts:
        window = toks[:max_tokens]
    else:
        index = {t.pos: k for k, t in enumerate(toks)}
        starts = [index[s] for s, _ in insts]
        best_k, best_n = 0, -1
        for lo in range(0, len(toks) - max_tokens + 1):
            n = sum(1 for k in starts if lo <= k < lo + max_tokens)
            if n > best_n:
                best_k, best_n = lo, n
        # centre the window on its first match when there is room
        first = next((k for k in starts if best_k <= k < best_k + max_tokens), best_k)
        lo = min(max(0, first - (max_tokens // 4)), len(toks) - max_tokens)
        if sum(1 for k in starts if lo <= k < lo + max_tokens) < best_n:
            lo = best_k
        window = toks[lo : lo + max_tokens]
    if not window:
        return ""
    lo_pos, hi_pos = window[0].pos, window[-1].pos
    inside = [(s, e) for s, e in insts if lo_pos <= s and e <= hi_pos]
    start, end = window[0].start, window[-1].end
    marked = highlight(text[start:end], [Token(t.term, t.pos, t.start - start, t.end - start) for t in window], inside, open_, close)
    return (ellipsis if window[0] is not toks[0] else "") + marked + (ellipsis if window[-1] is not toks[-1] else "")
