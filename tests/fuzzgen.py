"""Seeded generators of documents and queries for the comparison with FTS5.

Everything is a function of a ``random.Random``, so a failing seed can be repeated: ``KENDI_FUZZ_START=<seed> KENDI_FUZZ=1 pytest``.
"""
from __future__ import annotations

import random
from typing import List, Sequence

# words with their awkward forms: case, accents, Turkish letters, digits, other scripts
BASE_WORDS = (
    "cat cats dog dogs fox foxes run running runs ran quick quickly slow slowly red green blue brown black white "
    "the a of and to in is it that was for on are with as at be this have from or one had by word but not what all were "
    "search engine index query rank score term document phrase prefix fuzzy stem stemming stemmer happiness happy happily "
    "relational relate relating conditional rational valency digitizer operator feudalism decisiveness hopefulness "
    "Café café CAFÉ naïve Naïve résumé Ünïcödé über straße İstanbul ISPARTA ıslak ışık çalışma ağaç öğrenci Ωμέγα αθήνα "
    "42 3 14 2024 x2 a1b2 foo_bar foo bar hello Hello HELLO world World wörld 日本語 日本 テスト"
).split()

SEPARATORS = [" ", " ", " ", " ", ", ", ". ", " - ", "; ", "\n", "  ", " (", ") ", " \"", "\" ", "!", "?!"]


def zipf_word(rng: random.Random, vocab: Sequence[str]) -> str:
    """A word from the vocabulary with a skewed distribution: a few words are everywhere, most are rare."""
    n = len(vocab)
    i = int(n * (rng.random() ** 2.5))
    return vocab[min(i, n - 1)]


def make_vocab(rng: random.Random, size: int = 60) -> List[str]:
    words = list(BASE_WORDS)
    rng.shuffle(words)
    vocab = words[:size]
    # a few made-up words so that prefixes and fuzzy matches have neighbours
    for _ in range(max(0, size - len(vocab))):
        vocab.append("".join(rng.choice("abcdefghijklmnoprstuy") for _ in range(rng.randint(2, 8))))
    return vocab


def make_text(rng: random.Random, vocab: Sequence[str], max_words: int = 25) -> str:
    n = rng.choice([0, 0, 1, 2, 3, 5, 8, 12, max_words])
    n = rng.randint(0, max_words) if n == max_words else n
    parts = []
    for _ in range(n):
        parts.append(zipf_word(rng, vocab))
        parts.append(rng.choice(SEPARATORS))
    return "".join(parts)


def make_doc(rng: random.Random, vocab: Sequence[str], fields: Sequence[str]) -> dict:
    doc = {}
    for f in fields:
        if rng.random() < 0.9:
            doc[f] = make_text(rng, vocab)
    return doc


# ---------------------------------------------------------------------------------------------------------------- queries

def _word(rng: random.Random, vocab: Sequence[str]) -> str:
    w = zipf_word(rng, vocab)
    if rng.random() < 0.06:
        w = "".join(rng.choice("abcdefghijklmnoprstuy") for _ in range(rng.randint(2, 6)))  # probably not in the index
    return w


def _string(rng: random.Random, vocab: Sequence[str], alone: bool = True) -> str:
    """A bare word when it can be one, else a quoted string of one to three words."""
    r = rng.random()
    if r < 0.04 and alone:
        return rng.choice(['""', '"!!"', '"..."', '"- ."', '"x"', '"a"', '"zq"'])  # no words in it (or very short words)
    if r < 0.55:
        w = _word(rng, vocab)
        if all(ch.isalnum() or ch == "_" or ch >= "\x80" for ch in w):
            return w
        return '"%s"' % w.replace('"', '""')
    n = rng.choice([1, 2, 2, 3])
    words = " ".join(_word(rng, vocab) for _ in range(n))
    if rng.random() < 0.1:
        words += " " + rng.choice(["!!", "...", "-", ""])
    return '"%s"' % words


def _phrase(rng: random.Random, vocab: Sequence[str], star: float = 0.2) -> str:
    n = 1 if rng.random() < 0.8 else rng.randint(2, 3)
    parts = []
    for _ in range(n):
        s = _string(rng, vocab, n == 1)
        if rng.random() < star:
            if rng.random() < 0.7 and not s.startswith('"'):
                s = s[: rng.randint(1, len(s))]  # a real prefix
            s += "*"
        parts.append(s)
    return " + ".join(parts)


def _colset(rng: random.Random, fields: Sequence[str]) -> str:
    r = rng.random()
    if len(fields) == 1 or r < 0.5:
        return rng.choice(fields) + ":"
    k = rng.randint(1, len(fields))
    names = rng.sample(list(fields), k)
    if rng.random() < 0.3:
        if len(names) == len(fields):  # a filter that leaves no field is a case FTS5 handles oddly in a NOT: not generated
            names = names[:-1]
        return "-{%s}:" % " ".join(names) if rng.random() < 0.6 else "-%s:" % names[0]
    return "{%s}:" % " ".join(names)


def _item(rng: random.Random, vocab: Sequence[str], fields: Sequence[str], depth: int, allow_colset: bool = True) -> str:
    r = rng.random()
    prefix = ""
    if rng.random() < 0.25 and allow_colset:
        prefix = _colset(rng, fields)
    if r < 0.12:
        k = rng.randint(1, 3)
        phrases = " ".join(_phrase(rng, vocab, 0.1) for _ in range(k + 1))
        dist = rng.choice(["", "", ", 0", ", 1", ", 2", ", 5", ", 10", ", 30"])
        return "%sNEAR(%s%s)" % (prefix, phrases, dist)
    caret = "^" if rng.random() < 0.08 else ""
    return prefix + caret + _phrase(rng, vocab)


def _list(rng: random.Random, vocab: Sequence[str], fields: Sequence[str], depth: int, allow_colset: bool = True) -> str:
    if depth > 0 and rng.random() < 0.2:
        pre = _colset(rng, fields) if rng.random() < 0.2 and allow_colset else ""
        inner = _expr(rng, vocab, fields, depth - 1, allow_colset and not pre)  # no field filter inside one: they would be intersected
        return "%s(%s)" % (pre, inner)
    n = rng.choice([1, 1, 1, 2, 2, 3])
    return " ".join(_item(rng, vocab, fields, depth, allow_colset) for _ in range(n))


def _expr(rng: random.Random, vocab: Sequence[str], fields: Sequence[str], depth: int, allow_colset: bool = True) -> str:
    n = rng.choice([1, 1, 2, 2, 3])
    out = _list(rng, vocab, fields, depth, allow_colset)
    for _ in range(n - 1):
        op = rng.choice(["AND", "OR", "NOT", "OR", "AND"])
        out += " %s %s" % (op, _list(rng, vocab, fields, depth, allow_colset))
    return out


def make_query(rng: random.Random, vocab: Sequence[str], fields: Sequence[str], depth: int = 2) -> str:
    return _expr(rng, vocab, fields, depth)


def damage(rng: random.Random, text: str) -> str:
    """A query with a few characters deleted, inserted or swapped: most of them are no longer valid, some still are."""
    chars = list(text)
    for _ in range(rng.randint(1, 3)):
        if not chars:
            break
        i = rng.randrange(len(chars))
        r = rng.random()
        if r < 0.35:
            del chars[i]
        elif r < 0.7:
            chars.insert(i, rng.choice('()":*+-^{},~ ANDORT'))
        elif r < 0.85 and i + 1 < len(chars):
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
        else:
            chars[i] = rng.choice('()":*+-^{},~ .!_')
    return "".join(chars)
