"""The Porter stemming algorithm (M.F. Porter, 1980, "An algorithm for suffix stripping").

The rules are the ones of the paper, in the order of the paper: step 1a/1b/1c (plurals and -ed/-ing), 2 and 3 (double suffixes such as
-ational), 4 (single suffixes such as -ement) and 5 (a final -e and a final double l).  ``stem`` expects a lower case word.
"""
from __future__ import annotations

from functools import lru_cache

_VOWELS = frozenset("aeiou")

_STEP2 = (
    ("ational", "ate"), ("tional", "tion"), ("enci", "ence"), ("anci", "ance"), ("izer", "ize"), ("bli", "ble"), ("alli", "al"),
    ("entli", "ent"), ("eli", "e"), ("ousli", "ous"), ("ization", "ize"), ("ation", "ate"), ("ator", "ate"), ("alism", "al"),
    ("iveness", "ive"), ("fulness", "ful"), ("ousness", "ous"), ("aliti", "al"), ("iviti", "ive"), ("biliti", "ble"), ("logi", "log"),
)
_STEP3 = (
    ("icate", "ic"), ("ative", ""), ("alize", "al"), ("iciti", "ic"), ("ical", "ic"), ("ful", ""), ("ness", ""),
)
_STEP4 = (
    "al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement", "ment", "ent", "ion", "ou", "ism", "ate", "iti", "ous", "ive", "ize",
)


def _is_consonant(word: str, i: int) -> bool:
    c = word[i]
    if c in _VOWELS:
        return False
    if c == "y":
        return i == 0 or not _is_consonant(word, i - 1)
    return True


def _measure(stem: str) -> int:
    """The number m of vowel-consonant sequences in ``stem`` ([C](VC){m}[V])."""
    m = 0
    i = 0
    n = len(stem)
    while i < n and _is_consonant(stem, i):
        i += 1
    while i < n:
        while i < n and not _is_consonant(stem, i):
            i += 1
        if i >= n:
            break
        m += 1
        while i < n and _is_consonant(stem, i):
            i += 1
    return m


def _has_vowel(stem: str) -> bool:
    return any(not _is_consonant(stem, i) for i in range(len(stem)))


def _double_consonant(word: str) -> bool:
    return len(word) >= 2 and word[-1] == word[-2] and _is_consonant(word, len(word) - 1)


def _cvc(word: str) -> bool:
    """consonant-vowel-consonant at the end, the last consonant not w, x or y"""
    n = len(word)
    if n < 3 or not _is_consonant(word, n - 3) or _is_consonant(word, n - 2) or not _is_consonant(word, n - 1):
        return False
    return word[-1] not in "wxy"


MAX_WORD = 64  # longer words (in bytes) are left alone, like SQLite does


@lru_cache(maxsize=65536)
def stem(word: str) -> str:
    """The stem of a lower case word.  Like the stemmer of SQLite this works on the bytes of the UTF-8 text, in which a letter that is
    not ASCII is two or three consonants; words of two bytes or less, or longer than 64, are returned as they are."""
    if word.isascii():
        return _stem(word)
    raw = word.encode("utf-8", "surrogatepass")
    if len(raw) > MAX_WORD:
        return word
    return _stem(raw.decode("latin-1")).encode("latin-1").decode("utf-8", "surrogatepass")


def _stem(word: str) -> str:
    if len(word) <= 2 or len(word) > MAX_WORD:
        return word
    w = word

    # step 1a
    if w.endswith("sses"):
        w = w[:-2]
    elif w.endswith("ies"):
        w = w[:-2]
    elif w.endswith("ss"):
        pass
    elif w.endswith("s"):
        w = w[:-1]

    # step 1b
    flag = False
    if w.endswith("eed"):
        if _measure(w[:-3]) > 0:
            w = w[:-1]
    elif w.endswith("ed") and _has_vowel(w[:-2]):
        w = w[:-2]
        flag = True
    elif w.endswith("ing") and _has_vowel(w[:-3]):
        w = w[:-3]
        flag = True
    if flag:
        if w.endswith(("at", "bl", "iz")):
            w += "e"
        elif _double_consonant(w) and w[-1] not in "lsz":
            w = w[:-1]
        elif _measure(w) == 1 and _cvc(w):
            w += "e"

    # step 1c
    if w.endswith("y") and _has_vowel(w[:-1]):
        w = w[:-1] + "i"

    # step 2
    for suffix, repl in _STEP2:
        if w.endswith(suffix):
            if _measure(w[: -len(suffix)]) > 0:
                w = w[: -len(suffix)] + repl
            break

    # step 3
    for suffix, repl in _STEP3:
        if w.endswith(suffix):
            if _measure(w[: -len(suffix)]) > 0:
                w = w[: -len(suffix)] + repl
            break

    # step 4
    for suffix in _STEP4:
        if w.endswith(suffix):
            base = w[: -len(suffix)]
            if _measure(base) > 1 and (suffix != "ion" or base.endswith(("s", "t"))):
                w = base
            break

    # step 5a
    if w.endswith("e"):
        base = w[:-1]
        m = _measure(base)
        if m > 1 or (m == 1 and not _cvc(base)):
            w = base
    # step 5b
    if _measure(w) > 1 and _double_consonant(w) and w.endswith("l"):
        w = w[:-1]
    return w
