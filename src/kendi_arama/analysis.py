"""Text analysis: from the text of a field to a list of tokens.

A token is ``(term, position, start, end)``: the normalised word, its number in the field (0, 1, 2 ...; a stop word leaves a gap, so
phrases keep their meaning) and the span of the word in the original text (used to highlight).  The same analyzer is used when a document
is indexed and for the words of a query, so both sides agree on what a word is.

``unicode61`` follows the tokenizer of SQLite FTS5 of the same name (the tests compare the two): letters, numbers and private use
characters make words, everything else separates them; words are lower-cased and the diacritics of Latin letters are removed.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Dict, Iterable, List, NamedTuple, Optional

from . import porter


class Token(NamedTuple):
    term: str
    pos: int
    start: int
    end: int


# the combining marks that occur in the decompositions of Latin letters (é = e + U+0301): they are dropped from a word
_LATIN_MARKS = frozenset(
    mark
    for cp in range(0xC0, 0x2000)
    for d in (unicodedata.normalize("NFD", chr(cp)),)
    if len(d) >= 2 and d[0].isascii() and d[0].isalpha()
    for mark in d[1:]
)


def _simple_fold(ch: str) -> str:
    """Simple case folding of one character (never changes the length of the text): ς -> σ, ß stays ß, İ stays İ."""
    folded = ch.casefold()
    if len(folded) == 1:
        return folded
    lowered = ch.lower()
    return lowered if len(lowered) == 1 else ch


# ǡ (a with dot above and macron) keeps both marks in SQLite even in the mode that removes all marks (an omission in its table)
_MODE2_KEEPS = frozenset("\u01e0\u01e1")


class Analyzer:
    """Base class.  ``tokens`` is the whole job; ``spec`` describes the analyzer so that an index can be reopened with the same one."""

    name = "analyzer"

    def tokens(self, text: str) -> List[Token]:
        raise NotImplementedError

    def terms(self, text: str) -> List[str]:
        return [t.term for t in self.tokens(text)]

    def spec(self) -> dict:
        return {"name": self.name}

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Analyzer) and self.spec() == other.spec()

    def __hash__(self) -> int:
        return hash(repr(sorted(self.spec().items())))

    def __repr__(self) -> str:
        return "%s(%s)" % (type(self).__name__, ", ".join("%s=%r" % kv for kv in self.spec().items() if kv[0] != "name"))


_ASCII_WORD = re.compile(r"[0-9A-Za-z]+")


class Unicode61(Analyzer):
    name = "unicode61"

    def __init__(self, remove_diacritics: int = 1, tokenchars: str = "", separators: str = "") -> None:
        if remove_diacritics not in (0, 1, 2):
            raise ValueError("remove_diacritics must be 0, 1 or 2")
        self.remove_diacritics = remove_diacritics
        self.tokenchars = tokenchars
        self.separators = separators
        self._tokenchars = frozenset(tokenchars)
        self._separators = frozenset(separators)
        self._cache: Dict[str, Optional[str]] = {}
        self._plain_ascii = not tokenchars and not separators

    def spec(self) -> dict:
        return {"name": self.name, "remove_diacritics": self.remove_diacritics, "tokenchars": self.tokenchars, "separators": self.separators}

    def _fold(self, ch: str) -> Optional[str]:
        """The folded form of a character that belongs to a word ('' for a mark that is dropped), or None for a separator."""
        cache = self._cache
        try:
            return cache[ch]
        except KeyError:
            pass
        result: Optional[str]
        if ch in self._separators:
            result = None
        elif ch in self._tokenchars:
            result = _simple_fold(ch)
        elif ch < "\x80":
            result = ch.lower() if ch.isalnum() else None
        else:
            cat = unicodedata.category(ch)
            if (cat[0] in "LN" or cat == "Co" or cat == "Cn" and ord(ch) & 0xFFFE != 0xFFFE) or ch in _LATIN_MARKS:
                result = self._fold_letter(ch)
            else:
                result = None
        cache[ch] = result
        return result

    def _fold_letter(self, ch: str) -> str:
        folded = _simple_fold(ch)
        mode = self.remove_diacritics
        if mode == 0:
            return folded
        if folded in _LATIN_MARKS:
            return ""
        if folded >= "\x80" and folded not in _MODE2_KEEPS:
            decomposed = unicodedata.normalize("NFD", folded)
            if decomposed[0].isascii() and decomposed[0].isalpha() and len(decomposed) > 1:
                if (mode == 1 and len(decomposed) == 2 or mode == 2) and all(m in _LATIN_MARKS for m in decomposed[1:]):
                    return _simple_fold(decomposed[0])
        return folded

    def tokens(self, text: str) -> List[Token]:
        out: List[Token] = []
        if self._plain_ascii and text.isascii():
            for pos, m in enumerate(_ASCII_WORD.finditer(text)):
                out.append(Token(m.group().lower(), pos, m.start(), m.end()))
            return out
        fold = self._fold
        n = len(text)
        i = 0
        pos = 0
        marks = _LATIN_MARKS
        tokenchars = self._tokenchars
        while i < n:
            f = fold(text[i])
            if f is None or (text[i] in marks and text[i] not in tokenchars):  # a mark inside a word is part of it, it cannot begin one
                i += 1
                continue
            start = i
            parts = [f]
            i += 1
            while i < n:
                f = fold(text[i])
                if f is None:
                    break
                parts.append(f)
                i += 1
            term = "".join(parts)
            if term:
                out.append(Token(term, pos, start, i))
                pos += 1
        return out


class Ascii(Analyzer):
    """FTS5's ``ascii`` tokenizer: only ASCII letters and digits are folded and separate words, every other character is part of a word."""

    name = "ascii"

    def tokens(self, text: str) -> List[Token]:
        out: List[Token] = []
        n = len(text)
        i = 0
        pos = 0
        while i < n:
            if _is_ascii_sep(text[i]):
                i += 1
                continue
            start = i
            while i < n and not _is_ascii_sep(text[i]):
                i += 1
            word = text[start:i]
            out.append(Token("".join(chr(ord(c) + 32) if "A" <= c <= "Z" else c for c in word), pos, start, i))
            pos += 1
        return out


def _is_ascii_sep(ch: str) -> bool:
    return ch < "\x80" and not ch.isalnum()


class Trigram(Analyzer):
    """Every run of three characters is a term: a query for "ear" finds "search".  Needs at least three characters in a query word."""

    name = "trigram"

    def __init__(self, case_sensitive: bool = False, remove_diacritics: bool = False) -> None:
        self.case_sensitive = case_sensitive
        self.remove_diacritics = remove_diacritics
        self._base = Unicode61(remove_diacritics=1 if remove_diacritics else 0)

    def spec(self) -> dict:
        return {"name": self.name, "case_sensitive": self.case_sensitive, "remove_diacritics": self.remove_diacritics}

    def _chars(self, text: str) -> List[str]:
        out = []
        for ch in text:
            if not self.case_sensitive:
                ch = _simple_fold(ch)
            if self.remove_diacritics and ch >= "\x80":
                d = unicodedata.normalize("NFD", ch)
                if d[0].isascii() and d[0].isalpha() and len(d) == 2 and d[1] in _LATIN_MARKS:
                    ch = d[0]
                elif ch in _LATIN_MARKS:
                    continue
            out.append(ch)
        return out

    def tokens(self, text: str) -> List[Token]:
        chars = self._chars(text)
        return [Token("".join(chars[i : i + 3]), i, i, i + 3) for i in range(len(chars) - 2)]


class Porter(Analyzer):
    """Wraps another analyzer and stems its terms with the Porter algorithm (running, runs, ran -> run, run, ran)."""

    name = "porter"

    def __init__(self, base: Optional[Analyzer] = None) -> None:
        self.base = base if base is not None else Unicode61()

    def spec(self) -> dict:
        return {"name": self.name, "base": self.base.spec()}

    def tokens(self, text: str) -> List[Token]:
        return [Token(porter.stem(t.term), t.pos, t.start, t.end) for t in self.base.tokens(text)]


ENGLISH_STOP_WORDS = frozenset(
    "a an and are as at be but by for if in into is it no not of on or such that the their then there these they this to was will with".split()
)
TURKISH_STOP_WORDS = frozenset(
    "acaba ama ancak bazı bir biz bu çok çünkü da daha de diye en gibi hem hep her hiç için ile ise kadar ki kim mı mi mu mü ne neden nasıl "
    "o ve veya ya yani".split()
)


class Stop(Analyzer):
    """Drops the words of a list.  The positions of the other words are kept, so "the cat" is still one word after "of"."""

    name = "stop"

    def __init__(self, base: Optional[Analyzer] = None, words: Iterable[str] = ENGLISH_STOP_WORDS) -> None:
        self.base = base if base is not None else Unicode61()
        self.words = frozenset(words)

    def spec(self) -> dict:
        return {"name": self.name, "base": self.base.spec(), "words": sorted(self.words)}

    def tokens(self, text: str) -> List[Token]:
        return [t for t in self.base.tokens(text) if t.term not in self.words]


# Turkish: the capital letters of the language are I/ı and İ/i, which general purpose lower-casing gets wrong (I -> i)
_TURKISH_LOWER = {"I": "ı", "İ": "i"}
_TURKISH_ASCII = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")
_TR_WORD = re.compile(r"[^\W_]+", re.UNICODE)

# endings of nouns and verbs, longest first: plural, possessive, case.  This is a "light" stemmer: it cuts a few common endings and
# keeps a stem of at least three letters (accuracy is not measured against an oracle, unlike the rest of the package)
_TURKISH_SUFFIXES = tuple(
    sorted(
        {
            "larımızdan", "lerimizden", "larımız", "lerimiz", "larınız", "leriniz", "ların", "lerin", "ların", "lardan", "lerden", "larda", "lerde",
            "lara", "lere", "ları", "leri", "lar", "ler", "ımız", "imiz", "umuz", "ümüz", "ınız", "iniz", "unuz", "ünüz", "dan", "den", "tan", "ten",
            "nın", "nin", "nun", "nün", "ın", "in", "un", "ün", "da", "de", "ta", "te", "ya", "ye", "yı", "yi", "yu", "yü", "ı", "i", "u", "ü", "a", "e",
        },
        key=len,
        reverse=True,
    )
)


def turkish_lower(text: str) -> str:
    return "".join(_TURKISH_LOWER.get(c) or c.lower() for c in text)


class Turkish(Analyzer):
    """Turkish text: ``I``/``İ`` lower-case correctly, optional ASCII folding (ç -> c ...), stop words and a stemmer.

    ``stemmer``: ``None`` (words stay as they are), ``"light"`` (cut common noun endings) or ``"prefix5"`` (keep the first five letters:
    the simple baseline that did well for Turkish retrieval in the literature).
    """

    name = "turkish"

    def __init__(self, ascii_fold: bool = False, stemmer: Optional[str] = None, stop_words: bool = False) -> None:
        if stemmer not in (None, "light", "prefix5"):
            raise ValueError("stemmer must be None, 'light' or 'prefix5'")
        self.ascii_fold = ascii_fold
        self.stemmer = stemmer
        self.stop_words = stop_words

    def spec(self) -> dict:
        return {"name": self.name, "ascii_fold": self.ascii_fold, "stemmer": self.stemmer, "stop_words": self.stop_words}

    def stem(self, word: str) -> str:
        if self.stemmer == "prefix5":
            return word[:5]
        if self.stemmer == "light":
            for _ in range(2):  # a word can have two endings (kitap-lar-ı)
                for suffix in _TURKISH_SUFFIXES:
                    if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                        word = word[: -len(suffix)]
                        break
                else:
                    break
        return word

    def tokens(self, text: str) -> List[Token]:
        out: List[Token] = []
        pos = 0
        for m in _TR_WORD.finditer(text):
            word = turkish_lower(m.group())
            if self.stop_words and word in TURKISH_STOP_WORDS:
                pos += 1
                continue
            word = self.stem(word)
            if self.ascii_fold:
                word = word.translate(_TURKISH_ASCII)
            out.append(Token(word, pos, m.start(), m.end()))
            pos += 1
        return out


_ANALYZERS = {"unicode61": Unicode61, "ascii": Ascii, "trigram": Trigram, "turkish": Turkish}


def analyzer_from_spec(spec: dict) -> Analyzer:
    spec = dict(spec)
    name = spec.pop("name")
    if name == "porter":
        return Porter(analyzer_from_spec(spec["base"]))
    if name == "stop":
        return Stop(analyzer_from_spec(spec["base"]), spec["words"])
    return _ANALYZERS[name](**spec)


def get_analyzer(value: "str | Analyzer | dict | None") -> Analyzer:
    """An analyzer from an object, a spec or a short name: ``"unicode61"``, ``"porter"``, ``"english"`` (porter + stop words) ..."""
    if value is None:
        return Unicode61()
    if isinstance(value, Analyzer):
        return value
    if isinstance(value, dict):
        return analyzer_from_spec(value)
    if value == "porter":
        return Porter(Unicode61())
    if value == "english":
        return Porter(Stop(Unicode61(), ENGLISH_STOP_WORDS))
    if value in _ANALYZERS:
        return _ANALYZERS[value]()
    raise ValueError("unknown analyzer %r" % (value,))
