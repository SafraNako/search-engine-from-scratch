"""The query language and its parser.

The syntax is the one of SQLite FTS5 (the tests run the same queries through both engines), plus a fuzzy suffix::

    cat dog                  both words (a space is AND)
    cat OR dog               either
    cat NOT dog              cat without dog        (NOT binds tighter than AND, AND tighter than OR; (...) groups)
    "black cat"              a phrase: the words next to each other, in this order        (black + cat  is the same)
    cat*                     a prefix: cat, cats, category ...
    ^cat                     cat as the first word of a field
    title:cat                only in the field title   ({title body}:cat, -title:cat, title:(cat OR dog))
    NEAR(cat dog, 5)         the phrases within 5 words of each other (10 when the number is left out)
    colour~1                 fuzzy: words within one edit of colour (insert, delete, replace, swap)    [not in FTS5]

A bare word is letters, digits, ``_`` and any non-ASCII character; everything else has to be inside double quotes.  The words ``AND``,
``OR`` and ``NOT`` are operators only in capitals, ``NEAR`` only in capitals and in front of a ``(``.
"""
from __future__ import annotations

from typing import List, NamedTuple, Optional, Sequence

from .errors import QuerySyntaxError
from .schema import Schema


class Part(NamedTuple):
    """One string of a phrase: its text, whether its last word is a prefix (``*``) and the edit distance of the fuzzy suffix (0 for none)."""

    text: str
    prefix: bool = False
    fuzzy: int = 0


class Query:
    __slots__ = ()


class Phrase(Query):
    __slots__ = ("parts", "first", "colset")

    def __init__(self, parts: Sequence[Part], first: bool = False, colset: Optional[frozenset] = None) -> None:
        self.parts = tuple(parts)
        self.first = first
        self.colset = colset  # indexes of the fields where the phrase may match; None for all of them

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Phrase) and (self.parts, self.first, self.colset) == (other.parts, other.first, other.colset)

    def __hash__(self) -> int:
        return hash((self.parts, self.first, self.colset))

    def __repr__(self) -> str:
        return "Phrase(%r%s%s)" % (list(self.parts), ", first" if self.first else "", ", colset=%s" % sorted(self.colset) if self.colset is not None else "")


class Near(Query):
    __slots__ = ("phrases", "distance", "colset")

    def __init__(self, phrases: Sequence[Phrase], distance: int = 10, colset: Optional[frozenset] = None) -> None:
        self.phrases = tuple(phrases)
        self.distance = distance
        self.colset = colset

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Near) and (self.phrases, self.distance, self.colset) == (other.phrases, other.distance, other.colset)

    def __hash__(self) -> int:
        return hash((self.phrases, self.distance, self.colset))

    def __repr__(self) -> str:
        return "Near(%r, %d)" % (list(self.phrases), self.distance)


class _Binary(Query):
    __slots__ = ("left", "right")

    def __init__(self, left: Query, right: Query) -> None:
        self.left = left
        self.right = right

    def __eq__(self, other: object) -> bool:
        return type(other) is type(self) and (self.left, self.right) == (other.left, other.right)  # type: ignore[attr-defined]

    def __hash__(self) -> int:
        return hash((type(self).__name__, self.left, self.right))

    def __repr__(self) -> str:
        return "%s(%r, %r)" % (type(self).__name__, self.left, self.right)


class And(_Binary):
    """Both sides.  ``implicit`` is the AND of a list of phrases (``cat dog``): there an empty phrase is left out, while the explicit
    ``AND`` with a phrase that has no words matches nothing."""

    __slots__ = ("implicit",)

    def __init__(self, left: Query, right: Query, implicit: bool = False) -> None:
        super().__init__(left, right)
        self.implicit = implicit

    def __eq__(self, other: object) -> bool:
        return isinstance(other, And) and (self.left, self.right, self.implicit) == (other.left, other.right, other.implicit)

    def __hash__(self) -> int:
        return hash(("And", self.left, self.right, self.implicit))

    def __repr__(self) -> str:
        return "And(%r, %r%s)" % (self.left, self.right, ", implicit" if self.implicit else "")


class Or(_Binary):
    __slots__ = ()


class Not(_Binary):
    """``left`` without ``right``."""

    __slots__ = ()


# ---------------------------------------------------------------------------------------------------------------- lexer

STRING, AND, OR, NOT, NEAR, LP, RP, LCP, RCP, STAR, PLUS, MINUS, COMMA, COLON, CARET, EOF = (
    "STRING", "AND", "OR", "NOT", "NEAR", "(", ")", "{", "}", "*", "+", "-", ",", ":", "^", "EOF",
)
_PUNCT = {"(": LP, ")": RP, "{": LCP, "}": RCP, "*": STAR, "+": PLUS, "-": MINUS, ",": COMMA, ":": COLON, "^": CARET}
_KEYWORDS = {"AND": AND, "OR": OR, "NOT": NOT, "NEAR": NEAR}


class _Tok(NamedTuple):
    kind: str
    value: str
    pos: int
    quoted: bool = False


def _is_bareword(ch: str) -> bool:
    return ch >= "\x80" or ch.isalnum() or ch == "_" or ch == "\x1a"


def lex(text: str, fuzzy: bool = True) -> List[_Tok]:
    out: List[_Tok] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in " \t\n\r":
            i += 1
            continue
        if ch in _PUNCT:
            out.append(_Tok(_PUNCT[ch], ch, i))
            i += 1
        elif ch == '"':
            start = i
            i += 1
            buf = []
            while True:
                if i >= n:
                    raise QuerySyntaxError("unterminated string", start)
                if text[i] == '"':
                    if i + 1 < n and text[i + 1] == '"':
                        buf.append('"')
                        i += 2
                        continue
                    i += 1
                    break
                buf.append(text[i])
                i += 1
            out.append(_Tok(STRING, "".join(buf), start, True))
        elif _is_bareword(ch):
            start = i
            while i < n and _is_bareword(text[i]):
                i += 1
            word = text[start:i]
            kind = _KEYWORDS.get(word, STRING)
            if kind == NEAR:  # NEAR is an operator only in front of "(", else it is a word like any other
                j = i
                while j < n and text[j] in " \t\n\r":
                    j += 1
                if j >= n or text[j] != "(":
                    kind = STRING
            out.append(_Tok(kind, word, start))
        elif ch == "~" and fuzzy:
            start = i
            i += 1
            while i < n and text[i].isdigit() and text[i].isascii():
                i += 1
            out.append(_Tok("~", text[start:i], start))
        else:
            raise QuerySyntaxError("syntax error near %r" % ch, i)
    out.append(_Tok(EOF, "", n))
    return out


# ---------------------------------------------------------------------------------------------------------------- parser

class _Parser:
    def __init__(self, text: str, schema: Schema, fuzzy: bool) -> None:
        self.schema = schema
        self.fuzzy = fuzzy
        self.toks = lex(text, fuzzy)
        self.i = 0

    @property
    def tok(self) -> _Tok:
        return self.toks[self.i]

    def peek(self, k: int = 1) -> _Tok:
        return self.toks[min(self.i + k, len(self.toks) - 1)]

    def take(self) -> _Tok:
        t = self.toks[self.i]
        self.i += 1
        return t

    def fail(self, t: Optional[_Tok] = None) -> QuerySyntaxError:
        t = t or self.tok
        return QuerySyntaxError("syntax error near %r" % (t.value if t.kind != EOF else ""), t.pos)

    def expect(self, kind: str) -> _Tok:
        if self.tok.kind != kind:
            raise self.fail()
        return self.take()

    # expr := and ('OR' and)*      and := not ('AND' not)*      not := list ('NOT' list)*
    def parse(self) -> Query:
        q = self.parse_or()
        if self.tok.kind != EOF:
            raise self.fail()
        return q

    def parse_or(self) -> Query:
        left = self.parse_and()
        while self.tok.kind == OR:
            self.take()
            left = Or(left, self.parse_and())
        return left

    def parse_and(self) -> Query:
        left = self.parse_not()
        while self.tok.kind == AND:
            self.take()
            left = And(left, self.parse_not())
        return left

    def parse_not(self) -> Query:
        left = self.parse_list()
        while self.tok.kind == NOT:
            self.take()
            left = Not(left, self.parse_list())
        return left

    def starts_item(self) -> bool:
        return self.tok.kind in (STRING, CARET, NEAR, MINUS, LCP)

    def parse_list(self) -> Query:
        if self.tok.kind == LP:
            return self.parse_paren(None)
        if self.tok.kind in (MINUS, LCP) or (self.tok.kind == STRING and self.peek().kind == COLON):
            colset = self.parse_colset()
            self.expect(COLON)
            if self.tok.kind == LP:
                return self.parse_paren(colset)
            first = self.parse_nearset(colset)
        else:
            first = self.parse_nearset(None)
        q = first
        while self.starts_item():
            colset = None
            if self.tok.kind in (MINUS, LCP) or (self.tok.kind == STRING and self.peek().kind == COLON):
                colset = self.parse_colset()
                self.expect(COLON)
                if self.tok.kind == LP:
                    raise self.fail()
            q = And(q, self.parse_nearset(colset), implicit=True)
        return q

    def parse_paren(self, colset: Optional[frozenset]) -> Query:
        self.expect(LP)
        q = self.parse_or()
        self.expect(RP)
        return _restrict(q, colset) if colset is not None else q

    def parse_colset(self) -> frozenset:
        negate = False
        if self.tok.kind == MINUS:
            self.take()
            negate = True
        names: List[str] = []
        if self.tok.kind == LCP:
            self.take()
            while self.tok.kind == STRING:
                names.append(self.take().value)
            if not names:
                raise self.fail()
            self.expect(RCP)
        elif self.tok.kind == STRING:
            names.append(self.take().value)
        else:
            raise self.fail()
        idx = set()
        lower = {n.lower(): i for n, i in self.schema.index.items()}
        for name in names:
            key = name.lower() if name.isascii() else name  # field names are ASCII, and compared without regard to case like in SQL
            if key not in lower:
                raise QuerySyntaxError("no such field: %s" % name, self.toks[self.i - 1].pos)
            idx.add(lower[key])
        if negate:
            idx = set(range(len(self.schema))) - idx
        return frozenset(idx)

    def parse_phrase(self) -> List[Part]:
        parts: List[Part] = []
        while True:
            t = self.expect(STRING)
            star = False
            fuzzy = 0
            if self.tok.kind == STAR:
                self.take()
                star = True
            if self.tok.kind == "~":
                v = self.take().value[1:]
                fuzzy = int(v) if v else 1
            parts.append(Part(t.value, star, fuzzy))
            if self.tok.kind == PLUS:
                self.take()
                continue
            return parts

    def parse_nearset(self, colset: Optional[frozenset]) -> Query:
        if self.tok.kind == NEAR:
            self.take()
            self.expect(LP)
            phrases = [Phrase(self.parse_phrase())]
            while self.tok.kind == STRING:
                phrases.append(Phrase(self.parse_phrase()))
            distance = 10
            if self.tok.kind == COMMA:
                self.take()
                t = self.expect(STRING)
                if not t.value.isdigit() or not t.value.isascii():
                    raise QuerySyntaxError("expected integer, got %r" % t.value, t.pos)
                distance = int(t.value)
            self.expect(RP)
            return Near(phrases, distance, colset)
        first = False
        if self.tok.kind == CARET:
            self.take()
            first = True
        return Phrase(self.parse_phrase(), first, colset)


def _restrict(q: Query, colset: frozenset) -> Query:
    """Applies ``col:(...)`` to every phrase inside the parentheses (intersecting with a column filter that is already there)."""
    if isinstance(q, Phrase):
        return Phrase(q.parts, q.first, colset if q.colset is None else q.colset & colset)
    if isinstance(q, Near):
        return Near(q.phrases, q.distance, colset if q.colset is None else q.colset & colset)
    if isinstance(q, And):
        return And(_restrict(q.left, colset), _restrict(q.right, colset), q.implicit)
    return type(q)(_restrict(q.left, colset), _restrict(q.right, colset))  # type: ignore[attr-defined, call-arg]


def parse_query(text: str, schema: Schema, fuzzy: bool = True) -> Query:
    return _Parser(text, schema, fuzzy).parse()


# ---------------------------------------------------------------------------------------------------------------- printing

def _quote(s: str) -> str:
    return '"' + s.replace('"', '""') + '"'


def _phrase_text(p: Phrase, schema: Optional[Schema], with_colset: bool = True) -> str:
    s = " + ".join(_quote(x.text) + ("*" if x.prefix else "") + ("~%d" % x.fuzzy if x.fuzzy else "") for x in p.parts)
    if p.first:
        s = "^" + s
    return s


def _colset_text(colset: Optional[frozenset], schema: Schema) -> str:
    if colset is None:
        return ""
    if not colset:
        return "-{%s}:" % " ".join(schema.names())
    names = [schema.fields[i].name for i in sorted(colset)]
    return "{%s}:" % " ".join(names)


def to_string(q: Query, schema: Schema) -> str:
    """A query string that parses back to an equal query."""
    if isinstance(q, Phrase):
        return _colset_text(q.colset, schema) + _phrase_text(q, schema)
    if isinstance(q, Near):
        return _colset_text(q.colset, schema) + "NEAR(%s, %d)" % (" ".join(_phrase_text(p, schema) for p in q.phrases), q.distance)
    if isinstance(q, And) and q.implicit:
        return "%s %s" % (to_string(q.left, schema), to_string(q.right, schema))
    op = {And: "AND", Or: "OR", Not: "NOT"}[type(q)]
    return "(%s) %s (%s)" % (to_string(q.left, schema), op, to_string(q.right, schema))  # type: ignore[attr-defined]
