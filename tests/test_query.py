import pytest

from kendi_arama import Field, QuerySyntaxError, Schema
from kendi_arama.query import And, Near, Not, Or, Part, Phrase, lex, parse_query, to_string

SCHEMA = Schema([Field("title"), Field("body"), Field("tags")])


def parse(text, **kw):
    return parse_query(text, SCHEMA, **kw)


def P(*texts, first=False, colset=None):
    return Phrase([Part(t) if isinstance(t, str) else t for t in texts], first, colset)


def test_a_single_word_and_a_phrase():
    assert parse("cat") == P("cat")
    assert parse('"black cat"') == P("black cat")
    assert parse("black + cat") == P("black", "cat")
    assert parse('"say ""hi"""') == P('say "hi"')
    assert parse("cat*") == P(Part("cat", True))
    assert parse('"black cat" * ') == P(Part("black cat", True))
    assert parse("a* + b*") == P(Part("a", True), Part("b", True))
    assert parse("^cat") == P("cat", first=True)
    assert parse("^ cat") == P("cat", first=True)
    assert parse("colour~1") == P(Part("colour", False, 1))
    assert parse("colour~") == P(Part("colour", False, 1))
    assert parse("colour*~2") == P(Part("colour", True, 2))


def test_barewords():
    assert parse("foo_bar") == P("foo_bar")
    assert parse("café") == P("café")
    assert parse("日本語") == P("日本語")
    assert parse("x2") == P("x2")
    for bad in ["foo.bar", "foo!", "a@b", "#tag", "a;b", "[x]", "a/b"]:
        with pytest.raises(QuerySyntaxError):
            parse(bad)


def test_the_operators_are_operators_only_in_capitals():
    assert parse("cat and dog") == And(And(P("cat"), P("and"), implicit=True), P("dog"), implicit=True)
    assert parse("cat AND dog") == And(P("cat"), P("dog"))
    assert parse("not") == P("not")
    assert parse("near") == P("near")
    # NEAR is an operator only in front of a parenthesis
    assert parse("NEAR") == P("NEAR") and parse("a NEAR") == And(P("a"), P("NEAR"), implicit=True) and parse("NEAR a") == And(P("NEAR"), P("a"), implicit=True)
    assert parse('NEAR"a b"') == And(P("NEAR"), P("a b"), implicit=True)
    with pytest.raises(QuerySyntaxError):
        parse("NEAR:a")  # a word in front of a colon is a field, and there is no field of that name
    assert parse("NEAR (a b)") == Near([P("a"), P("b")], 10)
    assert parse('"AND"') == P("AND")  # quoted, it is a word
    with pytest.raises(QuerySyntaxError):
        parse("AND")
    with pytest.raises(QuerySyntaxError):
        parse("cat AND")
    with pytest.raises(QuerySyntaxError):
        parse("NOT cat")


def test_precedence_not_over_and_over_or():
    a, b, c, d = P("a"), P("b"), P("c"), P("d")
    assert parse("a OR b AND c") == Or(a, And(b, c))
    assert parse("a AND b OR c") == Or(And(a, b), c)
    assert parse("a OR b OR c") == Or(Or(a, b), c)
    assert parse("a NOT b AND c") == And(Not(a, b), c)
    assert parse("a AND b NOT c") == And(a, Not(b, c))
    assert parse("a NOT b NOT c") == Not(Not(a, b), c)
    assert parse("a b OR c d") == Or(And(a, b, implicit=True), And(c, d, implicit=True))
    assert parse("a b AND c") == And(And(a, b, implicit=True), c)
    assert parse("a NOT b c") == Not(a, And(b, c, implicit=True))  # the list on the right of NOT is one operand
    assert parse("(a OR b) AND (c OR d)") == And(Or(a, b), Or(c, d))
    assert parse("(a)") == a and parse("((a))") == a


def test_parentheses_cannot_be_part_of_a_list():
    for bad in ["(a) b", "a (b)", "(a) (b)", "(a OR b) c", "a b (c)", "()", "(a", "a)", "a b)", "((a)"]:
        with pytest.raises(QuerySyntaxError):
            parse(bad)
    assert parse("(a) AND (b)") == And(P("a"), P("b"))


def test_fields():
    assert parse("title:cat") == P("cat", colset=frozenset({0}))
    assert parse("{title tags}:cat") == P("cat", colset=frozenset({0, 2}))
    assert parse("-title:cat") == P("cat", colset=frozenset({1, 2}))
    assert parse("-{title body}:cat") == P("cat", colset=frozenset({2}))
    assert parse("-{title body tags}:cat") == P("cat", colset=frozenset())
    assert parse("TITLE:cat") == parse("title:cat")  # field names are case insensitive
    assert parse('"body":cat') == P("cat", colset=frozenset({1}))
    assert parse("title: cat") == parse("title:cat")
    assert parse("title:cat dog") == And(P("cat", colset=frozenset({0})), P("dog"), implicit=True)  # the filter is for the next phrase only
    assert parse("title:^cat") == P("cat", first=True, colset=frozenset({0}))
    for bad in ["nofield:cat", "title:", "title:title:cat", "^title:cat", "{}:cat", "{title}", "-cat", "- cat", "a -b"]:
        with pytest.raises(QuerySyntaxError):
            parse(bad)


def test_a_field_in_front_of_parentheses_applies_to_every_phrase_inside():
    c = frozenset({0})
    assert parse("title:(a OR b)") == Or(P("a", colset=c), P("b", colset=c))
    assert parse("title:(a b)") == And(P("a", colset=c), P("b", colset=c), implicit=True)
    assert parse("title:(body:a OR b)") == Or(P("a", colset=frozenset()), P("b", colset=c))  # the filters are intersected
    assert parse("{title body}:(tags:a)") == P("a", colset=frozenset())
    with pytest.raises(QuerySyntaxError):
        parse("title:(a) b")
    with pytest.raises(QuerySyntaxError):
        parse("a title:(b)")


def test_near():
    assert parse("NEAR(a b)") == Near([P("a"), P("b")], 10)
    assert parse("NEAR(a b, 3)") == Near([P("a"), P("b")], 3)
    assert parse("NEAR(a b, 0)") == Near([P("a"), P("b")], 0)
    assert parse('NEAR("a b" c* d + e, 7)') == Near([P("a b"), P(Part("c", True)), P("d", "e")], 7)
    assert parse("NEAR(a)") == Near([P("a")], 10)
    assert parse("title:NEAR(a b)") == Near([P("a"), P("b")], 10, frozenset({0}))
    assert parse("NEAR(a b) c") == And(Near([P("a"), P("b")], 10), P("c"), implicit=True)
    for bad in ["NEAR(a b,)", "NEAR(a b, x)", "NEAR(a b, -1)", "NEAR(, a b)", "NEAR()", "NEAR(a b", "NEAR(title:a b)", "NEAR(^a b)", "NEAR(a b, 1.5)", "NEAR(a b, 1 2)"]:
        with pytest.raises(QuerySyntaxError):
            parse(bad)


def test_other_errors():
    for bad in ["", "   ", '"unterminated', "a +", "+ a", "a + + b", "* a", "a **", "a ^ ^ b", "a^", "~1", "a ~~", ",", ":", "a:b:c"]:
        with pytest.raises(QuerySyntaxError):
            parse(bad)


def test_fuzzy_suffix_can_be_switched_off():
    with pytest.raises(QuerySyntaxError):
        parse("colour~1", fuzzy=False)
    assert parse("colour", fuzzy=False) == P("colour")


def test_tokens_and_positions_in_an_error():
    with pytest.raises(QuerySyntaxError) as e:
        parse("cat dog !")
    assert e.value.position == 8
    assert [t.kind for t in lex("a AND (b* OR \"c d\")")] == ["STRING", "AND", "(", "STRING", "*", "OR", "STRING", ")", "EOF"]


@pytest.mark.parametrize(
    "text",
    [
        "cat", "cat dog", "cat OR dog", "cat NOT dog", "a b OR c d", "a OR b AND c NOT d", '"a b" + c', "a* + b*", "^a", "title:a", "{title body}:a b", "-title:a",
        "title:(a OR b)", "NEAR(a b, 3)", "title:NEAR(a b)", "(a OR b) AND (c NOT d)", "a NOT b c", '"say ""hi"""', "x~2", "-{title body tags}:a", "tags:(a NOT b)",
    ],
)
def test_printing_a_query_gives_one_that_parses_to_the_same(text):
    q = parse(text)
    again = parse(to_string(q, SCHEMA))
    assert again == q
    assert to_string(again, SCHEMA) == to_string(q, SCHEMA)
