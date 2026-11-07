import random

import pytest

from kendi_arama import Field, Index, Schema
from kendi_arama.fuzzy import close_terms, edit_distance


def reference(a, b):
    """Optimal string alignment distance, as a plain table (the textbook form)."""
    d = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        d[i][0] = i
    for j in range(len(b) + 1):
        d[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            c = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + c)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[len(a)][len(b)]


@pytest.mark.parametrize(
    "a,b,d",
    [("kitten", "sitting", 3), ("", "", 0), ("", "abc", 3), ("abc", "", 3), ("abc", "abc", 0), ("ab", "ba", 1), ("ca", "abc", 3), ("abcdef", "abcfed", 2),
     ("flaw", "lawn", 2), ("saturday", "sunday", 3), ("colour", "color", 1), ("colour", "coluor", 1), ("a", "b", 1), ("İstanbul", "istanbul", 1)],
)
def test_known_distances(a, b, d):
    assert edit_distance(a, b) == d == edit_distance(b, a)


def test_the_distance_equals_the_reference():
    rng = random.Random(1)
    for _ in range(3000):
        a = "".join(rng.choice("abc") for _ in range(rng.randint(0, 8)))
        b = "".join(rng.choice("abc") for _ in range(rng.randint(0, 8)))
        assert edit_distance(a, b) == reference(a, b), (a, b)
        for limit in (0, 1, 2, 3):
            got = edit_distance(a, b, limit)
            assert (got <= limit and got == reference(a, b)) or (got == limit + 1 and reference(a, b) > limit), (a, b, limit)


def test_triangle_inequality_up_to_the_swap_rule():
    rng = random.Random(2)
    words = ["".join(rng.choice("abcd") for _ in range(rng.randint(1, 6))) for _ in range(30)]
    for a in words:
        assert edit_distance(a, a) == 0
        for b in words:
            assert edit_distance(a, b) == edit_distance(b, a)
            assert edit_distance(a, b) >= abs(len(a) - len(b))


def test_close_terms_equal_a_scan_with_the_reference():
    rng = random.Random(3)
    terms = sorted({"".join(rng.choice("abcde") for _ in range(rng.randint(1, 7))) for _ in range(400)})
    pairs = [(t, i) for i, t in enumerate(terms)]
    for _ in range(40):
        word = "".join(rng.choice("abcde") for _ in range(rng.randint(1, 7)))
        for dist in (0, 1, 2):
            for prefix in (0, 1, 2):
                want = [(t, i) for t, i in pairs if reference(word, t) <= dist and t[:prefix] == word[:prefix]]
                assert list(close_terms(word, pairs, dist, prefix)) == want


def test_fuzzy_query_matches_the_words_within_the_distance(tmp_path):
    rng = random.Random(4)
    vocab = sorted({"".join(rng.choice("abcd") for _ in range(rng.randint(2, 6))) for _ in range(120)})
    docs = {str(i): " ".join(rng.sample(vocab, 5)) for i in range(150)}
    idx = Index.create(str(tmp_path / "i"), Schema([Field("text")]))
    with idx.writer(max_buffered_docs=40) as w:
        for k, t in docs.items():
            w.add(k, {"text": t})
    with idx.searcher() as s:
        for word in rng.sample(vocab, 15) + ["abcdab", "dd", "a"]:
            for dist in (1, 2):
                near = {t for t in vocab if reference(word, t) <= dist}
                want = sorted(k for k, t in docs.items() if near & set(t.split()))
                assert sorted(s.search("%s~%d" % (word, dist), limit=None).keys()) == want, (word, dist)
        # "did you mean": the nearest words first, the more common one first among equals
        got = s.suggest("abcd", max_distance=2, limit=100)
        assert [t for t, _, _ in got] == [t for t, d, n in sorted(((t, reference("abcd", t), sum(1 for x in docs.values() if t in x.split())) for t in vocab if reference("abcd", t) <= 2), key=lambda x: (x[1], -x[2], x[0]))]
