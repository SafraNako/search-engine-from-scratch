import pytest

from kendi_arama import BM25, Field, Index, QuerySyntaxError, Schema, Stop, Porter, Unicode61, Trigram, Turkish


def build(path, docs, fields=None, **writer):
    fields = fields or [Field("title"), Field("body")]
    idx = Index.create(path, Schema(fields))
    with idx.writer(**writer) as w:
        for key, doc in docs.items():
            w.add(key, doc)
    return idx.searcher()


DOCS = {
    "1": {"title": "The red fox", "body": "the quick red fox jumps over the lazy dog"},
    "2": {"title": "Lazy dog", "body": "a lazy dog sleeps all day; the dog is lazy"},
    "3": {"title": "Quick start", "body": "quick quick quick brown fox"},
    "4": {"title": "Cats", "body": "completely unrelated text about cats and kittens"},
    "5": {"title": "", "body": "dog days"},
}


@pytest.fixture
def s(index_dir):
    searcher = build(index_dir, DOCS)
    yield searcher
    searcher.close()


def k(searcher, query, **kw):
    return sorted(searcher.search(query, limit=None, **kw).keys())


def test_words_and_boolean_operators(s):
    assert k(s, "fox") == ["1", "3"]
    assert k(s, "FOX") == ["1", "3"]  # case does not matter
    assert k(s, "lazy dog") == ["1", "2"]
    assert k(s, "lazy OR cats") == ["1", "2", "4"]
    assert k(s, "dog NOT lazy") == ["5"]
    assert k(s, "(fox OR cats) AND quick") == ["1", "3"]
    assert k(s, "nosuchword") == [] and k(s, "fox nosuchword") == [] and k(s, "fox OR nosuchword") == ["1", "3"]
    assert k(s, "fox NOT nosuchword") == ["1", "3"] and k(s, "nosuchword NOT fox") == []


def test_phrases_follow_the_order_and_adjacency(s):
    assert k(s, '"red fox"') == ["1"]
    assert k(s, '"fox red"') == []
    assert k(s, '"quick red"') == ["1"] and k(s, '"quick fox"') == []
    assert k(s, "red + fox") == ["1"]
    assert k(s, '"lazy dog"') == ["1", "2"]
    assert k(s, '"quick quick quick"') == ["3"] and k(s, '"quick quick quick quick"') == []
    assert k(s, '"the lazy"') == ["1"]  # across the stop word "the", which is a word here


def test_prefix_and_first_word(s):
    assert k(s, "qui*") == ["1", "3"]
    assert k(s, "kit*") == ["4"] and k(s, "zz*") == []
    assert k(s, '"red f"*') == ["1"]
    assert k(s, "^quick") == ["3"] and k(s, "^the") == ["1"]
    assert k(s, "body:^quick") == ["3"]
    assert k(s, "title:^lazy") == ["2"] and k(s, "title:^dog") == []
    assert k(s, "^dog") == ["5"]


def test_fields(s):
    assert k(s, "title:fox") == ["1"]
    assert k(s, "body:fox") == ["1", "3"]
    assert k(s, "-title:fox") == ["1", "3"]
    assert k(s, "{title body}:cats") == ["4"]
    assert k(s, "title:(lazy OR cats)") == ["2", "4"]
    assert k(s, "title:quick body:quick") == ["3"]
    with pytest.raises(QuerySyntaxError):
        s.search("author:fox")


def test_near(s):
    # "the quick red fox jumps over the lazy dog": between quick and dog there are six words
    assert k(s, "NEAR(quick dog, 5)") == []
    assert k(s, "NEAR(quick dog, 6)") == ["1"]
    assert k(s, "NEAR(dog quick, 6)") == ["1"]  # either order
    assert k(s, "NEAR(fox dog)") == ["1"]  # 10 words by default
    assert k(s, "NEAR(quick fox, 0)") == [] and k(s, "NEAR(quick brown, 0)") == ["3"] and k(s, "NEAR(brown fox, 0)") == ["3"]
    assert k(s, 'NEAR("quick quick" fox, 1)') == ["3"]
    assert k(s, "NEAR(lazy lazy, 3)") == ["1", "2"]
    assert k(s, "NEAR(quick)") == k(s, "quick")
    # NEAR works inside a field only
    assert k(s, "NEAR(fox lazy)") == ["1"] and k(s, "NEAR(red sleeps)") == [] and k(s, "NEAR(lazy dog, 0)") == ["1", "2"]
    # with three phrases every one has to be within the distance of the last one: quick and fox have one word between them
    assert k(s, "NEAR(quick red fox, 0)") == [] and k(s, "NEAR(quick red fox, 1)") == ["1"] and k(s, "NEAR(quick red lazy, 1)") == []


def test_phrases_without_words(s):
    assert k(s, '"!!"') == [] and k(s, '""') == []
    assert k(s, 'fox "!!"') == ["1", "3"]  # left out of a list
    assert k(s, 'fox AND "!!"') == []  # but nothing matches an explicit AND with it
    assert k(s, 'fox OR "!!"') == ["1", "3"]
    assert k(s, 'fox NOT "!!"') == ["1", "3"] and k(s, '"!!" NOT fox') == []
    assert k(s, 'NEAR(fox "!!", 3)') == ["1", "3"]


def test_results_are_ranked_and_paged(s):
    all_ = s.search("dog OR fox OR quick", limit=None, scorer=BM25())
    assert all_.total == 4 and len(all_) == 4
    scores = [h.score for h in all_]
    assert scores == sorted(scores, reverse=True)
    page1 = s.search("dog OR fox OR quick", limit=2)
    page2 = s.search("dog OR fox OR quick", limit=2, offset=2)
    page3 = s.search("dog OR fox OR quick", limit=2, offset=4)
    assert page1.total == page2.total == page3.total == 4
    assert page1.keys() + page2.keys() + page3.keys() == all_.keys()
    assert len(page3) == 0 and s.search("fox", limit=0).total == 2 and len(s.search("fox", limit=0)) == 0
    assert s.search("fox", offset=10).keys() == []
    # a page in the middle
    assert s.search("dog OR fox OR quick", limit=1, offset=1).keys() == all_.keys()[1:2]


def test_a_short_document_scores_higher_than_a_long_one_with_the_same_word(s):
    assert s.search("dog").keys()[0] == "5"  # "dog days": the word is half of the document
    assert s.search("quick").keys()[0] == "3"  # three occurrences


def test_hits_carry_the_stored_fields(s):
    h = s.search("cats").hits[0]
    assert h.key == "4" and h["title"] == "Cats" and h.get("nothing", "x") == "x" and "kittens" in h.fields["body"] and h.score > 0
    assert s.matches("fox") == ["1", "3"] and s.count("fox") == 2 and s.count("nosuch") == 0


def test_field_weights(index_dir):
    docs = {"a": {"title": "fox", "body": "x x x x x x x x"}, "b": {"title": "x", "body": "fox x x x x x x x"}}
    searcher = build(index_dir, docs)
    with searcher as sr:
        tie = sr.search("fox", limit=None)
        assert tie.keys() == ["a", "b"] and tie.hits[0].score == tie.hits[1].score  # the same document length and frequency: a tie
        assert sr.search("fox", weights={"title": 10}).keys() == ["a", "b"] and sr.search("fox", weights={"title": 10}).hits[0].score > tie.hits[0].score
        assert sr.search("fox", weights={"body": 10}).keys() == ["b", "a"]
        assert sr.search("fox", weights={"title": 0, "body": 1}).hits[0].key == "b"
        assert sr.search("fox", weights={"title": 0, "body": 0}).hits[0].score == 0.0
    # the weight in the schema is the default
    idx = Index.create(index_dir + "w", Schema([Field("title", weight=10.0), Field("body")]))
    with idx.writer() as w:
        for key, d in docs.items():
            w.add(key, d)
    with idx.searcher() as sr:
        assert sr.search("fox").keys() == ["a", "b"] and sr.search("fox").hits[0].score > sr.search("fox").hits[1].score
        assert sr.search("fox", weights={"title": 1}).hits[0].score == sr.search("fox", weights={"title": 1}).hits[1].score


def test_stemming_and_stop_words(index_dir):
    fields = [Field("text", Porter(Stop(Unicode61())))]
    docs = {"1": {"text": "The runners were running quickly"}, "2": {"text": "He runs"}, "3": {"text": "A rune"}, "4": {"text": "cats of dogs"}}
    with build(index_dir, docs, fields) as sr:
        assert k(sr, "run") == ["1", "2"] and k(sr, "running") == ["1", "2"] and k(sr, "runs") == ["1", "2"]
        assert k(sr, "runner") == ["1"] and k(sr, "rune") == ["3"]
        assert k(sr, "quickly") == ["1"] and k(sr, "quick") == []  # the Porter stem of quickly is quickli
        assert k(sr, "the") == [] and k(sr, "of") == []  # a stop word is not in the index, and not in a query either
        assert k(sr, '"the runners"') == ["1"]
        assert k(sr, '"runners were"') == ["1"] and k(sr, '"were running"') == ["1"] and k(sr, '"runners running"') == []
        # the position of the words after a stop word is kept: "cats of dogs" has a gap between cats and dogs
        assert k(sr, '"cats of dogs"') == ["4"] and k(sr, '"cats dogs"') == [] and k(sr, "NEAR(cats dogs, 1)") == ["4"] and k(sr, "NEAR(cats dogs, 0)") == []


def test_trigrams_find_substrings(index_dir):
    docs = {"1": {"text": "The searching engine"}, "2": {"text": "research papers"}, "3": {"text": "nothing"}}
    with build(index_dir, docs, [Field("text", Trigram())]) as sr:
        assert k(sr, "earch") == ["1", "2"] and k(sr, "arc") == ["1", "2"] and k(sr, "engin") == ["1"] and k(sr, "ing") == ["1", "3"]
        assert k(sr, "xyz") == [] and k(sr, "ea") == []  # shorter than three letters: no words in the query, so nothing matches
        assert k(sr, '"searching eng"') == ["1"] and k(sr, '"search eng"') == []  # the spaces are part of the substring


def test_turkish_text(index_dir):
    docs = {"1": {"text": "IŞIK hızı saniyede çok yüksektir"}, "2": {"text": "ışık kaynağı"}, "3": {"text": "Isı ve sıcaklık"}}
    with build(index_dir, docs, [Field("text", Turkish())]) as sr:
        assert k(sr, "ışık") == ["1", "2"] and k(sr, "IŞIK") == ["1", "2"]
        assert k(sr, "isi") == []  # İ/i and I/ı are different letters
        assert k(sr, "ısı") == ["3"] and k(sr, "ISI") == ["3"]
    with build(index_dir + "2", docs, [Field("text", Turkish(ascii_fold=True))]) as sr:
        assert k(sr, "isik") == ["1", "2"] and k(sr, "cok") == ["1"]
    with build(index_dir + "3", {"1": {"text": "kitaplarımızdan"}, "2": {"text": "kitap okumak"}, "3": {"text": "kalem"}}, [Field("text", Turkish(stemmer="light"))]) as sr:
        assert k(sr, "kitap") == ["1", "2"] and k(sr, "kitaplar") == ["1", "2"]


def test_fuzzy_queries(index_dir):
    docs = {"1": {"text": "colour of the sea"}, "2": {"text": "color wheel"}, "3": {"text": "the colonel"}, "4": {"text": "culture"}}
    with build(index_dir, docs, [Field("text")]) as sr:
        assert k(sr, "colour~1") == ["1", "2"]
        assert k(sr, "colour~2") == ["1", "2"] and k(sr, "colour~3") == ["1", "2", "3", "4"]
        assert k(sr, "colour~0") == ["1"] and k(sr, "colour") == ["1"]
        assert k(sr, "coluor~1") == ["1", "2"]  # one swap makes colour of it (or one deletion, color)
        assert k(sr, "cloour~1") == ["1"]  # one swap away from colour, two edits from color
        assert k(sr, "the + colour~1") == []  # the phrase "the colour": there is none (it is "colour of")
        assert k(sr, "colour~1 + of") == ["1"] and k(sr, "colour~1 + wheel") == ["2"] and k(sr, "colour~1 + sea") == []
        assert k(sr, '"colour of"~1') == ["1"]  # the suffix after a string is for every word of it
        assert k(sr, '"colour~1 of"') == []  # inside the quotes it is only punctuation
        assert [t for t, _, _ in sr.suggest("colur")][:2] == ["color", "colour"] and sr.suggest("zzzzzzzz") == [] and sr.suggest("two words") == []
        with pytest.raises(QuerySyntaxError):
            sr.search("colour~1", fuzzy=False)


def test_scores_do_not_depend_on_the_number_of_segments(index_dir):
    docs = {str(i): {"title": "t%d common" % (i % 4), "body": ("fox " * (i % 5)) + "filler words " * (i % 3)} for i in range(40)}
    one = build(index_dir + "a", docs)
    many = build(index_dir + "b", docs, max_buffered_docs=3, auto_merge=False)
    with one, many:
        assert len(many.segs) > 5 and len(one.segs) == 1
        for q in ["fox", "fox OR common", '"filler words"', "t1 AND fox", "NEAR(fox filler, 3)"]:
            a = [(h.key, round(h.score, 12)) for h in one.search(q, limit=None)]
            b = [(h.key, round(h.score, 12)) for h in many.search(q, limit=None)]
            assert sorted(a) == sorted(b)


def test_explain(s):
    ex = s.explain("lazy OR cats", "2", scorer=BM25())
    ranked = {h.key: h.score for h in s.search("lazy OR cats", limit=None)}
    assert ex.key == "2" and abs(ex.score - ranked["2"]) < 1e-12
    by = {leaf.text: leaf for leaf in ex.leaves}
    assert by["title:lazy | body:lazy"].freq == 3 and by["title:cats | body:cats"].freq == 0 and by["title:cats | body:cats"].contribution == 0
    assert by["title:lazy | body:lazy"].df == 2 and by["title:lazy | body:lazy"].counted
    assert s.explain("lazy", "4") is None and s.explain("lazy", "nosuchkey") is None
    assert abs(sum(leaf.contribution for leaf in ex.leaves) - ex.score) < 1e-12
