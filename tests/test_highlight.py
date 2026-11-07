import pytest

from kendi_arama import Field, Index, Porter, Schema, Unicode61
from kendi_arama.analysis import Token
from kendi_arama.highlight import highlight, snippet


def toks(text):
    return Unicode61().tokens(text)


def test_instances_are_wrapped_where_they_are():
    text = "the quick brown fox"
    assert highlight(text, toks(text), [(1, 1), (3, 3)], "[", "]") == "the [quick] brown [fox]"
    assert highlight(text, toks(text), [(1, 2)], "[", "]") == "the [quick brown] fox"
    assert highlight(text, toks(text), [], "[", "]") == text
    assert highlight("", [], [(0, 0)], "[", "]") == ""


def test_overlapping_instances_are_one_and_touching_ones_are_two():
    text = "a b c d e"
    assert highlight(text, toks(text), [(0, 1), (1, 2)], "[", "]") == "[a b c] d e"
    assert highlight(text, toks(text), [(0, 3), (1, 2)], "[", "]") == "[a b c d] e"
    assert highlight(text, toks(text), [(0, 0), (1, 1)], "[", "]") == "[a] [b] c d e"
    assert highlight(text, toks(text), [(2, 2), (0, 0), (2, 2)], "[", "]") == "[a] b [c] d e"


def test_the_text_between_words_is_kept():
    text = "Hello,  World!\n\tfoo_bar"
    assert highlight(text, toks(text), [(0, 1)], "<", ">") == "<Hello,  World>!\n\tfoo_bar"
    assert highlight(text, toks(text), [(3, 3)], "<", ">") == "Hello,  World!\n\tfoo_<bar>"


def test_stop_word_gaps_do_not_break_a_phrase():
    # "cats of dogs" with "of" dropped: positions 0 and 2; an instance covering 0..2 marks both words and what is between them
    text = "cats of dogs here"
    tokens = [Token("cat", 0, 0, 4), Token("dog", 2, 8, 12), Token("here", 3, 13, 17)]
    assert highlight(text, tokens, [(0, 2)], "[", "]") == "[cats of dogs] here"
    assert highlight(text, tokens, [(1, 1)], "[", "]") == text  # a position without a word is nothing


def test_snippet_picks_the_window_with_the_most_matches():
    words = ["w%d" % i for i in range(60)]
    words[10] = "fox"
    words[40] = "fox"
    words[41] = "dog"
    text = " ".join(words)
    tokens = toks(text)
    instances = [(10, 10), (40, 40), (41, 41)]
    out = snippet(text, tokens, instances, "[", "]", "...", 8)
    assert out.startswith("...") and out.endswith("...") and "[fox]" in out and "[dog]" in out and len(out.replace("...", "").split()) == 8
    short = snippet("a fox here", toks("a fox here"), [(1, 1)], "[", "]", "...", 8)
    assert short == "a [fox] here"
    assert snippet(text, tokens, [], "[", "]", "...", 4) == "w0 w1 w2 w3..."
    assert snippet("", [], [], "[", "]") == ""


def test_searcher_highlight_and_snippet(tmp_path):
    idx = Index.create(str(tmp_path / "i"), Schema([Field("title", Porter()), Field("body", Porter())]))
    with idx.writer() as w:
        w.add("1", {"title": "Running Foxes", "body": "The fox runs. A runner runs fast; the foxes run away."})
        w.add("2", {"title": "Cats", "body": "nothing here"})
    with idx.searcher() as s:
        assert s.highlight("run", "1", "body", "[", "]") == "The fox [runs]. A runner [runs] fast; the foxes [run] away."
        assert s.highlight("title:run", "1", "title", "[", "]") == "[Running] Foxes"
        assert s.highlight('"fox runs"', "1", "body", "<b>", "</b>") == "The <b>fox runs</b>. A runner runs fast; the <b>foxes run</b> away."  # stemming: "foxes run" is the phrase too
        assert s.highlight("fox*", "1", "title", "[", "]") == "Running [Foxes]"
        assert s.highlight("run", "2", "body") is None and s.highlight("run", "nokey", "body") is None
        assert s.highlight("run", "1", "title", "[", "]") == "[Running] Foxes"
        extract = s.snippet("runner", "1", "body", "[", "]", "…", 5)
        assert "[runner]" in extract and extract.startswith("…") and extract.endswith("…") and len(extract.strip("…").split()) == 5
        with pytest.raises(KeyError):
            s.highlight("run", "1", "nofield")
