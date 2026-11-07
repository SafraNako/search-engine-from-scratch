import os
import sqlite3

import pytest

from conftest import needs_fts5
from oracle import tokenizer_terms

from kendi_arama import Ascii, Porter, Stop, Trigram, Turkish, Unicode61, get_analyzer
from kendi_arama import porter
from kendi_arama.analysis import ENGLISH_STOP_WORDS, analyzer_from_spec, turkish_lower

HERE = os.path.dirname(__file__)


def terms(text, **kw):
    return Unicode61(**kw).terms(text)


def test_words_are_letters_and_digits():
    assert terms("Hello, World! foo_bar baz-qux 3.14") == ["hello", "world", "foo", "bar", "baz", "qux", "3", "14"]
    assert terms("") == [] and terms("  ,.;  ") == []


def test_positions_and_offsets():
    toks = Unicode61().tokens("The  cat, the DOG")
    assert [(t.term, t.pos, t.start, t.end) for t in toks] == [("the", 0, 0, 3), ("cat", 1, 5, 8), ("the", 2, 10, 13), ("dog", 3, 14, 17)]
    # the offsets also point at the right place with accents and in a text that is not ASCII
    text = "Café ÜBER straße"
    for t in Unicode61().tokens(text):
        assert text[t.start : t.end].lower().startswith(t.term[0]) or t.term[0] in "cuus"
    assert [text[t.start : t.end] for t in Unicode61().tokens(text)] == ["Café", "ÜBER", "straße"]


def test_accents_are_removed_from_latin_letters_only():
    assert terms("Café naïve Ünïcödé résumé") == ["cafe", "naive", "unicode", "resume"]
    assert terms("Café", remove_diacritics=0) == ["café"]
    assert terms("élève") == ["eleve"]  # the combining marks of a decomposed text are dropped
    assert terms("élève", remove_diacritics=0) == ["élève"]
    assert terms("αθήνα Σίσυφος ς") == ["αθήνα", "σίσυφοσ", "σ"]  # Greek keeps its accents; the final sigma is folded


def test_turkish_letters_in_the_generic_analyzer():
    # İ loses its dot like any accent; ı has no lower-case partner in Unicode 6.1, so it stays; ş, ç, ğ, ö, ü lose their marks
    assert terms("İstanbul ISPARTA ıslak ışık çalışma ağaç öğrenci") == ["istanbul", "isparta", "ıslak", "ısık", "calısma", "agac", "ogrenci"]


def test_case_folding():
    assert terms("STRASSE Straße ẞ") == ["strasse", "straße", "ß"]
    assert terms("ǅ ǆ Ǆ") == ["ǆ", "ǆ", "ǆ"]


def test_tokenchars_and_separators():
    assert terms("foo_bar-baz", tokenchars="_") == ["foo_bar", "baz"]
    assert terms("foo_bar-baz", tokenchars="_-") == ["foo_bar-baz"]
    assert terms("abcXdef", separators="X") == ["abc", "def"]


def test_other_scripts_and_a_run_of_ideographs_is_one_word():
    assert terms("日本語のテキスト 漢字") == ["日本語のテキスト", "漢字"]
    assert terms("Привет, МИР") == ["привет", "мир"]


def test_ascii_analyzer():
    assert Ascii().terms("Hello WORLD é-ü") == ["hello", "world", "é", "ü"]  # every non-ASCII character is part of a word
    assert Ascii().terms("ÉCOLE") == ["École"]  # only A-Z are folded


def test_trigram_analyzer():
    assert Trigram().terms("Search") == ["sea", "ear", "arc", "rch"]
    assert Trigram().terms("ab") == []
    assert Trigram().terms("a b") == ["a b"]
    assert Trigram(case_sensitive=True).terms("AbC") == ["AbC"]
    assert Trigram(remove_diacritics=True).terms("café") == ["caf", "afe"]
    assert [t.pos for t in Trigram().tokens("abcd")] == [0, 1]


PORTER_WORDS = {
    # the examples of the paper and a few more
    "caresses": "caress", "ponies": "poni", "ties": "ti", "caress": "caress", "cats": "cat", "feed": "feed", "agreed": "agre",
    "plastered": "plaster", "bled": "bled", "motoring": "motor", "sing": "sing", "conflated": "conflat", "troubled": "troubl",
    "sized": "size", "hopping": "hop", "tanned": "tan", "falling": "fall", "hissing": "hiss", "fizzed": "fizz", "failing": "fail",
    "filing": "file", "happy": "happi", "sky": "sky", "relational": "relat", "conditional": "condit", "rational": "ration",
    "valenci": "valenc", "hesitanci": "hesit", "digitizer": "digit", "conformabli": "conform", "radicalli": "radic",
    "differentli": "differ", "vileli": "vile", "analogousli": "analog", "vietnamization": "vietnam", "predication": "predic",
    "operator": "oper", "feudalism": "feudal", "decisiveness": "decis", "hopefulness": "hope", "callousness": "callous",
    "formaliti": "formal", "sensitiviti": "sensit", "sensibiliti": "sensibl", "triplicate": "triplic", "formative": "form",
    "formalize": "formal", "electriciti": "electr", "electrical": "electr", "hopeful": "hope", "goodness": "good",
    "revival": "reviv", "allowance": "allow", "inference": "infer", "airliner": "airlin", "gyroscopic": "gyroscop",
    "adjustable": "adjust", "defensible": "defens", "irritant": "irrit", "replacement": "replac", "adjustment": "adjust",
    "dependent": "depend", "adoption": "adopt", "homologou": "homolog", "communism": "commun", "activate": "activ",
    "angulariti": "angular", "homologous": "homolog", "effective": "effect", "bowdlerize": "bowdler", "probate": "probat",
    "rate": "rate", "cease": "ceas", "controll": "control", "roll": "roll", "generalizations": "gener", "oscillators": "oscil",
    "a": "a", "is": "is", "was": "wa", "running": "run", "runs": "run", "ran": "ran", "easily": "easili", "happiness": "happi",
}


@pytest.mark.parametrize("word,stem", sorted(PORTER_WORDS.items()))
def test_porter_stems(word, stem):
    assert porter.stem(word) == stem


@needs_fts5
def test_porter_hand_written_stems_are_the_stems_of_fts5_too():
    words = sorted(PORTER_WORDS)
    got = tokenizer_terms("porter unicode61", words)
    assert [g[0] for g in got] == [PORTER_WORDS[w] for w in words]


@needs_fts5
def test_porter_matches_fts5_on_a_list_of_19_thousand_words():
    words = open(os.path.join(HERE, "data", "words.txt"), encoding="utf-8").read().split()
    assert len(words) > 19000
    got = tokenizer_terms("porter unicode61", words)
    bad = [(w, g, porter.stem(w)) for w, g in zip(words, got) if g != [porter.stem(w)]]
    assert not bad, bad[:10]


@needs_fts5
def test_porter_matches_fts5_on_words_with_letters_that_are_not_ascii():
    # SQLite stems the bytes of the UTF-8 text: ß is two consonants there
    chars = ["ß", "ı", "ø", "đ", "ł", "æ", "α", "а", "日"]
    sufs = ["e", "es", "ed", "ing", "s", "ly", "ful", "ness", "ation", "er", "al", "y", "ies", "sses", "eed", "ement"]
    words = [pre + c + mid + suf for c in chars for pre in ("foo", "str", "ra", "") for mid in ("", "o") for suf in sufs]
    words += ["x" * 62 + "ing", "x" * 63 + "ing", "x" * 61 + "ßing", "ßs", "ıs", "ab", "abc", "ßß", "e" * 70, "hopeful" * 10, "ational" * 9 + "ing"]
    got = tokenizer_terms("porter unicode61", words)
    folded = [Unicode61().terms(w) for w in words]
    bad = [(w, g, [porter.stem(t) for t in f]) for w, g, f in zip(words, got, folded) if g != [porter.stem(t) for t in f]]
    assert not bad, bad[:10]


@needs_fts5
def test_porter_matches_fts5_on_words_with_stacked_suffixes():
    suffixes = ["", "s", "es", "ed", "ing", "ly", "ness", "ful", "ment", "ation", "ational", "izer", "ize", "ive", "iveness", "ousness", "alism", "ity", "able", "ible", "ical", "al", "ance", "ence", "er", "ic", "ant", "ent", "ion", "ou", "ism", "ate", "iti", "ous", "e", "ll", "y", "bli", "logi", "eli"]
    stems = ["relat", "condit", "run", "hop", "mat", "control", "hel", "sens", "form", "gener", "oscil", "digit", "feud", "fail", "plast", "troubl", "tan", "fiz", "siz", "ag"]
    words = [a + b + c for a in stems for b in suffixes for c in ("", "s", "ed")]
    got = tokenizer_terms("porter unicode61", words)
    bad = [(w, g, porter.stem(w)) for w, g in zip(words, got) if g != [porter.stem(w)]]
    assert not bad, bad[:10]


def test_stop_words_leave_gaps_in_the_positions():
    toks = Stop(Unicode61()).tokens("the cat of the dog")
    assert [(t.term, t.pos) for t in toks] == [("cat", 1), ("dog", 4)]
    assert "the" in ENGLISH_STOP_WORDS
    english = get_analyzer("english")
    assert english.terms("The running dogs of the village") == ["run", "dog", "villag"]


def test_turkish_lower_case():
    assert turkish_lower("IŞIK İSTANBUL") == "ışık istanbul"
    assert "IŞIK".lower() == "işik"  # what a general purpose lower() does to Turkish


def test_turkish_analyzer():
    t = Turkish()
    assert t.terms("IŞIK İstanbul'da çalışıyorum") == ["ışık", "istanbul", "da", "çalışıyorum"]
    assert Turkish(ascii_fold=True).terms("IŞIK Çağrı") == ["isik", "cagri"]
    assert Turkish(stop_words=True).terms("bu bir kitap ve defter") == ["kitap", "defter"]
    assert [x.pos for x in Turkish(stop_words=True).tokens("bu bir kitap")] == [2]
    assert Turkish(stemmer="prefix5").terms("kitaplarımızdan okullarda") == ["kitap", "okull"]
    light = Turkish(stemmer="light")
    assert light.terms("kitaplar kitapları kitabın evler evlerde") [0] == "kitap"
    assert light.stem("kitaplarımızdan") == "kitap"
    assert light.stem("ev") == "ev"  # short words are left alone


def test_specs_describe_analyzers_exactly():
    for a in [Unicode61(), Unicode61(0, "_", "-"), Ascii(), Trigram(True, True), Porter(), Stop(Unicode61(), ["a", "b"]), Turkish(True, "light", True),
              get_analyzer("english"), Porter(Stop(Ascii()))]:
        again = analyzer_from_spec(a.spec())
        assert again == a and again.spec() == a.spec()
        assert again.terms("The quick brown fox") == a.terms("The quick brown fox")
    with pytest.raises(ValueError):
        get_analyzer("klingon")
    with pytest.raises(ValueError):
        Unicode61(remove_diacritics=3)


# ----------------------------------------------------------------------------------------------- the comparison with FTS5

# Blocks of Unicode whose characters were all assigned long before Unicode 6.1, the version that SQLite's tables stand for (a script added
# after that is treated as one big "word character" range by SQLite and is told apart by us: emoji, for example)
OLD_BLOCKS = [
    (0x20, 0x24F), (0x250, 0x2FF), (0x300, 0x36F), (0x370, 0x37E), (0x380, 0x3FF), (0x400, 0x527), (0x531, 0x587), (0x5D0, 0x5EA),
    (0x620, 0x64A), (0x900, 0x97F), (0xE01, 0xE3A), (0x10A0, 0x10C5), (0x1100, 0x11FF), (0x1E00, 0x1EFF), (0x1F00, 0x1FFF),
    (0x2000, 0x2065), (0x2070, 0x209F), (0x20A0, 0x20B9), (0x2100, 0x2188), (0x2190, 0x22FF), (0x2460, 0x24FF), (0x2500, 0x257F),
    (0x3041, 0x3096), (0x30A1, 0x30FA), (0x4E00, 0x9FA5), (0xAC00, 0xAC80), (0xD790, 0xD7A3), (0xFF01, 0xFFEF),
]
LATIN = [(0x20, 0x24F), (0x300, 0x36F), (0x1E00, 0x1EFF)]


def survey(analyzer, tokenize, blocks):
    chars = [chr(c) for lo, hi in blocks for c in range(lo, hi + 1)]
    texts = []
    for ch in chars:
        texts.append("a" + ch + "b")
        texts.append(ch + "x " + ch)
    want = tokenizer_terms(tokenize, texts)
    return [(texts[i], want[i], analyzer.terms(texts[i])) for i in range(len(texts)) if want[i] != analyzer.terms(texts[i])]


@needs_fts5
def test_unicode61_agrees_with_fts5_on_every_character_of_the_old_blocks():
    bad = survey(Unicode61(), "unicode61", OLD_BLOCKS)
    assert not bad, bad[:10]


@needs_fts5
@pytest.mark.parametrize("mode", [0, 2])
def test_unicode61_other_diacritic_modes_agree_with_fts5(mode):
    bad = survey(Unicode61(remove_diacritics=mode), "unicode61 remove_diacritics %d" % mode, LATIN)
    assert not bad, bad[:10]


@needs_fts5
def test_ascii_agrees_with_fts5():
    bad = survey(Ascii(), "ascii", OLD_BLOCKS[:8])
    assert not bad, bad[:10]


@needs_fts5
def test_unicode61_agrees_with_fts5_on_texts():
    texts = [
        "Hello, World! foo_bar baz-qux", "İstanbul ISPARTA ıslak ışık", "Ünïcödé Café naïve", "a1b2 3.14 x²y ½ ①", "ΑΘΗΝΑ αθήνα Σίσυφος ς",
        "日本語のテキスト 漢字", "é é", "don't stop", "ß ẞ ǅ ǆ Ǆ", "́x", "x ́́ y", "á́b", "   ", "", "a" * 500 + " " + "b" * 30,
        "tab\tsep\nnew line\r\nwin", "price: $5.99 (50% off!)", "e-mail: a.b@c-d.org", "Ωμέγα ωμέγα ΩΜΈΓΑ", "ǕǗǙǛ ǖǘǚǜ", "ǠǡǢǣ",
    ]
    for tk, a in [("unicode61", Unicode61()), ("unicode61 remove_diacritics 0", Unicode61(0)), ("unicode61 remove_diacritics 2", Unicode61(2)),
                  ("unicode61 tokenchars '_'", Unicode61(tokenchars="_")), ("unicode61 separators 'ab'", Unicode61(separators="ab"))]:
        want = tokenizer_terms(tk, texts)
        for text, w in zip(texts, want):
            assert a.terms(text) == w, (tk, text)


@needs_fts5
def test_trigram_agrees_with_fts5():
    texts = ["Search", "ab", "a b", "Hello World", "ÉCOLE école", "x" * 10, "日本語のテキスト", "Hello, World!"]
    assert tokenizer_terms("trigram", texts) == [Trigram().terms(t) for t in texts]
    assert tokenizer_terms("trigram case_sensitive 1", texts) == [Trigram(case_sensitive=True).terms(t) for t in texts]
    try:
        want = tokenizer_terms("trigram remove_diacritics 1", texts)
    except sqlite3.OperationalError:  # the option is newer than the SQLite of this machine
        return
    assert want == [Trigram(remove_diacritics=True).terms(t) for t in texts]


@needs_fts5
def test_known_difference_characters_added_after_unicode_6_1():
    # SQLite's tables stand for Unicode 6.1: an emoji of a later version is a letter there.  We use the categories of the running Python.
    from oracle import tokenizer_terms as tt

    assert tt("unicode61", ["a\U0001F916b"]) == [["a\U0001F916b"]]
    assert Unicode61().terms("a\U0001F916b") == ["a", "b"]
