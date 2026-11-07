# search-engine-from-scratch

A full-text search engine written from scratch in pure Python (3.9+, no dependencies): an inverted index kept in immutable segments on disk,
BM25 and TF-IDF ranking, boolean / phrase / NEAR / prefix / fuzzy queries, highlighting, updates and deletes, merging, crash safety.

It is not a wrapper: it does not use `sqlite3`, Whoosh, Lucene or any library for the index, the query language or the ranking.
SQLite FTS5 is used by the tests only, as the **oracle**: the same documents, edits and queries go to both engines, and they have to agree
on which documents match and on the score of each. TF-IDF is checked against scikit-learn and against the worked example of a textbook.

```python
import tempfile
from kendi_arama import Field, Index, Schema

path = tempfile.mkdtemp() + "/library"
index = Index.create(path, Schema([Field("title", "english", weight=3.0), Field("body", "english")]))
with index.writer() as w:                       # one writer at a time; commit() when the block ends
    w.add("fox", {"title": "The quick red fox", "body": "The quick red fox jumps over the lazy dogs."})
    w.add("dog", {"title": "Lazy dogs", "body": "A lazy dog sleeps all day, and the dogs of the village sleep with it."})
    w.add("cat", {"title": "Cats", "body": "Cats like boxes."})

with index.searcher() as s:                      # a snapshot of the committed index
    results = s.search("lazy dog* NOT cats", limit=5)
    print(results.total, "documents match")
    for hit in results:
        print(hit.key, round(hit.score, 3), hit["title"])
    print(s.highlight('"lazy dog"', "fox", "body", "[", "]"))
    print(s.search("quik~1", limit=1).keys())     # fuzzy: one edit
```

```
2 documents match
dog 1.567 Lazy dogs
fox 0.853 The quick red fox
The quick red fox jumps over the [lazy dogs].
['fox']
```

There is also a command line tool: `arama index idx docs.jsonl`, then `arama search idx "lazy dog*" --snippet body --explain`.

## What it can do

| | |
|---|---|
| **Analysis** | `unicode61` (the word rules, case folding and accent removal of SQLite's tokenizer of that name), `ascii`, `trigram` (substrings), a Porter stemmer, stop words that keep the positions of the other words, a Turkish analyzer (`I`/`İ`, ASCII folding, light stemmer); each field has its own |
| **Queries** | words, `"phrases"` and `a + b`, `prefix*`, `^first`, `AND OR NOT ( )`, `field:word`, `{f1 f2}:word`, `-field:word`, `NEAR(a b, 5)`, fuzzy `word~1` (insert, delete, replace, swap) |
| **Ranking** | BM25 (two variants of idf), TF-IDF in the notation of the IR textbook: `tf` natural / log / sqrt / boolean, `idf` classic / smooth / lucene / prob / none, document norm cosine / length / none; field weights; paging; exact total |
| **Documents** | add, replace, delete by key; stored fields; highlighting and snippets; "did you mean" (`suggest`); `explain` (what each phrase adds to a score) |
| **Index on disk** | immutable segments written once, read through `mmap`; term dictionary with front coding; postings in blocks of 128 with a skip table, bit-packed; positions; compressed stored fields; checksums |
| **Updates** | an atomic manifest, tombstones and a `.del` file per segment, tiered merging, snapshot readers that never see a half-written state, one writer at a time (file lock), cleanup after a crash |
| **Tools** | the `arama` command: `index`, `search`, `get`, `delete`, `stats`, `terms`, `merge`, `check` |

## The query language

It is the language of FTS5 (the tests run the same strings through both) with one addition, the fuzzy suffix.

```
cat dog                  both words (a space is AND)         cat OR dog         cat NOT dog         (cat OR dog) AND bird
"black cat"   black + cat    the words next to each other, in this order
cat*                     a prefix: cat, cats, category ...    "black ca"*  (the last word is the prefix)
^cat                     cat as the first word of a field
title:cat   {title body}:cat   -title:cat   title:(cat OR dog)    field filters (names are not case sensitive)
NEAR(cat dog, 5)         both within 5 words of each other, 10 if the number is left out
colour~1                 [not in FTS5] words within one edit of colour; "colour of"~2 for every word of a phrase
```

`NOT` binds tighter than `AND`, `AND` tighter than `OR`; a list of phrases without an operator is an `AND` that binds tighter than all of them.
A parenthesised group cannot be part of such a list (`(a) b` is an error, `(a) AND b` is not). Operators are operators in capitals only.

## Ranking

The score of a document is the sum, over the phrases of the query, of what each is worth: the **weight** of the phrase (rare phrases weigh
more: idf, from the number of documents that have the phrase) times a function of its **frequency** in the document (more occurrences count
more, but less and less) and of the **length** of the document (a hit in a short text means more).

* `BM25(k1=1.2, b=0.75, idf="lucene")` is the default. `idf="fts5"` is the variant of SQLite, `log((N - n + 0.5) / (n + 0.5))` that is never
  below 1e-6; with it the scores are equal to the `bm25()` function of FTS5, which is what the tests compare.
* `TfIdf(tf, idf, norm, doc_idf, normalize_query)` is the vector space model. `TfIdf(tf="log", idf="smooth", norm="cosine", doc_idf=True)` is
  what scikit-learn's `TfidfVectorizer` computes (`scorer_from_name("tfidf-sklearn")`); the default (`lnc.ltc` in the book: the document side
  has no idf, so a norm needs only the document itself) and the classic similarity of Lucene are there too.
* A phrase is one unit: `"new york"` has its own document count and its own frequency in a document.
* Statistics (the number of documents, the average length, the document count of a phrase) are counted over all segments, without the
  deleted documents, so a score does not depend on how many segments there are or whether a merge has happened.

## How it works

```
text ──analyzer──▶ tokens ──SegmentBuilder──▶ segment file ◀──mmap── SegmentReader ──▶ matchers ──▶ scorer ──▶ top k
                                  │                                        ▲
 query string ──parser──▶ query ──plan (analysis of the words, empty phrases)┘
```

* `analysis.py`, `porter.py`: text to `(term, position, start, end)`; the same analyzer is used for the documents and for the words of a query.
* `segment.py`: the file format (the layout is in the docstring). A term is found by a binary search over the first terms of the blocks and a
  scan of one block of 32. The postings of a term are 128 gaps between document numbers and 128 frequencies per block, bit-packed with the
  width of the largest; a skip table (last document, offsets, largest frequency of each block) lets `advance(target)` jump over blocks.
  Positions are gaps in a separate stream, decoded only for the documents that need them (phrases, NEAR, scoring).
* `index.py`: the manifest, writers and merging. The manifest is replaced atomically (temp file, `fsync`, rename, `fsync` of the directory);
  segments and deletion lists are written once and never touched again. A crash leaves, at worst, files that nothing refers to, and the next
  writer deletes them.
* `query.py`: the parser (recursive descent, one function per level of precedence) and the printer. `matchers.py`: the plan and the
  cursors: `AND` is a leapfrog over cursors sorted by cost, `OR` moves the cursors that are on the smallest document, a phrase intersects the
  positions of its words, a prefix or fuzzy word is a union of the postings of the terms it stands for. `NEAR` is the algorithm of FTS5.
* `scoring.py`, `searcher.py`: the scorers, and the search over all segments with a heap of the best `offset + limit` hits.
* `highlight.py`: the places of the matches come from the same cursors, the text between them is cut from the stored field.

## How it was checked

There are 285 tests. The expected values in the hand-written ones were worked out on paper or taken from a published source (the
examples of Porter's paper, the numbers of the IR textbook); everything else is compared with a real implementation:

* **SQLite FTS5** (`tests/harness.py`, `tests/test_differential.py`). A scenario is a random schema (one to three fields; analyzers
  `unicode61` with three settings of accent removal, `porter`, `ascii`, `trigram`), a random sequence of edits (add, replace, delete, commit,
  merge, reopen the writer) applied to both engines, and, after the commits, random queries: nested `AND OR NOT`, phrases, prefixes,
  `NEAR`, field filters, empty phrases, and the same queries damaged by deleting, inserting and swapping characters. Both engines have to agree
  on whether the query is a syntax error, on **which documents match** and on **the score of each** (to 1e-9, with random field weights).
  Over **73,000 queries** in 1,000 scenarios (the last run; more than 300,000 queries during the work), on Python 3.9 with SQLite 3.39.5 and on Python 3.13 with SQLite 3.50.4,
  the sets of documents are the same except in the cases listed below.
* **The tokenizers**, character by character: every code point of 28 blocks of Unicode (all of Latin, Greek, Cyrillic, Armenian, Hebrew, the
  Arabic letters, Devanagari, Thai, Hangul, CJK, the punctuation and the symbols that were there in 2011), in the three accent modes, against
  FTS5's `unicode61`; `ascii` and `trigram` on samples. The **Porter stemmer** against the stemmer of FTS5 on 19,538 words, on 2,400 words made by
  stacking suffixes, and on words with letters that are not ASCII.
* **highlight()** of FTS5 against `Searcher.highlight`, for every column of every row that a random query matches (about 15,000 comparisons per run).
* **TF-IDF** against `TfidfVectorizer` of scikit-learn (over five hundred query/document cosines in several segments, with deleted documents, to
  1e-12), and against the worked example of *Introduction to Information Retrieval* (the three novels, figures 6.9 to 6.13).
* **Edit distance** against a plain table, **term lookups** against the sorted list, **segments** against the builder (every term of random
  corpora, with `advance` to random targets), **crashes** by killing a writer (`os._exit`) before the commit, in the middle of a merge and just
  before the manifest is replaced, then reopening: the last commit is intact and the leftovers are removed. Damaged files (a flipped byte,
  a cut file, a foreign header) are found by `Index.check()`.
* The random scenarios are seeded: `KENDI_FUZZ_START=<seed> KENDI_FUZZ=1 pytest tests/test_differential.py` repeats one. The CI runs 480.

### What FTS5 taught me

The oracle found a lot of behaviour that is in no manual. It is all in the tests now.

* The idf of `bm25()` is negative for a phrase that is in more than half of the documents, and FTS5 replaces it with 1e-6: the score of a very
  common word is almost zero. (That is why this engine's default idf is the Lucene one.)
* `NEAR(a b c, 0)` can never match: each phrase has to be within the distance of the *last one*, not of its neighbours.
* Empty phrases (`"!!"`, or a word shorter than three letters with the trigram tokenizer) are left out of a list of phrases and of `NEAR`,
  vanish around `OR` and the right side of `NOT`, and make an explicit `AND` match nothing: `cat "!!"` is `cat`, `cat AND "!!"` is empty.
* `highlight()` joins instances that overlap (`"runs runs"` in `runs runs runs`) and keeps neighbours apart.
* The word tables of `unicode61` are those of Unicode 6.1: a later emoji is a *letter* there (`a🤖b` is one word). Combining marks can be inside a
  word but not start one, and `İ` loses its dot only when accents are removed.
* The Porter stemmer of SQLite works on the **bytes** of the UTF-8 text: `ß` is two consonants, so `straße` loses its `e` and `raße` does too, while
  words longer than 64 bytes are not stemmed.
* A column named `Body` can be written `body`.
* What is a syntax error: `(a) b`, `a (b)`, `NOT a`, `a AND`, `NEAR(a b,)`, `^title:a`, `title:title:a`, `a -b`, a bare `-`: the 600 or more random and damaged queries of the
  syntax test are accepted and rejected in the same places by both parsers (the fuzzy `~` is the exception).

My own bugs, found the same way: a generator expression that captured a loop variable late (every field of a merged segment got the
postings of the last one: twice, in the merge and in the cosine norms), a first design that counted the phrases whose cursor stood on the
document (the right side of a `NOT` was counted twice), empty phrases dropped everywhere instead of by the rules above, a stemmer that worked on
characters and not on bytes, `highlight` that split overlapping matches, field names that were case sensitive, `advance(-1)` that returned
"before the first document".

### Known differences

* FTS5 sometimes counts a phrase that is in the row but not part of the match: the right side of a `NOT` whose left side matches nothing, or the words
  of a `NEAR` that are not near enough, when they are below an `OR`. Rarely it leaves a phrase out. This engine counts exactly the phrases that
  take part in the match. In the comparison tests such a difference is accepted only if it equals the sum of the contributions of some of the phrases of
  the query that sit on the row (the `explain()` of the engine lists them); it happens in about 0.1 % of the compared queries and the tests fail if it is
  more than 3 %. When a `NEAR` is combined with other operators and the scores differ otherwise, only the documents are compared (under 0.1 %).
* A field filter that excludes every field, inside the right side of a `NOT` (`x NOT (-{all fields}:y OR z)`) matches nothing in FTS5; here it matches
  `x NOT z`. The generator of the tests does not produce it.
* `unicode61`: characters added to Unicode after 6.1 are separators here and word characters in SQLite (emoji, newer scripts, new marks).
* The Turkish analyzer, the stop word lists and `suggest` have no oracle; they are tested by hand. Suggestions are terms of the index: with a
  stemmer they are stems.

### Limits

* No Windows support (the file lock needs `fcntl`; replacing a file that is open fails there).
* The whole result set of a query is scored (the total is exact); there is no WAND-style early stop. A cosine norm of TF-IDF with `doc_idf`
  needs one pass over the postings the first time it is used with a snapshot.
* Documents that are not committed are lost with the process; there is no write-ahead log.
* One writer at a time; a `Searcher` is not thread safe (open one per thread); there is no server.

## Speed

`python benchmarks/bench.py 5000`: the docstrings of the standard library of Python (words from real text), both engines answer with the
best ten documents by BM25. FTS5 is C, so it is faster; the table shows how far a pure Python engine is.

| | this engine | FTS5 | ratio |
|---|---:|---:|---:|
| indexing 5000 documents | 843 ms | 79 ms | 11× |
| size on disk | 1308 KiB | 2508 KiB | 0.52× |
| merge 7 segments into 1 | 951 ms | | |
| a very common word (3658 matches, all scored) | 25.0 ms | 6.2 ms | 4× |
| a rare word (3 matches) | 0.11 ms | 0.10 ms | 1× |
| two words, AND (40) | 1.22 ms | 0.29 ms | 4× |
| three words, OR (224) | 2.34 ms | 0.53 ms | 4× |
| phrase (159) | 6.85 ms | 1.66 ms | 4× |
| prefix (346) | 4.01 ms | 0.97 ms | 4× |
| NEAR (57) | 2.52 ms | 0.34 ms | 8× |
| NOT (497) | 4.20 ms | 0.97 ms | 4× |
| field filter (28) | 0.22 ms | 0.18 ms | 1× |

201,228 words, Python 3.13, SQLite 3.50.4, on an old laptop that was busy with other work, so take the ratios as rough. The index is half the
size of FTS5's: the postings are bit-packed in blocks and the stored text is compressed in blocks. Where it is slowest, a query with many matches, the time is the scoring of each match in Python.

## Development

```bash
pip install -e ".[dev]"                      # pytest and scikit-learn (the oracle of the TF-IDF test)
pytest                                       # 285 tests, about four minutes
KENDI_FUZZ=800 pytest tests/test_differential.py    # more random scenarios
python benchmarks/bench.py
```

MIT licence.
