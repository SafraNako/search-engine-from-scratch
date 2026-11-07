"""A full-text search engine written from scratch: inverted index in immutable segments, BM25 and TF-IDF ranking, boolean, phrase,
NEAR, prefix and fuzzy queries.  See the README; the tests compare it with SQLite FTS5."""
from .analysis import Analyzer, Ascii, Porter, Stop, Trigram, Turkish, Unicode61, get_analyzer
from .errors import CorruptIndexError, LockError, QuerySyntaxError, SchemaError, SearchError
from .index import Index, IndexWriter, MergePolicy
from .schema import Field, Schema
from .scoring import BM25, Scorer, TfIdf, scorer_from_name
from .searcher import Hit, Results, Searcher

__version__ = "0.1.0"

__all__ = [
    "Analyzer", "Ascii", "BM25", "CorruptIndexError", "Field", "Hit", "Index", "IndexWriter", "LockError", "MergePolicy", "Porter",
    "QuerySyntaxError", "Results", "Schema", "SchemaError", "Scorer", "SearchError", "Searcher", "Stop", "TfIdf", "Trigram", "Turkish",
    "Unicode61", "get_analyzer", "scorer_from_name",
]
