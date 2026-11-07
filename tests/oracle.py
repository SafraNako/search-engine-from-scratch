"""SQLite FTS5 as the oracle.

A real FTS5 table is kept next to the index under test: every document that is added, replaced or deleted goes to both, and the
same queries are run through both.  Nothing here is a stand-in for FTS5: it is SQLite's own code, from the ``sqlite3`` module of
the standard library.
"""
from __future__ import annotations

import sqlite3
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def fts5_available() -> bool:
    try:
        c = sqlite3.connect(":memory:")
        c.execute("create virtual table t using fts5(a)")
        return True
    except sqlite3.Error:
        return False


class Fts5:
    def __init__(self, columns: Sequence[str], tokenize: str = "unicode61") -> None:
        self.columns = list(columns)
        self.db = sqlite3.connect(":memory:")
        cols = ", ".join(self.columns)
        self.db.execute('create virtual table t using fts5(%s, tokenize="%s")' % (cols, tokenize))
        self.tokenize = tokenize
        self._marks = ",".join("?" for _ in self.columns)

    def insert(self, rowid: int, doc: Dict[str, str]) -> None:
        self.db.execute("insert into t(rowid, %s) values (?, %s)" % (", ".join(self.columns), self._marks), [rowid] + [doc.get(c) or "" for c in self.columns])

    def delete(self, rowid: int) -> None:
        self.db.execute("delete from t where rowid = ?", (rowid,))

    def replace(self, rowid: int, doc: Dict[str, str]) -> None:
        self.delete(rowid)
        self.insert(rowid, doc)

    def count(self) -> int:
        return self.db.execute("select count(*) from t").fetchone()[0]

    def match(self, query: str) -> List[int]:
        """The rowids that match (raises ``sqlite3.OperationalError`` for a query that FTS5 rejects)."""
        return [r[0] for r in self.db.execute("select rowid from t where t match ? order by rowid", (query,))]

    def ranked(self, query: str, weights: Optional[Sequence[float]] = None) -> List[Tuple[int, float]]:
        """(rowid, score) best first, the score being the negative of ``bm25()`` (so that more is better)."""
        args = ""
        if weights is not None:
            args = ", " + ", ".join(repr(float(w)) for w in weights)
        rows = self.db.execute("select rowid, bm25(t%s) from t where t match ? order by rank, rowid" % args, (query,)).fetchall()
        return [(r, -s) for r, s in rows]

    def highlight(self, query: str, column: int, open_: str = "[", close: str = "]") -> Dict[int, str]:
        rows = self.db.execute("select rowid, highlight(t, ?, ?, ?) from t where t match ? order by rowid", (column, open_, close, query)).fetchall()
        return dict(rows)

    def tokens(self, text: str) -> List[str]:
        """The terms the tokenizer makes of a text, in order."""
        db = sqlite3.connect(":memory:")
        db.execute('create virtual table t using fts5(a, tokenize="%s")' % self.tokenize)
        db.execute("create virtual table v using fts5vocab(t, instance)")
        db.execute("insert into t(a) values (?)", (text,))
        return [r[0] for r in db.execute("select term from v order by offset")]


def tokenizer_terms(tokenize: str, texts: Iterable[str]) -> List[List[str]]:
    """The terms of many texts through one tokenizer, using one table."""
    db = sqlite3.connect(":memory:")
    db.execute('create virtual table t using fts5(a, tokenize="%s")' % tokenize)
    db.execute("create virtual table v using fts5vocab(t, instance)")
    out = []
    for text in texts:
        db.execute("delete from t")
        db.execute("insert into t(a) values (?)", (text,))
        out.append([r[0] for r in db.execute("select term from v order by offset")])
    return out
