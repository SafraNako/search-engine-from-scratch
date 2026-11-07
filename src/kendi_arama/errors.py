"""The exceptions of the package."""
from __future__ import annotations


class SearchError(Exception):
    """Base class of every error raised on purpose."""


class SchemaError(SearchError):
    """A schema that is not valid, or a document that does not fit it."""


class QuerySyntaxError(SearchError):
    """A query string that cannot be parsed (the message says where)."""

    def __init__(self, message: str, position: int = -1) -> None:
        super().__init__(message)
        self.position = position


class CorruptIndexError(SearchError):
    """A file of the index is damaged: the checksum, the header or a length is wrong."""


class LockError(SearchError):
    """Another writer has the index open."""
