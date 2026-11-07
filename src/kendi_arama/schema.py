"""The schema of an index: the names of the fields and how the text of each one is analysed."""
from __future__ import annotations

import re
from typing import Dict, List, Sequence, Union

from .analysis import Analyzer, get_analyzer
from .errors import SchemaError

_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class Field:
    """A field of a document.  ``stored`` keeps the original text (needed for the results and for highlighting); ``weight`` is
    the default importance of the field when documents are scored (a match in a title can count more than one in a body)."""

    def __init__(self, name: str, analyzer: "str | Analyzer | dict | None" = None, stored: bool = True, weight: float = 1.0) -> None:
        if not _NAME.match(name):
            raise SchemaError("%r is not a valid field name" % (name,))
        self.name = name
        self.analyzer = get_analyzer(analyzer)
        self.stored = stored
        self.weight = float(weight)

    def to_json(self) -> dict:
        return {"name": self.name, "analyzer": self.analyzer.spec(), "stored": self.stored, "weight": self.weight}

    @classmethod
    def from_json(cls, data: dict) -> "Field":
        return cls(data["name"], data["analyzer"], data["stored"], data["weight"])

    def __repr__(self) -> str:
        return "Field(%r, %r, stored=%r, weight=%r)" % (self.name, self.analyzer, self.stored, self.weight)


class Schema:
    def __init__(self, fields: Union[Sequence[Union[Field, str]], None] = None) -> None:
        fields = list(fields) if fields is not None else [Field("text")]
        if not fields:
            raise SchemaError("a schema needs at least one field")
        self.fields: List[Field] = [f if isinstance(f, Field) else Field(f) for f in fields]
        self.index: Dict[str, int] = {}
        for i, f in enumerate(self.fields):
            if f.name in self.index:
                raise SchemaError("the field %r is defined twice" % f.name)
            self.index[f.name] = i

    def __len__(self) -> int:
        return len(self.fields)

    def names(self) -> List[str]:
        return [f.name for f in self.fields]

    def field(self, name: str) -> Field:
        try:
            return self.fields[self.index[name]]
        except KeyError:
            raise SchemaError("no such field: %s" % name) from None

    def to_json(self) -> list:
        return [f.to_json() for f in self.fields]

    @classmethod
    def from_json(cls, data: list) -> "Schema":
        return cls([Field.from_json(f) for f in data])

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Schema) and self.to_json() == other.to_json()

    def __repr__(self) -> str:
        return "Schema(%r)" % (self.fields,)
