import os

import pytest

from oracle import fts5_available


def fuzz_count(default: int) -> int:
    """How many random scenarios a comparison test runs (``KENDI_FUZZ=800 pytest`` for more)."""
    return int(os.environ.get("KENDI_FUZZ", default))


def fuzz_start() -> int:
    return int(os.environ.get("KENDI_FUZZ_START", 0))


needs_fts5 = pytest.mark.skipif(not fts5_available(), reason="this SQLite has no FTS5")


@pytest.fixture
def index_dir(tmp_path):
    return str(tmp_path / "index")
