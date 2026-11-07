import json
import os
import subprocess
import sys

import pytest

from kendi_arama.cli import main

SRC = os.path.join(os.path.dirname(__file__), "..", "src")

RECORDS = [
    {"id": "a", "title": "Red fox", "body": "The quick red fox jumps over the lazy dog"},
    {"id": "b", "title": "Lazy dog", "body": "A lazy dog sleeps; the dog is lazy"},
    {"id": "c", "title": "Quick start", "body": "quick quick brown fox"},
    {"id": "d", "title": "Cats", "body": "completely unrelated text about cats"},
]


@pytest.fixture
def built(tmp_path):
    src = tmp_path / "docs.jsonl"
    src.write_text("\n".join(json.dumps(r) for r in RECORDS) + "\n", encoding="utf-8")
    idx = str(tmp_path / "idx")
    assert main(["index", idx, str(src)]) == 0
    return idx


def run(argv, capsys):
    code = main(argv)
    out = capsys.readouterr()
    return code, out.out, out.err


def test_index_and_search(built, capsys):
    code, out, err = run(["search", built, "fox"], capsys)
    assert code == 0 and "2 documents match" in err
    lines = [l.split() for l in out.strip().splitlines()]
    assert {l[1] for l in lines} == {"a", "c"} and float(lines[0][0]) > 0


def test_json_output_and_snippets(built, capsys):
    code, out, _ = run(["search", built, '"lazy dog"', "--json", "--snippet", "body", "--fields-out", "--scorer", "bm25-fts5"], capsys)
    data = json.loads(out)
    assert code == 0 and data["total"] == 2
    top = data["hits"][0]
    assert top["id"] in ("a", "b") and "[lazy dog]" in top["snippet"] and top["fields"]["title"]


def test_explain_and_weights_and_paging(built, capsys):
    code, out, _ = run(["search", built, "fox OR dog", "--explain", "--weights", "title=5"], capsys)
    assert code == 0 and "df=" in out and "weight=" in out
    code, out, err = run(["search", built, "fox OR dog", "-n", "1", "--offset", "1"], capsys)
    assert len(out.strip().splitlines()) == 1 and "3 documents match" in err
    assert main(["search", built, "dog", "--weights", "nonsense"]) == 2


def test_other_scorers(built, capsys):
    for name in ("bm25", "bm25-fts5", "tfidf", "tfidf-sklearn", "tfidf-lucene"):
        code, out, _ = run(["search", built, "quick OR fox", "--scorer", name], capsys)
        assert code == 0 and out.strip()
    assert main(["search", built, "x", "--scorer", "magic"]) == 2


def test_get_delete_stats_terms_merge_check(built, capsys):
    code, out, _ = run(["get", built, "b"], capsys)
    assert code == 0 and json.loads(out)["title"] == "Lazy dog"
    assert run(["get", built, "zzz"], capsys)[0] == 1
    code, out, _ = run(["stats", built], capsys)
    assert code == 0 and "documents  4" in out and "field body" in out
    code, out, _ = run(["terms", built, "--prefix", "qu", "--field", "body"], capsys)
    assert "quick" in out and out.split()[0] == "2"
    code, out, _ = run(["delete", built, "a", "nosuch"], capsys)
    assert code == 0 and "1 of 2" in out
    assert run(["get", built, "a"], capsys)[0] == 1
    code, out, _ = run(["merge", built], capsys)
    assert code == 0 and "1 segments" in out
    code, out, _ = run(["check", built], capsys)
    assert code == 0 and "healthy" in out


def test_text_files_and_folders(tmp_path, capsys):
    folder = tmp_path / "notes"
    (folder / "sub").mkdir(parents=True)
    (folder / "one.txt").write_text("the first note about foxes")
    (folder / "sub" / "two.md").write_text("another note about dogs")
    (folder / "skip.bin").write_bytes(b"\x00\x01")
    idx = str(tmp_path / "idx")
    assert main(["index", idx, str(folder), "--analyzer", "english"]) == 0
    capsys.readouterr()
    code, out, _ = run(["search", idx, "fox"], capsys)
    assert code == 0 and "one.txt" in out and "two.md" not in out
    code, out, _ = run(["search", idx, "notes"], capsys)  # stemmed: note
    assert "one.txt" in out and "two.md" in out


def test_errors_are_reported_without_a_traceback(tmp_path, built, capsys):
    assert main(["search", str(tmp_path / "nothing"), "x"]) == 2
    assert "not an index" in capsys.readouterr().err
    assert main(["search", built, "AND"]) == 2
    assert "syntax error" in capsys.readouterr().err
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{not json}\n")
    assert main(["index", str(tmp_path / "i2"), str(bad)]) == 2
    assert "not valid JSON" in capsys.readouterr().err
    assert main(["terms", built, "--field", "nofield"]) == 2


def test_check_reports_damage(built, capsys):
    seg = [n for n in os.listdir(built) if n.endswith(".kseg")][0]
    path = os.path.join(built, seg)
    with open(path, "r+b") as fh:
        fh.seek(os.path.getsize(path) // 2)
        b = fh.read(1)
        fh.seek(-1, 1)
        fh.write(bytes([b[0] ^ 1]))
    code, _, err = run(["check", built], capsys)
    assert code == 1 and "PROBLEM" in err


def test_as_a_program(tmp_path):
    src = tmp_path / "d.jsonl"
    src.write_text(json.dumps({"id": 1, "text": "hello world"}) + "\n")
    env = dict(os.environ, PYTHONPATH=SRC)
    r = subprocess.run([sys.executable, "-m", "kendi_arama", "index", str(tmp_path / "i"), str(src)], capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 0, r.stderr
    r = subprocess.run([sys.executable, "-m", "kendi_arama", "search", str(tmp_path / "i"), "hello"], capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 0 and r.stdout.split()[1] == "1"
    r = subprocess.run([sys.executable, "-m", "kendi_arama", "--version"], capture_output=True, text=True, env=env, timeout=60)
    assert r.stdout.startswith("arama ")
