"""The ``arama`` command: build an index from files, search it, look inside it.

    arama index DIR FILE...          add documents: JSON Lines files ({"id": ..., "title": ..., "body": ...}), text files, or folders
    arama search DIR "query"         ranked results (-n, --offset, --scorer, --weights title=3, --json, --snippet FIELD, --explain)
    arama get DIR KEY                the stored fields of one document
    arama delete DIR KEY...          remove documents
    arama stats DIR                  documents, segments, terms
    arama terms DIR [--prefix p]     the terms of a field with their document frequency
    arama merge DIR [--segments N]   merge segments (the default is one)
    arama check DIR                  read every file and verify its checksum
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Iterator, Optional, Sequence, Tuple

from . import __version__
from .analysis import get_analyzer
from .errors import SearchError
from .index import Index
from .schema import Field, Schema
from .scoring import scorer_from_name

TEXT_SUFFIXES = {".txt", ".md", ".rst", ".text", ".log", ".csv", ".html", ".htm", ".py", ".json"}


def _read_jsonl(path: str) -> Iterator[Tuple[str, dict]]:
    with open(path, "r", encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError as e:
                raise SearchError("%s:%d: not valid JSON (%s)" % (path, n, e)) from None
            if not isinstance(rec, dict):
                raise SearchError("%s:%d: a document is a JSON object" % (path, n))
            key = rec.pop("id", None)
            if key is None:
                key = "%s:%d" % (path, n)
            yield str(key), rec


def _read_text(path: str) -> Iterator[Tuple[str, dict]]:
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        yield path, {"text": fh.read()}


def _walk(paths: Sequence[str]) -> Iterator[str]:
    for p in paths:
        if os.path.isdir(p):
            for root, dirs, files in os.walk(p):
                dirs.sort()
                for f in sorted(files):
                    if os.path.splitext(f)[1].lower() in TEXT_SUFFIXES | {".jsonl", ".ndjson"}:
                        yield os.path.join(root, f)
        else:
            yield p


def _documents(paths: Sequence[str]) -> Iterator[Tuple[str, dict]]:
    for path in _walk(paths):
        if path.endswith((".jsonl", ".ndjson")):
            yield from _read_jsonl(path)
        else:
            yield from _read_text(path)


def _parse_weights(items: Optional[Sequence[str]]) -> Optional[Dict[str, float]]:
    if not items:
        return None
    out = {}
    for item in items:
        name, _, value = item.partition("=")
        try:
            out[name] = float(value)
        except ValueError:
            raise SearchError("a weight is written field=number, not %r" % item) from None
    return out


def cmd_index(args) -> int:
    docs = _documents(args.files)
    if not Index.exists(args.dir):
        first = next(docs, None)
        names = args.fields.split(",") if args.fields else (sorted(first[1]) if first else ["text"])
        analyzer = get_analyzer(args.analyzer)
        schema = Schema([Field(n, analyzer) for n in names])
        Index.create(args.dir, schema)
        if first is not None:
            docs = _chain(first, docs)
    index = Index.open(args.dir)
    n = 0
    with index.writer() as w:
        for key, doc in docs:
            w.add(key, doc)
            n += 1
        w.commit()
        print("%d documents added; %d in the index" % (n, len(w)))
    return 0


def _chain(first, rest):
    yield first
    yield from rest


def cmd_search(args) -> int:
    index = Index.open(args.dir)
    scorer = scorer_from_name(args.scorer)
    with index.searcher() as s:
        res = s.search(args.query, limit=args.n, offset=args.offset, scorer=scorer, weights=_parse_weights(args.weights))
        if args.json:
            out = []
            for h in res:
                rec = {"id": h.key, "score": h.score}
                if args.snippet:
                    rec["snippet"] = s.snippet(args.query, h.key, args.snippet, "[", "]")
                if args.fields_out:
                    rec["fields"] = h.fields
                out.append(rec)
            print(json.dumps({"total": res.total, "hits": out}, ensure_ascii=False, indent=2))
            return 0
        print("%d documents match" % res.total, file=sys.stderr)
        for h in res:
            print("%8.4f  %s" % (h.score, h.key))
            if args.snippet:
                text = s.snippet(args.query, h.key, args.snippet, "[", "]")
                if text:
                    print("          " + text.replace("\n", " "))
            if args.explain:
                ex = s.explain(args.query, h.key, scorer=scorer, weights=_parse_weights(args.weights))
                for leaf in ex.leaves if ex else []:
                    print("          %-30s df=%-5d weight=%-8.4f freq=%-4g -> %.4f" % (leaf.text, leaf.df, leaf.weight, leaf.freq, leaf.contribution))
    return 0


def cmd_get(args) -> int:
    index = Index.open(args.dir)
    with index.searcher() as s:
        doc = s.get(args.key)
    if doc is None:
        print("no document with the key %r" % args.key, file=sys.stderr)
        return 1
    print(json.dumps(doc, ensure_ascii=False, indent=2))
    return 0


def cmd_delete(args) -> int:
    index = Index.open(args.dir)
    with index.writer() as w:
        n = sum(1 for k in args.keys if w.delete(k))
        w.commit()
    print("%d of %d documents deleted" % (n, len(args.keys)))
    return 0


def cmd_stats(args) -> int:
    index = Index.open(args.dir)
    with index.searcher() as s:
        print("documents  %d" % s.num_docs)
        print("segments   %d" % len(s.segs))
        print("words      %d (average %.1f per document)" % (s.stats.total_length, s.stats.avg_length))
        for fi, f in enumerate(index.schema.fields):
            terms = set()
            for seg in s.segs:
                terms.update(t for t, _ in seg.reader.terms(fi))
            print("field %-10s analyzer=%r weight=%g terms=%d" % (f.name, f.analyzer, f.weight, len(terms)))
        size = sum(os.path.getsize(os.path.join(args.dir, f)) for f in os.listdir(args.dir))
        print("size       %.1f KiB" % (size / 1024))
    return 0


def cmd_terms(args) -> int:
    index = Index.open(args.dir)
    fi = index.schema.index[args.field] if args.field else 0
    with index.searcher() as s:
        counts: Dict[str, int] = {}
        for seg in s.segs:
            for term, info in seg.reader.terms(fi, args.prefix or ""):
                counts[term] = counts.get(term, 0) + info.df
        items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[: args.n]
    for term, df in items:
        print("%8d  %s" % (df, term))
    return 0


def cmd_merge(args) -> int:
    index = Index.open(args.dir)
    with index.writer() as w:
        w.merge(args.segments)
        w.commit()
        print("%d segments" % len(w.segments()))
    return 0


def cmd_check(args) -> int:
    problems = Index.open(args.dir).check()
    for p in problems:
        print("PROBLEM: " + p, file=sys.stderr)
    if not problems:
        print("the index is healthy")
    return 1 if problems else 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="arama", description="A full-text search engine written from scratch.")
    p.add_argument("--version", action="version", version="arama " + __version__)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("index", help="add documents to an index (created if it does not exist)")
    s.add_argument("dir")
    s.add_argument("files", nargs="+", help=".jsonl files ({\"id\": ..., fields}), text files, or folders")
    s.add_argument("--fields", help="the fields of a new index, comma separated (default: the fields of the first JSON record, or 'text')")
    s.add_argument("--analyzer", default="unicode61", help="unicode61 (default), porter, english, ascii, trigram, turkish")
    s.set_defaults(func=cmd_index)

    s = sub.add_parser("search", help="search an index")
    s.add_argument("dir")
    s.add_argument("query")
    s.add_argument("-n", type=int, default=10, help="how many results (default 10)")
    s.add_argument("--offset", type=int, default=0)
    s.add_argument("--scorer", default="bm25", help="bm25 (default), bm25-fts5, tfidf, tfidf-sklearn, tfidf-lucene")
    s.add_argument("--weights", nargs="*", metavar="FIELD=WEIGHT", help="the importance of fields for this search")
    s.add_argument("--snippet", metavar="FIELD", help="show an extract of this field around the matches")
    s.add_argument("--explain", action="store_true", help="show how each phrase of the query counts in each score")
    s.add_argument("--json", action="store_true")
    s.add_argument("--fields-out", action="store_true", help="with --json, include the stored fields")
    s.set_defaults(func=cmd_search)

    s = sub.add_parser("get", help="the stored fields of a document")
    s.add_argument("dir")
    s.add_argument("key")
    s.set_defaults(func=cmd_get)

    s = sub.add_parser("delete", help="delete documents")
    s.add_argument("dir")
    s.add_argument("keys", nargs="+")
    s.set_defaults(func=cmd_delete)

    s = sub.add_parser("stats", help="numbers about an index")
    s.add_argument("dir")
    s.set_defaults(func=cmd_stats)

    s = sub.add_parser("terms", help="the terms of a field and how many documents have each")
    s.add_argument("dir")
    s.add_argument("--field")
    s.add_argument("--prefix")
    s.add_argument("-n", type=int, default=30)
    s.set_defaults(func=cmd_terms)

    s = sub.add_parser("merge", help="merge the segments of an index")
    s.add_argument("dir")
    s.add_argument("--segments", type=int, default=1, help="how many segments to leave (default 1)")
    s.set_defaults(func=cmd_merge)

    s = sub.add_parser("check", help="verify the files of an index")
    s.add_argument("dir")
    s.set_defaults(func=cmd_check)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (SearchError, OSError) as e:
        print("arama: %s" % e, file=sys.stderr)
        return 2
    except BrokenPipeError:  # pragma: no cover
        return 0
    except KeyError as e:  # an unknown field in --field
        print("arama: no such field: %s" % e.args[0], file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
