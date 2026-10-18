"""Command-line interface: ``python -m ragsearch ingest|search|eval``."""

import argparse
import logging
import sys
from pathlib import Path

from ragsearch.config import ConfigError, Settings, load_settings
from ragsearch.embedder import Embedder, EmbeddingError, make_embedder
from ragsearch.eval import EvalError, evaluate, load_cases
from ragsearch.index import index_backend
from ragsearch.loaders import LoaderError, load_documents
from ragsearch.pipeline import ingest
from ragsearch.retrieval import hybrid_search
from ragsearch.store import Store, StoreError
from ragsearch.web.budget import BudgetExceeded
from ragsearch.web.fetcher import FetchError
from ragsearch.web.sources import make_tavily, web_documents
from ragsearch.web.tavily import TavilyError

SNIPPET_CHARS = 160
# Expected failures: print one "error: ..." line instead of a traceback.
ERRORS = (EmbeddingError, StoreError, FetchError, TavilyError, BudgetExceeded, EvalError,
          LoaderError, OSError)


def _int_at_least(minimum: int):
    def parse(text: str) -> int:
        try:
            value = int(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{text!r} is not a whole number") from None
        if value < minimum:
            raise argparse.ArgumentTypeError(f"must be {minimum} or more, got {value}")
        return value

    return parse


def _build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--embedder", choices=["hash", "ollama"],
        help="ollama = real semantic embeddings; hash = offline, for testing "
        "(default: EMBEDDER setting, else ollama)",
    )
    common.add_argument("--data-dir", type=Path, help="where the index is stored")

    parser = argparse.ArgumentParser(prog="ragsearch", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    p_ingest = commands.add_parser("ingest", parents=[common], help="add files or web pages")
    p_ingest.add_argument("path", help="a .md/.txt/.pdf file, a directory, or an http(s) URL")
    p_ingest.add_argument("--crawl", action="store_true", help="crawl the site, not one page")
    p_ingest.add_argument("--max-pages", type=_int_at_least(1), default=30)
    p_ingest.add_argument("--max-depth", type=_int_at_least(0), default=2)
    p_ingest.add_argument(
        "--provider", choices=["local", "tavily"], default="local",
        help="local = free polite crawler; tavily = Tavily API (spends credits)",
    )

    p_search = commands.add_parser("search", parents=[common], help="search the index")
    p_search.add_argument("query")
    p_search.add_argument("-k", type=_int_at_least(1), default=5, help="number of results")
    p_search.add_argument("--web", action="store_true",
                          help="if local results are weak, also search Tavily (spends credits)")
    p_search.add_argument("--save-web", action="store_true",
                          help="add web results to the local index")

    p_eval = commands.add_parser("eval", parents=[common], help="measure retrieval quality")
    p_eval.add_argument("labelled", type=Path, help="JSON list of question/expected_source")
    p_eval.add_argument("-k", type=_int_at_least(1), default=5)
    return parser


def _snippet(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= SNIPPET_CHARS else text[: SNIPPET_CHARS - 3] + "..."


def _cmd_ingest(args, settings: Settings, embedder: Embedder, store: Store) -> None:
    prune_under = None
    if args.path.startswith(("http://", "https://")):
        documents = web_documents(
            args.path, settings, store.data_dir, crawl=args.crawl,
            max_pages=args.max_pages, max_depth=args.max_depth, provider=args.provider,
        )
    else:
        path = Path(args.path)
        if not path.exists():
            raise LoaderError(f"{path} does not exist")
        documents = load_documents(path)
        prune_under = path if path.is_dir() else None
    report = ingest(documents, embedder, store, prune_under=prune_under)
    total = len(report.updated) + len(report.skipped) + len(report.failed)
    removed = f", {len(report.removed)} removed" if report.removed else ""
    print(
        f"Ingested {report.added} new chunks from {total} sources "
        f"({len(report.skipped)} unchanged, {len(report.failed)} failed{removed}). "
        f"Index holds {len(store)} chunks."
    )


def _cmd_search(args, settings: Settings, embedder: Embedder, store: Store) -> None:
    web = make_tavily(settings, store.data_dir) if args.web else None
    hits = hybrid_search(args.query, embedder, store, k=args.k, web=web,
                         threshold=settings.web_fallback_threshold, persist=args.save_web)
    if not hits:
        print("No results. Ingest some documents first.")
        return
    for rank, hit in enumerate(hits, start=1):
        print(f"{rank}. {hit.score:.3f}  [{hit.origin}]  {hit.source}")
        print(f"   {_snippet(hit.text)}")


def _cmd_eval(args, settings: Settings, embedder: Embedder, store: Store) -> None:
    report = evaluate(load_cases(args.labelled), embedder, store, k=args.k)
    print(f"questions: {report.n}")
    print(f"recall@{report.k}: {report.recall:.3f}")
    print(f"MRR@{report.k}: {report.mrr:.3f}")


def main(argv: list[str] | None = None, settings: Settings | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="warning: %(message)s")
    try:
        settings = settings or load_settings()
    except ConfigError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2

    embedder = make_embedder(args.embedder or settings.embedder, host=settings.ollama_host,
                             model=settings.embed_model)
    try:
        store = Store(
            args.data_dir or settings.data_dir,
            embedder_name=embedder.name,
            index_factory=index_backend(settings.index_backend),
        )
        commands = {"ingest": _cmd_ingest, "search": _cmd_search, "eval": _cmd_eval}
        commands[args.command](args, settings, embedder, store)
    except ERRORS as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    return 0
