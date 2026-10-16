"""Command line interface.

edgar-rag ingest AAPL MSFT --per-ticker 2
edgar-rag index
edgar-rag ask "What were Apple's main risk factors in 2024?"
"""

from __future__ import annotations

import argparse

from . import pipeline


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="edgar-rag")
    sub = p.add_subparsers(dest="cmd", required=True)

    ing = sub.add_parser("ingest", help="download 10-K filings from EDGAR")
    ing.add_argument("tickers", nargs="+")
    ing.add_argument("--per-ticker", type=int, default=2)

    idx = sub.add_parser("index", help="parse, chunk and embed downloaded filings")
    idx.add_argument("--no-dense", action="store_true", help="BM25 only (no embedding model)")

    ask = sub.add_parser("ask", help="ask a question")
    ask.add_argument("question")
    ask.add_argument("-k", type=int, default=5)
    ask.add_argument("--no-dense", action="store_true")
    ask.add_argument("--no-rerank", action="store_true")
    ask.add_argument("--json", action="store_true")

    args = p.parse_args(argv)
    if args.cmd == "ingest":
        for path in pipeline.ingest(args.tickers, args.per_ticker):
            print(f"saved {path}")
    elif args.cmd == "index":
        n = pipeline.build_index(use_dense=not args.no_dense)
        print(f"indexed {n} chunks")
    elif args.cmd == "ask":
        qa = pipeline.QA(use_dense=not args.no_dense, use_reranker=not args.no_rerank)
        ans = qa.ask(args.question, k=args.k)
        if args.json:
            print(pipeline.answer_to_json(ans))
            return
        print(ans.answer, "\n\nSources:")
        for s in ans.sources:
            print(f"  [{s['n']}] {s['ticker']} FY{s['fiscal_year']} Item {s['item']}  {s['url']}")


if __name__ == "__main__":
    main()
