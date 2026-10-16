# EDGAR-RAG

Question answering over SEC 10-K filings with hybrid retrieval and source-linked answers.

```
$ edgar-rag ask "How did Apple's iPhone net sales change in fiscal 2025?"
iPhone net sales increased ... [1]

Sources:
  [1] AAPL FY2025 Item 7  https://www.sec.gov/Archives/edgar/data/320193/...
```

## How it works

```
EDGAR API ──> 10-K HTML ──> clean text + tables ──> split by Item (1, 1A, 7, 8 ...)
                                                          │
                                       parent chunks (~1200 words)
                                       └─ child chunks (~220 words, 40 overlap)
                                                          │
question ──> query router ──> BM25 ─┐                     │
             (tickers, years,       ├─ RRF fusion ──> cross-encoder rerank
              Items, rewrite)  dense┘                     │
                                                          ▼
                                    child hits expanded to parent context
                                                          ▼
                                         LLM answer with [n] citations
```

- **Ingestion** (`edgar.py`): ticker to CIK lookup, filing history from `data.sec.gov`, downloads the primary 10-K document. Respects the SEC User-Agent and rate-limit rules.
- **Parsing** (`parse.py`): strips HTML, keeps table rows on one line so numbers stay next to their labels, and splits the filing into Items. Every 10-K lists each Item twice (table of contents + real section), so the parser keeps the last header for each Item.
- **Parent-child chunking** (`chunking.py`): small child chunks are searched for precision, and the matching parent chunk is what the LLM sees, so it has enough context.
- **Hybrid retrieval** (`retrieval.py`): a hand-written BM25 for exact terms ("goodwill impairment", tickers) plus dense embeddings for paraphrases, merged with Reciprocal Rank Fusion, then reranked by a cross-encoder.
- **Query routing** (`router.py`): extracts tickers and fiscal years for metadata filters, routes the question to the Items most likely to hold the answer (risk questions to 1A, revenue to 7/8, and so on), expands finance shorthand ("R&D", "capex"), and detects comparison questions so each company/year gets its own evidence.
- **Answering** (`pipeline.py`, `llm.py`): any OpenAI-compatible endpoint (Ollama by default). The prompt forces answers to cite excerpts and to say so when the filings don't contain the answer.

## Quickstart

```bash
uv sync --extra dense          # or: uv sync  (BM25 only, no torch)
cp .env.example .env           # set SEC_USER_AGENT="Your Name you@email.com"

uv run edgar-rag ingest AAPL MSFT --per-ticker 2
uv run edgar-rag index
uv run edgar-rag ask "Compare AAPL and MSFT research and development spending in 2025"
```

Without the `dense` extra, use `--no-dense` / `--no-rerank` for a lightweight BM25-only setup.

## Tests

```bash
uv run pytest
```

Tests cover HTML/table parsing, Item splitting (including the table-of-contents duplicate), parent-child chunk overlap, BM25 ranking, RRF fusion, metadata filtering and query routing. No network or model downloads needed.

## Roadmap

- Evaluation set of question/answer pairs with retrieval recall@k and answer faithfulness
- XBRL financial data for exact numeric answers
- Persisted vector store instead of in-memory numpy
