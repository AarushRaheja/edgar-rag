"""End-to-end pipeline: ingest filings -> build index -> answer with citations."""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .chunking import Chunk, chunk_sections
from .config import settings
from .edgar import EdgarClient
from .parse import html_to_text, split_items
from .retrieval import HybridRetriever, Reranker
from .router import plan_query

SYSTEM_PROMPT = (
    "You answer questions about companies using ONLY the 10-K excerpts provided. "
    "Cite every claim with the excerpt number in square brackets, like [2]. "
    "If the excerpts do not contain the answer, say so instead of guessing. "
    "Quote numbers exactly as they appear."
)


@dataclass
class Answer:
    question: str
    answer: str
    sources: list[dict]


def ingest(tickers: list[str], per_ticker: int = 2) -> list[Path]:
    client = EdgarClient()
    paths = []
    for t in tickers:
        for filing in client.list_10ks(t, limit=per_ticker):
            paths.append(client.download(filing))
    return paths


def build_index(raw_dir: Path | None = None, index_dir: Path | None = None, use_dense: bool = True) -> int:
    raw_dir = raw_dir or settings.raw_dir
    index_dir = index_dir or settings.index_dir
    index_dir.mkdir(parents=True, exist_ok=True)

    parents: list[Chunk] = []
    children: list[Chunk] = []
    for meta_path in sorted(raw_dir.glob("*.json")):
        meta = json.loads(meta_path.read_text())
        html = meta_path.with_suffix(".html").read_bytes()
        sections = split_items(html_to_text(html))
        p, c = chunk_sections(
            sections,
            doc_id=meta_path.stem,
            ticker=meta["ticker"],
            fiscal_year=meta["fiscal_year"],
            url=meta["url"],
            parent_words=settings.parent_chunk_words,
            child_words=settings.child_chunk_words,
            child_overlap=settings.child_overlap_words,
        )
        parents += p
        children += c

    with open(index_dir / "chunks.pkl", "wb") as f:
        pickle.dump({"parents": parents, "children": children}, f)

    if use_dense and children:
        from sentence_transformers import SentenceTransformer

        enc = SentenceTransformer(settings.embedding_model)
        emb = enc.encode([c.text for c in children], normalize_embeddings=True, show_progress_bar=True)
        np.save(index_dir / "embeddings.npy", emb)
    return len(children)


class QA:
    def __init__(self, index_dir: Path | None = None, use_dense: bool = True, use_reranker: bool = True, llm=None):
        index_dir = index_dir or settings.index_dir
        with open(index_dir / "chunks.pkl", "rb") as f:
            data = pickle.load(f)
        self.parents = {p.chunk_id: p for p in data["parents"]}
        children = data["children"]

        encoder, emb = None, None
        emb_path = index_dir / "embeddings.npy"
        if use_dense and emb_path.exists():
            from sentence_transformers import SentenceTransformer

            encoder = SentenceTransformer(settings.embedding_model)
            emb = np.load(emb_path)
        self.retriever = HybridRetriever(children, encoder, emb)
        self.reranker = Reranker(settings.reranker_model) if use_reranker else None
        self.tickers = {c.ticker for c in children}
        if llm is None:
            from .llm import ChatLLM

            llm = ChatLLM()
        self.llm = llm

    def retrieve(self, question: str, k: int = 5) -> list[Chunk]:
        plan = plan_query(question, self.tickers)
        hits = self.retriever.search(
            plan.rewritten, k=30, tickers=plan.tickers or None, years=plan.years or None, items=plan.items or None
        )
        if not hits and plan.items:  # routing was too strict, retry across all Items
            hits = self.retriever.search(plan.rewritten, k=30, tickers=plan.tickers or None, years=plan.years or None)
        if self.reranker is not None:
            hits = self.reranker.rerank(question, hits, k=k * 2)

        # Comparison questions need evidence from every ticker/year, not just the top scorer.
        per_group = 2 if plan.route == "compare" else k
        picked, seen_parents, group_counts = [], set(), {}
        for child, _ in hits:
            group = (child.ticker, child.fiscal_year)
            if child.parent_id in seen_parents or group_counts.get(group, 0) >= per_group:
                continue
            seen_parents.add(child.parent_id)
            group_counts[group] = group_counts.get(group, 0) + 1
            picked.append(self.parents[child.parent_id])  # expand child -> parent for context
            if len(picked) >= k:
                break
        return picked

    def ask(self, question: str, k: int = 5) -> Answer:
        context = self.retrieve(question, k=k)
        if not context:
            return Answer(question, "I couldn't find anything relevant in the indexed filings.", [])
        blocks = [
            f"[{i}] {c.ticker} FY{c.fiscal_year} 10-K, Item {c.item} ({c.item_title}):\n{c.text}"
            for i, c in enumerate(context, start=1)
        ]
        answer = self.llm.chat(SYSTEM_PROMPT, "Excerpts:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}")
        sources = [
            {"n": i, "ticker": c.ticker, "fiscal_year": c.fiscal_year, "item": c.item, "url": c.url}
            for i, c in enumerate(context, start=1)
        ]
        return Answer(question, answer, sources)


def answer_to_json(a: Answer) -> str:
    return json.dumps(asdict(a), indent=2)
