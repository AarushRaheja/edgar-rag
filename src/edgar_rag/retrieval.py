"""Hybrid retrieval: BM25 + dense embeddings, fused with Reciprocal Rank Fusion.

BM25 is good at exact terms ("goodwill impairment", tickers, numbers) and
dense vectors are good at paraphrases ("how much did they spend on R&D"), so
we run both and merge the rankings.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Protocol

import numpy as np

from .chunking import Chunk

TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.'][a-z0-9]+)*")
STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "will",
        "with",
    ]
)


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


class BM25:
    """Okapi BM25, implemented directly so the ranking is easy to inspect."""

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_tokens = [tokenize(d) for d in docs]
        self.doc_len = np.array([len(t) for t in self.doc_tokens], dtype=float)
        self.avgdl = float(self.doc_len.mean()) if len(docs) else 0.0
        self.tf = [Counter(t) for t in self.doc_tokens]
        df: Counter[str] = Counter()
        for toks in self.doc_tokens:
            df.update(set(toks))
        n = len(docs)
        self.idf = {term: math.log(1 + (n - f + 0.5) / (f + 0.5)) for term, f in df.items()}

    def scores(self, query: str) -> np.ndarray:
        q = tokenize(query)
        out = np.zeros(len(self.doc_tokens))
        for i, tf in enumerate(self.tf):
            denom_norm = self.k1 * (1 - self.b + self.b * self.doc_len[i] / (self.avgdl or 1))
            for term in q:
                f = tf.get(term)
                if f:
                    out[i] += self.idf[term] * f * (self.k1 + 1) / (f + denom_norm)
        return out

    def search(self, query: str, k: int = 20) -> list[tuple[int, float]]:
        s = self.scores(query)
        top = np.argsort(-s)[:k]
        return [(int(i), float(s[i])) for i in top if s[i] > 0]


class Encoder(Protocol):
    def encode(self, texts: list[str], normalize_embeddings: bool = True) -> np.ndarray: ...


class DenseIndex:
    def __init__(self, encoder: Encoder, docs: list[str], embeddings: np.ndarray | None = None):
        self.encoder = encoder
        self.emb = embeddings if embeddings is not None else np.asarray(encoder.encode(docs, normalize_embeddings=True))

    def search(self, query: str, k: int = 20) -> list[tuple[int, float]]:
        q = np.asarray(self.encoder.encode([query], normalize_embeddings=True))[0]
        sims = self.emb @ q
        top = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in top]


def reciprocal_rank_fusion(rankings: list[list[tuple[int, float]]], k: int = 60) -> list[tuple[int, float]]:
    """Merge ranked lists: score(d) = sum over lists of 1 / (k + rank)."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, (doc, _) in enumerate(ranking, start=1):
            fused[doc] = fused.get(doc, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda x: -x[1])


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], encoder: Encoder | None = None, embeddings: np.ndarray | None = None):
        self.chunks = chunks
        texts = [c.text for c in chunks]
        self.bm25 = BM25(texts)
        self.dense = DenseIndex(encoder, texts, embeddings) if encoder is not None else None

    def search(
        self,
        query: str,
        k: int = 20,
        tickers: set[str] | None = None,
        years: set[int] | None = None,
        items: set[str] | None = None,
    ) -> list[tuple[Chunk, float]]:
        rankings = [self.bm25.search(query, k=k * 5)]
        if self.dense is not None:
            rankings.append(self.dense.search(query, k=k * 5))
        results = []
        for idx, score in reciprocal_rank_fusion(rankings):
            c = self.chunks[idx]
            if tickers and c.ticker not in tickers:
                continue
            if years and c.fiscal_year not in years:
                continue
            if items and c.item not in items:
                continue
            results.append((c, score))
            if len(results) >= k:
                break
        return results


class Reranker:
    """Cross-encoder reranking of the fused candidates (slower, more accurate)."""

    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, hits: list[tuple[Chunk, float]], k: int = 5) -> list[tuple[Chunk, float]]:
        if not hits:
            return []
        scores = self.model.predict([(query, c.text) for c, _ in hits])
        order = np.argsort(-np.asarray(scores))[:k]
        return [(hits[i][0], float(scores[i])) for i in order]
