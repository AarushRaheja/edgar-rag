import numpy as np

from edgar_rag.chunking import chunk_sections
from edgar_rag.parse import Section, html_to_text, split_items
from edgar_rag.retrieval import BM25, HybridRetriever, reciprocal_rank_fusion
from edgar_rag.router import plan_query

FILLER = " ".join(["lorem"] * 80)

HTML = f"""
<html><body>
<p>Table of Contents</p><p>Item 1. Business</p><p>Item 1A. Risk Factors</p><p>Item 7. MD&A</p>
<p>Item 1. Business</p><p>We design smartphones and wearables. {FILLER}</p>
<p>Item 1A. Risk Factors</p><p>Supply chain disruption could hurt results. {FILLER}</p>
<p>Item 7. Management's Discussion</p>
<table><tr><td>Net sales</td><td>$391,035</td></tr></table>
<p>Revenue grew because of services. {FILLER}</p>
</body></html>
"""


def test_html_to_text_keeps_table_rows_together():
    text = html_to_text(HTML)
    assert "Net sales | $391,035" in text


def test_split_items_skips_table_of_contents():
    sections = split_items(html_to_text(HTML))
    assert [s.item for s in sections] == ["1", "1A", "7"]
    assert "smartphones" in sections[0].text
    assert sections[1].title == "Risk Factors"


def test_parent_child_chunking_links_children_to_parents():
    words = " ".join(f"w{i}" for i in range(2500))
    parents, children = chunk_sections(
        [Section("7", "MD&A", words)],
        doc_id="X-2024-10K",
        ticker="X",
        fiscal_year=2024,
        parent_words=1000,
        child_words=200,
        child_overlap=50,
    )
    assert len(parents) == 3
    parent_ids = {p.chunk_id for p in parents}
    assert all(c.parent_id in parent_ids for c in children)
    # overlap: consecutive children share 50 words
    first, second = children[0].text.split(), children[1].text.split()
    assert first[-50:] == second[:50]


def test_bm25_ranks_exact_term_first():
    bm = BM25(["goodwill impairment was recorded", "revenue increased", "employees and culture"])
    assert bm.search("goodwill impairment")[0][0] == 0


def test_rrf_rewards_agreement():
    fused = reciprocal_rank_fusion([[(1, 9.0), (2, 5.0)], [(2, 0.9), (3, 0.8)]])
    assert fused[0][0] == 2


class FakeEncoder:
    """Bag-of-letters 'embedding' so tests don't need to download a model."""

    def encode(self, texts, normalize_embeddings=True):
        out = np.zeros((len(texts), 26))
        for i, t in enumerate(texts):
            for ch in t.lower():
                if "a" <= ch <= "z":
                    out[i, ord(ch) - 97] += 1
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


def test_hybrid_filters_by_ticker():
    from edgar_rag.chunking import Chunk

    chunks = [
        Chunk("a", "a", "AAPL-2024-10K", "AAPL", 2024, "1A", "Risk Factors", "supply chain risk in china"),
        Chunk("b", "b", "MSFT-2024-10K", "MSFT", 2024, "1A", "Risk Factors", "supply chain risk for cloud"),
    ]
    r = HybridRetriever(chunks, FakeEncoder())
    hits = r.search("supply chain risk", tickers={"MSFT"})
    assert [c.ticker for c, _ in hits] == ["MSFT"]


def test_router():
    plan = plan_query("Compare AAPL and MSFT R&D spending in 2023", known_tickers={"AAPL", "MSFT"})
    assert plan.tickers == {"AAPL", "MSFT"}
    assert plan.years == {2023}
    assert "7" in plan.items
    assert plan.route == "compare"
    assert "research and development" in plan.rewritten
