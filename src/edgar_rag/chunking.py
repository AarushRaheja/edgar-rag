"""Parent-child chunking.

Small "child" chunks are what we search over (precise matches), but we hand the
LLM the larger "parent" chunk they came from so it sees enough context to
answer. Every chunk keeps its filing + Item metadata for citations.
"""

from __future__ import annotations

from dataclasses import dataclass

from .parse import Section


@dataclass
class Chunk:
    chunk_id: str
    parent_id: str
    doc_id: str
    ticker: str
    fiscal_year: int
    item: str
    item_title: str
    text: str
    url: str = ""


def _windows(words: list[str], size: int, overlap: int) -> list[list[str]]:
    if size <= overlap:
        raise ValueError("size must be larger than overlap")
    out, step = [], size - overlap
    for start in range(0, max(len(words), 1), step):
        piece = words[start : start + size]
        if piece:
            out.append(piece)
        if start + size >= len(words):
            break
    return out


def chunk_sections(
    sections: list[Section],
    *,
    doc_id: str,
    ticker: str,
    fiscal_year: int,
    url: str = "",
    parent_words: int = 1200,
    child_words: int = 220,
    child_overlap: int = 40,
) -> tuple[list[Chunk], list[Chunk]]:
    """Return (parents, children)."""
    parents: list[Chunk] = []
    children: list[Chunk] = []
    for section in sections:
        words = section.text.split()
        parent_blocks = [words[i : i + parent_words] for i in range(0, len(words), parent_words)]
        for p_idx, p_words in enumerate(parent_blocks):
            parent_id = f"{doc_id}:{section.item}:p{p_idx}"
            parents.append(
                Chunk(
                    parent_id,
                    parent_id,
                    doc_id,
                    ticker,
                    fiscal_year,
                    section.item,
                    section.title,
                    " ".join(p_words),
                    url,
                )
            )
            for c_idx, c_words in enumerate(_windows(p_words, child_words, child_overlap)):
                children.append(
                    Chunk(
                        f"{parent_id}:c{c_idx}",
                        parent_id,
                        doc_id,
                        ticker,
                        fiscal_year,
                        section.item,
                        section.title,
                        " ".join(c_words),
                        url,
                    )
                )
    return parents, children
