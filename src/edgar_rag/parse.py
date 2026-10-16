"""Turn 10-K HTML into clean text split by Item (1, 1A, 7, 7A, 8, ...)."""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

# Inline XBRL filings are XHTML; lxml's HTML parser handles them fine.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# Items we care about and their human-readable names.
ITEM_NAMES = {
    "1": "Business",
    "1A": "Risk Factors",
    "1B": "Unresolved Staff Comments",
    "1C": "Cybersecurity",
    "2": "Properties",
    "3": "Legal Proceedings",
    "5": "Market for Registrant's Common Equity",
    "7": "Management's Discussion and Analysis",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements and Supplementary Data",
    "9A": "Controls and Procedures",
}

# "Item 1A." / "ITEM 7 -" / "Item 7A:" at the start of a line.
ITEM_RE = re.compile(r"^\s*item\s+(\d{1,2}[A-C]?)\s*[.:\-—–]?\s*(.*)$", re.IGNORECASE | re.MULTILINE)


@dataclass
class Section:
    item: str
    title: str
    text: str


def html_to_text(html: str | bytes) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "ix:header"]):
        tag.decompose()
    # Tables become pipe-separated rows so numbers stay next to their labels.
    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
            cells = [c for c in cells if c]
            if cells:
                rows.append(" | ".join(cells))
        table.replace_with("\n".join(rows) + "\n")
    text = soup.get_text("\n")
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def split_items(text: str) -> list[Section]:
    """Split filing text into Items.

    10-Ks list every Item twice: once in the table of contents and once for the
    real section. We keep the *last* occurrence of each Item header, which is
    almost always the real one, then cut the text between consecutive headers.
    """
    matches = list(ITEM_RE.finditer(text))
    last: dict[str, re.Match] = {}
    for m in matches:
        last[m.group(1).upper()] = m
    starts = sorted(last.values(), key=lambda m: m.start())

    sections = []
    for i, m in enumerate(starts):
        end = starts[i + 1].start() if i + 1 < len(starts) else len(text)
        body = text[m.end() : end].strip()
        item = m.group(1).upper()
        if len(body.split()) < 50:  # TOC leftovers or "Not applicable."
            continue
        sections.append(Section(item=item, title=ITEM_NAMES.get(item, m.group(2).strip()[:80]), text=body))

    if not sections:  # unusual formatting: fall back to one big section
        sections.append(Section(item="FULL", title="Full filing", text=text))
    return sections
