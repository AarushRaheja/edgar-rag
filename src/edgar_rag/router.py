"""Query understanding: pull out tickers/years, route to likely 10-K Items,
and rewrite the query with financial synonyms so BM25 doesn't miss obvious matches.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

YEAR_RE = re.compile(r"\b(20\d{2})\b")
TICKER_RE = re.compile(r"\b[A-Z]{1,5}\b")
NOT_TICKERS = {"A", "I", "AI", "US", "USA", "CEO", "CFO", "SEC", "MD", "RD", "R", "D", "Q", "FY", "EPS", "GAAP", "IT"}

# keyword -> Items that usually contain the answer
ROUTES: list[tuple[tuple[str, ...], set[str]]] = [
    (("risk", "threat", "uncertain", "litigation risk", "regulat"), {"1A"}),
    (("cyber", "security incident", "breach"), {"1C", "1A"}),
    (
        (
            "revenue",
            "sales",
            "margin",
            "income",
            "earnings",
            "cash flow",
            "guidance",
            "outlook",
            "growth",
            "segment",
            "expense",
            "r&d",
            "research and development",
        ),
        {"7", "8"},
    ),
    (("balance sheet", "assets", "liabilities", "debt", "goodwill", "impairment", "depreciation"), {"8", "7"}),
    (("interest rate", "currency", "foreign exchange", "market risk", "hedg"), {"7A"}),
    (("business", "products", "competition", "competitor", "employees", "customers", "strategy"), {"1"}),
    (("lawsuit", "legal proceeding", "litigation"), {"3", "1A"}),
    (("property", "properties", "facilities", "headquarters"), {"2"}),
    (("internal control", "controls and procedures"), {"9A"}),
]

SYNONYMS = {
    "r&d": "research and development",
    "revenue": "net sales total revenue",
    "profit": "net income operating income",
    "capex": "capital expenditures",
    "buyback": "share repurchase",
    "headcount": "employees",
    "debt": "borrowings notes payable",
}


@dataclass
class QueryPlan:
    original: str
    rewritten: str
    tickers: set[str] = field(default_factory=set)
    years: set[int] = field(default_factory=set)
    items: set[str] = field(default_factory=set)
    route: str = "lookup"  # "lookup" or "compare"


def plan_query(query: str, known_tickers: set[str] | None = None) -> QueryPlan:
    lowered = query.lower()
    years = {int(y) for y in YEAR_RE.findall(query)}
    tickers = {t for t in TICKER_RE.findall(query) if t not in NOT_TICKERS}
    if known_tickers is not None:
        tickers &= known_tickers

    items: set[str] = set()
    for keywords, route_items in ROUTES:
        if any(k in lowered for k in keywords):
            items |= route_items

    extra = [v for k, v in SYNONYMS.items() if k in lowered]
    rewritten = query if not extra else f"{query} {' '.join(extra)}"

    compare = (
        len(tickers) > 1 or len(years) > 1 or any(w in lowered for w in ("compare", " vs", "versus", "change from"))
    )
    return QueryPlan(query, rewritten, tickers, years, items, "compare" if compare else "lookup")
