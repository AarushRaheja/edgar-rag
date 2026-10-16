"""Download 10-K filings from SEC EDGAR.

Uses the public JSON endpoints:
  - https://www.sec.gov/files/company_tickers.json      (ticker -> CIK)
  - https://data.sec.gov/submissions/CIK##########.json (filing history)
SEC rate limit is 10 requests/second, so we sleep a little between calls.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx

from .config import settings

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"


@dataclass
class Filing:
    ticker: str
    cik: int
    company: str
    form: str
    fiscal_year: int
    filing_date: str
    accession: str
    primary_document: str

    @property
    def url(self) -> str:
        return ARCHIVE_URL.format(
            cik=self.cik, accession=self.accession.replace("-", ""), document=self.primary_document
        )

    @property
    def doc_id(self) -> str:
        return f"{self.ticker}-{self.fiscal_year}-10K"


class EdgarClient:
    def __init__(self, user_agent: str | None = None, pause: float = 0.15):
        self.client = httpx.Client(
            headers={"User-Agent": user_agent or settings.sec_user_agent, "Accept-Encoding": "gzip, deflate"},
            timeout=30.0,
            follow_redirects=True,
        )
        self.pause = pause
        self._tickers: dict[str, tuple[int, str]] | None = None

    def _get(self, url: str) -> httpx.Response:
        time.sleep(self.pause)
        resp = self.client.get(url)
        resp.raise_for_status()
        return resp

    def lookup(self, ticker: str) -> tuple[int, str]:
        """Return (cik, company name) for a ticker."""
        if self._tickers is None:
            rows = self._get(TICKERS_URL).json().values()
            self._tickers = {r["ticker"].upper(): (int(r["cik_str"]), r["title"]) for r in rows}
        try:
            return self._tickers[ticker.upper()]
        except KeyError as e:
            raise ValueError(f"Unknown ticker: {ticker}") from e

    def list_10ks(self, ticker: str, limit: int = 3) -> list[Filing]:
        cik, company = self.lookup(ticker)
        recent = self._get(SUBMISSIONS_URL.format(cik=cik)).json()["filings"]["recent"]
        filings = []
        for form, date, acc, doc, report in zip(
            recent["form"],
            recent["filingDate"],
            recent["accessionNumber"],
            recent["primaryDocument"],
            recent["reportDate"],
        ):
            if form != "10-K":
                continue
            filings.append(
                Filing(
                    ticker=ticker.upper(),
                    cik=cik,
                    company=company,
                    form=form,
                    fiscal_year=int(report[:4]) if report else int(date[:4]),
                    filing_date=date,
                    accession=acc,
                    primary_document=doc,
                )
            )
            if len(filings) >= limit:
                break
        return filings

    def download(self, filing: Filing, out_dir: Path | None = None) -> Path:
        """Save the filing HTML plus a metadata sidecar. Skips if already cached."""
        out_dir = out_dir or settings.raw_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        html_path = out_dir / f"{filing.doc_id}.html"
        if not html_path.exists():
            html_path.write_bytes(self._get(filing.url).content)
            (out_dir / f"{filing.doc_id}.json").write_text(json.dumps(asdict(filing) | {"url": filing.url}, indent=2))
        return html_path
