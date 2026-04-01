"""Evidence retrieval service.

Gathers real-world context for markets from multiple sources:
- DuckDuckGo instant answers (no API key needed)
- CoinGecko crypto prices (free, no key)
- Google News RSS (free, no key)
- Wikipedia summaries (free, no key)
"""
from __future__ import annotations

import asyncio
import hashlib
import re
from datetime import datetime, timezone
from typing import Any

import httpx
import structlog

logger = structlog.get_logger()


class EvidenceBundle:
    """Collection of evidence snippets for a market."""

    def __init__(self) -> None:
        self.snippets: list[str] = []
        self.sources: list[dict[str, Any]] = []

    def add(self, snippet: str, source_type: str, source_domain: str = "", url: str = "") -> None:
        if snippet and snippet not in self.snippets:
            self.snippets.append(snippet)
            self.sources.append({
                "source_type": source_type,
                "source_domain": source_domain,
                "url": url,
                "snippet": snippet,
            })

    @property
    def count(self) -> int:
        return len(self.snippets)

    def top_snippets(self, n: int = 8) -> list[str]:
        return self.snippets[:n]


class EvidenceRetriever:
    """Retrieves evidence from multiple free sources."""

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(timeout=15.0, follow_redirects=True)

    async def close(self) -> None:
        await self._client.aclose()

    async def gather(self, title: str, entity: str | None = None, category: str | None = None) -> EvidenceBundle:
        """Gather evidence from all sources in parallel."""
        bundle = EvidenceBundle()

        # Build search queries
        queries = [title]
        if entity:
            queries.append(f"{entity} latest news")

        tasks = []
        for q in queries[:2]:
            tasks.append(self._search_duckduckgo(q))
            tasks.append(self._search_google_news(q))

        # Crypto-specific: get price data
        crypto_ticker = self._extract_crypto(title)
        if crypto_ticker:
            tasks.append(self._get_crypto_price(crypto_ticker))

        # Wikipedia for entity context
        if entity:
            tasks.append(self._search_wikipedia(entity))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                continue
            if isinstance(result, list):
                for item in result:
                    bundle.add(**item)
            elif isinstance(result, dict) and result.get("snippet"):
                bundle.add(**result)

        logger.debug("evidence_gathered", title=title[:50], count=bundle.count)
        return bundle

    async def _search_duckduckgo(self, query: str) -> list[dict]:
        """DuckDuckGo instant answer API (no key needed)."""
        try:
            resp = await self._client.get(
                "https://api.duckduckgo.com/",
                params={"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"},
            )
            data = resp.json()
            results = []

            # Abstract (main answer)
            if data.get("AbstractText"):
                results.append({
                    "snippet": data["AbstractText"][:500],
                    "source_type": "official",
                    "source_domain": data.get("AbstractSource", "duckduckgo"),
                    "url": data.get("AbstractURL", ""),
                })

            # Related topics
            for topic in (data.get("RelatedTopics") or [])[:3]:
                if isinstance(topic, dict) and topic.get("Text"):
                    results.append({
                        "snippet": topic["Text"][:300],
                        "source_type": "news",
                        "source_domain": "duckduckgo",
                        "url": topic.get("FirstURL", ""),
                    })

            return results
        except Exception as e:
            logger.debug(f"DDG search failed: {e}")
            return []

    async def _search_google_news(self, query: str) -> list[dict]:
        """Google News RSS (no key needed)."""
        try:
            import feedparser
            url = f"https://news.google.com/rss/search?q={query.replace(' ', '+')}&hl=en-US&gl=US&ceid=US:en"
            resp = await self._client.get(url)
            feed = feedparser.parse(resp.text)

            results = []
            for entry in feed.entries[:5]:
                title_text = entry.get("title", "")
                published = entry.get("published", "")
                source = entry.get("source", {}).get("title", "")
                snippet = f"[{source}] {title_text} ({published})" if source else f"{title_text} ({published})"
                results.append({
                    "snippet": snippet[:400],
                    "source_type": "news",
                    "source_domain": source or "google_news",
                    "url": entry.get("link", ""),
                })
            return results
        except Exception as e:
            logger.debug(f"Google News search failed: {e}")
            return []

    async def _get_crypto_price(self, ticker: str) -> dict:
        """Get current crypto price from CoinGecko (free, no key)."""
        coin_map = {
            "BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana",
            "DOGE": "dogecoin", "XRP": "ripple", "ADA": "cardano",
            "DOT": "polkadot", "MATIC": "matic-network", "AVAX": "avalanche-2",
            "LINK": "chainlink", "BNB": "binancecoin", "NEAR": "near",
            "SUI": "sui", "APT": "aptos", "ARB": "arbitrum",
        }
        coin_id = coin_map.get(ticker.upper())
        if not coin_id:
            return {}

        try:
            resp = await self._client.get(
                f"https://api.coingecko.com/api/v3/simple/price",
                params={"ids": coin_id, "vs_currencies": "usd", "include_24hr_change": "true"},
            )
            data = resp.json()
            if coin_id in data:
                price = data[coin_id].get("usd", 0)
                change = data[coin_id].get("usd_24h_change", 0)
                return {
                    "snippet": f"{ticker} current price: ${price:,.2f} (24h change: {change:+.1f}%)",
                    "source_type": "onchain",
                    "source_domain": "coingecko",
                    "url": f"https://www.coingecko.com/en/coins/{coin_id}",
                }
        except Exception as e:
            logger.debug(f"CoinGecko failed for {ticker}: {e}")
        return {}

    async def _search_wikipedia(self, entity: str) -> dict:
        """Get Wikipedia summary for an entity."""
        try:
            resp = await self._client.get(
                "https://en.wikipedia.org/api/rest_v1/page/summary/" + entity.replace(" ", "_"),
            )
            if resp.status_code == 200:
                data = resp.json()
                extract = data.get("extract", "")
                if extract:
                    return {
                        "snippet": extract[:500],
                        "source_type": "official",
                        "source_domain": "wikipedia",
                        "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
                    }
        except Exception as e:
            logger.debug(f"Wikipedia failed for {entity}: {e}")
        return {}

    @staticmethod
    def _extract_crypto(title: str) -> str | None:
        """Extract crypto ticker from market title."""
        patterns = [
            r"\b(BTC|Bitcoin)\b",
            r"\b(ETH|Ethereum)\b",
            r"\b(SOL|Solana)\b",
            r"\b(DOGE|Dogecoin)\b",
            r"\b(XRP)\b",
            r"\b(ADA|Cardano)\b",
            r"\b(DOT|Polkadot)\b",
            r"\b(MATIC|Polygon)\b",
            r"\b(AVAX|Avalanche)\b",
            r"\b(LINK|Chainlink)\b",
            r"\b(BNB)\b",
            r"\b(SUI)\b",
            r"\b(APT|Aptos)\b",
            r"\b(ARB|Arbitrum)\b",
        ]
        ticker_map = {
            "Bitcoin": "BTC", "Ethereum": "ETH", "Solana": "SOL",
            "Dogecoin": "DOGE", "Cardano": "ADA", "Polkadot": "DOT",
            "Polygon": "MATIC", "Avalanche": "AVAX", "Chainlink": "LINK",
            "Aptos": "APT", "Arbitrum": "ARB",
        }
        for pattern in patterns:
            match = re.search(pattern, title, re.IGNORECASE)
            if match:
                found = match.group(1)
                return ticker_map.get(found, found.upper())
        return None
