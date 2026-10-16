"""Minimal client for any OpenAI-compatible chat endpoint (Ollama, vLLM, OpenAI)."""

from __future__ import annotations

import httpx

from .config import settings


class ChatLLM:
    def __init__(self, base_url: str | None = None, model: str | None = None, api_key: str | None = None):
        self.base_url = (base_url or settings.llm_base_url).rstrip("/")
        self.model = model or settings.llm_model
        self.client = httpx.Client(
            headers={"Authorization": f"Bearer {api_key or settings.llm_api_key}"}, timeout=120.0
        )

    def chat(self, system: str, user: str, temperature: float = 0.0) -> str:
        resp = self.client.post(
            f"{self.base_url}/chat/completions",
            json={
                "model": self.model,
                "temperature": temperature,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            },
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
