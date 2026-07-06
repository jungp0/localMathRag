from __future__ import annotations

import json
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from .config import GenerationConfig


class OpenAICompatibleClient:
    def __init__(self, config: GenerationConfig):
        self.config = config

    def chat_json(self, messages: list[dict[str, str]], schema: dict[str, Any] | None = None) -> dict[str, Any] | None:
        if not self.config.enabled:
            return None
        url = self.config.base_url.rstrip("/") + "/chat/completions"
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "stream": False,
        }
        if schema:
            payload["response_format"] = {"type": "json_schema", "json_schema": {"name": "lookup_result", "schema": schema}}
        request = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.config.timeout_seconds) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except (OSError, URLError, TimeoutError, json.JSONDecodeError):
            return None
        content = raw.get("choices", [{}])[0].get("message", {}).get("content")
        if not content:
            return None
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return None

    def chat_text(self, messages: list[dict[str, str]]) -> str | None:
        if not self.config.enabled:
            return None
        url = self.config.base_url.rstrip("/") + "/chat/completions"
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "stream": False,
        }
        request = Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.config.timeout_seconds) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except (OSError, URLError, TimeoutError, json.JSONDecodeError):
            return None
        content = raw.get("choices", [{}])[0].get("message", {}).get("content")
        return str(content).strip() if content else None


def generation_config_from_settings(settings: dict[str, Any]) -> GenerationConfig:
    return GenerationConfig(
        enabled=bool(settings.get("enabled", False)),
        provider=str(settings.get("provider", "openai_compatible")),
        base_url=str(settings.get("base_url", "http://127.0.0.1:8080/v1")),
        model=str(settings.get("model", "Qwen/Qwen3-8B-GGUF:Q4_K_M")),
        temperature=float(settings.get("temperature", 0)),
        timeout_seconds=int(settings.get("timeout_seconds", 60)),
    )


def synthesize_answer(settings: dict[str, Any], query: str, result: dict[str, Any]) -> dict[str, Any] | None:
    config = generation_config_from_settings(settings)
    client = OpenAICompatibleClient(config)
    evidence_lines: list[str] = []
    evidence = result.get("evidence", {})
    for ev_id, item in list(evidence.items())[:8]:
        preview = item.get("text_preview") or item.get("caption") or ""
        if preview:
            evidence_lines.append(f"{ev_id}: {preview}")
    if not evidence_lines:
        for item in result.get("items", [])[:8]:
            evidence_lines.append(json.dumps(item, ensure_ascii=False))
    prompt = (
        "You are a local technical document retrieval assistant. "
        "Answer only from the provided evidence. If evidence is insufficient, say so. "
        "Return concise Chinese text.\n\n"
        f"Question: {query}\n\nEvidence:\n" + "\n".join(evidence_lines)
    )
    answer = client.chat_text(
        [
            {"role": "system", "content": "You synthesize answers from local evidence with citations."},
            {"role": "user", "content": prompt},
        ]
    )
    if not answer:
        return None
    return {
        "id": "llm.answer.001",
        "type": "generated_answer",
        "text": answer,
        "model": config.model,
        "evidence": list(evidence.keys())[:8],
    }
