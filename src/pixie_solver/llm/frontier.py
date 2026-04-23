from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Protocol

from pixie_solver.rules.providers import (
    CompileRequest,
    CompileResponse,
    RepairRequest,
    RepairResponse,
)
from pixie_solver.utils.serialization import JsonValue, canonical_json


DEFAULT_ANTHROPIC_MODEL = "claude-opus-4-6"
DEFAULT_OPENAI_MODEL = "gpt-5.4"
ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
ANTHROPIC_VERSION = "2023-06-01"

_ANTHROPIC_EFFORTS = {"low", "medium", "high", "max"}
_OPENAI_EFFORTS = {"none", "low", "medium", "high", "xhigh"}

_SYSTEM_PROMPT = """You are the PixieChess piece-program compiler and repair engine.
PixieChess pieces are represented only by the JSON DSL described in the request.

Return exactly one JSON object and no markdown.
Do not invent operator names, event names, state field types, square references, or base piece types.
Prefer the smallest deterministic DSL program that explains the behavior.
Preserve the current piece_id during repair unless the request explicitly says otherwise.
"""


class JsonTransport(Protocol):
    def post_json(
        self,
        *,
        url: str,
        headers: dict[str, str],
        payload: dict[str, JsonValue],
        timeout_seconds: float,
    ) -> dict[str, JsonValue]:
        ...


class LLMProviderError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class LLMConfig:
    provider: str | None = None
    model: str | None = None
    api_key: str | None = None
    api_key_env: str | None = None
    endpoint: str | None = None
    max_tokens: int = 8192
    timeout_seconds: float = 120.0
    temperature: float | None = 0.0
    thinking: bool = True
    effort: str = "high"
    anthropic_version: str = ANTHROPIC_VERSION

    def resolved_provider(self) -> str:
        provider = (self.provider or os.environ.get("PIXIE_LLM_PROVIDER") or "anthropic").lower()
        if provider not in {"anthropic", "openai"}:
            raise ValueError("LLM provider must be 'anthropic' or 'openai'")
        return provider

    def resolved_model(self) -> str:
        if self.model:
            return self.model
        provider = self.resolved_provider()
        if provider == "anthropic":
            return os.environ.get("PIXIE_LLM_MODEL", DEFAULT_ANTHROPIC_MODEL)
        return os.environ.get("PIXIE_LLM_MODEL", DEFAULT_OPENAI_MODEL)

    def resolved_api_key(self) -> str:
        if self.api_key:
            return self.api_key
        env_name = self.resolved_api_key_env()
        api_key = os.environ.get(env_name)
        if not api_key:
            raise ValueError(f"{env_name} is required for live LLM calls")
        return api_key

    def resolved_api_key_env(self) -> str:
        if self.api_key_env:
            return self.api_key_env
        provider = self.resolved_provider()
        if provider == "anthropic":
            return "ANTHROPIC_API_KEY"
        return "OPENAI_API_KEY"

    def resolved_endpoint(self) -> str:
        if self.endpoint:
            return self.endpoint
        provider = self.resolved_provider()
        if provider == "anthropic":
            return ANTHROPIC_MESSAGES_URL
        return OPENAI_RESPONSES_URL

    def validate(self) -> None:
        if self.max_tokens < 1:
            raise ValueError("LLM max_tokens must be at least 1")
        if self.timeout_seconds <= 0:
            raise ValueError("LLM timeout_seconds must be positive")
        provider = self.resolved_provider()
        effort = self.effort.lower()
        if provider == "anthropic" and effort not in _ANTHROPIC_EFFORTS:
            raise ValueError("Anthropic effort must be one of low, medium, high, max")
        if provider == "openai" and effort not in _OPENAI_EFFORTS:
            raise ValueError("OpenAI effort must be one of none, low, medium, high, xhigh")

    def safe_metadata(self) -> dict[str, JsonValue]:
        return {
            "provider": self.resolved_provider(),
            "model": self.resolved_model(),
            "endpoint": self.resolved_endpoint(),
            "api_key_env": self.resolved_api_key_env(),
            "max_tokens": self.max_tokens,
            "timeout_seconds": self.timeout_seconds,
            "temperature": self.temperature,
            "thinking": self.thinking,
            "effort": self.effort.lower(),
        }


@dataclass(frozen=True, slots=True)
class LLMJsonResponse:
    data: dict[str, JsonValue]
    metadata: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "data", dict(self.data))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class UrllibJsonTransport:
    def post_json(
        self,
        *,
        url: str,
        headers: dict[str, str],
        payload: dict[str, JsonValue],
        timeout_seconds: float,
    ) -> dict[str, JsonValue]:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise LLMProviderError(
                f"LLM provider returned HTTP {exc.code}: {error_body}"
            ) from exc
        except urllib.error.URLError as exc:
            raise LLMProviderError(f"LLM provider request failed: {exc}") from exc
        payload_json = json.loads(raw)
        if not isinstance(payload_json, dict):
            raise LLMProviderError("LLM provider response must be a JSON object")
        return payload_json


@dataclass(frozen=True, slots=True)
class FrontierLLMClient:
    config: LLMConfig = field(default_factory=LLMConfig)
    transport: JsonTransport = field(default_factory=UrllibJsonTransport)

    def generate_json(
        self,
        *,
        user_payload: dict[str, JsonValue],
        system_prompt: str = _SYSTEM_PROMPT,
    ) -> LLMJsonResponse:
        self.config.validate()
        provider = self.config.resolved_provider()
        if provider == "anthropic":
            response = self._call_anthropic(
                system_prompt=system_prompt,
                user_payload=user_payload,
            )
            text = _anthropic_text(response)
        else:
            response = self._call_openai(
                system_prompt=system_prompt,
                user_payload=user_payload,
            )
            text = _openai_text(response)
        data = _extract_json_object(text)
        return LLMJsonResponse(
            data=data,
            metadata={
                **self.config.safe_metadata(),
                "response_id": response.get("id"),
                "usage": response.get("usage", {}),
            },
        )

    def _call_anthropic(
        self,
        *,
        system_prompt: str,
        user_payload: dict[str, JsonValue],
    ) -> dict[str, JsonValue]:
        payload: dict[str, JsonValue] = {
            "model": self.config.resolved_model(),
            "max_tokens": self.config.max_tokens,
            "system": system_prompt,
            "messages": [
                {
                    "role": "user",
                    "content": canonical_json(user_payload, indent=2),
                }
            ],
            "output_config": {
                "effort": self.config.effort.lower(),
            },
        }
        if self.config.temperature is not None:
            payload["temperature"] = self.config.temperature
        if self.config.thinking:
            payload["thinking"] = {
                "type": "adaptive",
                "display": "omitted",
            }
        return self.transport.post_json(
            url=self.config.resolved_endpoint(),
            headers={
                "content-type": "application/json",
                "x-api-key": self.config.resolved_api_key(),
                "anthropic-version": self.config.anthropic_version,
            },
            payload=payload,
            timeout_seconds=self.config.timeout_seconds,
        )

    def _call_openai(
        self,
        *,
        system_prompt: str,
        user_payload: dict[str, JsonValue],
    ) -> dict[str, JsonValue]:
        effort = self.config.effort.lower()
        payload: dict[str, JsonValue] = {
            "model": self.config.resolved_model(),
            "input": [
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": system_prompt}],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": canonical_json(user_payload, indent=2),
                        }
                    ],
                },
            ],
            "max_output_tokens": self.config.max_tokens,
        }
        if self.config.temperature is not None:
            payload["temperature"] = self.config.temperature
        if effort != "none":
            payload["reasoning"] = {"effort": effort}
        return self.transport.post_json(
            url=self.config.resolved_endpoint(),
            headers={
                "content-type": "application/json",
                "authorization": f"Bearer {self.config.resolved_api_key()}",
            },
            payload=payload,
            timeout_seconds=self.config.timeout_seconds,
        )


@dataclass(frozen=True, slots=True)
class FrontierLLMPieceProgramProvider:
    client: FrontierLLMClient = field(default_factory=FrontierLLMClient)

    def compile_piece(self, request: CompileRequest) -> CompileResponse:
        response = self.client.generate_json(
            user_payload={
                "task": "compile_piece_program",
                "description": request.description,
                "dsl_reference": request.dsl_reference,
                "metadata": request.metadata,
                "output_contract": {
                    "candidate_program": "DSL object for the piece",
                    "explanation": "brief rationale string",
                    "metadata": "optional JSON object",
                },
            }
        )
        parsed = CompileResponse.from_dict(response.data)
        return CompileResponse(
            candidate_program=parsed.candidate_program,
            explanation=parsed.explanation,
            metadata={**dict(parsed.metadata), **response.metadata},
        )

    def repair_piece(self, request: RepairRequest) -> RepairResponse:
        response = self.client.generate_json(
            user_payload={
                "task": "repair_piece_program",
                "repair_request": request.to_dict(),
                "output_contract": {
                    "patched_program": "complete patched DSL object",
                    "explanation": "brief rationale string",
                    "generated_tests": "optional list of replay/regression cases",
                    "metadata": "optional JSON object",
                },
            }
        )
        parsed = RepairResponse.from_dict(response.data)
        return RepairResponse(
            patched_program=parsed.patched_program,
            explanation=parsed.explanation,
            generated_tests=parsed.generated_tests,
            metadata={**dict(parsed.metadata), **response.metadata},
        )

    def repair_piece_candidates(
        self,
        request: RepairRequest,
        candidate_count: int,
    ) -> tuple[RepairResponse, ...]:
        response = self.client.generate_json(
            user_payload={
                "task": "repair_piece_program_candidates",
                "candidate_count": candidate_count,
                "repair_request": request.to_dict(),
                "output_contract": {
                    "candidates": "list of complete patched DSL repair responses",
                    "metadata": "optional JSON object",
                },
            }
        )
        payload = response.data
        if isinstance(payload, dict) and "candidates" in payload:
            candidate_payloads = payload["candidates"]
        elif isinstance(payload, list):
            candidate_payloads = payload
        else:
            candidate_payloads = [payload]
        candidates: list[RepairResponse] = []
        for item in candidate_payloads:
            if not isinstance(item, dict):
                continue
            parsed = RepairResponse.from_dict(item)
            candidates.append(
                RepairResponse(
                    patched_program=parsed.patched_program,
                    explanation=parsed.explanation,
                    generated_tests=parsed.generated_tests,
                    metadata={**dict(parsed.metadata), **response.metadata},
                )
            )
        return tuple(candidates[:candidate_count])


def _anthropic_text(response: dict[str, JsonValue]) -> str:
    chunks: list[str] = []
    for block in response.get("content", []):
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            chunks.append(str(block.get("text", "")))
    if not chunks:
        raise LLMProviderError("Anthropic response did not contain a text block")
    return "\n".join(chunks)


def _openai_text(response: dict[str, JsonValue]) -> str:
    output_text = response.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    chunks: list[str] = []
    for item in response.get("output", []):
        if not isinstance(item, dict):
            continue
        for block in item.get("content", []):
            if not isinstance(block, dict):
                continue
            if block.get("type") in {"output_text", "text"}:
                text = block.get("text")
                if isinstance(text, str):
                    chunks.append(text)
    if not chunks:
        raise LLMProviderError("OpenAI response did not contain output text")
    return "\n".join(chunks)


def _extract_json_object(text: str) -> dict[str, JsonValue]:
    stripped = text.strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        payload = _raw_decode_object(stripped)
    if not isinstance(payload, dict):
        raise LLMProviderError("LLM output must be a JSON object")
    return payload


def _raw_decode_object(text: str) -> JsonValue:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        return payload
    raise LLMProviderError("LLM output did not contain a JSON object")
