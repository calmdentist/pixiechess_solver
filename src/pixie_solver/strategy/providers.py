from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from pixie_solver.core.hash import stable_digest
from pixie_solver.llm.frontier import FrontierLLMClient
from pixie_solver.strategy.canonicalize import (
    canonicalize_strategy_hypothesis,
    strategy_digest as compute_strategy_digest,
)
from pixie_solver.strategy.schema import StrategyHypothesis
from pixie_solver.utils.serialization import JsonValue, to_primitive

_STRATEGY_SYSTEM_PROMPT = """You are the PixieChess strategy planner.
You are given the current world summary, board state, and optional prior strategy.

Return exactly one JSON object and no markdown.
The response must contain a single strategy hypothesis that is concise, concrete, and executable as high-level guidance.
Do not invent unsupported fields.
Prefer a small, focused strategy over a verbose one.
"""

NO_STRATEGY_CACHE_SCOPE = "none"
WORLD_PHASE_STRATEGY_CACHE_SCOPE = "world_phase"
FULL_REQUEST_STRATEGY_CACHE_SCOPE = "full_request"
SUPPORTED_STRATEGY_CACHE_SCOPES = (
    NO_STRATEGY_CACHE_SCOPE,
    WORLD_PHASE_STRATEGY_CACHE_SCOPE,
    FULL_REQUEST_STRATEGY_CACHE_SCOPE,
)


@dataclass(frozen=True, slots=True)
class StrategyRequest:
    state: dict[str, JsonValue]
    world_summary: dict[str, JsonValue]
    phase: str = "game_start"
    prior_strategy: dict[str, JsonValue] | None = None
    metadata: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "state", dict(self.state))
        object.__setattr__(self, "world_summary", dict(self.world_summary))
        object.__setattr__(
            self,
            "prior_strategy",
            None if self.prior_strategy is None else dict(self.prior_strategy),
        )
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "state": dict(self.state),
            "world_summary": dict(self.world_summary),
            "phase": self.phase,
            "prior_strategy": (
                None if self.prior_strategy is None else dict(self.prior_strategy)
            ),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class StrategyResponse:
    strategy: dict[str, JsonValue]
    explanation: str = ""
    metadata: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        canonical_strategy = canonicalize_strategy_hypothesis(self.strategy)
        object.__setattr__(self, "strategy", canonical_strategy)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "strategy": dict(self.strategy),
            "explanation": self.explanation,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, JsonValue]) -> "StrategyResponse":
        if "strategy" in data:
            strategy = dict(data["strategy"])
            explanation = str(data.get("explanation", ""))
            metadata = dict(data.get("metadata", {}))
        else:
            strategy = dict(data)
            explanation = ""
            metadata = {"response_shape": "raw_strategy"}
        return cls(
            strategy=strategy,
            explanation=explanation,
            metadata=metadata,
        )


class StrategyProvider(Protocol):
    def propose_strategy(self, request: StrategyRequest) -> StrategyResponse:
        ...


@dataclass(slots=True)
class CachedStrategyProvider:
    provider: StrategyProvider
    scope: str = WORLD_PHASE_STRATEGY_CACHE_SCOPE
    _cache: dict[str, dict[str, JsonValue]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.scope not in SUPPORTED_STRATEGY_CACHE_SCOPES:
            supported = ", ".join(SUPPORTED_STRATEGY_CACHE_SCOPES)
            raise ValueError(
                f"strategy cache scope must be one of {supported}, got {self.scope!r}"
            )

    def propose_strategy(self, request: StrategyRequest) -> StrategyResponse:
        if self.scope == NO_STRATEGY_CACHE_SCOPE:
            response = self.provider.propose_strategy(request)
            return _with_cache_metadata(
                response,
                scope=self.scope,
                cache_hit=False,
            )
        cache_key = _strategy_cache_key(request, scope=self.scope)
        cached_payload = self._cache.get(cache_key)
        if cached_payload is not None:
            return _with_cache_metadata(
                StrategyResponse.from_dict(cached_payload),
                scope=self.scope,
                cache_hit=True,
            )
        response = self.provider.propose_strategy(request)
        self._cache[cache_key] = response.to_dict()
        return _with_cache_metadata(
            response,
            scope=self.scope,
            cache_hit=False,
        )


@dataclass(frozen=True, slots=True)
class StaticStrategyProvider:
    strategy: StrategyHypothesis | dict[str, JsonValue]
    explanation: str = "static strategy response"

    def propose_strategy(self, request: StrategyRequest) -> StrategyResponse:
        return StrategyResponse(
            strategy=dict(to_primitive(self.strategy)),
            explanation=self.explanation,
            metadata={"provider": "static", "phase": request.phase},
        )


@dataclass(frozen=True, slots=True)
class JsonFileStrategyProvider:
    path: Path

    def propose_strategy(self, request: StrategyRequest) -> StrategyResponse:
        with self.path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError("strategy provider response must be a JSON object")
        response = StrategyResponse.from_dict(payload)
        return StrategyResponse(
            strategy=response.strategy,
            explanation=response.explanation,
            metadata={
                **dict(response.metadata),
                "provider": "json_file",
                "path": str(self.path),
                "phase": request.phase,
            },
        )


@dataclass(frozen=True, slots=True)
class FrontierLLMStrategyProvider:
    client: FrontierLLMClient = field(default_factory=FrontierLLMClient)

    def propose_strategy(self, request: StrategyRequest) -> StrategyResponse:
        response = self.client.generate_json(
            user_payload={
                "task": "generate_strategy_hypothesis",
                "strategy_request": request.to_dict(),
                "output_contract": {
                    "strategy": {
                        "strategy_id": "lowercase identifier",
                        "summary": "brief strategy summary",
                        "confidence": "float in [0,1]",
                        "scope": "game_start or refresh scope",
                        "subgoals": "optional list of strings",
                        "action_biases": "optional list of strings",
                        "avoid_biases": "optional list of strings",
                        "success_predicates": "optional list of strings",
                        "failure_triggers": "optional list of strings",
                        "metadata": "optional JSON object",
                    },
                    "explanation": "brief rationale string",
                    "metadata": "optional JSON object",
                },
            },
            system_prompt=_STRATEGY_SYSTEM_PROMPT,
        )
        parsed = StrategyResponse.from_dict(response.data)
        return StrategyResponse(
            strategy=parsed.strategy,
            explanation=parsed.explanation,
            metadata={**dict(parsed.metadata), **response.metadata},
        )


def _strategy_cache_key(request: StrategyRequest, *, scope: str) -> str:
    if scope == WORLD_PHASE_STRATEGY_CACHE_SCOPE:
        return stable_digest(
            {
                "world_summary": dict(request.world_summary),
                "phase": request.phase,
                "prior_strategy_digest": (
                    None
                    if request.prior_strategy is None
                    else compute_strategy_digest(request.prior_strategy)
                ),
            }
        )
    if scope == FULL_REQUEST_STRATEGY_CACHE_SCOPE:
        return stable_digest(request.to_dict())
    raise ValueError(f"unsupported strategy cache scope: {scope}")


def _with_cache_metadata(
    response: StrategyResponse,
    *,
    scope: str,
    cache_hit: bool,
) -> StrategyResponse:
    return StrategyResponse(
        strategy=response.strategy,
        explanation=response.explanation,
        metadata={
            **dict(response.metadata),
            "cache_scope": scope,
            "cache_hit": cache_hit,
        },
    )
