from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from pixie_solver.llm.frontier import FrontierLLMClient
from pixie_solver.strategy.canonicalize import canonicalize_strategy_hypothesis
from pixie_solver.strategy.schema import StrategyHypothesis
from pixie_solver.utils.serialization import JsonValue, to_primitive

_STRATEGY_SYSTEM_PROMPT = """You are the PixieChess strategy planner.
You are given the current world summary, board state, and optional prior strategy.

Return exactly one JSON object and no markdown.
The response must contain a single strategy hypothesis that is concise, concrete, and executable as high-level guidance.
Do not invent unsupported fields.
Prefer a small, focused strategy over a verbose one.
"""


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
