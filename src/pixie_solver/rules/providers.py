from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from pixie_solver.utils.serialization import JsonValue


@dataclass(frozen=True, slots=True)
class CompileRequest:
    description: str
    dsl_reference: dict[str, JsonValue] = field(default_factory=dict)
    metadata: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "dsl_reference", dict(self.dsl_reference))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "description": self.description,
            "dsl_reference": dict(self.dsl_reference),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class CompileResponse:
    candidate_program: dict[str, JsonValue]
    explanation: str = ""
    metadata: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_program", dict(self.candidate_program))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "candidate_program": dict(self.candidate_program),
            "explanation": self.explanation,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, JsonValue]) -> "CompileResponse":
        if "candidate_program" in data:
            candidate_program = dict(data["candidate_program"])
            explanation = str(data.get("explanation", ""))
            metadata = dict(data.get("metadata", {}))
        else:
            candidate_program = dict(data)
            explanation = ""
            metadata = {"response_shape": "raw_program"}
        return cls(
            candidate_program=candidate_program,
            explanation=explanation,
            metadata=metadata,
        )


@dataclass(frozen=True, slots=True)
class RepairRequest:
    description: str
    current_program: dict[str, JsonValue]
    before_state: dict[str, JsonValue]
    move: dict[str, JsonValue]
    predicted_state: dict[str, JsonValue]
    observed_state: dict[str, JsonValue]
    diff: dict[str, JsonValue]
    predicted_error: str | None = None
    implicated_piece_ids: tuple[str, ...] = ()
    implicated_piece_class_ids: tuple[str, ...] = ()
    dsl_reference: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "current_program", dict(self.current_program))
        object.__setattr__(self, "before_state", dict(self.before_state))
        object.__setattr__(self, "move", dict(self.move))
        object.__setattr__(self, "predicted_state", dict(self.predicted_state))
        object.__setattr__(self, "observed_state", dict(self.observed_state))
        object.__setattr__(self, "diff", dict(self.diff))
        object.__setattr__(
            self,
            "implicated_piece_ids",
            tuple(str(piece_id) for piece_id in self.implicated_piece_ids),
        )
        object.__setattr__(
            self,
            "implicated_piece_class_ids",
            tuple(str(class_id) for class_id in self.implicated_piece_class_ids),
        )
        object.__setattr__(self, "dsl_reference", dict(self.dsl_reference))

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "description": self.description,
            "current_program": dict(self.current_program),
            "before_state": dict(self.before_state),
            "move": dict(self.move),
            "predicted_state": dict(self.predicted_state),
            "observed_state": dict(self.observed_state),
            "diff": dict(self.diff),
            "predicted_error": self.predicted_error,
            "implicated_piece_ids": list(self.implicated_piece_ids),
            "implicated_piece_class_ids": list(self.implicated_piece_class_ids),
            "dsl_reference": dict(self.dsl_reference),
        }


@dataclass(frozen=True, slots=True)
class RepairResponse:
    patched_program: dict[str, JsonValue]
    explanation: str = ""
    generated_tests: tuple[dict[str, JsonValue], ...] = ()
    metadata: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "patched_program", dict(self.patched_program))
        object.__setattr__(
            self,
            "generated_tests",
            tuple(dict(item) for item in self.generated_tests),
        )
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "patched_program": dict(self.patched_program),
            "explanation": self.explanation,
            "generated_tests": [dict(item) for item in self.generated_tests],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, JsonValue]) -> "RepairResponse":
        if "patched_program" in data:
            patched_program = dict(data["patched_program"])
            explanation = str(data.get("explanation", ""))
            generated_tests = tuple(
                dict(item) for item in data.get("generated_tests", [])
            )
            metadata = dict(data.get("metadata", {}))
        else:
            patched_program = dict(data)
            explanation = ""
            generated_tests = ()
            metadata = {"response_shape": "raw_program"}
        return cls(
            patched_program=patched_program,
            explanation=explanation,
            generated_tests=generated_tests,
            metadata=metadata,
        )


class PieceProgramCompileProvider(Protocol):
    def compile_piece(self, request: CompileRequest) -> CompileResponse:
        ...


class PieceProgramRepairProvider(Protocol):
    def repair_piece(self, request: RepairRequest) -> RepairResponse:
        ...


@dataclass(frozen=True, slots=True)
class StaticCompileProvider:
    candidate_program: dict[str, JsonValue]
    explanation: str = "static compile response"

    def compile_piece(self, request: CompileRequest) -> CompileResponse:
        return CompileResponse(
            candidate_program=self.candidate_program,
            explanation=self.explanation,
            metadata={"provider": "static"},
        )


@dataclass(frozen=True, slots=True)
class JsonFileCompileProvider:
    path: Path

    def compile_piece(self, request: CompileRequest) -> CompileResponse:
        with self.path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError("compile provider response must be a JSON object")
        response = CompileResponse.from_dict(payload)
        return CompileResponse(
            candidate_program=response.candidate_program,
            explanation=response.explanation,
            metadata={
                **dict(response.metadata),
                "provider": "json_file",
                "path": str(self.path),
            },
        )


@dataclass(frozen=True, slots=True)
class StaticRepairProvider:
    patched_program: dict[str, JsonValue]
    explanation: str = "static repair response"

    def repair_piece(self, request: RepairRequest) -> RepairResponse:
        return RepairResponse(
            patched_program=self.patched_program,
            explanation=self.explanation,
            metadata={"provider": "static"},
        )

    def repair_piece_candidates(
        self,
        request: RepairRequest,
        candidate_count: int,
    ) -> tuple[RepairResponse, ...]:
        del candidate_count
        return (self.repair_piece(request),)


@dataclass(frozen=True, slots=True)
class JsonFileRepairProvider:
    path: Path

    def _load_repair_responses(self) -> tuple[RepairResponse, ...]:
        with self.path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payloads: list[dict[str, JsonValue]]
        if isinstance(payload, dict):
            payloads = [payload]
        elif isinstance(payload, list) and all(isinstance(item, dict) for item in payload):
            payloads = [dict(item) for item in payload]
        else:
            raise ValueError("repair provider response must be a JSON object or list of objects")
        return tuple(RepairResponse.from_dict(item) for item in payloads)

    def repair_piece(self, request: RepairRequest) -> RepairResponse:
        responses = self._load_repair_responses()
        if not responses:
            raise ValueError("repair provider response list was empty")
        response = responses[0]
        return RepairResponse(
            patched_program=response.patched_program,
            explanation=response.explanation,
            generated_tests=response.generated_tests,
            metadata={
                **dict(response.metadata),
                "provider": "json_file",
                "path": str(self.path),
            },
        )

    def repair_piece_candidates(
        self,
        request: RepairRequest,
        candidate_count: int,
    ) -> tuple[RepairResponse, ...]:
        del request
        responses = self._load_repair_responses()
        if not responses:
            raise ValueError("repair provider response list was empty")
        return tuple(
            RepairResponse(
                patched_program=response.patched_program,
                explanation=response.explanation,
                generated_tests=response.generated_tests,
                metadata={
                    **dict(response.metadata),
                    "provider": "json_file",
                    "path": str(self.path),
                },
            )
            for response in responses[:candidate_count]
        )
