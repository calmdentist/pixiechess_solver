from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Mapping

from pixie_solver.utils.serialization import JsonValue

_IDENTIFIER_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


def _validate_identifier(label: str, value: str) -> None:
    if not value:
        raise ValueError(f"{label} must not be empty")
    if not _IDENTIFIER_PATTERN.fullmatch(value):
        raise ValueError(
            f"{label} must match {_IDENTIFIER_PATTERN.pattern!r}, got {value!r}"
        )


@dataclass(frozen=True, slots=True)
class ObjectiveSpec:
    objective_id: str
    win_condition: str
    legality_mode: str
    terminal_timing: str
    metadata: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_identifier("objective id", self.objective_id)
        _validate_identifier("win condition", self.win_condition)
        _validate_identifier("legality mode", self.legality_mode)
        _validate_identifier("terminal timing", self.terminal_timing)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "objective_id": self.objective_id,
            "win_condition": self.win_condition,
            "legality_mode": self.legality_mode,
            "terminal_timing": self.terminal_timing,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, JsonValue]) -> "ObjectiveSpec":
        return cls(
            objective_id=str(data["objective_id"]),
            win_condition=str(data["win_condition"]),
            legality_mode=str(data["legality_mode"]),
            terminal_timing=str(data["terminal_timing"]),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True, slots=True)
class ExecutableWorldModelSpec:
    world_model_id: str
    world_schema: dict[str, JsonValue]
    entity_programs: dict[str, Mapping[str, Any]]
    objective: ObjectiveSpec
    global_rule_programs: dict[str, Mapping[str, Any]] = field(default_factory=dict)
    query_programs: dict[str, Mapping[str, Any]] = field(default_factory=dict)
    constants: dict[str, JsonValue] = field(default_factory=dict)
    metadata: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_identifier("world model id", self.world_model_id)
        object.__setattr__(self, "world_schema", deepcopy(dict(self.world_schema)))
        object.__setattr__(
            self,
            "entity_programs",
            {str(key): deepcopy(dict(value)) for key, value in self.entity_programs.items()},
        )
        object.__setattr__(
            self,
            "global_rule_programs",
            {
                str(key): deepcopy(dict(value))
                for key, value in self.global_rule_programs.items()
            },
        )
        object.__setattr__(
            self,
            "query_programs",
            {str(key): deepcopy(dict(value)) for key, value in self.query_programs.items()},
        )
        object.__setattr__(self, "constants", deepcopy(dict(self.constants)))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "world_model_id": self.world_model_id,
            "world_schema": deepcopy(dict(self.world_schema)),
            "entity_programs": {
                key: deepcopy(dict(value))
                for key, value in sorted(self.entity_programs.items())
            },
            "objective": self.objective.to_dict(),
            "global_rule_programs": {
                key: deepcopy(dict(value))
                for key, value in sorted(self.global_rule_programs.items())
            },
            "query_programs": {
                key: deepcopy(dict(value))
                for key, value in sorted(self.query_programs.items())
            },
            "constants": deepcopy(dict(self.constants)),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, JsonValue]) -> "ExecutableWorldModelSpec":
        return cls(
            world_model_id=str(data["world_model_id"]),
            world_schema=dict(data["world_schema"]),
            entity_programs={
                str(key): dict(value)
                for key, value in dict(data.get("entity_programs", {})).items()
            },
            objective=ObjectiveSpec.from_dict(dict(data["objective"])),
            global_rule_programs={
                str(key): dict(value)
                for key, value in dict(data.get("global_rule_programs", {})).items()
            },
            query_programs={
                str(key): dict(value)
                for key, value in dict(data.get("query_programs", {})).items()
            },
            constants=dict(data.get("constants", {})),
            metadata=dict(data.get("metadata", {})),
        )
