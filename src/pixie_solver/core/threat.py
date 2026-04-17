from __future__ import annotations

import re
from dataclasses import dataclass, field

from pixie_solver.core.query import QueryFact
from pixie_solver.utils.serialization import JsonValue
from pixie_solver.utils.squares import normalize_square

_IDENTIFIER_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


def _validate_identifier(label: str, value: str | None) -> None:
    if value is None:
        return
    if not value:
        raise ValueError(f"{label} must not be empty")
    if not _IDENTIFIER_PATTERN.fullmatch(value):
        raise ValueError(
            f"{label} must match {_IDENTIFIER_PATTERN.pattern!r}, got {value!r}"
        )


@dataclass(frozen=True, slots=True)
class ThreatMark:
    threat_kind: str
    target_square: str
    source_entity_id: str | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)
    params: dict[str, JsonValue] = field(default_factory=dict)
    metadata: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_identifier("threat kind", self.threat_kind)
        _validate_identifier("source entity id", self.source_entity_id)
        object.__setattr__(self, "target_square", normalize_square(self.target_square))
        if self.target_square is None:
            raise ValueError("target_square must be a valid square")
        object.__setattr__(self, "tags", tuple(str(tag) for tag in self.tags))
        object.__setattr__(self, "params", dict(self.params))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def stable_id(self) -> str:
        return stable_threat_id(self)

    def to_query_fact(self) -> QueryFact:
        return QueryFact(
            query_kind=self.threat_kind,
            subject_ref=self.source_entity_id,
            target_ref=self.target_square,
            tags=self.tags,
            params=self.params,
            metadata=self.metadata,
        )

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "threat_kind": self.threat_kind,
            "target_square": self.target_square,
            "source_entity_id": self.source_entity_id,
            "tags": list(self.tags),
            "params": dict(self.params),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, JsonValue]) -> "ThreatMark":
        return cls(
            threat_kind=str(data["threat_kind"]),
            target_square=str(data["target_square"]),
            source_entity_id=(
                str(data["source_entity_id"])
                if data.get("source_entity_id") is not None
                else None
            ),
            tags=tuple(str(tag) for tag in data.get("tags", [])),
            params=dict(data.get("params", {})),
            metadata=dict(data.get("metadata", {})),
        )

    @classmethod
    def from_query_fact(cls, query_fact: QueryFact) -> "ThreatMark":
        if query_fact.target_ref is None:
            raise ValueError("query_fact.target_ref must not be None for ThreatMark")
        return cls(
            threat_kind=query_fact.query_kind,
            target_square=query_fact.target_ref,
            source_entity_id=query_fact.subject_ref,
            tags=query_fact.tags,
            params=query_fact.params,
            metadata=query_fact.metadata,
        )


def stable_threat_id(threat: ThreatMark) -> str:
    from pixie_solver.core.hash import stable_digest

    return stable_digest(threat.to_dict())
