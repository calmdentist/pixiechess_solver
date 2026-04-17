from __future__ import annotations

import re
from dataclasses import dataclass, field

from pixie_solver.utils.serialization import JsonValue

_IDENTIFIER_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


def _validate_identifier(label: str, value: str) -> None:
    if not value:
        raise ValueError(f"{label} must not be empty")
    if not _IDENTIFIER_PATTERN.fullmatch(value):
        raise ValueError(
            f"{label} must match {_IDENTIFIER_PATTERN.pattern!r}, got {value!r}"
        )


def _validate_ref(label: str, value: str | None) -> None:
    if value is None:
        return
    if not str(value):
        raise ValueError(f"{label} must not be empty")


@dataclass(frozen=True, slots=True)
class QueryFact:
    query_kind: str
    subject_ref: str | None = None
    target_ref: str | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)
    params: dict[str, JsonValue] = field(default_factory=dict)
    metadata: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_identifier("query kind", self.query_kind)
        _validate_ref("subject ref", self.subject_ref)
        _validate_ref("target ref", self.target_ref)
        object.__setattr__(
            self,
            "subject_ref",
            str(self.subject_ref) if self.subject_ref is not None else None,
        )
        object.__setattr__(
            self,
            "target_ref",
            str(self.target_ref) if self.target_ref is not None else None,
        )
        object.__setattr__(self, "tags", tuple(str(tag) for tag in self.tags))
        object.__setattr__(self, "params", dict(self.params))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def stable_id(self) -> str:
        return stable_query_fact_id(self)

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "query_kind": self.query_kind,
            "subject_ref": self.subject_ref,
            "target_ref": self.target_ref,
            "tags": list(self.tags),
            "params": dict(self.params),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, JsonValue]) -> "QueryFact":
        return cls(
            query_kind=str(data["query_kind"]),
            subject_ref=(
                str(data["subject_ref"]) if data.get("subject_ref") is not None else None
            ),
            target_ref=(
                str(data["target_ref"]) if data.get("target_ref") is not None else None
            ),
            tags=tuple(str(tag) for tag in data.get("tags", [])),
            params=dict(data.get("params", {})),
            metadata=dict(data.get("metadata", {})),
        )


def stable_query_fact_id(query_fact: QueryFact) -> str:
    from pixie_solver.core.hash import stable_digest

    return stable_digest(query_fact.to_dict())
