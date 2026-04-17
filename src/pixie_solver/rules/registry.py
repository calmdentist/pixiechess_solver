from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pixie_solver.core import PieceClass
from pixie_solver.core.hash import stable_digest
from pixie_solver.dsl.canonicalize import canonicalize_piece_program
from pixie_solver.dsl.compiler import compile_piece_file
from pixie_solver.utils.serialization import JsonValue, canonical_json


REGISTRY_FORMAT_VERSION = 1


@dataclass(frozen=True, slots=True)
class PieceRegistryRecord:
    piece_id: str
    version: int
    status: str
    description: str
    dsl_path: str
    dsl_digest: str
    source: str = "repair"
    parent_digest: str | None = None
    verified_cases: int = 0
    repair_attempts: int = 1
    metadata: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "piece_id": self.piece_id,
            "version": self.version,
            "status": self.status,
            "description": self.description,
            "dsl_path": self.dsl_path,
            "dsl_digest": self.dsl_digest,
            "source": self.source,
            "parent_digest": self.parent_digest,
            "verified_cases": self.verified_cases,
            "repair_attempts": self.repair_attempts,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, JsonValue]) -> "PieceRegistryRecord":
        return cls(
            piece_id=str(data["piece_id"]),
            version=int(data["version"]),
            status=str(data["status"]),
            description=str(data.get("description", "")),
            dsl_path=str(data["dsl_path"]),
            dsl_digest=str(data["dsl_digest"]),
            source=str(data.get("source", "repair")),
            parent_digest=(
                str(data["parent_digest"])
                if data.get("parent_digest") is not None
                else None
            ),
            verified_cases=int(data.get("verified_cases", 0)),
            repair_attempts=int(data.get("repair_attempts", 1)),
            metadata=dict(data.get("metadata", {})),
        )


def load_piece_registry(path: str | Path) -> list[PieceRegistryRecord]:
    registry_path = Path(path)
    if not registry_path.exists():
        return []
    with registry_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        records_payload = payload
    elif isinstance(payload, dict):
        records_payload = payload.get("records", [])
    else:
        raise ValueError("piece registry must be a JSON object or list")
    return [PieceRegistryRecord.from_dict(dict(item)) for item in records_payload]


def write_piece_registry(
    path: str | Path,
    records: list[PieceRegistryRecord],
) -> None:
    registry_path = Path(path)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": REGISTRY_FORMAT_VERSION,
        "records": [record.to_dict() for record in sorted(records, key=_record_sort_key)],
    }
    registry_path.write_text(canonical_json(payload, indent=2), encoding="utf-8")


def append_verified_piece_version(
    *,
    registry_path: str | Path,
    out_dir: str | Path,
    program: dict[str, Any],
    description: str,
    source: str = "repair",
    parent_digest: str | None = None,
    verified_cases: int = 0,
    repair_attempts: int = 1,
    metadata: dict[str, JsonValue] | None = None,
) -> PieceRegistryRecord:
    canonical_program = canonicalize_piece_program(program)
    piece_id = str(canonical_program["piece_id"])
    records = load_piece_registry(registry_path)
    version = _next_version(records, piece_id)
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dsl_path = output_dir / f"{piece_id}_v{version:03d}.json"
    dsl_path.write_text(canonical_json(canonical_program, indent=2), encoding="utf-8")

    record = PieceRegistryRecord(
        piece_id=piece_id,
        version=version,
        status="verified",
        description=description,
        dsl_path=str(dsl_path),
        dsl_digest=stable_digest(canonical_program),
        source=source,
        parent_digest=parent_digest,
        verified_cases=verified_cases,
        repair_attempts=repair_attempts,
        metadata=metadata or {},
    )
    records.append(record)
    write_piece_registry(registry_path, records)
    return record


def load_verified_piece_records(path: str | Path) -> list[PieceRegistryRecord]:
    records = [record for record in load_piece_registry(path) if record.status == "verified"]
    latest_by_piece: dict[str, PieceRegistryRecord] = {}
    for record in records:
        current = latest_by_piece.get(record.piece_id)
        if current is None or record.version > current.version:
            latest_by_piece[record.piece_id] = record
    return [
        latest_by_piece[piece_id]
        for piece_id in sorted(latest_by_piece)
    ]


def load_verified_piece_classes(path: str | Path) -> list[PieceClass]:
    records = load_verified_piece_records(path)
    return load_piece_classes_for_records(path, records)


def load_piece_classes_for_records(
    registry_path: str | Path,
    records: list[PieceRegistryRecord],
) -> list[PieceClass]:
    return [compile_piece_file(_resolve_record_path(registry_path, record)) for record in records]


def registry_piece_digest_metadata(
    records: list[PieceRegistryRecord],
) -> dict[str, JsonValue]:
    return {
        record.piece_id: {
            "version": record.version,
            "dsl_digest": record.dsl_digest,
            "source": record.source,
        }
        for record in records
    }


def registry_piece_record_metadata(
    records: list[PieceRegistryRecord],
) -> dict[str, JsonValue]:
    return {
        record.piece_id: {
            "version": record.version,
            "source": record.source,
            "metadata": dict(record.metadata),
        }
        for record in records
    }


def registry_piece_training_metadata(
    records: list[PieceRegistryRecord],
) -> dict[str, JsonValue]:
    training_fields = (
        "family_id",
        "split",
        "novelty_tier",
        "admission_cycle",
        "task_id",
    )
    return {
        record.piece_id: {
            "version": record.version,
            "source": record.source,
            **{
                field: record.metadata[field]
                for field in training_fields
                if field in record.metadata
            },
        }
        for record in records
    }


def _resolve_record_path(
    registry_path: str | Path,
    record: PieceRegistryRecord,
) -> Path:
    dsl_path = Path(record.dsl_path)
    if dsl_path.is_absolute():
        return dsl_path
    registry_parent = Path(registry_path).parent
    candidate = registry_parent / dsl_path
    if candidate.exists():
        return candidate
    return dsl_path


def _next_version(records: list[PieceRegistryRecord], piece_id: str) -> int:
    versions = [record.version for record in records if record.piece_id == piece_id]
    return max(versions, default=0) + 1


def _record_sort_key(record: PieceRegistryRecord) -> tuple[str, int, str]:
    return (record.piece_id, record.version, record.dsl_digest)
