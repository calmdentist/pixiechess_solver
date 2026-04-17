from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from pixie_solver.core.piece import (
    BasePieceType,
    Condition,
    Effect,
    Hook,
    Modifier,
    PieceClass,
    StateField,
)
from pixie_solver.dsl.canonicalize import canonicalize_piece_program
from pixie_solver.dsl.parser import load_piece_program
from pixie_solver.dsl.schema import SCHEMA_VERSION
from pixie_solver.program.lower_legacy_dsl import lower_legacy_piece_program


@dataclass(frozen=True, slots=True)
class CompiledPieceArtifacts:
    piece_class: PieceClass
    program_ir: dict[str, Any]


def compile_piece_program(program: Mapping[str, Any]) -> PieceClass:
    return compile_piece_artifacts(program).piece_class


def compile_piece_program_ir(program: Mapping[str, Any]) -> dict[str, Any]:
    return compile_piece_artifacts(program).program_ir


def compile_piece_artifacts(program: Mapping[str, Any]) -> CompiledPieceArtifacts:
    canonical_program = canonicalize_piece_program(program)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        **dict(canonical_program.get("metadata", {})),
    }
    piece_class = PieceClass(
        class_id=str(canonical_program["piece_id"]),
        name=str(canonical_program["name"]),
        base_piece_type=BasePieceType(str(canonical_program["base_piece_type"])),
        movement_modifiers=tuple(
            Modifier(op=str(item["op"]), args=dict(item.get("args", {})))
            for item in canonical_program.get("movement_modifiers", [])
        ),
        capture_modifiers=tuple(
            Modifier(op=str(item["op"]), args=dict(item.get("args", {})))
            for item in canonical_program.get("capture_modifiers", [])
        ),
        hooks=tuple(
            Hook(
                event=str(item["event"]),
                conditions=tuple(
                    Condition(op=str(condition["op"]), args=dict(condition.get("args", {})))
                    for condition in item.get("conditions", [])
                ),
                effects=tuple(
                    Effect(op=str(effect["op"]), args=dict(effect.get("args", {})))
                    for effect in item.get("effects", [])
                ),
                priority=int(item.get("priority", 0)),
                metadata=dict(item.get("metadata", {})),
            )
            for item in canonical_program.get("hooks", [])
        ),
        instance_state_schema=tuple(
            StateField(
                name=str(field_spec["name"]),
                field_type=str(field_spec["type"]),
                default=field_spec.get("default"),
                description=str(field_spec.get("description", "")),
            )
            for field_spec in canonical_program.get("instance_state_schema", [])
        ),
        metadata=metadata,
    )
    return CompiledPieceArtifacts(
        piece_class=piece_class,
        program_ir=lower_legacy_piece_program(canonical_program),
    )


def compile_piece_file(path: str | Path) -> PieceClass:
    return compile_piece_program(load_piece_program(path))


def compile_piece_file_ir(path: str | Path) -> dict[str, Any]:
    return compile_piece_program_ir(load_piece_program(path))
