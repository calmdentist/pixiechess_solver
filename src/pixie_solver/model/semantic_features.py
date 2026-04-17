from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pixie_solver.core.piece import Color, PieceClass
from pixie_solver.core.state import GameState
from pixie_solver.model._features import normalize_scalar
from pixie_solver.simulator.query import (
    CAPTURE_CONTROL_QUERY_KIND,
    enumerate_query_facts,
    is_king_capturable,
)

SEMANTIC_PROBE_FEATURES = 6


@dataclass(frozen=True, slots=True)
class SemanticProbeSpec:
    probe_id: str
    namespace: str
    label: str
    numeric_features: tuple[float, ...]


def build_semantic_probe_specs(
    state: GameState,
    *,
    class_ids: Sequence[str],
    piece_classes: Mapping[str, PieceClass],
    program_registry: Mapping[str, Mapping[str, Any]],
) -> tuple[SemanticProbeSpec, ...]:
    white_facts = enumerate_query_facts(
        state,
        by_color=Color.WHITE,
        query_kind=CAPTURE_CONTROL_QUERY_KIND,
        program_registry=program_registry,
    )
    black_facts = enumerate_query_facts(
        state,
        by_color=Color.BLACK,
        query_kind=CAPTURE_CONTROL_QUERY_KIND,
        program_registry=program_registry,
    )
    specs = [
        _color_control_probe(
            state,
            facts=white_facts,
            attacking_color=Color.WHITE,
            defending_color=Color.BLACK,
            program_registry=program_registry,
        ),
        _color_control_probe(
            state,
            facts=black_facts,
            attacking_color=Color.BLACK,
            defending_color=Color.WHITE,
            program_registry=program_registry,
        ),
    ]
    active_pieces = tuple(state.active_pieces())
    for class_id in class_ids:
        piece_class = piece_classes[class_id]
        program = program_registry[class_id]
        class_pieces = [piece for piece in active_pieces if piece.piece_class_id == class_id]
        nondefault_state_fields = sum(
            _nondefault_state_fields(piece.state, piece_class)
            for piece in class_pieces
        )
        specs.append(
            SemanticProbeSpec(
                probe_id=f"class_summary:{class_id}",
                namespace="class_program_summary",
                label=class_id,
                numeric_features=(
                    normalize_scalar(len(class_pieces), scale=8.0),
                    normalize_scalar(len(program.get("state_schema", ())), scale=8.0),
                    normalize_scalar(len(program.get("action_blocks", ())), scale=8.0),
                    normalize_scalar(len(program.get("query_blocks", ())), scale=8.0),
                    normalize_scalar(len(program.get("reaction_blocks", ())), scale=8.0),
                    normalize_scalar(nondefault_state_fields, scale=8.0),
                ),
            )
        )
    return tuple(specs)


def _color_control_probe(
    state: GameState,
    *,
    facts,
    attacking_color: Color,
    defending_color: Color,
    program_registry: Mapping[str, Mapping[str, Any]],
) -> SemanticProbeSpec:
    unique_targets = {fact.target_ref for fact in facts}
    active_piece_count = sum(
        1
        for piece in state.active_pieces()
        if piece.color == attacking_color
    )
    return SemanticProbeSpec(
        probe_id=f"capture_control:{attacking_color.value}",
        namespace="capture_control_summary",
        label=attacking_color.value,
        numeric_features=(
            normalize_scalar(len(facts), scale=32.0),
            normalize_scalar(len(unique_targets), scale=32.0),
            float(
                is_king_capturable(
                    state,
                    defending_color,
                    program_registry=program_registry,
                )
            ),
            normalize_scalar(active_piece_count, scale=16.0),
            0.0,
            0.0,
        ),
    )


def _nondefault_state_fields(
    piece_state: Mapping[str, Any],
    piece_class: PieceClass,
) -> int:
    count = 0
    for field_spec in piece_class.instance_state_schema:
        if piece_state.get(field_spec.name, field_spec.default) != field_spec.default:
            count += 1
    return count
