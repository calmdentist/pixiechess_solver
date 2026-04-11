from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pixie_solver.core import GameState, Move
from pixie_solver.core.event import StateDelta
from pixie_solver.dsl.compiler import compile_piece_program
from pixie_solver.simulator.engine import apply_move
from pixie_solver.utils.serialization import JsonValue


@dataclass(frozen=True, slots=True)
class PieceDiff:
    piece_id: str
    predicted: dict[str, JsonValue] | None
    observed: dict[str, JsonValue] | None

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "piece_id": self.piece_id,
            "predicted": self.predicted,
            "observed": self.observed,
        }


@dataclass(frozen=True, slots=True)
class StateDiff:
    piece_diffs: tuple[PieceDiff, ...] = ()
    global_diffs: dict[str, dict[str, JsonValue]] = field(default_factory=dict)
    predicted_state_hash: str | None = None
    observed_state_hash: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "piece_diffs", tuple(self.piece_diffs))
        object.__setattr__(self, "global_diffs", dict(self.global_diffs))

    @property
    def is_empty(self) -> bool:
        return not self.piece_diffs and not self.global_diffs

    @property
    def changed_piece_ids(self) -> tuple[str, ...]:
        return tuple(diff.piece_id for diff in self.piece_diffs)

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "piece_diffs": [diff.to_dict() for diff in self.piece_diffs],
            "global_diffs": dict(self.global_diffs),
            "predicted_state_hash": self.predicted_state_hash,
            "observed_state_hash": self.observed_state_hash,
        }


@dataclass(frozen=True, slots=True)
class StateMismatch:
    before_state: GameState
    move: Move
    predicted_state: GameState
    observed_state: GameState
    diff: StateDiff
    implicated_piece_ids: tuple[str, ...] = ()
    implicated_piece_class_ids: tuple[str, ...] = ()
    predicted_delta: StateDelta | None = None
    predicted_error: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "implicated_piece_ids",
            tuple(sorted(set(self.implicated_piece_ids))),
        )
        object.__setattr__(
            self,
            "implicated_piece_class_ids",
            tuple(sorted(set(self.implicated_piece_class_ids))),
        )

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "before_state": self.before_state.to_dict(),
            "move": self.move.to_dict(),
            "predicted_state": self.predicted_state.to_dict(),
            "observed_state": self.observed_state.to_dict(),
            "diff": self.diff.to_dict(),
            "implicated_piece_ids": list(self.implicated_piece_ids),
            "implicated_piece_class_ids": list(self.implicated_piece_class_ids),
            "predicted_delta": (
                self.predicted_delta.to_dict()
                if self.predicted_delta is not None
                else None
            ),
            "predicted_error": self.predicted_error,
        }


def build_state_mismatch(
    *,
    before_state: GameState,
    move: Move,
    observed_state: GameState,
    current_program: dict[str, Any],
) -> StateMismatch:
    predicted_before = replace_piece_program(before_state, current_program)
    predicted_delta = None
    predicted_error = None
    try:
        predicted_state, predicted_delta = apply_move(predicted_before, move)
    except Exception as exc:
        predicted_state = predicted_before
        predicted_error = str(exc)
    diff = diff_states(predicted_state, observed_state)
    implicated_piece_ids = _implicated_piece_ids(
        move=move,
        predicted_delta=predicted_delta,
        diff=diff,
    )
    return StateMismatch(
        before_state=predicted_before,
        move=move,
        predicted_state=predicted_state,
        observed_state=observed_state,
        diff=diff,
        implicated_piece_ids=implicated_piece_ids,
        implicated_piece_class_ids=_piece_class_ids_for(
            predicted_before,
            observed_state,
            implicated_piece_ids,
        ),
        predicted_delta=predicted_delta,
        predicted_error=predicted_error,
    )


def replace_piece_program(state: GameState, program: dict[str, Any]) -> GameState:
    piece_class = compile_piece_program(program)
    piece_classes = dict(state.piece_classes)
    piece_classes[piece_class.class_id] = piece_class
    return GameState(
        piece_classes=piece_classes,
        piece_instances=state.piece_instances,
        side_to_move=state.side_to_move,
        castling_rights=state.castling_rights,
        en_passant_square=state.en_passant_square,
        halfmove_clock=state.halfmove_clock,
        fullmove_number=state.fullmove_number,
        repetition_counts=state.repetition_counts,
        pending_events=state.pending_events,
        metadata=state.metadata,
    )


def diff_states(predicted: GameState, observed: GameState) -> StateDiff:
    predicted_signature = behavioral_state_signature(predicted)
    observed_signature = behavioral_state_signature(observed)
    piece_diffs: list[PieceDiff] = []
    predicted_pieces = dict(predicted_signature["piece_instances"])
    observed_pieces = dict(observed_signature["piece_instances"])
    for piece_id in sorted(set(predicted_pieces) | set(observed_pieces)):
        predicted_piece = predicted_pieces.get(piece_id)
        observed_piece = observed_pieces.get(piece_id)
        if predicted_piece != observed_piece:
            piece_diffs.append(
                PieceDiff(
                    piece_id=piece_id,
                    predicted=(
                        dict(predicted_piece)
                        if isinstance(predicted_piece, dict)
                        else None
                    ),
                    observed=(
                        dict(observed_piece)
                        if isinstance(observed_piece, dict)
                        else None
                    ),
                )
            )

    global_diffs: dict[str, dict[str, JsonValue]] = {}
    for key in sorted(set(predicted_signature) | set(observed_signature)):
        if key == "piece_instances":
            continue
        predicted_value = predicted_signature.get(key)
        observed_value = observed_signature.get(key)
        if predicted_value != observed_value:
            global_diffs[key] = {
                "predicted": predicted_value,
                "observed": observed_value,
            }

    return StateDiff(
        piece_diffs=tuple(piece_diffs),
        global_diffs=global_diffs,
        predicted_state_hash=predicted.state_hash(),
        observed_state_hash=observed.state_hash(),
    )


def behavioral_state_signature(state: GameState) -> dict[str, JsonValue]:
    return {
        "piece_instances": {
            piece_id: piece.to_dict()
            for piece_id, piece in sorted(state.piece_instances.items())
        },
        "side_to_move": state.side_to_move.value,
        "castling_rights": {
            color: list(rights)
            for color, rights in sorted(state.castling_rights.items())
        },
        "en_passant_square": state.en_passant_square,
        "halfmove_clock": state.halfmove_clock,
        "fullmove_number": state.fullmove_number,
        "repetition_counts": dict(sorted(state.repetition_counts.items())),
        "pending_events": [event.to_dict() for event in state.pending_events],
    }


def _implicated_piece_ids(
    *,
    move: Move,
    predicted_delta: StateDelta | None,
    diff: StateDiff,
) -> tuple[str, ...]:
    piece_ids = {move.piece_id, *diff.changed_piece_ids}
    if predicted_delta is not None:
        piece_ids.update(predicted_delta.changed_piece_ids)
    if move.captured_piece_id is not None:
        piece_ids.add(move.captured_piece_id)
    target_piece_id = move.metadata.get("target_piece_id")
    if target_piece_id is not None:
        piece_ids.add(str(target_piece_id))
    return tuple(sorted(piece_ids))


def _piece_class_ids_for(
    before_state: GameState,
    observed_state: GameState,
    piece_ids: tuple[str, ...],
) -> tuple[str, ...]:
    class_ids: set[str] = set()
    for state in (before_state, observed_state):
        for piece_id in piece_ids:
            piece = state.piece_instances.get(piece_id)
            if piece is not None:
                class_ids.add(piece.piece_class_id)
    return tuple(sorted(class_ids))
