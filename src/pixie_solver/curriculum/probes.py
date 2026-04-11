from __future__ import annotations

from dataclasses import dataclass

from pixie_solver.core import (
    BasePieceType,
    Color,
    GameState,
    Modifier,
    Move,
    PieceClass,
    PieceInstance,
)
from pixie_solver.dsl.compiler import compile_piece_program
from pixie_solver.simulator.engine import apply_move
from pixie_solver.simulator.movegen import legal_moves
from pixie_solver.utils.serialization import JsonValue


@dataclass(frozen=True, slots=True)
class DiagnosticProbe:
    label: str
    before_state: GameState
    move: Move
    observed_state: GameState

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "label": self.label,
            "before_state": self.before_state.to_dict(),
            "move": self.move.to_dict(),
            "observed_state": self.observed_state.to_dict(),
        }


def generate_diagnostic_probes(
    teacher_program: dict[str, JsonValue],
) -> tuple[DiagnosticProbe, ...]:
    teacher_class = compile_piece_program(teacher_program)
    probes: list[DiagnosticProbe] = []
    if _has_movement_modifier(teacher_class, "phase_through_allies"):
        probes.append(_phase_probe(teacher_class))
    if _has_capture_modifier(teacher_class, "replace_capture_with_push"):
        probes.append(_push_capture_probe(teacher_class))
    if any(hook.event == "piece_captured" for hook in teacher_class.hooks):
        probes.append(_piece_captured_hook_probe(teacher_class))
    if any(hook.event == "turn_start" for hook in teacher_class.hooks):
        probes.append(_turn_start_hook_probe(teacher_class))
    if not probes:
        probes.append(_basic_move_probe(teacher_class))
    return tuple(probes)


def _phase_probe(teacher_class: PieceClass) -> DiagnosticProbe:
    blocker = _baseline_piece(BasePieceType.PAWN)
    state = GameState(
        piece_classes={
            teacher_class.class_id: teacher_class,
            blocker.class_id: blocker,
        },
        piece_instances={
            "white_teacher": PieceInstance(
                instance_id="white_teacher",
                piece_class_id=teacher_class.class_id,
                color=Color.WHITE,
                square="a1",
            ),
            "white_blocker": PieceInstance(
                instance_id="white_blocker",
                piece_class_id=blocker.class_id,
                color=Color.WHITE,
                square="a2",
            ),
        },
        side_to_move=Color.WHITE,
    )
    move = Move(piece_id="white_teacher", from_square="a1", to_square="a3")
    observed_state, _ = apply_move(state, move)
    return DiagnosticProbe(
        label="phase_through_allies",
        before_state=state,
        move=move,
        observed_state=observed_state,
    )


def _push_capture_probe(teacher_class: PieceClass) -> DiagnosticProbe:
    target = _baseline_piece(BasePieceType.PAWN)
    edge_behavior = next(
        modifier.args.get("edge_behavior")
        for modifier in teacher_class.capture_modifiers
        if modifier.op == "replace_capture_with_push"
    )
    teacher_square = "a6" if edge_behavior == "remove_if_pushed_off_board" else "a5"
    target_square = "a8" if edge_behavior == "remove_if_pushed_off_board" else "a7"
    state = GameState(
        piece_classes={
            teacher_class.class_id: teacher_class,
            target.class_id: target,
        },
        piece_instances={
            "white_teacher": PieceInstance(
                instance_id="white_teacher",
                piece_class_id=teacher_class.class_id,
                color=Color.WHITE,
                square=teacher_square,
            ),
            "black_target": PieceInstance(
                instance_id="black_target",
                piece_class_id=target.class_id,
                color=Color.BLACK,
                square=target_square,
            ),
        },
        side_to_move=Color.WHITE,
    )
    move = next(
        candidate
        for candidate in legal_moves(state)
        if candidate.piece_id == "white_teacher" and candidate.to_square == target_square
    )
    observed_state, _ = apply_move(state, move)
    return DiagnosticProbe(
        label="push_capture_edge",
        before_state=state,
        move=move,
        observed_state=observed_state,
    )


def _piece_captured_hook_probe(teacher_class: PieceClass) -> DiagnosticProbe:
    actor = _baseline_piece(BasePieceType.ROOK)
    target = _baseline_piece(BasePieceType.PAWN)
    state = GameState(
        piece_classes={
            teacher_class.class_id: teacher_class,
            actor.class_id: actor,
            target.class_id: target,
        },
        piece_instances={
            "white_actor": PieceInstance(
                instance_id="white_actor",
                piece_class_id=actor.class_id,
                color=Color.WHITE,
                square="h1",
            ),
            "white_teacher": PieceInstance(
                instance_id="white_teacher",
                piece_class_id=teacher_class.class_id,
                color=Color.WHITE,
                square="a2",
            ),
            "black_target": PieceInstance(
                instance_id="black_target",
                piece_class_id=target.class_id,
                color=Color.BLACK,
                square="h3",
            ),
        },
        side_to_move=Color.WHITE,
    )
    move = next(
        candidate
        for candidate in legal_moves(state)
        if candidate.piece_id == "white_actor" and candidate.to_square == "h3"
    )
    observed_state, _ = apply_move(state, move)
    return DiagnosticProbe(
        label="piece_captured_hook",
        before_state=state,
        move=move,
        observed_state=observed_state,
    )


def _turn_start_hook_probe(teacher_class: PieceClass) -> DiagnosticProbe:
    actor = _baseline_piece(BasePieceType.ROOK)
    state = GameState(
        piece_classes={
            teacher_class.class_id: teacher_class,
            actor.class_id: actor,
        },
        piece_instances={
            "white_actor": PieceInstance(
                instance_id="white_actor",
                piece_class_id=actor.class_id,
                color=Color.WHITE,
                square="h1",
            ),
            "black_teacher": PieceInstance(
                instance_id="black_teacher",
                piece_class_id=teacher_class.class_id,
                color=Color.BLACK,
                square="a7",
            ),
        },
        side_to_move=Color.WHITE,
    )
    move = Move(piece_id="white_actor", from_square="h1", to_square="h2")
    observed_state, _ = apply_move(state, move)
    return DiagnosticProbe(
        label="turn_start_hook",
        before_state=state,
        move=move,
        observed_state=observed_state,
    )


def _basic_move_probe(teacher_class: PieceClass) -> DiagnosticProbe:
    state = GameState(
        piece_classes={teacher_class.class_id: teacher_class},
        piece_instances={
            "white_teacher": PieceInstance(
                instance_id="white_teacher",
                piece_class_id=teacher_class.class_id,
                color=Color.WHITE,
                square="a2",
            )
        },
        side_to_move=Color.WHITE,
    )
    move = next(move for move in legal_moves(state) if move.piece_id == "white_teacher")
    observed_state, _ = apply_move(state, move)
    return DiagnosticProbe(
        label="basic_move",
        before_state=state,
        move=move,
        observed_state=observed_state,
    )


def _baseline_piece(base_piece_type: BasePieceType) -> PieceClass:
    return PieceClass(
        class_id=f"baseline_{base_piece_type.value}",
        name=f"Baseline {base_piece_type.value.title()}",
        base_piece_type=base_piece_type,
        movement_modifiers=(Modifier(op="inherit_base"),),
        capture_modifiers=(Modifier(op="inherit_base"),),
    )


def _has_movement_modifier(piece_class: PieceClass, op: str) -> bool:
    return any(modifier.op == op for modifier in piece_class.movement_modifiers)


def _has_capture_modifier(piece_class: PieceClass, op: str) -> bool:
    return any(modifier.op == op for modifier in piece_class.capture_modifiers)
