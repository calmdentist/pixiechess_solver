from __future__ import annotations

import random
from collections.abc import Sequence

from pixie_solver.core.piece import BasePieceType, Color, Modifier, PieceClass, PieceInstance
from pixie_solver.core.state import GameState

_BACK_RANK = (
    ("rook", "a"),
    ("knight", "b"),
    ("bishop", "c"),
    ("queen", "d"),
    ("king", "e"),
    ("bishop", "f"),
    ("knight", "g"),
    ("rook", "h"),
)
_BASELINE_CLASS_IDS = {
    "pawn": "baseline_pawn",
    "knight": "baseline_knight",
    "bishop": "baseline_bishop",
    "rook": "baseline_rook",
    "queen": "baseline_queen",
    "king": "baseline_king",
}
_STANDARD_SLOT_FILES = {
    BasePieceType.PAWN: ("a", "b", "c", "d", "e", "f", "g", "h"),
    BasePieceType.KNIGHT: ("b", "g"),
    BasePieceType.BISHOP: ("c", "f"),
    BasePieceType.ROOK: ("a", "h"),
    BasePieceType.QUEEN: ("d",),
    BasePieceType.KING: ("e",),
}


def standard_piece_classes() -> dict[str, PieceClass]:
    classes: dict[str, PieceClass] = {}
    for piece_name, class_id in sorted(_BASELINE_CLASS_IDS.items()):
        base_piece_type = BasePieceType(piece_name)
        classes[class_id] = PieceClass(
            class_id=class_id,
            name=f"Baseline {piece_name.title()}",
            base_piece_type=base_piece_type,
            movement_modifiers=(Modifier(op="inherit_base"),),
            capture_modifiers=(Modifier(op="inherit_base"),),
        )
    return classes


def standard_initial_state() -> GameState:
    piece_classes = standard_piece_classes()
    piece_instances: dict[str, PieceInstance] = {}

    for color, home_rank, pawn_rank in (
        (Color.WHITE, "1", "2"),
        (Color.BLACK, "8", "7"),
    ):
        color_name = color.value
        for piece_name, file_name in _BACK_RANK:
            piece_id = (
                f"{color_name}_{piece_name}"
                if piece_name in {"king", "queen"}
                else f"{color_name}_{piece_name}_{file_name}"
            )
            piece_instances[piece_id] = PieceInstance(
                instance_id=piece_id,
                piece_class_id=_BASELINE_CLASS_IDS[piece_name],
                color=color,
                square=f"{file_name}{home_rank}",
            )

        for file_name in "abcdefgh":
            piece_id = f"{color_name}_pawn_{file_name}"
            piece_instances[piece_id] = PieceInstance(
                instance_id=piece_id,
                piece_class_id=_BASELINE_CLASS_IDS["pawn"],
                color=color,
                square=f"{file_name}{pawn_rank}",
            )

    return GameState(
        piece_classes=piece_classes,
        piece_instances=piece_instances,
        side_to_move=Color.WHITE,
        castling_rights={
            Color.WHITE.value: ("king", "queen"),
            Color.BLACK.value: ("king", "queen"),
        },
        halfmove_clock=0,
        fullmove_number=1,
    )


def sample_standard_initial_state(
    rng: random.Random,
    *,
    special_piece_classes: Sequence[PieceClass] = (),
    inclusion_probability: float = 0.5,
) -> GameState:
    if not 0.0 <= inclusion_probability <= 1.0:
        raise ValueError("inclusion_probability must be between 0.0 and 1.0")

    piece_classes = standard_piece_classes()
    available_slots = {
        base_piece_type: list(slot_files)
        for base_piece_type, slot_files in _STANDARD_SLOT_FILES.items()
    }
    sampled_specials: list[tuple[PieceClass, str]] = []
    special_candidates = list(special_piece_classes)
    rng.shuffle(special_candidates)
    for piece_class in special_candidates:
        eligible_slots = available_slots[piece_class.base_piece_type]
        if not eligible_slots or rng.random() > inclusion_probability:
            continue
        chosen_file = rng.choice(eligible_slots)
        eligible_slots.remove(chosen_file)
        sampled_specials.append((piece_class, chosen_file))
        piece_classes[piece_class.class_id] = piece_class

    overrides = {
        (piece_class.base_piece_type, file_name): piece_class.class_id
        for piece_class, file_name in sampled_specials
    }

    piece_instances: dict[str, PieceInstance] = {}
    for color, home_rank, pawn_rank in (
        (Color.WHITE, "1", "2"),
        (Color.BLACK, "8", "7"),
    ):
        color_name = color.value
        for piece_name, file_name in _BACK_RANK:
            base_piece_type = BasePieceType(piece_name)
            class_id = overrides.get(
                (base_piece_type, file_name),
                _BASELINE_CLASS_IDS[piece_name],
            )
            piece_instances[_instance_id_for_slot(color_name, class_id, piece_name, file_name)] = PieceInstance(
                instance_id=_instance_id_for_slot(color_name, class_id, piece_name, file_name),
                piece_class_id=class_id,
                color=color,
                square=f"{file_name}{home_rank}",
            )

        for file_name in "abcdefgh":
            class_id = overrides.get(
                (BasePieceType.PAWN, file_name),
                _BASELINE_CLASS_IDS["pawn"],
            )
            piece_instances[_instance_id_for_slot(color_name, class_id, "pawn", file_name)] = PieceInstance(
                instance_id=_instance_id_for_slot(color_name, class_id, "pawn", file_name),
                piece_class_id=class_id,
                color=color,
                square=f"{file_name}{pawn_rank}",
            )

    return GameState(
        piece_classes=piece_classes,
        piece_instances=piece_instances,
        side_to_move=Color.WHITE,
        castling_rights={
            Color.WHITE.value: ("king", "queen"),
            Color.BLACK.value: ("king", "queen"),
        },
        halfmove_clock=0,
        fullmove_number=1,
    )


def _instance_id_for_slot(
    color_name: str,
    class_id: str,
    baseline_piece_name: str,
    file_name: str,
) -> str:
    baseline_class_id = _BASELINE_CLASS_IDS[baseline_piece_name]
    if class_id == baseline_class_id:
        if baseline_piece_name in {"king", "queen"}:
            return f"{color_name}_{baseline_piece_name}"
        return f"{color_name}_{baseline_piece_name}_{file_name}"
    return f"{color_name}_{class_id}_{file_name}"
