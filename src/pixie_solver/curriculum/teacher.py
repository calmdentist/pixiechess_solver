from __future__ import annotations

import random
from dataclasses import dataclass, field

from pixie_solver.utils.serialization import JsonValue


RECIPES = ("capture_sprint", "edge_sumo", "phase_rook", "turn_charge")


@dataclass(frozen=True, slots=True)
class SyntheticPiece:
    piece_id: str
    description: str
    teacher_program: dict[str, JsonValue]
    recipe: str
    seed: int
    metadata: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "teacher_program", dict(self.teacher_program))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "piece_id": self.piece_id,
            "description": self.description,
            "teacher_program": dict(self.teacher_program),
            "recipe": self.recipe,
            "seed": self.seed,
            "metadata": dict(self.metadata),
        }


def generate_teacher_piece(
    *,
    seed: int,
    recipe: str | None = None,
    piece_id: str | None = None,
) -> SyntheticPiece:
    rng = random.Random(seed)
    selected_recipe = recipe or RECIPES[seed % len(RECIPES)]
    if selected_recipe not in RECIPES:
        raise ValueError(f"recipe must be one of {RECIPES!r}")
    generated_piece_id = piece_id or f"synthetic_{selected_recipe}_{seed}"

    if selected_recipe == "capture_sprint":
        distance = 2 if rng.random() < 0.7 else 1
        program = _capture_sprint_program(generated_piece_id, distance=distance)
        description = (
            "A pawn that surges forward whenever any piece is taken."
            if distance == 2
            else "A pawn that steps forward whenever any piece is taken."
        )
        metadata = {"hidden_forward_distance": distance}
    elif selected_recipe == "edge_sumo":
        remove_at_edge = rng.random() < 0.75
        edge_behavior = "remove_if_pushed_off_board" if remove_at_edge else "block"
        program = _edge_sumo_program(generated_piece_id, edge_behavior=edge_behavior)
        description = (
            "A rook that shoves enemies instead of capturing; at the rim they fall away."
            if remove_at_edge
            else "A rook that shoves enemies instead of capturing, but the board edge stops them."
        )
        metadata = {"hidden_edge_behavior": edge_behavior}
    elif selected_recipe == "phase_rook":
        program = _phase_rook_program(generated_piece_id)
        description = "A rook that can slide through friendly pieces."
        metadata = {"hidden_movement": "phase_through_allies"}
    else:
        amount = 2 if rng.random() < 0.5 else 1
        program = _turn_charge_program(generated_piece_id, amount=amount)
        description = "A pawn that gathers charge as its turn begins."
        metadata = {"hidden_charge_amount": amount}

    return SyntheticPiece(
        piece_id=generated_piece_id,
        description=description,
        teacher_program=program,
        recipe=selected_recipe,
        seed=seed,
        metadata=metadata,
    )


def _base_program(
    *,
    piece_id: str,
    name: str,
    base_piece_type: str,
) -> dict[str, JsonValue]:
    return {
        "piece_id": piece_id,
        "name": name,
        "base_piece_type": base_piece_type,
        "instance_state_schema": [],
        "movement_modifiers": [{"op": "inherit_base", "args": {}}],
        "capture_modifiers": [{"op": "inherit_base", "args": {}}],
        "hooks": [],
        "metadata": {"synthetic": True},
    }


def _capture_sprint_program(piece_id: str, *, distance: int) -> dict[str, JsonValue]:
    program = _base_program(
        piece_id=piece_id,
        name="Synthetic Capture Sprint",
        base_piece_type="pawn",
    )
    program["hooks"] = [
        {
            "event": "piece_captured",
            "conditions": [
                {"op": "self_on_board", "args": {}},
                {
                    "op": "square_empty",
                    "args": {
                        "square": {
                            "relative_to": "self",
                            "offset": [0, distance],
                        }
                    },
                },
            ],
            "effects": [
                {
                    "op": "move_piece",
                    "args": {
                        "piece": "self",
                        "to": {
                            "relative_to": "self",
                            "offset": [0, distance],
                        },
                    },
                }
            ],
        }
    ]
    return program


def _edge_sumo_program(piece_id: str, *, edge_behavior: str) -> dict[str, JsonValue]:
    program = _base_program(
        piece_id=piece_id,
        name="Synthetic Edge Sumo",
        base_piece_type="rook",
    )
    program["capture_modifiers"] = [
        {
            "op": "replace_capture_with_push",
            "args": {
                "distance": 1,
                "edge_behavior": edge_behavior,
            },
        }
    ]
    return program


def _phase_rook_program(piece_id: str) -> dict[str, JsonValue]:
    program = _base_program(
        piece_id=piece_id,
        name="Synthetic Phasing Rook",
        base_piece_type="rook",
    )
    program["movement_modifiers"] = [
        {"op": "inherit_base", "args": {}},
        {"op": "phase_through_allies", "args": {}},
    ]
    return program


def _turn_charge_program(piece_id: str, *, amount: int) -> dict[str, JsonValue]:
    program = _base_program(
        piece_id=piece_id,
        name="Synthetic Turn Charge",
        base_piece_type="pawn",
    )
    program["instance_state_schema"] = [
        {
            "name": "charge",
            "type": "int",
            "default": 0,
            "description": "Stored charge.",
        }
    ]
    program["hooks"] = [
        {
            "event": "turn_start",
            "conditions": [{"op": "self_on_board", "args": {}}],
            "effects": [
                {
                    "op": "increment_state",
                    "args": {
                        "piece": "self",
                        "name": "charge",
                        "amount": amount,
                    },
                }
            ],
        }
    ]
    return program
