from __future__ import annotations

from dataclasses import dataclass, field

from pixie_solver.core.event import Event
from pixie_solver.core.piece import Color, PieceClass, PieceInstance
from pixie_solver.utils.serialization import JsonValue
from pixie_solver.utils.squares import normalize_square


@dataclass(frozen=True, slots=True)
class GameState:
    piece_classes: dict[str, PieceClass] = field(default_factory=dict)
    piece_instances: dict[str, PieceInstance] = field(default_factory=dict)
    side_to_move: Color = Color.WHITE
    castling_rights: dict[str, tuple[str, ...]] = field(default_factory=dict)
    en_passant_square: str | None = None
    halfmove_clock: int = 0
    fullmove_number: int = 1
    repetition_counts: dict[str, int] = field(default_factory=dict)
    pending_events: tuple[Event, ...] = ()
    metadata: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "piece_classes", dict(self.piece_classes))
        piece_instances = dict(self.piece_instances)
        object.__setattr__(
            self,
            "castling_rights",
            {
                str(color): tuple(str(side) for side in rights)
                for color, rights in self.castling_rights.items()
            },
        )
        object.__setattr__(
            self,
            "en_passant_square",
            normalize_square(self.en_passant_square),
        )
        object.__setattr__(
            self,
            "repetition_counts",
            {str(key): int(value) for key, value in self.repetition_counts.items()},
        )
        object.__setattr__(self, "pending_events", tuple(self.pending_events))
        object.__setattr__(self, "metadata", dict(self.metadata))
        normalized_piece_instances: dict[str, PieceInstance] = {}
        for piece_id, piece in piece_instances.items():
            piece_class = self.piece_classes.get(piece.piece_class_id)
            if piece_class is None:
                normalized_piece_instances[piece_id] = piece
                continue
            normalized_piece_instances[piece_id] = PieceInstance(
                instance_id=piece.instance_id,
                piece_class_id=piece.piece_class_id,
                color=piece.color,
                square=piece.square,
                state=piece_class.normalize_instance_state(piece.state),
            )
        object.__setattr__(self, "piece_instances", normalized_piece_instances)
        self.validate()

    @classmethod
    def empty(cls, *, side_to_move: Color = Color.WHITE) -> "GameState":
        return cls(side_to_move=side_to_move)

    def validate(self) -> None:
        if self.halfmove_clock < 0:
            raise ValueError("halfmove_clock must be non-negative")
        if self.fullmove_number < 1:
            raise ValueError("fullmove_number must be at least 1")

        occupied_squares: dict[str, str] = {}
        for piece_id, piece in self.piece_instances.items():
            if piece.instance_id != piece_id:
                raise ValueError(
                    f"piece instance id mismatch for key {piece_id!r}: "
                    f"{piece.instance_id!r}"
                )
            if piece.piece_class_id not in self.piece_classes:
                raise ValueError(
                    f"piece {piece_id!r} references missing class "
                    f"{piece.piece_class_id!r}"
                )
            if piece.square is None:
                continue
            if piece.square in occupied_squares:
                other_piece_id = occupied_squares[piece.square]
                raise ValueError(
                    f"duplicate occupancy on {piece.square}: "
                    f"{other_piece_id!r} and {piece_id!r}"
                )
            occupied_squares[piece.square] = piece_id

    def active_pieces(self) -> tuple[PieceInstance, ...]:
        return tuple(piece for piece in self.piece_instances.values() if piece.is_active)

    def occupancy(self) -> dict[str, str]:
        return {
            piece.square: piece_id
            for piece_id, piece in self.piece_instances.items()
            if piece.square is not None
        }

    def piece_on(self, square: str) -> PieceInstance | None:
        normalized_square = normalize_square(square)
        for piece in self.piece_instances.values():
            if piece.square == normalized_square:
                return piece
        return None

    def state_hash(self) -> str:
        from pixie_solver.core.hash import stable_state_hash

        return stable_state_hash(self)

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "piece_classes": {
                class_id: piece_class.to_dict()
                for class_id, piece_class in sorted(self.piece_classes.items())
            },
            "piece_instances": {
                piece_id: piece.to_dict()
                for piece_id, piece in sorted(self.piece_instances.items())
            },
            "side_to_move": self.side_to_move.value,
            "castling_rights": {
                color: list(rights)
                for color, rights in sorted(self.castling_rights.items())
            },
            "en_passant_square": self.en_passant_square,
            "halfmove_clock": self.halfmove_clock,
            "fullmove_number": self.fullmove_number,
            "repetition_counts": dict(sorted(self.repetition_counts.items())),
            "pending_events": [event.to_dict() for event in self.pending_events],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, JsonValue]) -> "GameState":
        return cls(
            piece_classes={
                class_id: PieceClass.from_dict(piece_class)
                for class_id, piece_class in data.get("piece_classes", {}).items()
            },
            piece_instances={
                piece_id: PieceInstance.from_dict(piece_instance)
                for piece_id, piece_instance in data.get("piece_instances", {}).items()
            },
            side_to_move=Color(str(data.get("side_to_move", Color.WHITE.value))),
            castling_rights={
                color: tuple(str(side) for side in rights)
                for color, rights in data.get("castling_rights", {}).items()
            },
            en_passant_square=(
                str(data["en_passant_square"])
                if data.get("en_passant_square") is not None
                else None
            ),
            halfmove_clock=int(data.get("halfmove_clock", 0)),
            fullmove_number=int(data.get("fullmove_number", 1)),
            repetition_counts={
                str(key): int(value)
                for key, value in data.get("repetition_counts", {}).items()
            },
            pending_events=tuple(
                Event.from_dict(event) for event in data.get("pending_events", [])
            ),
            metadata=dict(data.get("metadata", {})),
        )
