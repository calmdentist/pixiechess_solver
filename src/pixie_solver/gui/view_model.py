from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

from pixie_solver.core import GameState, Move, PieceClass, StateDelta, stable_move_id
from pixie_solver.search import SearchResult
from pixie_solver.simulator.engine import apply_move
from pixie_solver.training.dataset import SelfPlayGame
from pixie_solver.utils.serialization import JsonValue

if TYPE_CHECKING:
    from pixie_solver.training.selfplay import SelfPlayTraceEvent

PIECE_SYMBOLS = {
    "white": {
        "king": "\u2654",
        "queen": "\u2655",
        "rook": "\u2656",
        "bishop": "\u2657",
        "knight": "\u2658",
        "pawn": "\u2659",
    },
    "black": {
        "king": "\u265a",
        "queen": "\u265b",
        "rook": "\u265c",
        "bishop": "\u265d",
        "knight": "\u265e",
        "pawn": "\u265f",
    },
}


def board_snapshot_from_state(state: GameState) -> dict[str, JsonValue]:
    """Return a browser-friendly board snapshot without rule behavior."""
    pieces = []
    for piece in sorted(
        state.piece_instances.values(),
        key=lambda item: (item.square is None, item.square or "", item.instance_id),
    ):
        piece_class = state.piece_classes[piece.piece_class_id]
        base_piece_type = piece_class.base_piece_type.value
        color = piece.color.value
        pieces.append(
            {
                "id": piece.instance_id,
                "class_id": piece.piece_class_id,
                "class_name": piece_class.name,
                "color": color,
                "square": piece.square,
                "active": piece.is_active,
                "base_piece_type": base_piece_type,
                "symbol": PIECE_SYMBOLS[color][base_piece_type],
                "display_letter": display_letter_for_piece_class(piece_class),
                "magical": is_magical_piece_class(piece_class),
                "state": dict(piece.state),
            }
        )
    return {
        "state_hash": state.state_hash(),
        "side_to_move": state.side_to_move.value,
        "fullmove_number": state.fullmove_number,
        "halfmove_clock": state.halfmove_clock,
        "en_passant_square": state.en_passant_square,
        "castling_rights": {
            color: list(rights)
            for color, rights in sorted(state.castling_rights.items())
        },
        "pieces": pieces,
        "active_piece_count": sum(1 for piece in pieces if piece["active"]),
        "metadata": dict(state.metadata),
    }


def move_summary(move: Move | None) -> dict[str, JsonValue] | None:
    if move is None:
        return None
    return {
        "id": stable_move_id(move),
        "short_id": stable_move_id(move)[:8],
        "piece_id": move.piece_id,
        "from": move.from_square,
        "to": move.to_square,
        "label": move_label(move),
        "move_kind": move.move_kind,
        "captured_piece_id": move.captured_piece_id,
        "promotion_piece_type": move.promotion_piece_type,
        "tags": list(move.tags),
        "metadata": dict(move.metadata),
    }


def delta_summary(
    *,
    before_state: GameState | None,
    after_state: GameState | None,
    delta: StateDelta | None,
) -> dict[str, JsonValue] | None:
    if delta is None:
        return None
    changed_piece_ids = list(delta.changed_piece_ids)
    changed_squares: set[str] = set()
    if before_state is not None and after_state is not None:
        for piece_id in changed_piece_ids:
            before_piece = before_state.piece_instances.get(piece_id)
            after_piece = after_state.piece_instances.get(piece_id)
            if before_piece is not None and before_piece.square is not None:
                changed_squares.add(before_piece.square)
            if after_piece is not None and after_piece.square is not None:
                changed_squares.add(after_piece.square)
    return {
        "move": move_summary(delta.move),
        "events": [event.to_dict() for event in delta.events],
        "changed_piece_ids": changed_piece_ids,
        "changed_squares": sorted(changed_squares),
        "notes": list(delta.notes),
        "metadata": dict(delta.metadata),
    }


def search_summary(search_result: SearchResult | None) -> dict[str, JsonValue] | None:
    if search_result is None:
        return None
    moves_by_id = {
        stable_move_id(move): move
        for move in search_result.legal_moves
    }
    ranked_move_ids = sorted(
        moves_by_id,
        key=lambda move_id: (
            -int(search_result.visit_counts.get(move_id, 0)),
            move_label(moves_by_id[move_id]),
            move_id,
        ),
    )
    top_moves = []
    for move_id in ranked_move_ids[:10]:
        move = moves_by_id[move_id]
        top_moves.append(
            {
                "id": move_id,
                "short_id": move_id[:8],
                "label": move_label(move),
                "move": move_summary(move),
                "visits": int(search_result.visit_counts.get(move_id, 0)),
                "probability": float(search_result.visit_distribution.get(move_id, 0.0)),
                "policy_logit": (
                    float(search_result.policy_logits[move_id])
                    if move_id in search_result.policy_logits
                    else None
                ),
                "selected": move_id == search_result.selected_move_id,
            }
        )
    return {
        "selected_move_id": search_result.selected_move_id,
        "selected_move": move_summary(search_result.selected_move),
        "legal_move_count": len(search_result.legal_moves),
        "root_value": search_result.root_value,
        "top_moves": top_moves,
        "metadata": dict(search_result.metadata),
    }


def selfplay_trace_event_to_frame(
    event: "SelfPlayTraceEvent",
    *,
    cycle: int | None = None,
    phase: str = "selfplay",
) -> dict[str, JsonValue]:
    return {
        "event": event.event,
        "cycle": cycle,
        "phase": phase,
        "game_index": event.game_index,
        "games_total": event.games_total,
        "ply": event.ply,
        "used_model": event.used_model,
        "before": (
            board_snapshot_from_state(event.before_state)
            if event.before_state is not None
            else None
        ),
        "after": (
            board_snapshot_from_state(event.after_state)
            if event.after_state is not None
            else None
        ),
        "move": move_summary(event.move),
        "delta": delta_summary(
            before_state=event.before_state,
            after_state=event.after_state,
            delta=event.delta,
        ),
        "search": search_summary(event.search_result),
        "outcome": event.outcome,
        "termination_reason": event.termination_reason,
        "metadata": dict(event.metadata),
    }


def replay_payload_from_games(
    games: Sequence[SelfPlayGame],
    *,
    source: str | None = None,
) -> dict[str, JsonValue]:
    frames: list[dict[str, JsonValue]] = []
    for game_index, game in enumerate(games):
        frames.extend(replay_frames_from_game(game, game_index=game_index))
    return {
        "mode": "replay",
        "source": source,
        "frames": frames,
        "game_count": len(games),
    }


def replay_frames_from_game(
    game: SelfPlayGame,
    *,
    game_index: int | None = None,
) -> list[dict[str, JsonValue]]:
    index = int(game.metadata.get("game_index", game_index or 0))
    state = game.replay_trace.initial_state
    frames: list[dict[str, JsonValue]] = [
        {
            "event": "game_started",
            "cycle": game.replay_trace.metadata.get("cycle"),
            "phase": str(game.replay_trace.metadata.get("phase", "replay")),
            "game_index": index,
            "games_total": None,
            "ply": 0,
            "used_model": bool(game.replay_trace.metadata.get("used_model", False)),
            "before": None,
            "after": board_snapshot_from_state(state),
            "move": None,
            "delta": None,
            "search": None,
            "outcome": None,
            "termination_reason": None,
            "metadata": dict(game.replay_trace.metadata),
        }
    ]
    for step in game.replay_trace.steps:
        before_state = state
        after_state, delta = apply_move(before_state, step.move)
        if delta != step.delta:
            raise ValueError(f"Replay delta mismatch at ply {step.ply}")
        if before_state.state_hash() != step.before_state_hash:
            raise ValueError(f"Replay before-state hash mismatch at ply {step.ply}")
        if after_state.state_hash() != step.after_state_hash:
            raise ValueError(f"Replay after-state hash mismatch at ply {step.ply}")
        frames.append(
            {
                "event": "ply_completed",
                "cycle": game.replay_trace.metadata.get("cycle"),
                "phase": str(game.replay_trace.metadata.get("phase", "replay")),
                "game_index": index,
                "games_total": None,
                "ply": step.ply,
                "used_model": bool(game.replay_trace.metadata.get("used_model", False)),
                "before": board_snapshot_from_state(before_state),
                "after": board_snapshot_from_state(after_state),
                "move": move_summary(step.move),
                "delta": delta_summary(
                    before_state=before_state,
                    after_state=after_state,
                    delta=delta,
                ),
                "search": None,
                "outcome": None,
                "termination_reason": None,
                "metadata": {
                    "before_state_hash": step.before_state_hash,
                    "after_state_hash": step.after_state_hash,
                },
            }
        )
        state = after_state
    frames.append(
        {
            "event": "game_completed",
            "cycle": game.replay_trace.metadata.get("cycle"),
            "phase": str(game.replay_trace.metadata.get("phase", "replay")),
            "game_index": index,
            "games_total": None,
            "ply": len(game.replay_trace.steps),
            "used_model": bool(game.replay_trace.metadata.get("used_model", False)),
            "before": None,
            "after": board_snapshot_from_state(state),
            "move": None,
            "delta": None,
            "search": None,
            "outcome": game.outcome,
            "termination_reason": game.metadata.get("termination_reason"),
            "metadata": {
                **dict(game.metadata),
                "final_state_hash": game.final_state_hash,
            },
        }
    )
    return frames


def training_progress_frame(
    *,
    event: str,
    cycle: int | None,
    phase: str,
    metadata: dict[str, JsonValue],
) -> dict[str, JsonValue]:
    return {
        "event": event,
        "cycle": cycle,
        "phase": phase,
        "game_index": None,
        "games_total": None,
        "ply": None,
        "used_model": None,
        "before": None,
        "after": None,
        "move": None,
        "delta": None,
        "search": None,
        "outcome": None,
        "termination_reason": None,
        "metadata": metadata,
    }


def viewer_status_frame(message: str, **metadata: JsonValue) -> dict[str, JsonValue]:
    return training_progress_frame(
        event="viewer_status",
        cycle=None,
        phase="viewer",
        metadata={"message": message, **metadata},
    )


def move_label(move: Move) -> str:
    capture = "x" if move.move_kind in {"capture", "en_passant_capture", "push_capture"} else "-"
    promotion = f"={move.promotion_piece_type}" if move.promotion_piece_type else ""
    return f"{move.from_square}{capture}{move.to_square}{promotion}"


def display_letter_for_piece_class(piece_class: PieceClass) -> str | None:
    if not is_magical_piece_class(piece_class):
        return None
    configured = piece_class.metadata.get("display_letter")
    if isinstance(configured, str) and configured.strip():
        return configured.strip()[0].upper()
    for token in piece_class.class_id.split("_"):
        if token and token not in {"baseline", "orthodox"}:
            return token[0].upper()
    for character in piece_class.name:
        if character.isalnum():
            return character.upper()
    return "?"


def is_magical_piece_class(piece_class: PieceClass) -> bool:
    if piece_class.class_id.startswith(("baseline_", "orthodox_")):
        return False
    if piece_class.name.lower().startswith(("baseline ", "orthodox ")):
        return False
    return True


def publish_frames(
    publisher,
    frames: Iterable[dict[str, JsonValue]],
) -> None:
    for frame in frames:
        publisher(frame)
