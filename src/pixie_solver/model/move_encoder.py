from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from pixie_solver.core.move import Move, stable_move_id
from pixie_solver.core.state import GameState
from pixie_solver.model._features import (
    COMMON_MOVE_KIND_IDS,
    PROMOTION_PIECE_TYPE_IDS,
    json_value_token,
    material_score,
    move_kind_id,
    normalize_scalar,
    square_index,
)
from pixie_solver.simulator.engine import apply_move, result
from pixie_solver.simulator.movegen import is_in_check
from pixie_solver.simulator.transition import other_color

CONSEQUENCE_FEATURES = 12


@dataclass(slots=True)
class EncodedMoves:
    move_ids: tuple[str, ...]
    candidate_embeddings: Tensor


class MoveEncoder(nn.Module):
    def __init__(
        self,
        *,
        d_model: int = 192,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        inner_dim = d_model * 2 if hidden_dim is None else hidden_dim
        self.d_model = d_model
        self.square_embedding = nn.Embedding(65, d_model)
        self.move_kind_embedding = nn.Embedding(len(COMMON_MOVE_KIND_IDS) + 32 + 1, d_model)
        self.promotion_embedding = nn.Embedding(len(PROMOTION_PIECE_TYPE_IDS), d_model)
        self.metadata_embedding = nn.Embedding(128 + 1, d_model)
        self.flag_projection = nn.Sequential(
            nn.Linear(7, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, d_model),
        )
        self.consequence_projection = nn.Sequential(
            nn.Linear(CONSEQUENCE_FEATURES, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, d_model),
        )
        self.output_mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, d_model),
        )
        self.null_piece_embedding = nn.Parameter(torch.zeros(d_model, dtype=torch.float32))

    def encode_move(
        self,
        state: GameState,
        move: Move,
        *,
        piece_context_by_id: dict[str, Tensor],
        global_context: Tensor,
    ) -> Tensor:
        return self.encode_moves(
            state,
            (move,),
            piece_context_by_id=piece_context_by_id,
            global_context=global_context,
        ).candidate_embeddings[0]

    def encode_moves(
        self,
        state: GameState,
        legal_moves: tuple[Move, ...] | list[Move],
        *,
        piece_context_by_id: dict[str, Tensor],
        global_context: Tensor,
    ) -> EncodedMoves:
        if not legal_moves:
            return EncodedMoves(
                move_ids=(),
                candidate_embeddings=torch.zeros(
                    (0, self.d_model),
                    dtype=torch.float32,
                    device=global_context.device,
                ),
            )

        candidate_embeddings: list[Tensor] = []
        move_ids: list[str] = []
        for move in legal_moves:
            move_ids.append(stable_move_id(move))
            candidate_embeddings.append(
                self._encode_single_move(
                    state,
                    move,
                    piece_context_by_id=piece_context_by_id,
                    global_context=global_context,
                )
            )

        return EncodedMoves(
            move_ids=tuple(move_ids),
            candidate_embeddings=torch.stack(candidate_embeddings, dim=0),
        )

    def _encode_single_move(
        self,
        state: GameState,
        move: Move,
        *,
        piece_context_by_id: dict[str, Tensor],
        global_context: Tensor,
    ) -> Tensor:
        device = global_context.device
        moving_context = piece_context_by_id[move.piece_id]
        target_piece_id = _target_piece_id(move)
        target_context = piece_context_by_id.get(
            target_piece_id,
            self.null_piece_embedding.to(device),
        )

        token = global_context + moving_context + target_context
        token = token + self._embed_scalar(
            self.square_embedding,
            square_index(move.from_square),
        )
        token = token + self._embed_scalar(
            self.square_embedding,
            square_index(move.to_square),
        )
        token = token + self._embed_scalar(
            self.move_kind_embedding,
            move_kind_id(move.move_kind),
        )
        token = token + self._embed_scalar(
            self.promotion_embedding,
            PROMOTION_PIECE_TYPE_IDS[move.promotion_piece_type],
        )
        token = token + self._embed_scalar(
            self.metadata_embedding,
            json_value_token(
                tuple(sorted(move.tags)),
                namespace="move_tags",
                bucket_count=128,
            ),
        )

        token = token + self.flag_projection(
            torch.tensor(
                [
                    float(move.captured_piece_id is not None),
                    float(move.move_kind == "castle"),
                    float(move.move_kind == "push_capture"),
                    float(move.move_kind == "en_passant_capture"),
                    float(move.promotion_piece_type is not None),
                    float(bool(move.tags)),
                    float(target_piece_id is not None),
                ],
                dtype=torch.float32,
                device=device,
            )
        )
        token = token + self.consequence_projection(
            torch.tensor(
                _consequence_features(state, move),
                dtype=torch.float32,
                device=device,
            )
        )
        return self.output_mlp(token)

    def _embed_scalar(self, embedding: nn.Embedding, token_id: int) -> Tensor:
        device = embedding.weight.device
        return embedding(torch.tensor(token_id, dtype=torch.long, device=device))


def _consequence_features(state: GameState, move: Move) -> tuple[float, ...]:
    moving_piece = state.piece_instances[move.piece_id]
    mover = moving_piece.color
    opponent = other_color(mover)
    before_advantage = material_score(state, mover) - material_score(state, opponent)
    before_self_check = is_in_check(state, mover)
    before_opponent_check = is_in_check(state, opponent)

    after_state, delta = apply_move(state, move)
    after_advantage = material_score(after_state, mover) - material_score(after_state, opponent)
    after_self_check = is_in_check(after_state, mover)
    after_opponent_check = is_in_check(after_state, opponent)
    active_after = len(after_state.active_pieces())
    active_before = len(state.active_pieces())
    piece_state_changed = any(
        state.piece_instances[piece_id].state != after_state.piece_instances[piece_id].state
        for piece_id in delta.changed_piece_ids
        if piece_id in state.piece_instances and piece_id in after_state.piece_instances
    )
    terminal = result(after_state)

    return (
        normalize_scalar(after_advantage - before_advantage, scale=8.0),
        float(before_self_check),
        float(before_opponent_check),
        float(after_self_check),
        float(after_opponent_check),
        normalize_scalar(len(delta.events), scale=8.0),
        normalize_scalar(len(delta.changed_piece_ids), scale=8.0),
        float(active_after < active_before),
        float(piece_state_changed),
        float(after_state.side_to_move == mover),
        1.0 if terminal == mover.value else -1.0 if terminal == opponent.value else 0.0,
        float(terminal == "draw"),
    )


def _target_piece_id(move: Move) -> str | None:
    if move.captured_piece_id is not None:
        return move.captured_piece_id
    target_piece_id = move.metadata.get("target_piece_id")
    if target_piece_id is None:
        return None
    return str(target_piece_id)
