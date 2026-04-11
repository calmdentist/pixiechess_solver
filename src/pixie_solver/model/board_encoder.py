from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from pixie_solver.core.piece import BasePieceType, PieceClass, PieceInstance
from pixie_solver.core.state import GameState
from pixie_solver.model._features import (
    BASE_PIECE_TYPE_IDS,
    COLOR_IDS,
    FIELD_TYPE_IDS,
    json_scalar_features,
    json_value_token,
    normalize_scalar,
    square_index,
)
from pixie_solver.model.dsl_encoder import DSLFeatureEncoder

STATE_VALUE_BUCKETS = 128
STATE_FEATURES = 11
GLOBAL_CLOCK_FEATURES = 6


@dataclass(slots=True)
class EncodedBoard:
    piece_ids: tuple[str, ...]
    piece_tokens: Tensor
    global_token: Tensor
    piece_index_by_id: dict[str, int]


class BoardEncoder(nn.Module):
    def __init__(
        self,
        *,
        d_model: int = 192,
        dsl_encoder: DSLFeatureEncoder | None = None,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        inner_dim = d_model * 2 if hidden_dim is None else hidden_dim
        self.d_model = d_model
        self.dsl_encoder = DSLFeatureEncoder(d_model=d_model) if dsl_encoder is None else dsl_encoder
        self.square_embedding = nn.Embedding(65, d_model)
        self.color_embedding = nn.Embedding(len(COLOR_IDS) + 1, d_model)
        self.base_piece_embedding = nn.Embedding(len(BASE_PIECE_TYPE_IDS) + 1, d_model)
        self.side_to_move_embedding = nn.Embedding(len(COLOR_IDS) + 1, d_model)
        self.en_passant_embedding = nn.Embedding(65, d_model)
        self.state_name_embedding = nn.Embedding(64 + 1, d_model)
        self.state_type_embedding = nn.Embedding(len(FIELD_TYPE_IDS) + 1, d_model)
        self.state_value_embedding = nn.Embedding(STATE_VALUE_BUCKETS + 1, d_model)
        self.state_scalar_projection = nn.Sequential(
            nn.Linear(STATE_FEATURES, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, d_model),
        )
        self.castling_projection = nn.Sequential(
            nn.Linear(4, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, d_model),
        )
        self.clock_projection = nn.Sequential(
            nn.Linear(GLOBAL_CLOCK_FEATURES, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, d_model),
        )
        self.piece_token_mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, d_model),
        )
        self.global_token_mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, d_model),
        )

    def encode_state(self, state: GameState) -> EncodedBoard:
        return self.forward(state)

    def forward(self, state: GameState) -> EncodedBoard:
        device = self.square_embedding.weight.device
        active_piece_ids = tuple(
            piece_id
            for piece_id, piece in sorted(state.piece_instances.items())
            if piece.is_active
        )
        active_pieces = [state.piece_instances[piece_id] for piece_id in active_piece_ids]
        piece_index_by_id = {
            piece_id: index
            for index, piece_id in enumerate(active_piece_ids)
        }

        unique_classes: dict[str, PieceClass] = {}
        for piece in active_pieces:
            unique_classes[piece.piece_class_id] = state.piece_classes[piece.piece_class_id]
        class_ids = tuple(sorted(unique_classes))
        class_embeddings = self.dsl_encoder(
            [unique_classes[class_id] for class_id in class_ids]
        )
        class_embedding_by_id = {
            class_id: class_embeddings[index]
            for index, class_id in enumerate(class_ids)
        }

        piece_tokens: list[Tensor] = []
        for piece in active_pieces:
            piece_class = state.piece_classes[piece.piece_class_id]
            token = class_embedding_by_id[piece.piece_class_id]
            token = token + self._embed_scalar(
                self.square_embedding,
                square_index(piece.square),
            )
            token = token + self._embed_scalar(
                self.color_embedding,
                COLOR_IDS[piece.color],
            )
            token = token + self._embed_scalar(
                self.base_piece_embedding,
                BASE_PIECE_TYPE_IDS[piece_class.base_piece_type],
            )
            token = token + self._encode_piece_state(piece, piece_class, device=device)
            piece_tokens.append(self.piece_token_mlp(token))

        if piece_tokens:
            piece_token_tensor = torch.stack(piece_tokens, dim=0)
        else:
            piece_token_tensor = torch.zeros(
                (0, self.d_model),
                dtype=torch.float32,
                device=device,
            )

        global_token = self._embed_scalar(
            self.side_to_move_embedding,
            COLOR_IDS[state.side_to_move],
        )
        global_token = global_token + self._embed_scalar(
            self.en_passant_embedding,
            square_index(state.en_passant_square),
        )
        global_token = global_token + self.castling_projection(
            torch.tensor(
                [
                    float("king" in state.castling_rights.get("white", ())),
                    float("queen" in state.castling_rights.get("white", ())),
                    float("king" in state.castling_rights.get("black", ())),
                    float("queen" in state.castling_rights.get("black", ())),
                ],
                dtype=torch.float32,
                device=device,
            )
        )
        repetition_max = max(state.repetition_counts.values(), default=0)
        global_token = global_token + self.clock_projection(
            torch.tensor(
                [
                    normalize_scalar(state.halfmove_clock, scale=50.0),
                    normalize_scalar(state.fullmove_number, scale=80.0),
                    normalize_scalar(len(active_piece_ids), scale=32.0),
                    normalize_scalar(len(state.pending_events), scale=12.0),
                    normalize_scalar(len(state.repetition_counts), scale=16.0),
                    normalize_scalar(repetition_max, scale=4.0),
                ],
                dtype=torch.float32,
                device=device,
            )
        )

        return EncodedBoard(
            piece_ids=active_piece_ids,
            piece_tokens=piece_token_tensor,
            global_token=self.global_token_mlp(global_token),
            piece_index_by_id=piece_index_by_id,
        )

    def _encode_piece_state(
        self,
        piece: PieceInstance,
        piece_class: PieceClass,
        *,
        device: torch.device,
    ) -> Tensor:
        if not piece_class.instance_state_schema:
            return torch.zeros(self.d_model, dtype=torch.float32, device=device)

        field_tokens: list[Tensor] = []
        for field_spec in piece_class.instance_state_schema:
            value = piece.state.get(field_spec.name, field_spec.default)
            scalar_features = (
                *json_scalar_features(value),
                *json_scalar_features(field_spec.default),
                1.0 if value != field_spec.default else 0.0,
            )
            field_token = self._embed_scalar(
                self.state_name_embedding,
                json_value_token(
                    field_spec.name,
                    namespace="state_field_name",
                    bucket_count=64,
                ),
            )
            field_token = field_token + self._embed_scalar(
                self.state_type_embedding,
                FIELD_TYPE_IDS[field_spec.field_type],
            )
            field_token = field_token + self._embed_scalar(
                self.state_value_embedding,
                json_value_token(
                    value,
                    namespace="piece_state_value",
                    bucket_count=STATE_VALUE_BUCKETS,
                ),
            )
            field_token = field_token + self.state_scalar_projection(
                torch.tensor(
                    scalar_features,
                    dtype=torch.float32,
                    device=device,
                )
            )
            field_tokens.append(field_token)

        return torch.stack(field_tokens, dim=0).mean(dim=0)

    def _embed_scalar(self, embedding: nn.Embedding, token_id: int) -> Tensor:
        device = embedding.weight.device
        return embedding(torch.tensor(token_id, dtype=torch.long, device=device))
