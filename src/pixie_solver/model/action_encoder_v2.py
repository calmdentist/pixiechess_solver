from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import time
from typing import Any

import torch
from torch import Tensor, nn

from pixie_solver.core.move import Move, stable_move_id
from pixie_solver.core.state import GameState
from pixie_solver.model._features import (
    PROMOTION_PIECE_TYPE_IDS,
    json_scalar_features,
    move_kind_id,
    normalize_scalar,
    square_index,
    stable_bucket,
)

ACTION_FEATURES = 8
ACTION_VALUE_FEATURES = 10
ACTION_ID_BUCKETS = 128
ACTION_CLASS_BUCKETS = 64
ACTION_TAG_BUCKETS = 64
ACTION_VALUE_NAMESPACE_BUCKETS = 128
ACTION_VALUE_LABEL_BUCKETS = 256
ACTION_VALUE_PATH_BUCKETS = 256
ACTION_VALUE_DEPTH_BUCKETS = 16
ACTION_VALUE_POSITION_BUCKETS = 64


@dataclass(frozen=True, slots=True)
class ActionValueSpec:
    namespace: str
    label: str
    path: str
    depth: int
    position: int
    numeric_features: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class ActionTokenSpec:
    move_id: str
    action_id: str
    actor_piece_id: str
    actor_class_id: str
    target_piece_id: str | None
    target_class_id: str | None
    action_kind: str
    from_square: str | None
    to_square: str | None
    promotion_piece_type: str | None
    tags: tuple[str, ...]
    param_specs: tuple[ActionValueSpec, ...]
    numeric_features: tuple[float, ...]


@dataclass(slots=True)
class EncodedActionsV2:
    move_ids: tuple[str, ...]
    action_ids: tuple[str, ...]
    action_specs: tuple[ActionTokenSpec, ...]
    candidate_embeddings: Tensor


@dataclass(frozen=True, slots=True)
class ActionEncodingMetricsV2:
    actions_encoded: int = 0
    total_ms: float = 0.0
    total_tag_count: int = 0
    total_param_tokens: int = 0

    def to_dict(self) -> dict[str, float | int]:
        return {
            "actions_encoded": self.actions_encoded,
            "total_ms": self.total_ms,
            "total_tag_count": self.total_tag_count,
            "total_param_tokens": self.total_param_tokens,
        }


class ActionTokenEncoderV2(nn.Module):
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
        self.action_kind_embedding = nn.Embedding(64, d_model)
        self.promotion_embedding = nn.Embedding(len(PROMOTION_PIECE_TYPE_IDS), d_model)
        self.actor_piece_embedding = nn.Embedding(ACTION_ID_BUCKETS + 1, d_model)
        self.actor_class_embedding = nn.Embedding(ACTION_CLASS_BUCKETS + 1, d_model)
        self.target_piece_embedding = nn.Embedding(ACTION_ID_BUCKETS + 1, d_model)
        self.target_class_embedding = nn.Embedding(ACTION_CLASS_BUCKETS + 1, d_model)
        self.tag_embedding = nn.Embedding(ACTION_TAG_BUCKETS + 1, d_model)
        self.value_namespace_embedding = nn.Embedding(
            ACTION_VALUE_NAMESPACE_BUCKETS + 1,
            d_model,
        )
        self.value_label_embedding = nn.Embedding(
            ACTION_VALUE_LABEL_BUCKETS + 1,
            d_model,
        )
        self.value_path_embedding = nn.Embedding(
            ACTION_VALUE_PATH_BUCKETS + 1,
            d_model,
        )
        self.value_depth_embedding = nn.Embedding(
            ACTION_VALUE_DEPTH_BUCKETS + 1,
            d_model,
        )
        self.value_position_embedding = nn.Embedding(
            ACTION_VALUE_POSITION_BUCKETS + 1,
            d_model,
        )
        self.flag_projection = nn.Sequential(
            nn.Linear(ACTION_FEATURES, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, d_model),
        )
        self.value_projection = nn.Sequential(
            nn.Linear(ACTION_VALUE_FEATURES, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, d_model),
        )
        self.output_mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, d_model),
        )
        self.null_target_embedding = nn.Parameter(torch.zeros(d_model, dtype=torch.float32))

    def encode_move(
        self,
        state: GameState,
        move: Move,
    ) -> Tensor:
        return self.encode_moves(state, (move,)).candidate_embeddings[0]

    def encode_moves(
        self,
        state: GameState,
        legal_moves: tuple[Move, ...] | list[Move],
    ) -> EncodedActionsV2:
        return self.encode_moves_with_metrics(state, legal_moves)[0]

    def encode_moves_with_metrics(
        self,
        state: GameState,
        legal_moves: tuple[Move, ...] | list[Move],
    ) -> tuple[EncodedActionsV2, ActionEncodingMetricsV2]:
        start = time.perf_counter()
        if not legal_moves:
            empty = torch.zeros(
                (0, self.d_model),
                dtype=torch.float32,
                device=self.actor_piece_embedding.weight.device,
            )
            return (
                EncodedActionsV2(
                    move_ids=(),
                    action_ids=(),
                    action_specs=(),
                    candidate_embeddings=empty,
                ),
                ActionEncodingMetricsV2(),
            )

        action_specs = action_token_specs(state, legal_moves)
        embeddings = torch.stack(
            [self._encode_action_spec(spec) for spec in action_specs],
            dim=0,
        )
        return (
            EncodedActionsV2(
                move_ids=tuple(spec.move_id for spec in action_specs),
                action_ids=tuple(spec.action_id for spec in action_specs),
                action_specs=action_specs,
                candidate_embeddings=embeddings,
            ),
            ActionEncodingMetricsV2(
                actions_encoded=len(action_specs),
                total_ms=(time.perf_counter() - start) * 1000.0,
                total_tag_count=sum(len(spec.tags) for spec in action_specs),
                total_param_tokens=sum(len(spec.param_specs) for spec in action_specs),
            ),
        )

    def _encode_action_spec(self, spec: ActionTokenSpec) -> Tensor:
        device = self.actor_piece_embedding.weight.device
        token = self._embed_scalar(
            self.actor_piece_embedding,
            stable_bucket("action_actor_piece", spec.actor_piece_id, ACTION_ID_BUCKETS),
        )
        token = token + self._embed_scalar(
            self.actor_class_embedding,
            stable_bucket("action_actor_class", spec.actor_class_id, ACTION_CLASS_BUCKETS),
        )
        if spec.target_piece_id is not None:
            token = token + self._embed_scalar(
                self.target_piece_embedding,
                stable_bucket("action_target_piece", spec.target_piece_id, ACTION_ID_BUCKETS),
            )
        else:
            token = token + self.null_target_embedding
        if spec.target_class_id is not None:
            token = token + self._embed_scalar(
                self.target_class_embedding,
                stable_bucket("action_target_class", spec.target_class_id, ACTION_CLASS_BUCKETS),
            )
        else:
            token = token + self.null_target_embedding
        token = token + self._embed_scalar(
            self.square_embedding,
            square_index(spec.from_square),
        )
        token = token + self._embed_scalar(
            self.square_embedding,
            square_index(spec.to_square),
        )
        token = token + self._embed_scalar(
            self.action_kind_embedding,
            move_kind_id(spec.action_kind),
        )
        token = token + self._embed_scalar(
            self.promotion_embedding,
            PROMOTION_PIECE_TYPE_IDS[spec.promotion_piece_type],
        )
        token = token + self._pooled_tag_embedding(spec.tags, device=device)
        token = token + self._pooled_param_embedding(spec.param_specs, device=device)
        token = token + self.flag_projection(
            torch.tensor(
                spec.numeric_features,
                dtype=torch.float32,
                device=device,
            )
        )
        return self.output_mlp(token)

    def _pooled_tag_embedding(
        self,
        tags: tuple[str, ...],
        *,
        device: torch.device,
    ) -> Tensor:
        if not tags:
            return torch.zeros(self.d_model, dtype=torch.float32, device=device)
        tag_ids = [
            stable_bucket("action_tag", tag, ACTION_TAG_BUCKETS)
            for tag in tags
        ]
        return self.tag_embedding(
            torch.tensor(tag_ids, dtype=torch.long, device=device)
        ).mean(dim=0)

    def _pooled_param_embedding(
        self,
        param_specs: tuple[ActionValueSpec, ...],
        *,
        device: torch.device,
    ) -> Tensor:
        if not param_specs:
            return torch.zeros(self.d_model, dtype=torch.float32, device=device)
        return torch.stack(
            [self._encode_value_spec(spec) for spec in param_specs],
            dim=0,
        ).mean(dim=0)

    def _encode_value_spec(self, spec: ActionValueSpec) -> Tensor:
        device = self.actor_piece_embedding.weight.device
        token = self._embed_scalar(
            self.value_namespace_embedding,
            stable_bucket(
                "action_value_namespace",
                spec.namespace,
                ACTION_VALUE_NAMESPACE_BUCKETS,
            ),
        )
        token = token + self._embed_scalar(
            self.value_label_embedding,
            stable_bucket(
                "action_value_label",
                spec.label,
                ACTION_VALUE_LABEL_BUCKETS,
            ),
        )
        token = token + self._embed_scalar(
            self.value_path_embedding,
            stable_bucket(
                "action_value_path",
                spec.path,
                ACTION_VALUE_PATH_BUCKETS,
            ),
        )
        token = token + self._embed_scalar(
            self.value_depth_embedding,
            min(spec.depth, ACTION_VALUE_DEPTH_BUCKETS),
        )
        token = token + self._embed_scalar(
            self.value_position_embedding,
            min(spec.position + 1, ACTION_VALUE_POSITION_BUCKETS),
        )
        token = token + self.value_projection(
            torch.tensor(
                spec.numeric_features,
                dtype=torch.float32,
                device=device,
            )
        )
        return token

    def _embed_scalar(self, embedding: nn.Embedding, token_id: int) -> Tensor:
        device = embedding.weight.device
        return embedding(torch.tensor(token_id, dtype=torch.long, device=device))


def action_token_specs(
    state: GameState,
    legal_moves: tuple[Move, ...] | list[Move],
) -> tuple[ActionTokenSpec, ...]:
    return tuple(_action_token_spec(state, move) for move in legal_moves)


def _action_token_spec(state: GameState, move: Move) -> ActionTokenSpec:
    action = move.to_action_intent()
    actor_piece = state.piece_instances[action.actor_piece_id]
    target_piece = (
        state.piece_instances.get(action.target_piece_id)
        if action.target_piece_id is not None
        else None
    )
    param_specs = tuple(
        _emit_value_specs(
            namespace="params",
            path="params",
            value=dict(action.params),
            depth=0,
            position=0,
        )
    ) if action.params else ()
    tags = tuple(sorted(action.tags))
    return ActionTokenSpec(
        move_id=stable_move_id(move),
        action_id=action.stable_id(),
        actor_piece_id=action.actor_piece_id,
        actor_class_id=actor_piece.piece_class_id,
        target_piece_id=action.target_piece_id,
        target_class_id=(
            target_piece.piece_class_id
            if target_piece is not None
            else None
        ),
        action_kind=action.action_kind,
        from_square=action.from_square,
        to_square=action.to_square,
        promotion_piece_type=action.promotion_piece_type,
        tags=tags,
        param_specs=param_specs,
        numeric_features=(
            float(action.from_square is not None),
            float(action.to_square is not None),
            float(action.target_piece_id is not None),
            float(action.promotion_piece_type is not None),
            normalize_scalar(len(tags), scale=8.0),
            normalize_scalar(len(param_specs), scale=16.0),
            normalize_scalar(len(action.action_kind), scale=16.0),
            float(target_piece is not None),
        ),
    )


def _emit_value_specs(
    *,
    namespace: str,
    path: str,
    value: Any,
    depth: int,
    position: int,
) -> tuple[ActionValueSpec, ...]:
    specs = [
        ActionValueSpec(
            namespace=namespace,
            label=_label_for_value(value),
            path=path,
            depth=depth,
            position=position,
            numeric_features=_value_numeric_features(
                value=value,
                label=_label_for_value(value),
                depth=depth,
                position=position,
                child_count=_child_count(value),
            ),
        )
    ]
    if isinstance(value, Mapping):
        for child_index, key in enumerate(sorted(value)):
            specs.extend(
                _emit_value_specs(
                    namespace=str(key),
                    path=f"{path}.{key}",
                    value=value[key],
                    depth=depth + 1,
                    position=child_index,
                )
            )
    elif isinstance(value, list):
        for child_index, child in enumerate(value):
            specs.extend(
                _emit_value_specs(
                    namespace=f"{namespace}[]",
                    path=f"{path}[{child_index}]",
                    value=child,
                    depth=depth + 1,
                    position=child_index,
                )
            )
    return tuple(specs)


def _value_numeric_features(
    *,
    value: Any,
    label: str,
    depth: int,
    position: int,
    child_count: int,
) -> tuple[float, ...]:
    return (
        *json_scalar_features(value),
        normalize_scalar(depth, scale=8.0),
        normalize_scalar(position, scale=16.0),
        normalize_scalar(child_count, scale=12.0),
        1.0 if child_count else 0.0,
        normalize_scalar(len(label), scale=32.0),
    )


def _label_for_value(value: Any) -> str:
    if isinstance(value, Mapping):
        return "dict"
    if isinstance(value, list):
        return "list"
    return repr(value)


def _child_count(value: Any) -> int:
    if isinstance(value, (Mapping, list)):
        return len(value)
    return 0
