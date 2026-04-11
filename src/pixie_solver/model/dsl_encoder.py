from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from pixie_solver.core.piece import PieceClass
from pixie_solver.model._features import (
    BASE_PIECE_TYPE_IDS,
    FIELD_TYPE_IDS,
    json_scalar_features,
    normalize_scalar,
    stable_bucket,
)
from pixie_solver.utils.serialization import JsonValue

OP_BUCKETS = 128
EVENT_BUCKETS = 64
STATE_NAME_BUCKETS = 64
NUMERIC_FEATURES = 14


@dataclass(frozen=True, slots=True)
class PieceClassFeatureSpec:
    base_piece_type_id: int
    movement_op_ids: tuple[int, ...]
    capture_op_ids: tuple[int, ...]
    hook_event_ids: tuple[int, ...]
    condition_op_ids: tuple[int, ...]
    effect_op_ids: tuple[int, ...]
    state_name_ids: tuple[int, ...]
    state_type_ids: tuple[int, ...]
    numeric_features: tuple[float, ...]


class DSLFeatureEncoder(nn.Module):
    def __init__(
        self,
        *,
        d_model: int = 192,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        inner_dim = d_model * 2 if hidden_dim is None else hidden_dim
        self.d_model = d_model
        self.base_piece_embedding = nn.Embedding(len(BASE_PIECE_TYPE_IDS) + 1, d_model)
        self.movement_op_embedding = nn.Embedding(OP_BUCKETS + 1, d_model)
        self.capture_op_embedding = nn.Embedding(OP_BUCKETS + 1, d_model)
        self.hook_event_embedding = nn.Embedding(EVENT_BUCKETS + 1, d_model)
        self.condition_op_embedding = nn.Embedding(OP_BUCKETS + 1, d_model)
        self.effect_op_embedding = nn.Embedding(OP_BUCKETS + 1, d_model)
        self.state_name_embedding = nn.Embedding(STATE_NAME_BUCKETS + 1, d_model)
        self.state_type_embedding = nn.Embedding(len(FIELD_TYPE_IDS) + 1, d_model)
        self.numeric_projection = nn.Sequential(
            nn.Linear(NUMERIC_FEATURES, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, d_model),
        )
        self.output_mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, d_model),
        )

    def encode_piece_class(self, piece_class: PieceClass) -> Tensor:
        return self.forward((piece_class,))[0]

    def forward(self, piece_classes: Sequence[PieceClass]) -> Tensor:
        device = self.base_piece_embedding.weight.device
        if not piece_classes:
            return torch.zeros((0, self.d_model), dtype=torch.float32, device=device)

        encoded_rows = [self._encode_spec(piece_class_feature_spec(piece_class)) for piece_class in piece_classes]
        return torch.stack(encoded_rows, dim=0)

    def _encode_spec(self, spec: PieceClassFeatureSpec) -> Tensor:
        device = self.base_piece_embedding.weight.device
        combined = self._single_embedding(self.base_piece_embedding, spec.base_piece_type_id)
        combined = combined + self._pooled_embedding(
            self.movement_op_embedding,
            spec.movement_op_ids,
            device=device,
        )
        combined = combined + self._pooled_embedding(
            self.capture_op_embedding,
            spec.capture_op_ids,
            device=device,
        )
        combined = combined + self._pooled_embedding(
            self.hook_event_embedding,
            spec.hook_event_ids,
            device=device,
        )
        combined = combined + self._pooled_embedding(
            self.condition_op_embedding,
            spec.condition_op_ids,
            device=device,
        )
        combined = combined + self._pooled_embedding(
            self.effect_op_embedding,
            spec.effect_op_ids,
            device=device,
        )
        combined = combined + self._pooled_embedding(
            self.state_name_embedding,
            spec.state_name_ids,
            device=device,
        )
        combined = combined + self._pooled_embedding(
            self.state_type_embedding,
            spec.state_type_ids,
            device=device,
        )
        numeric_tensor = torch.tensor(
            spec.numeric_features,
            dtype=torch.float32,
            device=device,
        )
        combined = combined + self.numeric_projection(numeric_tensor)
        return self.output_mlp(combined)

    def _single_embedding(self, embedding: nn.Embedding, token_id: int) -> Tensor:
        device = embedding.weight.device
        token_tensor = torch.tensor(token_id, dtype=torch.long, device=device)
        return embedding(token_tensor)

    def _pooled_embedding(
        self,
        embedding: nn.Embedding,
        token_ids: Iterable[int],
        *,
        device: torch.device,
    ) -> Tensor:
        ids = tuple(token_ids)
        if not ids:
            return torch.zeros(self.d_model, dtype=torch.float32, device=device)
        token_tensor = torch.tensor(ids, dtype=torch.long, device=device)
        return embedding(token_tensor).mean(dim=0)


def piece_class_feature_spec(piece_class: PieceClass) -> PieceClassFeatureSpec:
    movement_op_ids = tuple(
        stable_bucket("movement_op", modifier.op, OP_BUCKETS)
        for modifier in piece_class.movement_modifiers
    )
    capture_op_ids = tuple(
        stable_bucket("capture_op", modifier.op, OP_BUCKETS)
        for modifier in piece_class.capture_modifiers
    )
    hook_event_ids = tuple(
        stable_bucket("hook_event", hook.event, EVENT_BUCKETS)
        for hook in piece_class.hooks
    )
    condition_op_ids = tuple(
        stable_bucket("condition_op", condition.op, OP_BUCKETS)
        for hook in piece_class.hooks
        for condition in hook.conditions
    )
    effect_op_ids = tuple(
        stable_bucket("effect_op", effect.op, OP_BUCKETS)
        for hook in piece_class.hooks
        for effect in hook.effects
    )
    state_name_ids = tuple(
        stable_bucket("state_field", field_spec.name, STATE_NAME_BUCKETS)
        for field_spec in piece_class.instance_state_schema
    )
    state_type_ids = tuple(
        FIELD_TYPE_IDS[field_spec.field_type]
        for field_spec in piece_class.instance_state_schema
    )

    hook_conditions = sum(len(hook.conditions) for hook in piece_class.hooks)
    hook_effects = sum(len(hook.effects) for hook in piece_class.hooks)
    hook_priorities = [hook.priority for hook in piece_class.hooks]
    arg_summary = _json_summary(
        modifier.args for modifier in piece_class.movement_modifiers
    )
    capture_arg_summary = _json_summary(
        modifier.args for modifier in piece_class.capture_modifiers
    )
    hook_metadata_summary = _json_summary(hook.metadata for hook in piece_class.hooks)
    state_default_summary = _json_summary(
        {"default": field_spec.default}
        for field_spec in piece_class.instance_state_schema
    )

    numeric_features = (
        normalize_scalar(len(piece_class.movement_modifiers), scale=6.0),
        normalize_scalar(len(piece_class.capture_modifiers), scale=6.0),
        normalize_scalar(len(piece_class.hooks), scale=6.0),
        normalize_scalar(hook_conditions, scale=12.0),
        normalize_scalar(hook_effects, scale=12.0),
        normalize_scalar(len(piece_class.instance_state_schema), scale=8.0),
        normalize_scalar(arg_summary["numeric_count"], scale=12.0),
        normalize_scalar(arg_summary["numeric_sum"], scale=16.0),
        normalize_scalar(capture_arg_summary["numeric_sum"], scale=16.0),
        normalize_scalar(state_default_summary["numeric_sum"], scale=16.0),
        normalize_scalar(
            sum(hook_priorities) / len(hook_priorities) if hook_priorities else 0.0,
            scale=4.0,
        ),
        normalize_scalar(hook_metadata_summary["string_count"], scale=8.0),
        normalize_scalar(
            state_default_summary["bool_count"] + state_default_summary["string_count"],
            scale=8.0,
        ),
        normalize_scalar(len(piece_class.metadata), scale=8.0),
    )
    return PieceClassFeatureSpec(
        base_piece_type_id=BASE_PIECE_TYPE_IDS[piece_class.base_piece_type],
        movement_op_ids=movement_op_ids,
        capture_op_ids=capture_op_ids,
        hook_event_ids=hook_event_ids,
        condition_op_ids=condition_op_ids,
        effect_op_ids=effect_op_ids,
        state_name_ids=state_name_ids,
        state_type_ids=state_type_ids,
        numeric_features=numeric_features,
    )


def _json_summary(values: Iterable[dict[str, JsonValue]]) -> dict[str, float]:
    numeric_values: list[float] = []
    bool_count = 0
    string_count = 0
    for value in values:
        for scalar in _iter_json_scalars(value):
            if isinstance(scalar, bool):
                bool_count += 1
            elif isinstance(scalar, (int, float)):
                numeric_values.append(float(scalar))
            elif isinstance(scalar, str):
                string_count += 1

    return {
        "numeric_count": float(len(numeric_values)),
        "numeric_sum": float(sum(numeric_values)),
        "bool_count": float(bool_count),
        "string_count": float(string_count),
    }


def _iter_json_scalars(value: JsonValue) -> Iterable[JsonValue]:
    if isinstance(value, dict):
        for nested_value in value.values():
            yield from _iter_json_scalars(nested_value)
        return
    if isinstance(value, list):
        for nested_value in value:
            yield from _iter_json_scalars(nested_value)
        return
    yield value
