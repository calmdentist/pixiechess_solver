from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from pixie_solver.core.piece import PieceClass
from pixie_solver.model._features import json_scalar_features, normalize_scalar, stable_bucket
from pixie_solver.program.compiler import compile_program_ir
from pixie_solver.program.lower_legacy_dsl import lower_legacy_piece_class

PROGRAM_TOKEN_ROLE_IDS = {
    "program": 1,
    "state_field": 2,
    "constant": 3,
    "action_block": 4,
    "query_block": 5,
    "reaction_block": 6,
    "trigger": 7,
    "condition": 8,
    "effect": 9,
    "arg": 10,
}
PROGRAM_TOKEN_NUMERIC_FEATURES = 10
PROGRAM_NAMESPACE_BUCKETS = 128
PROGRAM_LABEL_BUCKETS = 256
PROGRAM_PATH_BUCKETS = 512
PROGRAM_DEPTH_BUCKETS = 32
PROGRAM_POSITION_BUCKETS = 128


@dataclass(frozen=True, slots=True)
class ProgramTokenSpec:
    role: str
    namespace: str
    label: str
    path: str
    depth: int
    position: int
    numeric_features: tuple[float, ...]


@dataclass(slots=True)
class EncodedProgramBatch:
    program_ids: tuple[str, ...]
    programs: tuple[dict[str, Any], ...]
    token_specs: tuple[tuple[ProgramTokenSpec, ...], ...]
    token_embeddings: Tensor
    padding_mask: Tensor


class ProgramIRTokenEncoder(nn.Module):
    def __init__(
        self,
        *,
        d_model: int = 192,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        inner_dim = d_model * 2 if hidden_dim is None else hidden_dim
        self.d_model = d_model
        self.role_embedding = nn.Embedding(len(PROGRAM_TOKEN_ROLE_IDS) + 1, d_model)
        self.namespace_embedding = nn.Embedding(PROGRAM_NAMESPACE_BUCKETS + 1, d_model)
        self.label_embedding = nn.Embedding(PROGRAM_LABEL_BUCKETS + 1, d_model)
        self.path_embedding = nn.Embedding(PROGRAM_PATH_BUCKETS + 1, d_model)
        self.depth_embedding = nn.Embedding(PROGRAM_DEPTH_BUCKETS + 1, d_model)
        self.position_embedding = nn.Embedding(PROGRAM_POSITION_BUCKETS + 1, d_model)
        self.numeric_projection = nn.Sequential(
            nn.Linear(PROGRAM_TOKEN_NUMERIC_FEATURES, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, d_model),
        )
        self.output_mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, d_model),
        )

    def tokenize_program(
        self,
        program_source: Mapping[str, Any] | PieceClass,
    ) -> tuple[ProgramTokenSpec, ...]:
        return program_token_specs(program_source)

    def forward(
        self,
        program_sources: Sequence[Mapping[str, Any] | PieceClass],
    ) -> EncodedProgramBatch:
        device = self.role_embedding.weight.device
        if not program_sources:
            empty_tokens = torch.zeros((0, 0, self.d_model), dtype=torch.float32, device=device)
            empty_mask = torch.zeros((0, 0), dtype=torch.bool, device=device)
            return EncodedProgramBatch(
                program_ids=(),
                programs=(),
                token_specs=(),
                token_embeddings=empty_tokens,
                padding_mask=empty_mask,
            )

        canonical_programs = [_canonical_program_ir(source) for source in program_sources]
        token_specs = tuple(
            _program_token_specs_from_canonical(program)
            for program in canonical_programs
        )
        program_ids = tuple(str(program["program_id"]) for program in canonical_programs)
        max_tokens = max(len(specs) for specs in token_specs)
        token_rows: list[Tensor] = []
        padding_rows: list[Tensor] = []
        for specs in token_specs:
            encoded_tokens = torch.stack(
                [self._encode_token(spec) for spec in specs],
                dim=0,
            )
            padding_length = max_tokens - len(specs)
            if padding_length:
                padding = torch.zeros(
                    (padding_length, self.d_model),
                    dtype=torch.float32,
                    device=device,
                )
                encoded_tokens = torch.cat((encoded_tokens, padding), dim=0)
            token_rows.append(encoded_tokens)
            padding_rows.append(
                torch.tensor(
                    [False] * len(specs) + [True] * padding_length,
                    dtype=torch.bool,
                    device=device,
                )
            )

        return EncodedProgramBatch(
            program_ids=program_ids,
            programs=tuple(canonical_programs),
            token_specs=token_specs,
            token_embeddings=torch.stack(token_rows, dim=0),
            padding_mask=torch.stack(padding_rows, dim=0),
        )

    def _encode_token(self, spec: ProgramTokenSpec) -> Tensor:
        device = self.role_embedding.weight.device
        combined = self.role_embedding(
            torch.tensor(
                PROGRAM_TOKEN_ROLE_IDS[spec.role],
                dtype=torch.long,
                device=device,
            )
        )
        combined = combined + self.namespace_embedding(
            torch.tensor(
                stable_bucket(
                    "program_token_namespace",
                    spec.namespace,
                    PROGRAM_NAMESPACE_BUCKETS,
                ),
                dtype=torch.long,
                device=device,
            )
        )
        combined = combined + self.label_embedding(
            torch.tensor(
                stable_bucket(
                    "program_token_label",
                    spec.label,
                    PROGRAM_LABEL_BUCKETS,
                ),
                dtype=torch.long,
                device=device,
            )
        )
        combined = combined + self.path_embedding(
            torch.tensor(
                stable_bucket(
                    "program_token_path",
                    spec.path,
                    PROGRAM_PATH_BUCKETS,
                ),
                dtype=torch.long,
                device=device,
            )
        )
        combined = combined + self.depth_embedding(
            torch.tensor(
                min(spec.depth, PROGRAM_DEPTH_BUCKETS),
                dtype=torch.long,
                device=device,
            )
        )
        combined = combined + self.position_embedding(
            torch.tensor(
                min(spec.position + 1, PROGRAM_POSITION_BUCKETS),
                dtype=torch.long,
                device=device,
            )
        )
        combined = combined + self.numeric_projection(
            torch.tensor(
                spec.numeric_features,
                dtype=torch.float32,
                device=device,
            )
        )
        return self.output_mlp(combined)


def program_token_specs(
    program_source: Mapping[str, Any] | PieceClass,
) -> tuple[ProgramTokenSpec, ...]:
    return _program_token_specs_from_canonical(_canonical_program_ir(program_source))


def _canonical_program_ir(
    program_source: Mapping[str, Any] | PieceClass,
) -> dict[str, Any]:
    if isinstance(program_source, PieceClass):
        return lower_legacy_piece_class(program_source)
    return compile_program_ir(program_source)


def _program_token_specs_from_canonical(
    program: Mapping[str, Any],
) -> tuple[ProgramTokenSpec, ...]:
    tokens: list[ProgramTokenSpec] = []
    total_children = (
        len(program.get("state_schema", []))
        + len(program.get("constants", {}))
        + len(program.get("action_blocks", []))
        + len(program.get("query_blocks", []))
        + len(program.get("reaction_blocks", []))
    )
    tokens.append(
        _build_token(
            role="program",
            namespace="base_archetype",
            label=str(program["base_archetype"]),
            path="program",
            value=program["base_archetype"],
            depth=0,
            position=0,
            child_count=total_children,
        )
    )

    for index, field_spec in enumerate(program.get("state_schema", [])):
        field_path = f"state_schema[{index}]"
        tokens.append(
            _build_token(
                role="state_field",
                namespace=str(field_spec["type"]),
                label=str(field_spec["name"]),
                path=field_path,
                value=field_spec["default"],
                depth=1,
                position=index,
                child_count=2,
            )
        )
        tokens.extend(
            _emit_value_tokens(
                role="arg",
                namespace="state_default",
                path=f"{field_path}.default",
                value=field_spec["default"],
                depth=2,
                position=0,
            )
        )
        if field_spec.get("description", ""):
            tokens.extend(
                _emit_value_tokens(
                    role="arg",
                    namespace="state_description",
                    path=f"{field_path}.description",
                    value=str(field_spec["description"]),
                    depth=2,
                    position=1,
                )
            )

    if program.get("constants"):
        tokens.extend(
            _emit_value_tokens(
                role="constant",
                namespace="constants",
                path="constants",
                value=dict(program["constants"]),
                depth=1,
                position=0,
            )
        )

    for index, block in enumerate(program.get("action_blocks", [])):
        block_path = f"action_blocks[{index}]"
        tokens.append(
            _build_token(
                role="action_block",
                namespace=str(block["kind"]),
                label=str(block["block_id"]),
                path=block_path,
                value=block["kind"],
                depth=1,
                position=index,
                child_count=1,
            )
        )
        tokens.extend(
            _emit_value_tokens(
                role="arg",
                namespace="action_params",
                path=f"{block_path}.params",
                value=dict(block["params"]),
                depth=2,
                position=0,
            )
        )

    for index, block in enumerate(program.get("query_blocks", [])):
        block_path = f"query_blocks[{index}]"
        tokens.append(
            _build_token(
                role="query_block",
                namespace=str(block["kind"]),
                label=str(block["block_id"]),
                path=block_path,
                value=block["kind"],
                depth=1,
                position=index,
                child_count=1,
            )
        )
        tokens.extend(
            _emit_value_tokens(
                role="arg",
                namespace="query_params",
                path=f"{block_path}.params",
                value=dict(block["params"]),
                depth=2,
                position=0,
            )
        )

    for block_index, block in enumerate(program.get("reaction_blocks", [])):
        block_path = f"reaction_blocks[{block_index}]"
        child_count = 2 + len(block.get("conditions", [])) + len(block.get("effects", []))
        tokens.append(
            _build_token(
                role="reaction_block",
                namespace=str(block["kind"]),
                label=str(block["block_id"]),
                path=block_path,
                value=block["kind"],
                depth=1,
                position=block_index,
                child_count=child_count,
            )
        )
        tokens.append(
            _build_token(
                role="trigger",
                namespace="event_type",
                label=str(block["trigger"]["event_type"]),
                path=f"{block_path}.trigger",
                value=block["trigger"]["event_type"],
                depth=2,
                position=0,
                child_count=1 if block["trigger"].get("payload_filter") else 0,
            )
        )
        if block["trigger"].get("payload_filter"):
            tokens.extend(
                _emit_value_tokens(
                    role="arg",
                    namespace="trigger_payload_filter",
                    path=f"{block_path}.trigger.payload_filter",
                    value=dict(block["trigger"]["payload_filter"]),
                    depth=3,
                    position=0,
                )
            )
        tokens.extend(
            _emit_value_tokens(
                role="arg",
                namespace="reaction_priority",
                path=f"{block_path}.priority",
                value=int(block.get("priority", 0)),
                depth=2,
                position=1,
            )
        )
        for condition_index, condition in enumerate(block.get("conditions", [])):
            condition_path = f"{block_path}.conditions[{condition_index}]"
            tokens.append(
                _build_token(
                    role="condition",
                    namespace=str(condition["condition_kind"]),
                    label=str(condition["condition_kind"]),
                    path=condition_path,
                    value=condition["condition_kind"],
                    depth=2,
                    position=condition_index + 2,
                    child_count=1,
                )
            )
            tokens.extend(
                _emit_value_tokens(
                    role="arg",
                    namespace="condition_args",
                    path=f"{condition_path}.args",
                    value=dict(condition.get("args", {})),
                    depth=3,
                    position=0,
                )
            )
        for effect_index, effect in enumerate(block.get("effects", [])):
            effect_path = f"{block_path}.effects[{effect_index}]"
            tokens.append(
                _build_token(
                    role="effect",
                    namespace=str(effect["effect_kind"]),
                    label=str(effect["effect_kind"]),
                    path=effect_path,
                    value=effect["effect_kind"],
                    depth=2,
                    position=effect_index + 2 + len(block.get("conditions", [])),
                    child_count=1,
                )
            )
            tokens.extend(
                _emit_value_tokens(
                    role="arg",
                    namespace="effect_args",
                    path=f"{effect_path}.args",
                    value=dict(effect.get("args", {})),
                    depth=3,
                    position=0,
                )
            )
    return tuple(tokens)


def _emit_value_tokens(
    *,
    role: str,
    namespace: str,
    path: str,
    value: Any,
    depth: int,
    position: int,
) -> tuple[ProgramTokenSpec, ...]:
    tokens = [
        _build_token(
            role=role,
            namespace=namespace,
            label=_label_for_value(value),
            path=path,
            value=value,
            depth=depth,
            position=position,
            child_count=_child_count(value),
        )
    ]
    if isinstance(value, Mapping):
        for child_index, key in enumerate(sorted(value)):
            child_value = value[key]
            tokens.extend(
                _emit_value_tokens(
                    role=role,
                    namespace=str(key),
                    path=f"{path}.{key}",
                    value=child_value,
                    depth=depth + 1,
                    position=child_index,
                )
            )
    elif isinstance(value, list):
        for child_index, child_value in enumerate(value):
            tokens.extend(
                _emit_value_tokens(
                    role=role,
                    namespace=f"{namespace}[]",
                    path=f"{path}[{child_index}]",
                    value=child_value,
                    depth=depth + 1,
                    position=child_index,
                )
            )
    return tuple(tokens)


def _build_token(
    *,
    role: str,
    namespace: str,
    label: str,
    path: str,
    value: Any,
    depth: int,
    position: int,
    child_count: int,
) -> ProgramTokenSpec:
    return ProgramTokenSpec(
        role=role,
        namespace=namespace,
        label=label,
        path=path,
        depth=depth,
        position=position,
        numeric_features=_numeric_features(
            value=value,
            label=label,
            depth=depth,
            position=position,
            child_count=child_count,
        ),
    )


def _numeric_features(
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
    if isinstance(value, Mapping | list):
        return len(value)
    return 0
