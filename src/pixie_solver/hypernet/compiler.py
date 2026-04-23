from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import time

import torch
from torch import Tensor, nn

from pixie_solver.core.hash import stable_digest
from pixie_solver.core.piece import PieceClass
from pixie_solver.core.state import GameState
from pixie_solver.hypernet.schema import AdapterBundle, GatingValues, LayerModulation
from pixie_solver.model._features import stable_bucket
from pixie_solver.model.program_encoder import ProgramIRTokenEncoder
from pixie_solver.program.lower_legacy_dsl import lower_legacy_piece_class
from pixie_solver.strategy import (
    StrategyHypothesis,
    canonicalize_strategy_hypothesis,
    strategy_digest as compute_strategy_digest,
)

STRATEGY_DIGEST_BUCKETS = 256
STRATEGY_SCOPE_BUCKETS = 64
DEFAULT_ADAPTER_TARGET_LAYERS = (
    "context_input",
    "context_output",
    "action_input",
    "action_output",
)


@dataclass(frozen=True, slots=True)
class CompiledAdapterMetrics:
    compile_ms: float = 0.0
    program_encode_ms: float = 0.0
    program_count: int = 0
    target_layer_count: int = 0

    def to_dict(self) -> dict[str, float | int]:
        return {
            "compile_ms": self.compile_ms,
            "program_encode_ms": self.program_encode_ms,
            "program_count": self.program_count,
            "target_layer_count": self.target_layer_count,
        }


class WorldCompilerHypernetwork(nn.Module):
    def __init__(
        self,
        *,
        d_model: int = 192,
        hidden_dim: int | None = None,
        target_layers: Sequence[str] = DEFAULT_ADAPTER_TARGET_LAYERS,
    ) -> None:
        super().__init__()
        inner_dim = d_model * 2 if hidden_dim is None else hidden_dim
        self.d_model = d_model
        self.target_layers = tuple(target_layers)
        self.program_encoder = ProgramIRTokenEncoder(
            d_model=d_model,
            hidden_dim=inner_dim,
        )
        self.strategy_digest_embedding = nn.Embedding(
            STRATEGY_DIGEST_BUCKETS + 1,
            d_model,
        )
        self.strategy_scope_embedding = nn.Embedding(
            STRATEGY_SCOPE_BUCKETS + 1,
            d_model,
        )
        self.strategy_summary_projection = nn.Sequential(
            nn.Linear(8, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, d_model),
        )
        self.compile_projection = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, d_model),
        )
        output_dim = len(self.target_layers) * d_model
        self.scale_head = nn.Linear(d_model, output_dim)
        self.shift_head = nn.Linear(d_model, output_dim)
        self.gate_head = nn.Linear(d_model, output_dim)
        nn.init.zeros_(self.scale_head.weight)
        nn.init.zeros_(self.scale_head.bias)
        nn.init.zeros_(self.shift_head.weight)
        nn.init.zeros_(self.shift_head.bias)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.zeros_(self.gate_head.bias)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def world_digest_for_state(self, state: GameState) -> str:
        return self.world_digest_for_piece_classes(state.piece_classes)

    def world_digest_for_piece_classes(
        self,
        piece_classes: Mapping[str, PieceClass] | Sequence[PieceClass],
    ) -> str:
        canonical_programs = tuple(
            lower_legacy_piece_class(piece_class)
            for piece_class in _sorted_piece_classes(piece_classes)
        )
        return stable_digest(canonical_programs)

    def compile_from_state(
        self,
        state: GameState,
        *,
        strategy: StrategyHypothesis | Mapping[str, object] | None = None,
        strategy_digest: str | None = None,
    ) -> AdapterBundle:
        return self.compile_from_state_with_metrics(
            state,
            strategy=strategy,
            strategy_digest=strategy_digest,
        )[0]

    def compile_from_state_with_metrics(
        self,
        state: GameState,
        *,
        strategy: StrategyHypothesis | Mapping[str, object] | None = None,
        strategy_digest: str | None = None,
    ) -> tuple[AdapterBundle, CompiledAdapterMetrics]:
        return self.compile_from_piece_classes_with_metrics(
            state.piece_classes,
            strategy=strategy,
            strategy_digest=strategy_digest,
        )

    def compile_from_piece_classes_with_metrics(
        self,
        piece_classes: Mapping[str, PieceClass] | Sequence[PieceClass],
        *,
        strategy: StrategyHypothesis | Mapping[str, object] | None = None,
        strategy_digest: str | None = None,
    ) -> tuple[AdapterBundle, CompiledAdapterMetrics]:
        total_start = time.perf_counter()
        sorted_piece_classes = _sorted_piece_classes(piece_classes)
        program_encode_start = time.perf_counter()
        encoded_programs = self.program_encoder(sorted_piece_classes)
        program_encode_ms = (time.perf_counter() - program_encode_start) * 1000.0
        canonical_strategy = (
            canonicalize_strategy_hypothesis(strategy)
            if strategy is not None
            else None
        )
        effective_strategy_digest = (
            compute_strategy_digest(canonical_strategy)
            if canonical_strategy is not None
            else strategy_digest
        )
        compile_embedding = self._compile_embedding(
            encoded_programs.token_embeddings,
            encoded_programs.padding_mask,
            strategy=canonical_strategy,
            strategy_digest=effective_strategy_digest,
        )
        scale_values = self._reshape_scale(self.scale_head(compile_embedding))
        shift_values = self._reshape_shift(self.shift_head(compile_embedding))
        gate_values = self._reshape_gate(self.gate_head(compile_embedding))
        world_digest = self.world_digest_for_piece_classes(sorted_piece_classes)
        bundle = AdapterBundle(
            bundle_id=_bundle_id(world_digest, effective_strategy_digest),
            world_digest=world_digest,
            strategy_digest=effective_strategy_digest,
            layer_modulations=tuple(
                LayerModulation(
                    layer_name=layer_name,
                    scale=tuple(scale_values[layer_index].detach().cpu().tolist()),
                    shift=tuple(shift_values[layer_index].detach().cpu().tolist()),
                )
                for layer_index, layer_name in enumerate(self.target_layers)
            ),
            gating_values=tuple(
                GatingValues(
                    layer_name=layer_name,
                    values=tuple(gate_values[layer_index].detach().cpu().tolist()),
                )
                for layer_index, layer_name in enumerate(self.target_layers)
            ),
            metadata={
                "program_ids": [
                    piece_class.class_id
                    for piece_class in sorted_piece_classes
                ],
                "strategy_scope": (
                    canonical_strategy["scope"]
                    if canonical_strategy is not None
                    else None
                ),
                "target_layers": list(self.target_layers),
                "program_count": len(sorted_piece_classes),
            },
        )
        return bundle, CompiledAdapterMetrics(
            compile_ms=(time.perf_counter() - total_start) * 1000.0,
            program_encode_ms=program_encode_ms,
            program_count=len(sorted_piece_classes),
            target_layer_count=len(self.target_layers),
        )

    def _compile_embedding(
        self,
        token_embeddings: Tensor,
        padding_mask: Tensor,
        *,
        strategy: Mapping[str, object] | None,
        strategy_digest: str | None,
    ) -> Tensor:
        if token_embeddings.shape[0] == 0:
            world_embedding = torch.zeros(
                self.d_model,
                dtype=torch.float32,
                device=self.device,
            )
        else:
            pooled_programs: list[Tensor] = []
            for index in range(token_embeddings.shape[0]):
                valid_mask = ~padding_mask[index]
                if valid_mask.any():
                    pooled_programs.append(token_embeddings[index][valid_mask].mean(dim=0))
            world_embedding = (
                torch.stack(pooled_programs, dim=0).mean(dim=0)
                if pooled_programs
                else torch.zeros(
                    self.d_model,
                    dtype=torch.float32,
                    device=self.device,
                )
            )
        if strategy_digest is None:
            strategy_embedding = torch.zeros_like(world_embedding)
        else:
            strategy_embedding = self.strategy_digest_embedding(
                torch.tensor(
                    stable_bucket(
                        "adapter_strategy_digest",
                        strategy_digest,
                        STRATEGY_DIGEST_BUCKETS,
                    ),
                    dtype=torch.long,
                    device=self.device,
                )
            )
        if strategy is None:
            strategy_summary = torch.zeros_like(world_embedding)
        else:
            strategy_summary = self._encode_strategy(strategy)
        return self.compile_projection(
            world_embedding + strategy_embedding + strategy_summary
        )

    def _reshape_scale(self, values: Tensor) -> Tensor:
        reshaped = values.view(len(self.target_layers), self.d_model)
        return 1.0 + 0.1 * torch.tanh(reshaped)

    def _reshape_shift(self, values: Tensor) -> Tensor:
        reshaped = values.view(len(self.target_layers), self.d_model)
        return 0.1 * torch.tanh(reshaped)

    def _reshape_gate(self, values: Tensor) -> Tensor:
        reshaped = values.view(len(self.target_layers), self.d_model)
        return 1.0 + 0.1 * torch.tanh(reshaped)

    def _encode_strategy(self, strategy: Mapping[str, object]) -> Tensor:
        scope_embedding = self.strategy_scope_embedding(
            torch.tensor(
                stable_bucket(
                    "adapter_strategy_scope",
                    str(strategy["scope"]),
                    STRATEGY_SCOPE_BUCKETS,
                ),
                dtype=torch.long,
                device=self.device,
            )
        )
        summary_features = torch.tensor(
            [
                float(strategy["confidence"]),
                _normalized_count(strategy["subgoals"], scale=8.0),
                _normalized_count(strategy["action_biases"], scale=8.0),
                _normalized_count(strategy["avoid_biases"], scale=8.0),
                _normalized_count(strategy["success_predicates"], scale=8.0),
                _normalized_count(strategy["failure_triggers"], scale=8.0),
                min(len(str(strategy["summary"])) / 128.0, 1.0),
                min(len(str(strategy["strategy_id"])) / 32.0, 1.0),
            ],
            dtype=torch.float32,
            device=self.device,
        )
        return scope_embedding + self.strategy_summary_projection(summary_features)


def _sorted_piece_classes(
    piece_classes: Mapping[str, PieceClass] | Sequence[PieceClass],
) -> tuple[PieceClass, ...]:
    if isinstance(piece_classes, Mapping):
        return tuple(piece_classes[class_id] for class_id in sorted(piece_classes))
    return tuple(sorted(piece_classes, key=lambda piece_class: piece_class.class_id))


def _bundle_id(world_digest: str, strategy_digest: str | None) -> str:
    source = {
        "world_digest": world_digest,
        "strategy_digest": strategy_digest,
    }
    digest = stable_digest(source)
    return f"adapter_{digest[:16]}"


def _normalized_count(values: object, *, scale: float) -> float:
    return min(len(values) / scale, 1.0)
