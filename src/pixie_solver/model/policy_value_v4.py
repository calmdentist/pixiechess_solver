from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
import time

import torch
from torch import Tensor

from pixie_solver.core.move import Move
from pixie_solver.core.state import GameState
from pixie_solver.hypernet.cache import AdapterBundleCache, AdapterCacheStats
from pixie_solver.hypernet.compiler import WorldCompilerHypernetwork
from pixie_solver.hypernet.layers import (
    PreparedAdapterBundle,
    apply_layer_adapter,
    prepare_adapter_bundle,
)
from pixie_solver.hypernet.schema import AdapterBundle
from pixie_solver.model.action_encoder_v2 import ActionEncodingMetricsV2
from pixie_solver.model.policy_value import (
    HYPERNETWORK_MODEL_ARCHITECTURE,
    PolicyValueBatchMetrics,
    PolicyValueConfig,
    PolicyValueForwardOutput,
    WORLD_CONDITIONED_MODEL_ARCHITECTURE,
)
from pixie_solver.model.policy_value_v2 import (
    PolicyValueModelV2,
    _mask_padded_rows,
    _pad_rows,
    _stabilize_all_masked_rows,
)
from pixie_solver.strategy import StrategyHypothesis, strategy_digest as compute_strategy_digest


class PolicyValueModelV4(PolicyValueModelV2):
    def __init__(
        self,
        config: PolicyValueConfig | None = None,
        *,
        device: str | torch.device | None = None,
    ) -> None:
        active_config = (
            PolicyValueConfig(architecture=HYPERNETWORK_MODEL_ARCHITECTURE)
            if config is None
            else config
        )
        if active_config.architecture != HYPERNETWORK_MODEL_ARCHITECTURE:
            raise ValueError(
                "PolicyValueModelV4 only supports "
                f"{HYPERNETWORK_MODEL_ARCHITECTURE!r}"
            )
        super().__init__(
            replace(
                active_config,
                architecture=WORLD_CONDITIONED_MODEL_ARCHITECTURE,
            ),
            device=device,
        )
        self.config = active_config
        hidden_dim = self.config.d_model * self.config.feedforward_multiplier
        self.world_compiler = WorldCompilerHypernetwork(
            d_model=self.config.d_model,
            hidden_dim=hidden_dim,
        )
        self.uncertainty_head = torch.nn.Sequential(
            torch.nn.LayerNorm(self.config.d_model),
            torch.nn.Linear(self.config.d_model, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 1),
        )
        self._active_strategy: StrategyHypothesis | dict[str, object] | None = None
        self._active_strategy_digest: str | None = None
        self._adapter_cache: AdapterBundleCache[PreparedAdapterBundle] = (
            AdapterBundleCache()
        )

    @property
    def active_strategy(self) -> StrategyHypothesis | dict[str, object] | None:
        return self._active_strategy

    @property
    def active_strategy_digest(self) -> str | None:
        return self._active_strategy_digest

    def set_active_strategy(
        self,
        strategy: StrategyHypothesis | dict[str, object] | None,
    ) -> None:
        self._active_strategy = strategy
        self._active_strategy_digest = (
            None if strategy is None else compute_strategy_digest(strategy)
        )
        self.clear_adapter_cache()

    def set_active_strategy_digest(self, strategy_digest: str | None) -> None:
        normalized = None if strategy_digest is None else str(strategy_digest)
        if normalized != self._active_strategy_digest:
            self._active_strategy = None
            self._active_strategy_digest = normalized
            self.clear_adapter_cache()

    def clear_adapter_cache(self) -> None:
        self._adapter_cache.clear()

    def adapter_cache_stats(self) -> AdapterCacheStats:
        return self._adapter_cache.stats()

    def compile_adapter_bundle_for_state(
        self,
        state: GameState,
        *,
        strategy: StrategyHypothesis | dict[str, object] | None = None,
        strategy_digest: str | None = None,
    ) -> AdapterBundle:
        effective_strategy = self._active_strategy if strategy is None else strategy
        if strategy_digest is not None:
            effective_strategy_digest = strategy_digest
        elif effective_strategy is not None:
            effective_strategy_digest = compute_strategy_digest(effective_strategy)
        else:
            effective_strategy_digest = self._active_strategy_digest
        return self.world_compiler.compile_from_state(
            state,
            strategy=effective_strategy,
            strategy_digest=effective_strategy_digest,
        )

    def prepare_adapter_bundle_for_state(
        self,
        state: GameState,
        *,
        strategy: StrategyHypothesis | dict[str, object] | None = None,
        strategy_digest: str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> PreparedAdapterBundle:
        effective_strategy = self._active_strategy if strategy is None else strategy
        if strategy_digest is not None:
            effective_strategy_digest = strategy_digest
        elif effective_strategy is not None:
            effective_strategy_digest = compute_strategy_digest(effective_strategy)
        else:
            effective_strategy_digest = self._active_strategy_digest
        world_digest = self.world_compiler.world_digest_for_state(state)
        cache_key = (world_digest, effective_strategy_digest)
        cached_bundle = self._adapter_cache.get(cache_key)
        if cached_bundle is not None:
            return cached_bundle
        compiled_bundle = self.compile_adapter_bundle_for_state(
            state,
            strategy=effective_strategy,
            strategy_digest=effective_strategy_digest,
        )
        prepared_bundle = prepare_adapter_bundle(
            compiled_bundle,
            device=self.device,
            dtype=dtype,
        )
        if prepared_bundle is None:
            raise RuntimeError("prepare_adapter_bundle unexpectedly returned None")
        self._adapter_cache.put(cache_key, prepared_bundle)
        return prepared_bundle

    def _forward_batch_with_metrics(
        self,
        requests: Sequence[tuple[GameState, Sequence[Move]]],
        *,
        strategy: object | None = None,
    ) -> tuple[tuple[PolicyValueForwardOutput, ...], PolicyValueBatchMetrics]:
        total_start = time.perf_counter()
        if not requests:
            return (), PolicyValueBatchMetrics()

        board_encode_start = time.perf_counter()
        board_encodings = []
        prepared_bundles = []
        for state, _ in requests:
            board_encoding = self.board_encoder.encode_state(state)
            board_encodings.append(board_encoding)
            prepared_bundles.append(
                self.prepare_adapter_bundle_for_state(
                    state,
                    strategy=strategy,
                    dtype=board_encoding.context_tokens.dtype,
                )
            )
        board_encode_ms = (time.perf_counter() - board_encode_start) * 1000.0

        context_rows = [
            apply_layer_adapter(
                board_encoding.context_tokens
                + self.context_token_type_embedding(board_encoding.context_token_type_ids),
                prepared_bundle,
                layer_name="context_input",
            )
            for board_encoding, prepared_bundle in zip(
                board_encodings,
                prepared_bundles,
                strict=True,
            )
        ]
        padded_contexts, context_padding_mask = _pad_rows(
            context_rows,
            d_model=self.config.d_model,
            device=self.device,
        )

        attention_start = time.perf_counter()
        contextual_batch = self.context_transformer(
            padded_contexts,
            src_key_padding_mask=context_padding_mask,
        )
        contextual_batch = torch.stack(
            [
                apply_layer_adapter(
                    _mask_padded_rows(contextual_batch[index], context_padding_mask[index]),
                    prepared_bundles[index],
                    layer_name="context_output",
                )
                for index in range(len(requests))
            ],
            dim=0,
        )

        encoded_actions_by_request = []
        action_metrics_total = ActionEncodingMetricsV2()
        action_encode_start = time.perf_counter()
        for state, legal_moves in requests:
            encoded_actions, action_metrics = self.action_encoder.encode_moves_with_metrics(
                state,
                tuple(legal_moves),
            )
            encoded_actions_by_request.append(encoded_actions)
            action_metrics_total = ActionEncodingMetricsV2(
                actions_encoded=(
                    action_metrics_total.actions_encoded + action_metrics.actions_encoded
                ),
                total_ms=action_metrics_total.total_ms + action_metrics.total_ms,
                total_tag_count=(
                    action_metrics_total.total_tag_count + action_metrics.total_tag_count
                ),
                total_param_tokens=(
                    action_metrics_total.total_param_tokens
                    + action_metrics.total_param_tokens
                ),
            )
        action_encode_ms = (time.perf_counter() - action_encode_start) * 1000.0

        action_rows = [
            apply_layer_adapter(
                self.action_projection(encoded_actions.candidate_embeddings),
                prepared_bundle,
                layer_name="action_input",
            )
            for encoded_actions, prepared_bundle in zip(
                encoded_actions_by_request,
                prepared_bundles,
                strict=True,
            )
        ]
        if any(row.shape[0] for row in action_rows):
            padded_actions, action_padding_mask = _pad_rows(
                action_rows,
                d_model=self.config.d_model,
                device=self.device,
            )
            effective_action_padding_mask = _stabilize_all_masked_rows(
                action_padding_mask
            )
            cross_attended = self.cross_attention(
                padded_actions,
                context_tokens=contextual_batch,
                context_padding_mask=context_padding_mask,
                action_padding_mask=action_padding_mask,
            )
            action_context = self.action_transformer(
                cross_attended,
                src_key_padding_mask=effective_action_padding_mask,
            )
            action_context = torch.stack(
                [
                    apply_layer_adapter(
                        _mask_padded_rows(action_context[index], action_padding_mask[index]),
                        prepared_bundles[index],
                        layer_name="action_output",
                    )
                    for index in range(len(requests))
                ],
                dim=0,
            )
        else:
            padded_actions = torch.zeros(
                (len(requests), 0, self.config.d_model),
                dtype=torch.float32,
                device=self.device,
            )
            action_padding_mask = torch.zeros(
                (len(requests), 0),
                dtype=torch.bool,
                device=self.device,
            )
            action_context = padded_actions
        transformer_ms = (time.perf_counter() - attention_start) * 1000.0

        policy_head_start = time.perf_counter()
        policy_logits_batch = (
            self.policy_head(action_context).squeeze(-1)
            if action_context.shape[1] > 0
            else torch.zeros((len(requests), 0), dtype=torch.float32, device=self.device)
        )
        policy_head_ms = (time.perf_counter() - policy_head_start) * 1000.0

        value_head_start = time.perf_counter()
        global_contexts = contextual_batch[:, 0, :]
        values = torch.tanh(self.value_head(global_contexts)).squeeze(-1)
        uncertainties = torch.sigmoid(self.uncertainty_head(global_contexts)).squeeze(-1)
        value_head_ms = (time.perf_counter() - value_head_start) * 1000.0

        outputs: list[PolicyValueForwardOutput] = []
        for request_index, encoded_actions in enumerate(encoded_actions_by_request):
            length = len(encoded_actions.move_ids)
            if length:
                policy_logits = policy_logits_batch[request_index, :length]
            else:
                policy_logits = torch.zeros(0, dtype=torch.float32, device=self.device)
            outputs.append(
                PolicyValueForwardOutput(
                    move_ids=encoded_actions.move_ids,
                    policy_logits=policy_logits,
                    value=values[request_index],
                    uncertainty=uncertainties[request_index],
                )
            )

        total_legal_moves = sum(len(legal_moves) for _, legal_moves in requests)
        return tuple(outputs), PolicyValueBatchMetrics(
            requests=len(requests),
            total_legal_moves=total_legal_moves,
            total_ms=(time.perf_counter() - total_start) * 1000.0,
            board_encode_ms=board_encode_ms,
            transformer_ms=transformer_ms,
            move_encode_ms=action_encode_ms,
            policy_head_ms=policy_head_ms,
            value_head_ms=value_head_ms,
            consequence_total_ms=0.0,
            consequence_apply_move_ms=0.0,
            consequence_terminal_check_ms=0.0,
            consequence_check_eval_ms=0.0,
        )
