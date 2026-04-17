from __future__ import annotations

from collections.abc import Sequence
import time

import torch
from torch import Tensor, nn

from pixie_solver.core.move import Move
from pixie_solver.core.state import GameState
from pixie_solver.model.action_encoder_v2 import ActionEncodingMetricsV2, ActionTokenEncoderV2
from pixie_solver.model.board_encoder import BoardEncoder
from pixie_solver.model.policy_value import (
    PolicyValueBatchMetrics,
    PolicyValueConfig,
    PolicyValueForwardOutput,
    PolicyValueOutput,
    WORLD_CONDITIONED_MODEL_ARCHITECTURE,
    resolve_device,
)


class _CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        num_heads: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        action_tokens: Tensor,
        *,
        context_tokens: Tensor,
        context_padding_mask: Tensor,
        action_padding_mask: Tensor,
    ) -> Tensor:
        attended, _ = self.cross_attention(
            action_tokens,
            context_tokens,
            context_tokens,
            key_padding_mask=context_padding_mask,
            need_weights=False,
        )
        action_tokens = self.norm1(action_tokens + self.dropout(attended))
        feedforward = self.feedforward(action_tokens)
        action_tokens = self.norm2(action_tokens + self.dropout(feedforward))
        return _mask_padded_rows(action_tokens, action_padding_mask)


class PolicyValueModelV2(nn.Module):
    def __init__(
        self,
        config: PolicyValueConfig | None = None,
        *,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__()
        self.config = PolicyValueConfig() if config is None else config
        if self.config.architecture != WORLD_CONDITIONED_MODEL_ARCHITECTURE:
            raise ValueError(
                "PolicyValueModelV2 only supports "
                f"{WORLD_CONDITIONED_MODEL_ARCHITECTURE!r}"
            )
        if self.config.d_model % self.config.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        hidden_dim = self.config.d_model * self.config.feedforward_multiplier
        self.board_encoder = BoardEncoder(
            d_model=self.config.d_model,
            hidden_dim=hidden_dim,
        )
        self.action_encoder = ActionTokenEncoderV2(
            d_model=self.config.d_model,
            hidden_dim=hidden_dim,
        )
        self.context_token_type_embedding = nn.Embedding(4, self.config.d_model)
        context_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.d_model,
            nhead=self.config.num_heads,
            dim_feedforward=hidden_dim,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.context_transformer = nn.TransformerEncoder(
            context_encoder_layer,
            num_layers=self.config.num_layers,
            enable_nested_tensor=False,
        )
        self.action_projection = nn.Sequential(
            nn.LayerNorm(self.config.d_model),
            nn.Linear(self.config.d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.config.d_model),
        )
        self.cross_attention = _CrossAttentionBlock(
            d_model=self.config.d_model,
            num_heads=self.config.num_heads,
            hidden_dim=hidden_dim,
            dropout=self.config.dropout,
        )
        action_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.d_model,
            nhead=self.config.num_heads,
            dim_feedforward=hidden_dim,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.action_transformer = nn.TransformerEncoder(
            action_encoder_layer,
            num_layers=1,
            enable_nested_tensor=False,
        )
        self.policy_head = nn.Sequential(
            nn.LayerNorm(self.config.d_model),
            nn.Linear(self.config.d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.value_head = nn.Sequential(
            nn.LayerNorm(self.config.d_model),
            nn.Linear(self.config.d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.to(resolve_device(device))

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(
        self,
        state: GameState,
        legal_moves: Sequence[Move],
    ) -> PolicyValueForwardOutput:
        return self.forward_batch(((state, legal_moves),))[0]

    def forward_batch(
        self,
        requests: Sequence[tuple[GameState, Sequence[Move]]],
    ) -> tuple[PolicyValueForwardOutput, ...]:
        return self._forward_batch_with_metrics(requests)[0]

    def _forward_batch_with_metrics(
        self,
        requests: Sequence[tuple[GameState, Sequence[Move]]],
    ) -> tuple[tuple[PolicyValueForwardOutput, ...], PolicyValueBatchMetrics]:
        total_start = time.perf_counter()
        if not requests:
            return (), PolicyValueBatchMetrics()

        board_encode_start = time.perf_counter()
        board_encodings = [
            self.board_encoder.encode_state(state)
            for state, _ in requests
        ]
        board_encode_ms = (time.perf_counter() - board_encode_start) * 1000.0

        context_rows = [
            board_encoding.context_tokens
            + self.context_token_type_embedding(board_encoding.context_token_type_ids)
            for board_encoding in board_encodings
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

        encoded_actions_by_request = []
        total_action_metrics = ActionEncodingMetricsV2()
        action_encode_start = time.perf_counter()
        for state, legal_moves in requests:
            encoded_actions, action_metrics = self.action_encoder.encode_moves_with_metrics(
                state,
                tuple(legal_moves),
            )
            encoded_actions_by_request.append(encoded_actions)
            total_action_metrics = ActionEncodingMetricsV2(
                actions_encoded=(
                    total_action_metrics.actions_encoded + action_metrics.actions_encoded
                ),
                total_ms=total_action_metrics.total_ms + action_metrics.total_ms,
                total_tag_count=(
                    total_action_metrics.total_tag_count + action_metrics.total_tag_count
                ),
                total_param_tokens=(
                    total_action_metrics.total_param_tokens + action_metrics.total_param_tokens
                ),
            )
        action_encode_ms = (time.perf_counter() - action_encode_start) * 1000.0

        action_rows = [
            self.action_projection(encoded_actions.candidate_embeddings)
            for encoded_actions in encoded_actions_by_request
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
            action_context = _mask_padded_rows(action_context, action_padding_mask)
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

    def infer(
        self,
        state: GameState,
        legal_moves: Sequence[Move],
    ) -> PolicyValueOutput:
        return self.infer_batch(((state, legal_moves),))[0]

    def infer_batch(
        self,
        requests: Sequence[tuple[GameState, Sequence[Move]]],
    ) -> tuple[PolicyValueOutput, ...]:
        return self.infer_batch_with_metrics(requests)[0]

    def infer_batch_with_metrics(
        self,
        requests: Sequence[tuple[GameState, Sequence[Move]]],
    ) -> tuple[tuple[PolicyValueOutput, ...], PolicyValueBatchMetrics]:
        was_training = self.training
        self.eval()
        try:
            with torch.inference_mode():
                forward_outputs, metrics = self._forward_batch_with_metrics(requests)
        finally:
            if was_training:
                self.train()

        outputs: list[PolicyValueOutput] = []
        for forward_output in forward_outputs:
            outputs.append(
                PolicyValueOutput(
                    policy_logits={
                        move_id: float(logit)
                        for move_id, logit in zip(
                            forward_output.move_ids,
                            forward_output.policy_logits.detach().cpu().tolist(),
                            strict=True,
                        )
                    },
                    value=float(forward_output.value.detach().cpu().item()),
                )
            )
        return tuple(outputs), metrics


def _pad_rows(
    rows: Sequence[Tensor],
    *,
    d_model: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    max_tokens = max(row.shape[0] for row in rows)
    padded_rows: list[Tensor] = []
    padding_masks: list[Tensor] = []
    for row in rows:
        padding_length = max_tokens - row.shape[0]
        if padding_length:
            padding = torch.zeros(
                (padding_length, d_model),
                dtype=row.dtype,
                device=device,
            )
            padded_rows.append(torch.cat((row, padding), dim=0))
        else:
            padded_rows.append(row)
        padding_masks.append(
            torch.tensor(
                [False] * row.shape[0] + [True] * padding_length,
                dtype=torch.bool,
                device=device,
            )
        )
    return torch.stack(padded_rows, dim=0), torch.stack(padding_masks, dim=0)


def _mask_padded_rows(tokens: Tensor, padding_mask: Tensor) -> Tensor:
    if tokens.numel() == 0:
        return tokens
    return tokens.masked_fill(padding_mask.unsqueeze(-1), 0.0)


def _stabilize_all_masked_rows(padding_mask: Tensor) -> Tensor:
    if padding_mask.numel() == 0:
        return padding_mask
    stabilized = padding_mask.clone()
    all_masked_rows = stabilized.all(dim=1)
    if all_masked_rows.any():
        stabilized[all_masked_rows, 0] = False
    return stabilized
