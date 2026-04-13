from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import time

import torch
from torch import Tensor, nn

from pixie_solver.core.move import Move
from pixie_solver.core.state import GameState
from pixie_solver.model.board_encoder import BoardEncoder
from pixie_solver.model.dsl_encoder import DSLFeatureEncoder
from pixie_solver.model.move_encoder import MoveEncoder, MoveEncodingMetrics


@dataclass(frozen=True, slots=True)
class PolicyValueConfig:
    d_model: int = 192
    num_heads: int = 8
    num_layers: int = 4
    dropout: float = 0.1
    feedforward_multiplier: int = 4


@dataclass(slots=True)
class PolicyValueForwardOutput:
    move_ids: tuple[str, ...]
    policy_logits: Tensor
    value: Tensor


@dataclass(slots=True)
class PolicyValueOutput:
    policy_logits: dict[str, float] = field(default_factory=dict)
    value: float = 0.0


@dataclass(frozen=True, slots=True)
class PolicyValueBatchMetrics:
    requests: int = 0
    total_legal_moves: int = 0
    total_ms: float = 0.0
    board_encode_ms: float = 0.0
    transformer_ms: float = 0.0
    move_encode_ms: float = 0.0
    policy_head_ms: float = 0.0
    value_head_ms: float = 0.0
    consequence_total_ms: float = 0.0
    consequence_apply_move_ms: float = 0.0
    consequence_terminal_check_ms: float = 0.0
    consequence_check_eval_ms: float = 0.0

    def to_dict(self) -> dict[str, float | int]:
        return {
            "requests": self.requests,
            "total_legal_moves": self.total_legal_moves,
            "total_ms": self.total_ms,
            "board_encode_ms": self.board_encode_ms,
            "transformer_ms": self.transformer_ms,
            "move_encode_ms": self.move_encode_ms,
            "policy_head_ms": self.policy_head_ms,
            "value_head_ms": self.value_head_ms,
            "consequence_total_ms": self.consequence_total_ms,
            "consequence_apply_move_ms": self.consequence_apply_move_ms,
            "consequence_terminal_check_ms": self.consequence_terminal_check_ms,
            "consequence_check_eval_ms": self.consequence_check_eval_ms,
        }


class PolicyValueModel(nn.Module):
    def __init__(
        self,
        config: PolicyValueConfig | None = None,
        *,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__()
        self.config = PolicyValueConfig() if config is None else config
        if self.config.d_model % self.config.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        hidden_dim = self.config.d_model * self.config.feedforward_multiplier

        self.dsl_encoder = DSLFeatureEncoder(
            d_model=self.config.d_model,
            hidden_dim=hidden_dim,
        )
        self.board_encoder = BoardEncoder(
            d_model=self.config.d_model,
            dsl_encoder=self.dsl_encoder,
            hidden_dim=hidden_dim,
        )
        self.move_encoder = MoveEncoder(
            d_model=self.config.d_model,
            hidden_dim=hidden_dim,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.d_model,
            nhead=self.config.num_heads,
            dim_feedforward=hidden_dim,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.config.num_layers,
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
        token_rows = [
            torch.cat(
                (
                    board_encoding.global_token.unsqueeze(0),
                    board_encoding.piece_tokens,
                ),
                dim=0,
            )
            for board_encoding in board_encodings
        ]
        max_tokens = max(row.shape[0] for row in token_rows)
        padded_rows: list[Tensor] = []
        padding_masks: list[Tensor] = []
        for row in token_rows:
            padding_length = max_tokens - row.shape[0]
            if padding_length:
                padding = torch.zeros(
                    (padding_length, self.config.d_model),
                    dtype=row.dtype,
                    device=row.device,
                )
                padded_rows.append(torch.cat((row, padding), dim=0))
            else:
                padded_rows.append(row)
            padding_masks.append(
                torch.tensor(
                    [False] * row.shape[0] + [True] * padding_length,
                    dtype=torch.bool,
                    device=row.device,
                )
            )

        transformer_start = time.perf_counter()
        contextual_batch = self.transformer(
            torch.stack(padded_rows, dim=0),
            src_key_padding_mask=torch.stack(padding_masks, dim=0),
        )
        transformer_ms = (time.perf_counter() - transformer_start) * 1000.0

        outputs: list[PolicyValueForwardOutput] = []
        global_contexts: list[Tensor] = []
        encoded_moves_by_request = []
        total_move_metrics = MoveEncodingMetrics()
        move_encode_start = time.perf_counter()
        for request_index, ((state, legal_moves), board_encoding) in enumerate(
            zip(requests, board_encodings, strict=True)
        ):
            contextual_tokens = contextual_batch[request_index]
            global_context = contextual_tokens[0]
            global_contexts.append(global_context)
            piece_context_by_id = {
                piece_id: contextual_tokens[index + 1]
                for index, piece_id in enumerate(board_encoding.piece_ids)
            }
            encoded_moves, move_metrics = self.move_encoder.encode_moves_with_metrics(
                state,
                tuple(legal_moves),
                piece_context_by_id=piece_context_by_id,
                global_context=global_context,
            )
            encoded_moves_by_request.append(encoded_moves)
            total_move_metrics = MoveEncodingMetrics(
                moves_encoded=total_move_metrics.moves_encoded + move_metrics.moves_encoded,
                total_ms=total_move_metrics.total_ms + move_metrics.total_ms,
                consequence_total_ms=(
                    total_move_metrics.consequence_total_ms + move_metrics.consequence_total_ms
                ),
                consequence_apply_move_ms=(
                    total_move_metrics.consequence_apply_move_ms + move_metrics.consequence_apply_move_ms
                ),
                consequence_terminal_check_ms=(
                    total_move_metrics.consequence_terminal_check_ms + move_metrics.consequence_terminal_check_ms
                ),
                consequence_check_eval_ms=(
                    total_move_metrics.consequence_check_eval_ms + move_metrics.consequence_check_eval_ms
                ),
            )
        move_encode_ms = (time.perf_counter() - move_encode_start) * 1000.0

        candidate_lengths = [
            len(encoded_moves.move_ids)
            for encoded_moves in encoded_moves_by_request
        ]
        candidate_tensors = [
            encoded_moves.candidate_embeddings
            for encoded_moves in encoded_moves_by_request
            if encoded_moves.move_ids
        ]
        policy_head_start = time.perf_counter()
        if candidate_tensors:
            all_policy_logits = self.policy_head(
                torch.cat(candidate_tensors, dim=0)
            ).squeeze(-1)
        else:
            all_policy_logits = torch.zeros(0, dtype=torch.float32, device=self.device)
        policy_head_ms = (time.perf_counter() - policy_head_start) * 1000.0
        value_head_start = time.perf_counter()
        values = torch.tanh(self.value_head(torch.stack(global_contexts, dim=0))).squeeze(-1)
        value_head_ms = (time.perf_counter() - value_head_start) * 1000.0

        offset = 0
        for request_index, encoded_moves in enumerate(encoded_moves_by_request):
            length = candidate_lengths[request_index]
            if length:
                policy_logits = all_policy_logits[offset : offset + length]
            else:
                policy_logits = torch.zeros(0, dtype=torch.float32, device=self.device)
            offset += length
            outputs.append(
                PolicyValueForwardOutput(
                    move_ids=encoded_moves.move_ids,
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
            move_encode_ms=move_encode_ms,
            policy_head_ms=policy_head_ms,
            value_head_ms=value_head_ms,
            consequence_total_ms=total_move_metrics.consequence_total_ms,
            consequence_apply_move_ms=total_move_metrics.consequence_apply_move_ms,
            consequence_terminal_check_ms=total_move_metrics.consequence_terminal_check_ms,
            consequence_check_eval_ms=total_move_metrics.consequence_check_eval_ms,
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
            policy_logits = {
                move_id: float(logit)
                for move_id, logit in zip(
                    forward_output.move_ids,
                    forward_output.policy_logits.detach().cpu().tolist(),
                    strict=True,
                )
            }
            outputs.append(
                PolicyValueOutput(
                    policy_logits=policy_logits,
                    value=float(forward_output.value.detach().cpu().item()),
                )
            )
        return tuple(outputs), metrics


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
