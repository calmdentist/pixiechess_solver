from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

from pixie_solver.core.move import Move
from pixie_solver.core.state import GameState
from pixie_solver.model.board_encoder import BoardEncoder
from pixie_solver.model.dsl_encoder import DSLFeatureEncoder
from pixie_solver.model.move_encoder import MoveEncoder


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
        board_encoding = self.board_encoder.encode_state(state)
        tokens = torch.cat(
            (
                board_encoding.global_token.unsqueeze(0),
                board_encoding.piece_tokens,
            ),
            dim=0,
        ).unsqueeze(0)
        contextual_tokens = self.transformer(tokens).squeeze(0)
        global_context = contextual_tokens[0]
        piece_context_by_id = {
            piece_id: contextual_tokens[index + 1]
            for index, piece_id in enumerate(board_encoding.piece_ids)
        }

        encoded_moves = self.move_encoder.encode_moves(
            state,
            tuple(legal_moves),
            piece_context_by_id=piece_context_by_id,
            global_context=global_context,
        )
        if encoded_moves.move_ids:
            policy_logits = self.policy_head(encoded_moves.candidate_embeddings).squeeze(-1)
        else:
            policy_logits = torch.zeros(0, dtype=torch.float32, device=self.device)
        value = torch.tanh(self.value_head(global_context)).squeeze(-1)
        return PolicyValueForwardOutput(
            move_ids=encoded_moves.move_ids,
            policy_logits=policy_logits,
            value=value,
        )

    def infer(
        self,
        state: GameState,
        legal_moves: Sequence[Move],
    ) -> PolicyValueOutput:
        was_training = self.training
        self.eval()
        try:
            with torch.inference_mode():
                forward_output = self.forward(state, legal_moves)
        finally:
            if was_training:
                self.train()

        policy_logits = {
            move_id: float(logit)
            for move_id, logit in zip(
                forward_output.move_ids,
                forward_output.policy_logits.detach().cpu().tolist(),
                strict=True,
            )
        }
        return PolicyValueOutput(
            policy_logits=policy_logits,
            value=float(forward_output.value.detach().cpu().item()),
        )


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
