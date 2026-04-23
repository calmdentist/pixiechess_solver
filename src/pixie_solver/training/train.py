from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from pixie_solver.model.policy_value import (
    PolicyValueConfig,
    PolicyValueModel,
    build_policy_value_model,
    resolve_device,
)
from pixie_solver.training.dataset import SelfPlayExample
from pixie_solver.utils.serialization import JsonValue, canonical_json

UNIFORM_REPLAY_SAMPLING_STRATEGY = "uniform"
BUCKET_BALANCED_REPLAY_SAMPLING_STRATEGY = "bucket_balanced"
SUPPORTED_REPLAY_SAMPLING_STRATEGIES = (
    UNIFORM_REPLAY_SAMPLING_STRATEGY,
    BUCKET_BALANCED_REPLAY_SAMPLING_STRATEGY,
)
RECENT_REPLAY_BUCKET = "recent"
VERIFIED_REPLAY_BUCKET = "verified"
FOUNDATION_REPLAY_BUCKET = "foundation"


class SelfPlayDataset(Dataset[SelfPlayExample]):
    def __init__(self, examples: Sequence[SelfPlayExample]) -> None:
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> SelfPlayExample:
        return self.examples[index]


def collate_selfplay_examples(batch: list[SelfPlayExample]) -> list[SelfPlayExample]:
    return list(batch)


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    epochs: int = 1
    batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    policy_weight: float = 1.0
    value_weight: float = 1.0
    device: str | None = None
    shuffle: bool = True
    seed: int = 0
    sampling_strategy: str = UNIFORM_REPLAY_SAMPLING_STRATEGY
    recent_cycle_window: int = 1
    recent_bucket_weight: float = 1.0
    verified_bucket_weight: float = 1.0
    foundation_bucket_weight: float = 1.0
    sampling_reference_cycle: int | None = None
    model_config: PolicyValueConfig = field(default_factory=PolicyValueConfig)


@dataclass(slots=True)
class TrainingMetrics:
    epochs_completed: int
    examples_seen: int
    batches_completed: int
    average_policy_loss: float
    average_value_loss: float
    average_total_loss: float
    device: str


@dataclass(frozen=True, slots=True)
class TrainingProgress:
    event: str
    epoch: int | None = None
    epochs_total: int | None = None
    batch_index: int | None = None
    batches_total: int | None = None
    valid_examples: int | None = None
    examples_seen: int | None = None
    policy_loss: float | None = None
    value_loss: float | None = None
    total_loss: float | None = None
    device: str | None = None


@dataclass(slots=True)
class TrainingRunResult:
    model: PolicyValueModel
    metrics: TrainingMetrics
    optimizer_state_dict: dict[str, Any]


def train_from_replays(
    replays: Sequence[SelfPlayExample],
    *,
    model: PolicyValueModel | None = None,
    optimizer_state_dict: dict[str, Any] | None = None,
    config: TrainingConfig | None = None,
    progress_callback: Callable[[TrainingProgress], None] | None = None,
) -> TrainingRunResult:
    if not replays:
        raise ValueError("replays must contain at least one SelfPlayExample")

    active_config = TrainingConfig() if config is None else config
    if active_config.epochs < 1:
        raise ValueError("epochs must be at least 1")
    if active_config.batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if active_config.sampling_strategy not in SUPPORTED_REPLAY_SAMPLING_STRATEGIES:
        supported = ", ".join(SUPPORTED_REPLAY_SAMPLING_STRATEGIES)
        raise ValueError(
            f"Unsupported replay sampling strategy {active_config.sampling_strategy!r}. "
            f"Supported strategies: {supported}"
        )
    if active_config.recent_cycle_window < 1:
        raise ValueError("recent_cycle_window must be at least 1")
    for field_name, weight in (
        ("recent_bucket_weight", active_config.recent_bucket_weight),
        ("verified_bucket_weight", active_config.verified_bucket_weight),
        ("foundation_bucket_weight", active_config.foundation_bucket_weight),
    ):
        if weight <= 0.0:
            raise ValueError(f"{field_name} must be positive")

    device = resolve_device(active_config.device)
    _set_training_seed(active_config.seed)
    model_impl = model
    if model_impl is None:
        model_impl = build_policy_value_model(active_config.model_config, device=device)
    else:
        model_impl.to(device)
    model_impl.train()

    sampler = _build_replay_sampler(replays, active_config)
    data_loader = DataLoader(
        SelfPlayDataset(replays),
        batch_size=active_config.batch_size,
        shuffle=active_config.shuffle if sampler is None else False,
        sampler=sampler,
        collate_fn=collate_selfplay_examples,
        num_workers=0,
        generator=_data_loader_generator(active_config.seed),
    )
    optimizer = torch.optim.AdamW(
        model_impl.parameters(),
        lr=active_config.learning_rate,
        weight_decay=active_config.weight_decay,
    )
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
    batches_total = len(data_loader)
    if progress_callback is not None:
        progress_callback(
            TrainingProgress(
                event="training_started",
                epochs_total=active_config.epochs,
                batches_total=batches_total,
                examples_seen=0,
                device=str(device),
            )
        )

    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_total_loss = 0.0
    examples_seen = 0
    batches_completed = 0

    for epoch_index in range(active_config.epochs):
        if progress_callback is not None:
            progress_callback(
                TrainingProgress(
                    event="epoch_started",
                    epoch=epoch_index + 1,
                    epochs_total=active_config.epochs,
                    batches_total=batches_total,
                    examples_seen=examples_seen,
                    device=str(device),
                )
            )
        for batch_index, batch in enumerate(data_loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            batch_policy_loss = torch.zeros((), dtype=torch.float32, device=device)
            batch_value_loss = torch.zeros((), dtype=torch.float32, device=device)
            valid_batch = [
                example
                for example in batch
                if example.legal_moves
            ]
            valid_examples = len(valid_batch)

            if valid_examples == 0:
                continue

            forward_outputs = _forward_batch_for_examples(
                model_impl,
                valid_batch,
            )

            for example, forward_output in zip(
                valid_batch,
                forward_outputs,
                strict=True,
            ):
                if len(forward_output.move_ids) != len(example.legal_moves):
                    raise ValueError("model forward output must align with example legal moves")

                target_policy = _target_policy_tensor(
                    move_ids=forward_output.move_ids,
                    example=example,
                    device=device,
                )
                log_probs = torch.log_softmax(forward_output.policy_logits, dim=0)
                policy_loss = -(target_policy * log_probs).sum()
                target_value = torch.tensor(
                    float(example.outcome),
                    dtype=torch.float32,
                    device=device,
                )
                value_loss = F.mse_loss(
                    forward_output.value.float(),
                    target_value,
                )
                batch_policy_loss = batch_policy_loss + policy_loss
                batch_value_loss = batch_value_loss + value_loss
                examples_seen += 1

            mean_policy_loss = batch_policy_loss / valid_examples
            mean_value_loss = batch_value_loss / valid_examples
            total_loss = (
                active_config.policy_weight * mean_policy_loss
                + active_config.value_weight * mean_value_loss
            )
            total_loss.backward()
            optimizer.step()

            batches_completed += 1
            mean_policy_loss_value = float(mean_policy_loss.detach().cpu().item())
            mean_value_loss_value = float(mean_value_loss.detach().cpu().item())
            total_loss_value = float(total_loss.detach().cpu().item())
            total_policy_loss += mean_policy_loss_value
            total_value_loss += mean_value_loss_value
            total_total_loss += total_loss_value
            if progress_callback is not None:
                progress_callback(
                    TrainingProgress(
                        event="batch_completed",
                        epoch=epoch_index + 1,
                        epochs_total=active_config.epochs,
                        batch_index=batch_index,
                        batches_total=batches_total,
                        valid_examples=valid_examples,
                        examples_seen=examples_seen,
                        policy_loss=mean_policy_loss_value,
                        value_loss=mean_value_loss_value,
                        total_loss=total_loss_value,
                        device=str(device),
                    )
                )
        if progress_callback is not None:
            epoch_batches_completed = min(batches_total, max(0, batches_completed))
            progress_callback(
                TrainingProgress(
                    event="epoch_completed",
                    epoch=epoch_index + 1,
                    epochs_total=active_config.epochs,
                    batch_index=epoch_batches_completed,
                    batches_total=batches_total,
                    examples_seen=examples_seen,
                    device=str(device),
                )
            )

    completed_batches = max(batches_completed, 1)
    result = TrainingRunResult(
        model=model_impl,
        metrics=TrainingMetrics(
            epochs_completed=active_config.epochs,
            examples_seen=examples_seen,
            batches_completed=batches_completed,
            average_policy_loss=total_policy_loss / completed_batches,
            average_value_loss=total_value_loss / completed_batches,
            average_total_loss=total_total_loss / completed_batches,
            device=str(device),
        ),
        optimizer_state_dict=optimizer.state_dict(),
    )
    if progress_callback is not None:
        progress_callback(
            TrainingProgress(
                event="training_completed",
                epoch=active_config.epochs,
                epochs_total=active_config.epochs,
                batch_index=batches_completed,
                batches_total=batches_total,
                examples_seen=examples_seen,
                policy_loss=result.metrics.average_policy_loss,
                value_loss=result.metrics.average_value_loss,
                total_loss=result.metrics.average_total_loss,
                device=str(device),
            )
        )
    return result


def _forward_batch_for_examples(
    model: PolicyValueModel,
    examples: Sequence[SelfPlayExample],
):
    outputs: list[Any] = [None] * len(examples)
    strategy_groups: dict[str | None, list[int]] = {}
    strategy_payloads: dict[str | None, dict[str, JsonValue] | None] = {}
    for index, example in enumerate(examples):
        strategy = _strategy_from_example_metadata(example.metadata)
        strategy_key = _strategy_group_key(strategy)
        strategy_groups.setdefault(strategy_key, []).append(index)
        strategy_payloads[strategy_key] = strategy

    for strategy_key, indices in strategy_groups.items():
        requests = tuple(
            (examples[index].state, examples[index].legal_moves)
            for index in indices
        )
        strategy = strategy_payloads[strategy_key]
        if strategy is None:
            group_outputs = model.forward_batch(requests)
        else:
            group_outputs = model.forward_batch(
                requests,
                strategy=strategy,
            )
        for index, output in zip(indices, group_outputs, strict=True):
            outputs[index] = output
    return tuple(outputs)


def _strategy_from_example_metadata(
    metadata: dict[str, JsonValue],
) -> dict[str, JsonValue] | None:
    strategy = metadata.get("strategy")
    if not isinstance(strategy, dict):
        return None
    return dict(strategy)


def _strategy_group_key(strategy: dict[str, JsonValue] | None) -> str | None:
    if strategy is None:
        return None
    return canonical_json(strategy)


def _target_policy_tensor(
    *,
    move_ids: Sequence[str],
    example: SelfPlayExample,
    device: torch.device,
) -> torch.Tensor:
    distribution = torch.tensor(
        [
            float(example.visit_distribution.get(move_id, 0.0))
            for move_id in move_ids
        ],
        dtype=torch.float32,
        device=device,
    )
    total = float(distribution.sum().detach().cpu().item())
    if total > 0.0:
        return distribution / total
    if not move_ids:
        return distribution
    return torch.full(
        (len(move_ids),),
        1.0 / len(move_ids),
        dtype=torch.float32,
        device=device,
    )


def _set_training_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _data_loader_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def replay_bucket_for_example(
    example: SelfPlayExample,
    *,
    reference_cycle: int | None,
    recent_cycle_window: int,
) -> str:
    example_cycle = _example_cycle(example)
    if (
        reference_cycle is not None
        and example_cycle is not None
        and example_cycle >= reference_cycle - recent_cycle_window + 1
    ):
        return RECENT_REPLAY_BUCKET
    if _has_verified_piece_metadata(example):
        return VERIFIED_REPLAY_BUCKET
    return FOUNDATION_REPLAY_BUCKET


def summarize_replay_buckets(
    replays: Sequence[SelfPlayExample],
    *,
    config: TrainingConfig | None = None,
) -> dict[str, int]:
    active_config = TrainingConfig() if config is None else config
    reference_cycle = _resolve_sampling_reference_cycle(
        replays,
        active_config.sampling_reference_cycle,
    )
    counts = {
        FOUNDATION_REPLAY_BUCKET: 0,
        VERIFIED_REPLAY_BUCKET: 0,
        RECENT_REPLAY_BUCKET: 0,
    }
    for example in replays:
        bucket = replay_bucket_for_example(
            example,
            reference_cycle=reference_cycle,
            recent_cycle_window=active_config.recent_cycle_window,
        )
        counts[bucket] += 1
    return counts


def _build_replay_sampler(
    replays: Sequence[SelfPlayExample],
    config: TrainingConfig,
) -> WeightedRandomSampler | None:
    if config.sampling_strategy != BUCKET_BALANCED_REPLAY_SAMPLING_STRATEGY:
        return None
    reference_cycle = _resolve_sampling_reference_cycle(
        replays,
        config.sampling_reference_cycle,
    )
    weights = torch.tensor(
        [
            _bucket_weight_for_example(
                example,
                reference_cycle=reference_cycle,
                config=config,
            )
            for example in replays
        ],
        dtype=torch.double,
    )
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(replays),
        replacement=True,
    )


def _bucket_weight_for_example(
    example: SelfPlayExample,
    *,
    reference_cycle: int | None,
    config: TrainingConfig,
) -> float:
    bucket = replay_bucket_for_example(
        example,
        reference_cycle=reference_cycle,
        recent_cycle_window=config.recent_cycle_window,
    )
    if bucket == RECENT_REPLAY_BUCKET:
        return float(config.recent_bucket_weight)
    if bucket == VERIFIED_REPLAY_BUCKET:
        return float(config.verified_bucket_weight)
    return float(config.foundation_bucket_weight)


def _resolve_sampling_reference_cycle(
    replays: Sequence[SelfPlayExample],
    configured_reference_cycle: int | None,
) -> int | None:
    if configured_reference_cycle is not None:
        return configured_reference_cycle
    cycle_values = [
        cycle_value
        for cycle_value in (_example_cycle(example) for example in replays)
        if cycle_value is not None
    ]
    if not cycle_values:
        return None
    return max(cycle_values)


def _example_cycle(example: SelfPlayExample) -> int | None:
    value = example.metadata.get("cycle")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _has_verified_piece_metadata(example: SelfPlayExample) -> bool:
    value = example.metadata.get("verified_piece_digests")
    return isinstance(value, dict) and bool(value)
