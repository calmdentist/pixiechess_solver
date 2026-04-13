from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from pixie_solver.model.policy_value import (
    PolicyValueConfig,
    PolicyValueModel,
    resolve_device,
)
from pixie_solver.training.train import TrainingConfig, TrainingMetrics
from pixie_solver.utils.serialization import JsonValue

CHECKPOINT_FORMAT_VERSION = 1


@dataclass(slots=True)
class LoadedTrainingCheckpoint:
    model: PolicyValueModel
    model_config: PolicyValueConfig
    training_config: TrainingConfig | None = None
    training_metrics: TrainingMetrics | None = None
    optimizer_state_dict: dict[str, Any] | None = None
    metadata: dict[str, JsonValue] = field(default_factory=dict)


def save_training_checkpoint(
    path: str | Path,
    *,
    model: PolicyValueModel,
    training_config: TrainingConfig | None = None,
    training_metrics: TrainingMetrics | None = None,
    optimizer_state_dict: dict[str, Any] | None = None,
    metadata: dict[str, JsonValue] | None = None,
) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_config": asdict(model.config),
        "model_state_dict": model.state_dict(),
        "training_config": _serialize_training_config(training_config),
        "training_metrics": asdict(training_metrics) if training_metrics is not None else None,
        "optimizer_state_dict": optimizer_state_dict,
        "metadata": dict(metadata or {}),
    }
    temp_path = checkpoint_path.with_name(f".{checkpoint_path.name}.tmp")
    try:
        torch.save(payload, temp_path)
        temp_path.replace(checkpoint_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def load_training_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device | None = None,
) -> LoadedTrainingCheckpoint:
    map_location = resolve_device(device)
    payload = torch.load(
        Path(path),
        map_location=map_location,
        weights_only=True,
    )
    format_version = int(payload.get("format_version", 0))
    if format_version != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint format version {format_version}; "
            f"expected {CHECKPOINT_FORMAT_VERSION}"
        )

    model_config = PolicyValueConfig(**dict(payload["model_config"]))
    model = PolicyValueModel(model_config, device=map_location)
    model.load_state_dict(dict(payload["model_state_dict"]))

    training_config_payload = payload.get("training_config")
    training_metrics_payload = payload.get("training_metrics")
    return LoadedTrainingCheckpoint(
        model=model,
        model_config=model_config,
        training_config=_deserialize_training_config(training_config_payload),
        training_metrics=(
            TrainingMetrics(**dict(training_metrics_payload))
            if training_metrics_payload is not None
            else None
        ),
        optimizer_state_dict=payload.get("optimizer_state_dict"),
        metadata=dict(payload.get("metadata", {})),
    )


def _serialize_training_config(
    training_config: TrainingConfig | None,
) -> dict[str, JsonValue] | None:
    if training_config is None:
        return None
    payload = asdict(training_config)
    payload["model_config"] = asdict(training_config.model_config)
    return payload


def _deserialize_training_config(
    payload: dict[str, Any] | None,
) -> TrainingConfig | None:
    if payload is None:
        return None
    training_payload = dict(payload)
    model_config = PolicyValueConfig(**dict(training_payload.pop("model_config")))
    return TrainingConfig(model_config=model_config, **training_payload)
