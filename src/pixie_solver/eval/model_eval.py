from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from pixie_solver.model import PolicyValueModel
from pixie_solver.training.dataset import SelfPlayExample
from pixie_solver.training.targets import uncertainty_target_for_example


@dataclass(frozen=True, slots=True)
class ModelEvalProgress:
    event: str
    examples_seen: int
    examples_total: int


@dataclass(frozen=True, slots=True)
class ModelEvalMetrics:
    examples: int
    skipped_examples: int
    average_policy_cross_entropy: float
    average_target_entropy: float
    average_policy_kl: float
    top1_agreement: float
    value_mse: float
    value_mae: float
    value_correlation: float | None
    average_predicted_uncertainty: float
    uncertainty_mse: float
    uncertainty_mae: float
    average_legal_moves: float
    average_predicted_value: float
    average_target_value: float

    def to_dict(self) -> dict[str, float | int | None]:
        return {
            "examples": self.examples,
            "skipped_examples": self.skipped_examples,
            "average_policy_cross_entropy": self.average_policy_cross_entropy,
            "average_target_entropy": self.average_target_entropy,
            "average_policy_kl": self.average_policy_kl,
            "top1_agreement": self.top1_agreement,
            "value_mse": self.value_mse,
            "value_mae": self.value_mae,
            "value_correlation": self.value_correlation,
            "average_predicted_uncertainty": self.average_predicted_uncertainty,
            "uncertainty_mse": self.uncertainty_mse,
            "uncertainty_mae": self.uncertainty_mae,
            "average_legal_moves": self.average_legal_moves,
            "average_predicted_value": self.average_predicted_value,
            "average_target_value": self.average_target_value,
        }


def evaluate_policy_value_model(
    *,
    model: PolicyValueModel,
    examples: Sequence[SelfPlayExample],
    progress_callback: Callable[[ModelEvalProgress], None] | None = None,
) -> ModelEvalMetrics:
    if not examples:
        raise ValueError("examples must contain at least one SelfPlayExample")

    was_training = model.training
    model.eval()
    total_policy_cross_entropy = 0.0
    total_target_entropy = 0.0
    total_value_mse = 0.0
    total_value_mae = 0.0
    total_uncertainty_mse = 0.0
    total_uncertainty_mae = 0.0
    total_legal_moves = 0
    top1_matches = 0
    predicted_values: list[float] = []
    target_values: list[float] = []
    predicted_uncertainties: list[float] = []
    valid_examples = 0
    skipped_examples = 0

    if progress_callback is not None:
        progress_callback(
            ModelEvalProgress(
                event="eval_started",
                examples_seen=0,
                examples_total=len(examples),
            )
        )

    try:
        with torch.inference_mode():
            for example_index, example in enumerate(examples, start=1):
                if not example.legal_moves:
                    skipped_examples += 1
                    continue

                strategy = example.metadata.get("strategy")
                forward_output = model(
                    example.state,
                    example.legal_moves,
                    strategy=strategy,
                )
                target_policy = _target_policy(
                    move_ids=forward_output.move_ids,
                    example=example,
                    device=forward_output.policy_logits.device,
                )
                log_probs = torch.log_softmax(forward_output.policy_logits, dim=0)
                policy_cross_entropy = float(
                    -(target_policy * log_probs).sum().detach().cpu().item()
                )
                target_entropy = float(
                    -(target_policy * torch.log(target_policy.clamp_min(1e-12)))
                    .sum()
                    .detach()
                    .cpu()
                    .item()
                )
                predicted_value = float(forward_output.value.detach().cpu().item())
                target_value = float(example.outcome)
                value_error = predicted_value - target_value
                predicted_uncertainty = float(
                    forward_output.uncertainty.detach().cpu().item()
                )
                target_uncertainty = uncertainty_target_for_example(example)
                uncertainty_error = predicted_uncertainty - target_uncertainty

                total_policy_cross_entropy += policy_cross_entropy
                total_target_entropy += target_entropy
                total_value_mse += value_error * value_error
                total_value_mae += abs(value_error)
                total_uncertainty_mse += uncertainty_error * uncertainty_error
                total_uncertainty_mae += abs(uncertainty_error)
                total_legal_moves += len(example.legal_moves)
                top1_matches += int(
                    _predicted_top_move_id(forward_output.move_ids, forward_output.policy_logits)
                    == _target_top_move_id(example)
                )
                predicted_values.append(predicted_value)
                target_values.append(target_value)
                predicted_uncertainties.append(predicted_uncertainty)
                valid_examples += 1

                if progress_callback is not None:
                    progress_callback(
                        ModelEvalProgress(
                            event="example_evaluated",
                            examples_seen=example_index,
                            examples_total=len(examples),
                        )
                    )
    finally:
        if was_training:
            model.train()

    if valid_examples == 0:
        raise ValueError("examples did not contain any legal-move training rows")

    average_policy_cross_entropy = total_policy_cross_entropy / valid_examples
    average_target_entropy = total_target_entropy / valid_examples
    metrics = ModelEvalMetrics(
        examples=valid_examples,
        skipped_examples=skipped_examples,
        average_policy_cross_entropy=average_policy_cross_entropy,
        average_target_entropy=average_target_entropy,
        average_policy_kl=average_policy_cross_entropy - average_target_entropy,
        top1_agreement=top1_matches / valid_examples,
        value_mse=total_value_mse / valid_examples,
        value_mae=total_value_mae / valid_examples,
        value_correlation=_pearson_correlation(predicted_values, target_values),
        average_predicted_uncertainty=sum(predicted_uncertainties) / valid_examples,
        uncertainty_mse=total_uncertainty_mse / valid_examples,
        uncertainty_mae=total_uncertainty_mae / valid_examples,
        average_legal_moves=total_legal_moves / valid_examples,
        average_predicted_value=sum(predicted_values) / valid_examples,
        average_target_value=sum(target_values) / valid_examples,
    )
    if progress_callback is not None:
        progress_callback(
            ModelEvalProgress(
                event="eval_completed",
                examples_seen=len(examples),
                examples_total=len(examples),
            )
        )
    return metrics


def _target_policy(
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
    return torch.full(
        (len(move_ids),),
        1.0 / len(move_ids),
        dtype=torch.float32,
        device=device,
    )


def _predicted_top_move_id(move_ids: Sequence[str], policy_logits: torch.Tensor) -> str:
    top_index = int(torch.argmax(policy_logits).detach().cpu().item())
    return move_ids[top_index]


def _target_top_move_id(example: SelfPlayExample) -> str:
    if example.visit_distribution:
        return min(
            example.visit_distribution,
            key=lambda move_id: (-example.visit_distribution[move_id], move_id),
        )
    if example.visit_counts:
        return min(
            example.visit_counts,
            key=lambda move_id: (-example.visit_counts[move_id], move_id),
        )
    if example.selected_move_id is not None:
        return example.selected_move_id
    return min(example.legal_move_ids)


def _pearson_correlation(
    predicted_values: Sequence[float],
    target_values: Sequence[float],
) -> float | None:
    if len(predicted_values) < 2:
        return None
    mean_predicted = sum(predicted_values) / len(predicted_values)
    mean_target = sum(target_values) / len(target_values)
    numerator = sum(
        (predicted - mean_predicted) * (target - mean_target)
        for predicted, target in zip(predicted_values, target_values, strict=True)
    )
    predicted_variance = sum(
        (predicted - mean_predicted) ** 2 for predicted in predicted_values
    )
    target_variance = sum((target - mean_target) ** 2 for target in target_values)
    denominator = math.sqrt(predicted_variance * target_variance)
    if denominator == 0.0:
        return None
    return numerator / denominator
