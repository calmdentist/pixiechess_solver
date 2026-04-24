from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from pixie_solver.eval.model_eval import (
    ModelEvalMetrics,
    _pearson_correlation,
    _predicted_top_move_id,
    _target_policy,
    _target_top_move_id,
)
from pixie_solver.model import PolicyValueModel
from pixie_solver.training import (
    SelfPlayExample,
    flatten_selfplay_examples,
    read_selfplay_examples_jsonl,
    read_selfplay_games_jsonl,
    summarize_benchmark_metadata_records,
)
from pixie_solver.training.targets import uncertainty_target_for_example
from pixie_solver.utils.serialization import JsonValue

BENCHMARK_MANIFEST_FORMAT_VERSION = 1


@dataclass(frozen=True, slots=True)
class BenchmarkProgress:
    event: str
    suite_id: str
    suite_index: int
    suites_total: int
    examples: int | None = None
    games: int | None = None


@dataclass(frozen=True, slots=True)
class _ExampleBenchmarkDetail:
    game_index: int | None
    world_model_digest: str | None
    family_id: str | None
    top1_match: bool
    predicted_value: float
    target_value: float
    value_abs_error: float


def run_benchmark_manifest(
    *,
    model: PolicyValueModel,
    manifest_path: str | Path,
    progress_callback: Callable[[BenchmarkProgress], None] | None = None,
) -> dict[str, JsonValue]:
    resolved_manifest_path = Path(manifest_path)
    manifest = _load_manifest(resolved_manifest_path)
    suite_specs = manifest["suites"]
    suite_reports: list[dict[str, JsonValue]] = []
    base_dir = resolved_manifest_path.parent

    for suite_index, suite_spec in enumerate(suite_specs, start=1):
        suite_id = str(suite_spec["suite_id"])
        if progress_callback is not None:
            progress_callback(
                BenchmarkProgress(
                    event="suite_started",
                    suite_id=suite_id,
                    suite_index=suite_index,
                    suites_total=len(suite_specs),
                )
            )
        suite_report = _run_suite(
            model=model,
            suite_spec=suite_spec,
            base_dir=base_dir,
        )
        suite_reports.append(suite_report)
        if progress_callback is not None:
            progress_callback(
                BenchmarkProgress(
                    event="suite_completed",
                    suite_id=suite_id,
                    suite_index=suite_index,
                    suites_total=len(suite_specs),
                    examples=int(suite_report["examples_loaded"]),
                    games=(
                        int(suite_report["games_loaded"])
                        if suite_report["games_loaded"] is not None
                        else None
                    ),
                )
            )

    return {
        "format_version": BENCHMARK_MANIFEST_FORMAT_VERSION,
        "manifest_path": str(resolved_manifest_path),
        "manifest_name": manifest.get("name"),
        "manifest_description": manifest.get("description"),
        "suite_count": len(suite_reports),
        "suites": suite_reports,
        "aggregate": _aggregate_suite_reports(suite_reports),
    }


def _run_suite(
    *,
    model: PolicyValueModel,
    suite_spec: Mapping[str, JsonValue],
    base_dir: Path,
) -> dict[str, JsonValue]:
    suite_id = str(suite_spec["suite_id"])
    filters = dict(suite_spec.get("filters", {}))
    games_path = _resolve_optional_path(suite_spec.get("games"), base_dir)
    examples_path = _resolve_optional_path(suite_spec.get("examples"), base_dir)

    games = None
    if games_path is not None:
        loaded_games = read_selfplay_games_jsonl(games_path)
        games = _filter_games(loaded_games, filters=filters)

    if examples_path is not None:
        loaded_examples = read_selfplay_examples_jsonl(examples_path)
        examples = _filter_examples(loaded_examples, filters=filters)
    elif games is not None:
        examples = flatten_selfplay_examples(games)
    else:
        raise ValueError(
            f"Benchmark suite {suite_id!r} must define at least one of "
            "'games' or 'examples'"
        )

    metrics, extras, details = _evaluate_examples_with_details(model, examples)
    return {
        "suite_id": suite_id,
        "description": suite_spec.get("description"),
        "tags": list(suite_spec.get("tags", [])),
        "games_path": str(games_path) if games_path is not None else None,
        "examples_path": str(examples_path) if examples_path is not None else None,
        "filters": filters,
        "games_loaded": None if games is None else len(games),
        "examples_loaded": len(examples),
        "benchmark_metadata_summary": summarize_benchmark_metadata_records(
            example.metadata for example in examples
        ),
        "outcomes": None if games is None else _outcome_counts(games),
        "metrics": {
            **metrics.to_dict(),
            **extras,
        },
        "adaptation_curve": _adaptation_curve(details),
    }


def _evaluate_examples_with_details(
    model: PolicyValueModel,
    examples: Sequence[SelfPlayExample],
) -> tuple[ModelEvalMetrics, dict[str, JsonValue], list[_ExampleBenchmarkDetail]]:
    if not examples:
        raise ValueError("examples must contain at least one SelfPlayExample")

    was_training = model.training
    model.eval()
    total_policy_cross_entropy = 0.0
    total_target_entropy = 0.0
    total_value_mse = 0.0
    total_value_mae = 0.0
    total_legal_moves = 0
    top1_matches = 0
    predicted_values: list[float] = []
    target_values: list[float] = []
    win_probabilities: list[float] = []
    target_probabilities: list[float] = []
    predicted_uncertainties: list[float] = []
    target_uncertainties: list[float] = []
    details: list[_ExampleBenchmarkDetail] = []
    valid_examples = 0
    skipped_examples = 0

    try:
        with torch.inference_mode():
            for example in examples:
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
                top1_match = (
                    _predicted_top_move_id(
                        forward_output.move_ids,
                        forward_output.policy_logits,
                    )
                    == _target_top_move_id(example)
                )

                total_policy_cross_entropy += policy_cross_entropy
                total_target_entropy += target_entropy
                total_value_mse += value_error * value_error
                total_value_mae += abs(value_error)
                total_legal_moves += len(example.legal_moves)
                top1_matches += int(top1_match)
                predicted_values.append(predicted_value)
                target_values.append(target_value)
                win_probabilities.append(_value_to_win_probability(predicted_value))
                target_probabilities.append(_value_to_win_probability(target_value))
                predicted_uncertainties.append(predicted_uncertainty)
                target_uncertainties.append(target_uncertainty)
                details.append(
                    _ExampleBenchmarkDetail(
                        game_index=_coerce_int(example.metadata.get("game_index")),
                        world_model_digest=(
                            None
                            if example.metadata.get("world_model_digest") is None
                            else str(example.metadata["world_model_digest"])
                        ),
                        family_id=(
                            None
                            if example.metadata.get("family_id") is None
                            else str(example.metadata["family_id"])
                        ),
                        top1_match=top1_match,
                        predicted_value=predicted_value,
                        target_value=target_value,
                        value_abs_error=abs(value_error),
                    )
                )
                valid_examples += 1
    finally:
        if was_training:
            model.train()

    if valid_examples == 0:
        raise ValueError("examples did not contain any legal-move training rows")

    metrics = ModelEvalMetrics(
        examples=valid_examples,
        skipped_examples=skipped_examples,
        average_policy_cross_entropy=total_policy_cross_entropy / valid_examples,
        average_target_entropy=total_target_entropy / valid_examples,
        average_policy_kl=(
            (total_policy_cross_entropy - total_target_entropy) / valid_examples
        ),
        top1_agreement=top1_matches / valid_examples,
        value_mse=total_value_mse / valid_examples,
        value_mae=total_value_mae / valid_examples,
        value_correlation=_pearson_correlation(predicted_values, target_values),
        average_predicted_uncertainty=sum(predicted_uncertainties) / valid_examples,
        uncertainty_mse=_brier_score(predicted_uncertainties, target_uncertainties)
        or 0.0,
        uncertainty_mae=sum(
            abs(predicted - target)
            for predicted, target in zip(
                predicted_uncertainties,
                target_uncertainties,
                strict=True,
            )
        )
        / valid_examples,
        average_legal_moves=total_legal_moves / valid_examples,
        average_predicted_value=sum(predicted_values) / valid_examples,
        average_target_value=sum(target_values) / valid_examples,
    )
    extras = {
        "value_brier_score": _brier_score(win_probabilities, target_probabilities),
        "value_expected_calibration_error": _expected_calibration_error(
            win_probabilities,
            target_probabilities,
        ),
        "uncertainty_expected_calibration_error": _expected_calibration_error(
            predicted_uncertainties,
            target_uncertainties,
        ),
    }
    return metrics, extras, details


def _load_manifest(path: Path) -> dict[str, JsonValue]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("benchmark manifest must be a JSON object")
    format_version = int(payload.get("format_version", BENCHMARK_MANIFEST_FORMAT_VERSION))
    if format_version != BENCHMARK_MANIFEST_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported benchmark manifest format version {format_version}; "
            f"expected {BENCHMARK_MANIFEST_FORMAT_VERSION}"
        )
    suites = payload.get("suites")
    if not isinstance(suites, list) or not suites:
        raise ValueError("benchmark manifest must contain a non-empty suites list")
    normalized_suites: list[dict[str, JsonValue]] = []
    for suite in suites:
        if not isinstance(suite, dict):
            raise ValueError("benchmark suite entries must be JSON objects")
        suite_id = suite.get("suite_id")
        if suite_id is None or not str(suite_id).strip():
            raise ValueError("benchmark suites require a non-empty suite_id")
        normalized_suites.append(dict(suite))
    return {
        "name": payload.get("name"),
        "description": payload.get("description"),
        "suites": normalized_suites,
    }


def _resolve_optional_path(
    value: JsonValue | None,
    base_dir: Path,
) -> Path | None:
    if value is None:
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = base_dir / path
    return path


def _filter_games(games, *, filters: Mapping[str, JsonValue]) -> list:
    if not filters:
        return list(games)
    return [
        game
        for game in games
        if _metadata_matches_filters(game.metadata, filters)
    ]


def _filter_examples(
    examples: Sequence[SelfPlayExample],
    *,
    filters: Mapping[str, JsonValue],
) -> list[SelfPlayExample]:
    if not filters:
        return list(examples)
    return [
        example
        for example in examples
        if _metadata_matches_filters(example.metadata, filters)
    ]


def _metadata_matches_filters(
    metadata: Mapping[str, JsonValue],
    filters: Mapping[str, JsonValue],
) -> bool:
    for key, expected in filters.items():
        if not _value_matches_filter(metadata.get(str(key)), expected):
            return False
    return True


def _value_matches_filter(actual: JsonValue | None, expected: JsonValue) -> bool:
    expected_values = expected if isinstance(expected, list) else [expected]
    if isinstance(actual, list):
        return any(_scalar_filter_match(item, candidate) for item in actual for candidate in expected_values)
    return any(_scalar_filter_match(actual, candidate) for candidate in expected_values)


def _outcome_counts(games) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for game in games:
        counts[str(game.outcome)] += 1
    return {
        outcome: counts[outcome]
        for outcome in sorted(counts)
    }


def _value_to_win_probability(value: float) -> float:
    return max(0.0, min(1.0, (value + 1.0) / 2.0))


def _brier_score(
    predicted_probabilities: Sequence[float],
    target_probabilities: Sequence[float],
) -> float | None:
    if not predicted_probabilities:
        return None
    return sum(
        (predicted - target) ** 2
        for predicted, target in zip(
            predicted_probabilities,
            target_probabilities,
            strict=True,
        )
    ) / len(predicted_probabilities)


def _expected_calibration_error(
    predicted_probabilities: Sequence[float],
    target_probabilities: Sequence[float],
    *,
    bins: int = 10,
) -> float | None:
    if not predicted_probabilities:
        return None
    total = len(predicted_probabilities)
    error = 0.0
    for bucket_index in range(bins):
        lower = bucket_index / bins
        upper = (bucket_index + 1) / bins
        bucket_predictions: list[float] = []
        bucket_targets: list[float] = []
        for predicted, target in zip(
            predicted_probabilities,
            target_probabilities,
            strict=True,
        ):
            in_bucket = (
                lower <= predicted <= upper
                if bucket_index == bins - 1
                else lower <= predicted < upper
            )
            if not in_bucket:
                continue
            bucket_predictions.append(predicted)
            bucket_targets.append(target)
        if not bucket_predictions:
            continue
        average_prediction = sum(bucket_predictions) / len(bucket_predictions)
        average_target = sum(bucket_targets) / len(bucket_targets)
        error += (
            abs(average_prediction - average_target)
            * len(bucket_predictions)
            / total
        )
    return error


def _adaptation_curve(
    details: Sequence[_ExampleBenchmarkDetail],
) -> list[dict[str, JsonValue]] | None:
    details_with_games = [
        detail
        for detail in details
        if detail.game_index is not None
    ]
    if not details_with_games:
        return None

    per_game: dict[int, dict[str, JsonValue]] = {}
    for detail in details_with_games:
        game_index = int(detail.game_index)
        game = per_game.setdefault(
            game_index,
            {
                "world_model_digest": detail.world_model_digest,
                "family_id": detail.family_id,
                "examples": 0,
                "top1_matches": 0,
                "value_abs_error_sum": 0.0,
            },
        )
        game["examples"] = int(game["examples"]) + 1
        game["top1_matches"] = int(game["top1_matches"]) + int(detail.top1_match)
        game["value_abs_error_sum"] = float(game["value_abs_error_sum"]) + detail.value_abs_error

    ordered_games = [
        (game_index, per_game[game_index])
        for game_index in sorted(per_game)
    ]
    segments: list[dict[str, JsonValue]] = []
    current_world_digest: str | None = None
    running_examples = 0
    running_top1_matches = 0
    running_value_abs_error = 0.0
    current_segment: dict[str, JsonValue] | None = None

    for game_in_suite, (game_index, game_metrics) in enumerate(ordered_games, start=1):
        world_digest = (
            None
            if game_metrics["world_model_digest"] is None
            else str(game_metrics["world_model_digest"])
        )
        if current_segment is None or world_digest != current_world_digest:
            current_world_digest = world_digest
            running_examples = 0
            running_top1_matches = 0
            running_value_abs_error = 0.0
            current_segment = {
                "segment_index": len(segments),
                "world_model_digest": world_digest,
                "family_id": game_metrics["family_id"],
                "points": [],
            }
            segments.append(current_segment)

        game_examples = int(game_metrics["examples"])
        running_examples += game_examples
        running_top1_matches += int(game_metrics["top1_matches"])
        running_value_abs_error += float(game_metrics["value_abs_error_sum"])
        current_segment["points"].append(
            {
                "game_index": game_index,
                "game_in_suite": game_in_suite,
                "game_in_segment": len(current_segment["points"]) + 1,
                "examples": game_examples,
                "top1_agreement": int(game_metrics["top1_matches"]) / game_examples,
                "value_mae": float(game_metrics["value_abs_error_sum"]) / game_examples,
                "cumulative_top1_agreement": running_top1_matches / running_examples,
                "cumulative_value_mae": running_value_abs_error / running_examples,
            }
        )

    return segments


def _aggregate_suite_reports(
    suite_reports: Sequence[Mapping[str, JsonValue]],
) -> dict[str, JsonValue]:
    if not suite_reports:
        return {
            "examples": 0,
            "games": 0,
            "metrics": {},
            "outcomes": {},
        }

    total_examples = 0
    total_games = 0
    weighted_metrics: dict[str, float] = {}
    metric_weights: Counter[str] = Counter()
    outcome_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    novelty_counts: Counter[str] = Counter()
    model_architecture_counts: Counter[str] = Counter()
    unique_worlds = 0
    unique_strategy_digests = 0

    for suite_report in suite_reports:
        suite_metrics = dict(suite_report["metrics"])
        suite_examples = int(suite_metrics.get("examples", 0))
        total_examples += suite_examples
        if suite_report.get("games_loaded") is not None:
            total_games += int(suite_report["games_loaded"])

        for metric_name, metric_value in suite_metrics.items():
            if metric_name in {"examples", "skipped_examples"}:
                continue
            if metric_value is None or isinstance(metric_value, bool):
                continue
            if not isinstance(metric_value, (int, float)):
                continue
            weighted_metrics[metric_name] = (
                weighted_metrics.get(metric_name, 0.0)
                + float(metric_value) * suite_examples
            )
            metric_weights[metric_name] += suite_examples

        outcomes = suite_report.get("outcomes")
        if isinstance(outcomes, Mapping):
            for outcome, count in outcomes.items():
                outcome_counts[str(outcome)] += int(count)

        metadata_summary = suite_report.get("benchmark_metadata_summary")
        if isinstance(metadata_summary, Mapping):
            for family_id, count in dict(metadata_summary.get("family_counts", {})).items():
                family_counts[str(family_id)] += int(count)
            for split, count in dict(metadata_summary.get("split_counts", {})).items():
                split_counts[str(split)] += int(count)
            for novelty_tier, count in dict(
                metadata_summary.get("novelty_tier_counts", {})
            ).items():
                novelty_counts[str(novelty_tier)] += int(count)
            for architecture, count in dict(
                metadata_summary.get("model_architecture_counts", {})
            ).items():
                model_architecture_counts[str(architecture)] += int(count)
            unique_worlds += int(metadata_summary.get("unique_worlds", 0))
            unique_strategy_digests += int(
                metadata_summary.get("unique_strategy_digests", 0)
            )

    aggregate_metrics = {
        metric_name: weighted_metrics[metric_name] / metric_weights[metric_name]
        for metric_name in sorted(weighted_metrics)
        if metric_weights[metric_name] > 0
    }
    aggregate_metrics["examples"] = total_examples

    return {
        "examples": total_examples,
        "games": total_games,
        "metrics": aggregate_metrics,
        "outcomes": {
            outcome: outcome_counts[outcome]
            for outcome in sorted(outcome_counts)
        },
        "benchmark_metadata_summary": {
            "records": total_examples,
            "family_counts": {
                family_id: family_counts[family_id]
                for family_id in sorted(family_counts)
            },
            "split_counts": {
                split: split_counts[split]
                for split in sorted(split_counts)
            },
            "novelty_tier_counts": {
                novelty_tier: novelty_counts[novelty_tier]
                for novelty_tier in sorted(novelty_counts)
            },
            "model_architecture_counts": {
                architecture: model_architecture_counts[architecture]
                for architecture in sorted(model_architecture_counts)
            },
            "unique_worlds": unique_worlds,
            "unique_strategy_digests": unique_strategy_digests,
        },
    }


def _coerce_int(value: JsonValue | None) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return int(stripped)
        except ValueError:
            return None
    return None


def _scalar_filter_match(actual: JsonValue | None, expected: JsonValue) -> bool:
    if actual == expected:
        return True
    if actual is None or expected is None:
        return False
    if isinstance(actual, (bool, int, float, str)) and isinstance(
        expected,
        (bool, int, float, str),
    ):
        return str(actual) == str(expected)
    return False
