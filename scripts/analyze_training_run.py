#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


JsonObject = dict[str, Any]


@dataclass(slots=True)
class JsonlReadResult:
    rows: list[JsonObject] = field(default_factory=list)
    skipped_lines: int = 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read-only analyzer for pixie train-loop output directories."
    )
    parser.add_argument("output_dir", type=Path, help="Run directory produced by pixie train-loop.")
    parser.add_argument(
        "--watch",
        type=float,
        default=0.0,
        help="Refresh every N seconds. Default: print one snapshot and exit.",
    )
    parser.add_argument(
        "--top-moves",
        type=int,
        default=5,
        help="Show the N most frequent selected move ids per split.",
    )
    args = parser.parse_args()

    if args.watch > 0:
        while True:
            print("\033[2J\033[H", end="")
            print_snapshot(args.output_dir, top_moves=args.top_moves)
            time.sleep(args.watch)
    else:
        print_snapshot(args.output_dir, top_moves=args.top_moves)
    return 0


def print_snapshot(output_dir: Path, *, top_moves: int) -> None:
    print(f"Pixie training run: {output_dir}")
    print(f"Snapshot time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    _print_cycle_metrics(output_dir)
    print()
    _print_selfplay_summary(output_dir, split="train", top_moves=top_moves)
    print()
    _print_selfplay_summary(output_dir, split="val", top_moves=top_moves)
    print()
    _print_readiness_notes(output_dir)


def _print_cycle_metrics(output_dir: Path) -> None:
    metric_paths = sorted((output_dir / "metrics").glob("cycle_*.json"))
    print("Cycle Metrics")
    if not metric_paths:
        print("  No metrics files found yet.")
        return

    header = (
        "  cycle  "
        "train_ce  val_ce  train_kl  val_kl  "
        "train_top1  val_top1  train_v_mse  val_v_mse"
    )
    print(header)
    for path in metric_paths:
        payload = _read_json_object(path)
        if payload is None:
            print(f"  {path.name}: unreadable or currently incomplete")
            continue
        train = dict(payload.get("train_eval_metrics") or {})
        val = dict(payload.get("val_eval_metrics") or {})
        print(
            "  "
            f"{int(payload.get('cycle', 0)):>5}  "
            f"{_fmt(train.get('average_policy_cross_entropy')):>8}  "
            f"{_fmt(val.get('average_policy_cross_entropy')):>6}  "
            f"{_fmt(train.get('average_policy_kl')):>8}  "
            f"{_fmt(val.get('average_policy_kl')):>6}  "
            f"{_fmt(train.get('top1_agreement')):>10}  "
            f"{_fmt(val.get('top1_agreement')):>8}  "
            f"{_fmt(train.get('value_mse')):>11}  "
            f"{_fmt(val.get('value_mse')):>9}"
        )


def _print_selfplay_summary(output_dir: Path, *, split: str, top_moves: int) -> None:
    games_paths = sorted((output_dir / "selfplay").glob(f"cycle_*_{split}_games.jsonl"))
    examples_paths = sorted((output_dir / "selfplay").glob(f"cycle_*_{split}_examples.jsonl"))
    print(f"{split.title()} Self-Play")
    if not games_paths and not examples_paths:
        print(f"  No {split} self-play files found yet.")
        return

    for path in games_paths:
        games = _read_jsonl_objects(path)
        outcome_counts = Counter(str(row.get("outcome", "unknown")) for row in games.rows)
        reason_counts = Counter(
            str(dict(row.get("metadata") or {}).get("termination_reason", "unknown"))
            for row in games.rows
        )
        adjudicated = sum(
            1
            for row in games.rows
            if dict(row.get("metadata") or {}).get("cutoff_adjudication")
        )
        streaks = [_longest_selected_move_streak(row) for row in games.rows]
        unique_move_counts = [_unique_selected_move_count(row) for row in games.rows]
        print(f"  {path.name}")
        print(
            "    games="
            f"{len(games.rows)} outcomes={dict(sorted(outcome_counts.items()))} "
            f"reasons={dict(sorted(reason_counts.items()))} adjudicated={adjudicated}"
        )
        if streaks:
            print(
                "    repeated_move_streak: "
                f"max={max(streaks)} avg={_mean(streaks):.2f}; "
                f"unique_selected_moves/game avg={_mean(unique_move_counts):.2f}"
            )
        if games.skipped_lines:
            print(f"    skipped_partial_or_invalid_lines={games.skipped_lines}")

    for path in examples_paths:
        examples = _read_jsonl_objects(path)
        value_counts = Counter(float(row.get("outcome", 0.0)) for row in examples.rows)
        selected_move_counts = Counter(
            str(row.get("selected_move_id"))
            for row in examples.rows
            if row.get("selected_move_id") is not None
        )
        target_entropy_values = [
            _policy_entropy(dict(row.get("visit_distribution") or {}))
            for row in examples.rows
        ]
        print(f"  {path.name}")
        print(
            "    examples="
            f"{len(examples.rows)} value_targets={dict(sorted(value_counts.items()))} "
            f"avg_target_entropy={_mean(target_entropy_values):.4f}"
        )
        if selected_move_counts:
            most_common = selected_move_counts.most_common(top_moves)
            rendered = ", ".join(f"{move_id[:10]}...:{count}" for move_id, count in most_common)
            print(f"    top_selected_moves={rendered}")
        if examples.skipped_lines:
            print(f"    skipped_partial_or_invalid_lines={examples.skipped_lines}")


def _print_readiness_notes(output_dir: Path) -> None:
    latest_metrics = _latest_json((output_dir / "metrics").glob("cycle_*.json"))
    latest_train_examples = _latest_path((output_dir / "selfplay").glob("cycle_*_train_examples.jsonl"))
    latest_train_games = _latest_path((output_dir / "selfplay").glob("cycle_*_train_games.jsonl"))

    print("Interpretation")
    if latest_metrics is None:
        print("  Metrics are not available yet. Wait for the current cycle eval phase.")
    else:
        train = dict(latest_metrics.get("train_eval_metrics") or {})
        val = dict(latest_metrics.get("val_eval_metrics") or {})
        train_ce = _as_float(train.get("average_policy_cross_entropy"))
        val_ce = _as_float(val.get("average_policy_cross_entropy"))
        train_top1 = _as_float(train.get("top1_agreement"))
        val_top1 = _as_float(val.get("top1_agreement"))
        if train_ce is not None and val_ce is not None:
            print(
                "  Policy learning signal: "
                f"train_ce={train_ce:.4f}, val_ce={val_ce:.4f}; lower is better."
            )
        if train_top1 is not None and val_top1 is not None:
            print(
                "  Move agreement signal: "
                f"train_top1={train_top1:.3f}, val_top1={val_top1:.3f}; higher is better."
            )

    if latest_train_examples is not None:
        examples = _read_jsonl_objects(latest_train_examples)
        values = Counter(float(row.get("outcome", 0.0)) for row in examples.rows)
        nonzero = sum(count for value, count in values.items() if value != 0.0)
        if examples.rows and nonzero == 0:
            print("  Value warning: latest train examples all have target 0.0.")
        elif examples.rows:
            print(
                "  Value signal: "
                f"{nonzero}/{len(examples.rows)} latest train examples have nonzero targets."
            )

    if latest_train_games is not None:
        games = _read_jsonl_objects(latest_train_games)
        max_streak = max((_longest_selected_move_streak(row) for row in games.rows), default=0)
        max_plies = max((len(_examples(row)) for row in games.rows), default=0)
        if max_streak >= max(8, max_plies // 3):
            print(
                "  Repetition warning: selected move ids show a long repeated streak; "
                "next pass should add repetition termination/noise if this persists."
            )


def _read_json_object(path: Path) -> JsonObject | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _read_jsonl_objects(path: Path) -> JsonlReadResult:
    result = JsonlReadResult()
    try:
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    result.skipped_lines += 1
                    continue
                if isinstance(payload, dict):
                    result.rows.append(payload)
                else:
                    result.skipped_lines += 1
    except OSError:
        return result
    return result


def _latest_json(paths: Any) -> JsonObject | None:
    path = _latest_path(paths)
    if path is None:
        return None
    return _read_json_object(path)


def _latest_path(paths: Any) -> Path | None:
    sorted_paths = sorted(paths)
    return sorted_paths[-1] if sorted_paths else None


def _examples(game_row: JsonObject) -> list[JsonObject]:
    examples = game_row.get("examples") or []
    return [row for row in examples if isinstance(row, dict)]


def _longest_selected_move_streak(game_row: JsonObject) -> int:
    longest = 0
    current = 0
    previous: str | None = None
    for example in _examples(game_row):
        move_id = example.get("selected_move_id")
        if move_id is None:
            current = 0
            previous = None
            continue
        move_id = str(move_id)
        if move_id == previous:
            current += 1
        else:
            current = 1
            previous = move_id
        longest = max(longest, current)
    return longest


def _unique_selected_move_count(game_row: JsonObject) -> int:
    return len(
        {
            str(example.get("selected_move_id"))
            for example in _examples(game_row)
            if example.get("selected_move_id") is not None
        }
    )


def _policy_entropy(distribution: dict[str, Any]) -> float:
    total = sum(max(0.0, _as_float(value) or 0.0) for value in distribution.values())
    if total <= 0.0:
        return 0.0
    entropy = 0.0
    for value in distribution.values():
        probability = max(0.0, _as_float(value) or 0.0) / total
        if probability > 0.0:
            entropy -= probability * math.log(probability)
    return entropy


def _mean(values: list[float] | list[int]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt(value: Any) -> str:
    numeric = _as_float(value)
    if numeric is None:
        return "-"
    return f"{numeric:.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
