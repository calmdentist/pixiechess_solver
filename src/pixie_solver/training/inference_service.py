from __future__ import annotations

import os
import queue
import time
import uuid
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any

from pixie_solver.core import GameState, Move
from pixie_solver.model.policy_value import PolicyValueBatchMetrics, PolicyValueOutput
from pixie_solver.strategy import strategy_digest as compute_strategy_digest
from pixie_solver.training.checkpoint import load_training_checkpoint
from pixie_solver.utils.serialization import JsonValue, to_primitive


@dataclass(frozen=True, slots=True)
class BatchedInferenceConfig:
    max_batch_size: int = 32
    max_wait_ms: float = 5.0
    response_timeout_seconds: float = 300.0

    def __post_init__(self) -> None:
        if self.max_batch_size < 1:
            raise ValueError("max_batch_size must be at least 1")
        if self.max_wait_ms < 0.0:
            raise ValueError("max_wait_ms must be non-negative")
        if self.response_timeout_seconds <= 0.0:
            raise ValueError("response_timeout_seconds must be positive")


@dataclass(frozen=True, slots=True)
class BatchedInferenceStats:
    requests_completed: int
    batches_completed: int
    max_batch_size_seen: int
    errors: int
    total_legal_moves: int = 0
    uptime_ms: float = 0.0
    queue_wait_ms_total: float = 0.0
    request_latency_ms_total: float = 0.0
    deserialize_ms_total: float = 0.0
    model_total_ms: float = 0.0
    model_board_encode_ms_total: float = 0.0
    model_transformer_ms_total: float = 0.0
    model_move_encode_ms_total: float = 0.0
    model_policy_head_ms_total: float = 0.0
    model_value_head_ms_total: float = 0.0
    model_consequence_total_ms_total: float = 0.0
    model_consequence_apply_move_ms_total: float = 0.0
    model_consequence_terminal_check_ms_total: float = 0.0
    model_consequence_check_eval_ms_total: float = 0.0

    def to_dict(self) -> dict[str, JsonValue]:
        requests_completed = max(self.requests_completed, 0)
        batches_completed = max(self.batches_completed, 0)
        uptime_seconds = self.uptime_ms / 1000.0
        return {
            "requests_completed": self.requests_completed,
            "batches_completed": self.batches_completed,
            "max_batch_size_seen": self.max_batch_size_seen,
            "errors": self.errors,
            "total_legal_moves": self.total_legal_moves,
            "uptime_ms": self.uptime_ms,
            "queue_wait_ms_total": self.queue_wait_ms_total,
            "request_latency_ms_total": self.request_latency_ms_total,
            "deserialize_ms_total": self.deserialize_ms_total,
            "model_total_ms": self.model_total_ms,
            "model_board_encode_ms_total": self.model_board_encode_ms_total,
            "model_transformer_ms_total": self.model_transformer_ms_total,
            "model_move_encode_ms_total": self.model_move_encode_ms_total,
            "model_policy_head_ms_total": self.model_policy_head_ms_total,
            "model_value_head_ms_total": self.model_value_head_ms_total,
            "model_consequence_total_ms_total": self.model_consequence_total_ms_total,
            "model_consequence_apply_move_ms_total": self.model_consequence_apply_move_ms_total,
            "model_consequence_terminal_check_ms_total": self.model_consequence_terminal_check_ms_total,
            "model_consequence_check_eval_ms_total": self.model_consequence_check_eval_ms_total,
            "average_batch_size": _safe_divide(requests_completed, batches_completed),
            "average_legal_moves_per_request": _safe_divide(
                self.total_legal_moves,
                requests_completed,
            ),
            "average_queue_wait_ms": _safe_divide(
                self.queue_wait_ms_total,
                requests_completed,
            ),
            "average_request_latency_ms": _safe_divide(
                self.request_latency_ms_total,
                requests_completed,
            ),
            "average_deserialize_ms_per_batch": _safe_divide(
                self.deserialize_ms_total,
                batches_completed,
            ),
            "average_model_ms_per_batch": _safe_divide(
                self.model_total_ms,
                batches_completed,
            ),
            "average_model_ms_per_request": _safe_divide(
                self.model_total_ms,
                requests_completed,
            ),
            "requests_per_second": _safe_divide(requests_completed, uptime_seconds),
            "batches_per_second": _safe_divide(batches_completed, uptime_seconds),
        }


@dataclass(slots=True)
class BatchedInferenceClient:
    request_queue: Any
    response_dict: Any
    timeout_seconds: float

    def infer(
        self,
        state: GameState,
        legal_moves: tuple[Move, ...] | list[Move],
        *,
        strategy: object | None = None,
    ) -> PolicyValueOutput:
        request_id = f"{os.getpid()}-{time.monotonic_ns()}-{uuid.uuid4().hex}"
        strategy_payload = None if strategy is None else dict(to_primitive(strategy))
        self.request_queue.put(
            {
                "type": "infer",
                "request_id": request_id,
                "submitted_at_ns": time.monotonic_ns(),
                "state": state.to_dict(),
                "legal_moves": [move.to_dict() for move in legal_moves],
                "strategy": strategy_payload,
                "strategy_digest": (
                    None
                    if strategy_payload is None
                    else compute_strategy_digest(strategy_payload)
                ),
            }
        )
        deadline = time.monotonic() + self.timeout_seconds
        while time.monotonic() < deadline:
            response = self.response_dict.pop(request_id, None)
            if response is None:
                time.sleep(0.001)
                continue
            if response.get("error") is not None:
                raise RuntimeError(str(response["error"]))
            return PolicyValueOutput(
                policy_logits={
                    str(move_id): float(logit)
                    for move_id, logit in dict(response["policy_logits"]).items()
                },
                value=float(response["value"]),
                uncertainty=float(response.get("uncertainty", 0.0)),
            )
        raise TimeoutError(f"Timed out waiting for inference response {request_id}")


class BatchedInferenceService:
    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str | None = None,
        config: BatchedInferenceConfig | None = None,
    ) -> None:
        self.checkpoint_path = str(checkpoint_path)
        self.device = device
        self.config = BatchedInferenceConfig() if config is None else config
        self._ctx = get_context("spawn")
        self._manager = self._ctx.Manager()
        self._request_queue = self._manager.Queue()
        self._response_dict = self._manager.dict()
        self._stats_dict = self._manager.dict(
            {
                "requests_completed": 0,
                "batches_completed": 0,
                "max_batch_size_seen": 0,
                "errors": 0,
                "total_legal_moves": 0,
                "uptime_ms": 0.0,
                "queue_wait_ms_total": 0.0,
                "request_latency_ms_total": 0.0,
                "deserialize_ms_total": 0.0,
                "model_total_ms": 0.0,
                "model_board_encode_ms_total": 0.0,
                "model_transformer_ms_total": 0.0,
                "model_move_encode_ms_total": 0.0,
                "model_policy_head_ms_total": 0.0,
                "model_value_head_ms_total": 0.0,
                "model_consequence_total_ms_total": 0.0,
                "model_consequence_apply_move_ms_total": 0.0,
                "model_consequence_terminal_check_ms_total": 0.0,
                "model_consequence_check_eval_ms_total": 0.0,
            }
        )
        self._process = self._ctx.Process(
            target=_serve_inference,
            args=(
                self.checkpoint_path,
                self.device,
                self.config,
                self._request_queue,
                self._response_dict,
                self._stats_dict,
            ),
        )

    def __enter__(self) -> "BatchedInferenceService":
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.stop()

    def start(self) -> None:
        if not self._process.is_alive():
            self._process.start()

    def stop(self) -> None:
        if self._process.is_alive():
            self._request_queue.put({"type": "stop"})
            self._process.join(timeout=10.0)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=5.0)
        self._manager.shutdown()

    def client(self) -> BatchedInferenceClient:
        return BatchedInferenceClient(
            request_queue=self._request_queue,
            response_dict=self._response_dict,
            timeout_seconds=self.config.response_timeout_seconds,
        )

    def stats(self) -> BatchedInferenceStats:
        return BatchedInferenceStats(
            requests_completed=int(self._stats_dict.get("requests_completed", 0)),
            batches_completed=int(self._stats_dict.get("batches_completed", 0)),
            max_batch_size_seen=int(self._stats_dict.get("max_batch_size_seen", 0)),
            errors=int(self._stats_dict.get("errors", 0)),
            total_legal_moves=int(self._stats_dict.get("total_legal_moves", 0)),
            uptime_ms=float(self._stats_dict.get("uptime_ms", 0.0)),
            queue_wait_ms_total=float(self._stats_dict.get("queue_wait_ms_total", 0.0)),
            request_latency_ms_total=float(
                self._stats_dict.get("request_latency_ms_total", 0.0)
            ),
            deserialize_ms_total=float(self._stats_dict.get("deserialize_ms_total", 0.0)),
            model_total_ms=float(self._stats_dict.get("model_total_ms", 0.0)),
            model_board_encode_ms_total=float(
                self._stats_dict.get("model_board_encode_ms_total", 0.0)
            ),
            model_transformer_ms_total=float(
                self._stats_dict.get("model_transformer_ms_total", 0.0)
            ),
            model_move_encode_ms_total=float(
                self._stats_dict.get("model_move_encode_ms_total", 0.0)
            ),
            model_policy_head_ms_total=float(
                self._stats_dict.get("model_policy_head_ms_total", 0.0)
            ),
            model_value_head_ms_total=float(
                self._stats_dict.get("model_value_head_ms_total", 0.0)
            ),
            model_consequence_total_ms_total=float(
                self._stats_dict.get("model_consequence_total_ms_total", 0.0)
            ),
            model_consequence_apply_move_ms_total=float(
                self._stats_dict.get("model_consequence_apply_move_ms_total", 0.0)
            ),
            model_consequence_terminal_check_ms_total=float(
                self._stats_dict.get("model_consequence_terminal_check_ms_total", 0.0)
            ),
            model_consequence_check_eval_ms_total=float(
                self._stats_dict.get("model_consequence_check_eval_ms_total", 0.0)
            ),
        )


def _serve_inference(
    checkpoint_path: str,
    device: str | None,
    config: BatchedInferenceConfig,
    request_queue,
    response_dict,
    stats_dict,
) -> None:
    model = load_training_checkpoint(checkpoint_path, device=device).model
    model.eval()
    service_start_ns = time.monotonic_ns()
    stats = {
        "requests_completed": 0,
        "batches_completed": 0,
        "max_batch_size_seen": 0,
        "errors": 0,
        "total_legal_moves": 0,
        "uptime_ms": 0.0,
        "queue_wait_ms_total": 0.0,
        "request_latency_ms_total": 0.0,
        "deserialize_ms_total": 0.0,
        "model_total_ms": 0.0,
        "model_board_encode_ms_total": 0.0,
        "model_transformer_ms_total": 0.0,
        "model_move_encode_ms_total": 0.0,
        "model_policy_head_ms_total": 0.0,
        "model_value_head_ms_total": 0.0,
        "model_consequence_total_ms_total": 0.0,
        "model_consequence_apply_move_ms_total": 0.0,
        "model_consequence_terminal_check_ms_total": 0.0,
        "model_consequence_check_eval_ms_total": 0.0,
    }
    while True:
        first_request = request_queue.get()
        if first_request.get("type") == "stop":
            stats["uptime_ms"] = _elapsed_ms(service_start_ns)
            _write_stats(stats_dict, stats)
            return
        batch = [first_request]
        deadline = time.monotonic() + config.max_wait_ms / 1000.0
        while len(batch) < config.max_batch_size:
            timeout = max(0.0, deadline - time.monotonic())
            try:
                request = request_queue.get(timeout=timeout)
            except queue.Empty:
                break
            if request.get("type") == "stop":
                request_queue.put(request)
                break
            batch.append(request)
            if config.max_wait_ms == 0.0:
                break

        batch_start_ns = time.monotonic_ns()
        queue_wait_ms_total = sum(
            _elapsed_ms(int(request.get("submitted_at_ns", batch_start_ns)), end_ns=batch_start_ns)
            for request in batch
        )
        deserialize_ms = 0.0
        model_metrics = PolicyValueBatchMetrics()
        try:
            deserialize_start = time.perf_counter()
            states = [
                GameState.from_dict(dict(request["state"]))
                for request in batch
            ]
            move_batches = [
                tuple(Move.from_dict(dict(move)) for move in request["legal_moves"])
                for request in batch
            ]
            deserialize_ms = (time.perf_counter() - deserialize_start) * 1000.0
            outputs = [None] * len(batch)
            strategy_groups: dict[str | None, list[int]] = {}
            for index, request in enumerate(batch):
                strategy_groups.setdefault(
                    (
                        None
                        if request.get("strategy_digest") is None
                        else str(request["strategy_digest"])
                    ),
                    [],
                ).append(index)
            for group_indices in strategy_groups.values():
                group_strategy = batch[group_indices[0]].get("strategy")
                group_requests = tuple(
                    (states[index], move_batches[index])
                    for index in group_indices
                )
                group_outputs, group_metrics = model.infer_batch_with_metrics(
                    group_requests,
                    strategy=group_strategy,
                )
                model_metrics = PolicyValueBatchMetrics(
                    requests=model_metrics.requests + group_metrics.requests,
                    total_legal_moves=(
                        model_metrics.total_legal_moves + group_metrics.total_legal_moves
                    ),
                    total_ms=model_metrics.total_ms + group_metrics.total_ms,
                    board_encode_ms=(
                        model_metrics.board_encode_ms + group_metrics.board_encode_ms
                    ),
                    transformer_ms=(
                        model_metrics.transformer_ms + group_metrics.transformer_ms
                    ),
                    move_encode_ms=(
                        model_metrics.move_encode_ms + group_metrics.move_encode_ms
                    ),
                    policy_head_ms=(
                        model_metrics.policy_head_ms + group_metrics.policy_head_ms
                    ),
                    value_head_ms=(
                        model_metrics.value_head_ms + group_metrics.value_head_ms
                    ),
                    consequence_total_ms=(
                        model_metrics.consequence_total_ms
                        + group_metrics.consequence_total_ms
                    ),
                    consequence_apply_move_ms=(
                        model_metrics.consequence_apply_move_ms
                        + group_metrics.consequence_apply_move_ms
                    ),
                    consequence_terminal_check_ms=(
                        model_metrics.consequence_terminal_check_ms
                        + group_metrics.consequence_terminal_check_ms
                    ),
                    consequence_check_eval_ms=(
                        model_metrics.consequence_check_eval_ms
                        + group_metrics.consequence_check_eval_ms
                    ),
                )
                for index, output in zip(group_indices, group_outputs, strict=True):
                    outputs[index] = output
        except Exception as exc:
            stats["errors"] += len(batch)
            stats["uptime_ms"] = _elapsed_ms(service_start_ns)
            _write_stats(stats_dict, stats)
            for request in batch:
                response_dict[str(request["request_id"])] = {"error": str(exc)}
            continue

        response_ready_ns = time.monotonic_ns()
        for request, output in zip(batch, outputs, strict=True):
            response_dict[str(request["request_id"])] = {
                "policy_logits": dict(output.policy_logits),
                "value": output.value,
                "uncertainty": output.uncertainty,
                "error": None,
            }
        stats["requests_completed"] += len(batch)
        stats["batches_completed"] += 1
        stats["max_batch_size_seen"] = max(
            int(stats["max_batch_size_seen"]),
            len(batch),
        )
        stats["total_legal_moves"] += model_metrics.total_legal_moves
        stats["queue_wait_ms_total"] += queue_wait_ms_total
        stats["request_latency_ms_total"] += sum(
            _elapsed_ms(
                int(request.get("submitted_at_ns", response_ready_ns)),
                end_ns=response_ready_ns,
            )
            for request in batch
        )
        stats["deserialize_ms_total"] += deserialize_ms
        stats["model_total_ms"] += model_metrics.total_ms
        stats["model_board_encode_ms_total"] += model_metrics.board_encode_ms
        stats["model_transformer_ms_total"] += model_metrics.transformer_ms
        stats["model_move_encode_ms_total"] += model_metrics.move_encode_ms
        stats["model_policy_head_ms_total"] += model_metrics.policy_head_ms
        stats["model_value_head_ms_total"] += model_metrics.value_head_ms
        stats["model_consequence_total_ms_total"] += model_metrics.consequence_total_ms
        stats["model_consequence_apply_move_ms_total"] += (
            model_metrics.consequence_apply_move_ms
        )
        stats["model_consequence_terminal_check_ms_total"] += (
            model_metrics.consequence_terminal_check_ms
        )
        stats["model_consequence_check_eval_ms_total"] += (
            model_metrics.consequence_check_eval_ms
        )
        stats["uptime_ms"] = _elapsed_ms(service_start_ns)
        _write_stats(stats_dict, stats)


def _safe_divide(numerator: float | int, denominator: float | int) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def _elapsed_ms(start_ns: int, *, end_ns: int | None = None) -> float:
    finish_ns = time.monotonic_ns() if end_ns is None else end_ns
    return max(0.0, (finish_ns - start_ns) / 1_000_000.0)


def _write_stats(stats_dict, stats: dict[str, int | float]) -> None:
    for key, value in stats.items():
        stats_dict[key] = value
