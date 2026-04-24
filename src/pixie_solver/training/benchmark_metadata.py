from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence

from pixie_solver.core.hash import stable_digest
from pixie_solver.core.piece import PieceClass
from pixie_solver.core.state import GameState
from pixie_solver.program.lower_legacy_dsl import lower_legacy_piece_class
from pixie_solver.strategy import strategy_digest as compute_strategy_digest
from pixie_solver.utils.serialization import JsonValue

FOUNDATION_FAMILY_ID = "foundation"
FOUNDATION_SPLIT = "foundation"
FOUNDATION_NOVELTY_TIER = "foundation"
COMPOSITION_FAMILY_ID = "composition"
MIXED_SPLIT = "mixed"
COMPOSITION_NOVELTY_TIER = "composition"
SEARCH_ONLY_MODEL_ARCHITECTURE = "search_only"


def world_model_digest_for_state(state: GameState) -> str:
    return world_model_digest_for_piece_classes(state.piece_classes)


def world_model_digest_for_piece_classes(
    piece_classes: Mapping[str, PieceClass] | Sequence[PieceClass],
) -> str:
    canonical_programs = tuple(
        lower_legacy_piece_class(piece_class)
        for piece_class in _sorted_piece_classes(piece_classes)
    )
    return stable_digest(canonical_programs)


def benchmark_metadata_for_state(
    state: GameState,
    metadata: Mapping[str, JsonValue] | None = None,
    *,
    search_budget: int | None = None,
    model_architecture: str | None = None,
) -> dict[str, JsonValue]:
    payload = dict(metadata or {})
    payload["world_model_digest"] = world_model_digest_for_state(state)

    strategy_digest = _strategy_digest_from_metadata(payload)
    if strategy_digest is not None:
        payload["strategy_digest"] = strategy_digest

    effective_search_budget = (
        _coerce_int(search_budget)
        if search_budget is not None
        else search_budget_from_metadata(payload)
    )
    if effective_search_budget is not None:
        payload["search_budget"] = effective_search_budget

    if model_architecture is not None:
        payload["model_architecture"] = str(model_architecture)

    piece_training_metadata = _coerce_piece_metadata(
        payload.get("verified_piece_training_metadata")
    )
    if piece_training_metadata:
        payload.update(world_labels_from_piece_training_metadata(piece_training_metadata))
    else:
        payload.setdefault("family_id", FOUNDATION_FAMILY_ID)
        payload.setdefault("split", FOUNDATION_SPLIT)
        payload.setdefault("novelty_tier", FOUNDATION_NOVELTY_TIER)
        payload.setdefault("admission_cycle", None)
        payload.setdefault("world_family_ids", [])
        payload.setdefault("world_splits", [])
        payload.setdefault("world_novelty_tiers", [])
        payload.setdefault("world_admission_cycles", [])

    return payload


def present_piece_metadata_for_state(
    state: GameState,
    metadata_by_piece_id: Mapping[str, JsonValue],
) -> dict[str, JsonValue]:
    present_piece_ids = sorted(
        piece_id
        for piece_id in state.piece_classes
        if piece_id in metadata_by_piece_id
    )
    return {
        piece_id: metadata_by_piece_id[piece_id]
        for piece_id in present_piece_ids
    }


def world_labels_from_piece_training_metadata(
    metadata_by_piece_id: Mapping[str, JsonValue],
) -> dict[str, JsonValue]:
    normalized = _coerce_piece_metadata(metadata_by_piece_id)
    family_ids = sorted(
        {
            str(piece_metadata["family_id"])
            for piece_metadata in normalized.values()
            if piece_metadata.get("family_id") is not None
        }
    )
    splits = sorted(
        {
            str(piece_metadata["split"])
            for piece_metadata in normalized.values()
            if piece_metadata.get("split") is not None
        }
    )
    novelty_tiers = sorted(
        {
            str(piece_metadata["novelty_tier"])
            for piece_metadata in normalized.values()
            if piece_metadata.get("novelty_tier") is not None
        }
    )
    admission_cycles = sorted(
        {
            admission_cycle
            for piece_metadata in normalized.values()
            for admission_cycle in (_coerce_int(piece_metadata.get("admission_cycle")),)
            if admission_cycle is not None
        }
    )

    if not family_ids:
        family_id = FOUNDATION_FAMILY_ID
    elif len(family_ids) == 1:
        family_id = family_ids[0]
    else:
        family_id = COMPOSITION_FAMILY_ID

    if not splits:
        split = FOUNDATION_SPLIT
    elif len(splits) == 1:
        split = splits[0]
    else:
        split = MIXED_SPLIT

    if not novelty_tiers:
        novelty_tier = FOUNDATION_NOVELTY_TIER
    elif len(novelty_tiers) == 1:
        novelty_tier = novelty_tiers[0]
    else:
        novelty_tier = COMPOSITION_NOVELTY_TIER

    return {
        "family_id": family_id,
        "split": split,
        "novelty_tier": novelty_tier,
        "admission_cycle": (
            admission_cycles[-1] if admission_cycles else None
        ),
        "world_family_ids": family_ids,
        "world_splits": splits,
        "world_novelty_tiers": novelty_tiers,
        "world_admission_cycles": admission_cycles,
    }


def summarize_benchmark_metadata_records(
    metadata_records: Iterable[Mapping[str, JsonValue]],
) -> dict[str, JsonValue]:
    record_count = 0
    family_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    novelty_counts: Counter[str] = Counter()
    model_architecture_counts: Counter[str] = Counter()
    world_digests: set[str] = set()
    strategy_digests: set[str] = set()
    search_budgets: list[int] = []
    admission_cycles: list[int] = []

    for metadata_record in metadata_records:
        record_count += 1
        if metadata_record.get("family_id") is not None:
            family_counts[str(metadata_record["family_id"])] += 1
        if metadata_record.get("split") is not None:
            split_counts[str(metadata_record["split"])] += 1
        if metadata_record.get("novelty_tier") is not None:
            novelty_counts[str(metadata_record["novelty_tier"])] += 1
        if metadata_record.get("model_architecture") is not None:
            model_architecture_counts[str(metadata_record["model_architecture"])] += 1
        if metadata_record.get("world_model_digest") is not None:
            world_digests.add(str(metadata_record["world_model_digest"]))
        if metadata_record.get("strategy_digest") is not None:
            strategy_digests.add(str(metadata_record["strategy_digest"]))
        search_budget = search_budget_from_metadata(metadata_record)
        if search_budget is not None:
            search_budgets.append(search_budget)
        admission_cycle = _coerce_int(metadata_record.get("admission_cycle"))
        if admission_cycle is not None:
            admission_cycles.append(admission_cycle)

    return {
        "records": record_count,
        "family_counts": _sorted_counter_dict(family_counts),
        "split_counts": _sorted_counter_dict(split_counts),
        "novelty_tier_counts": _sorted_counter_dict(novelty_counts),
        "model_architecture_counts": _sorted_counter_dict(model_architecture_counts),
        "unique_worlds": len(world_digests),
        "unique_strategy_digests": len(strategy_digests),
        "search_budget": _numeric_summary(search_budgets),
        "admission_cycle": _numeric_summary(admission_cycles),
    }


def search_budget_from_metadata(
    metadata: Mapping[str, JsonValue],
) -> int | None:
    for key in (
        "search_budget",
        "simulations_used",
        "simulations",
        "simulations_requested",
    ):
        value = _coerce_int(metadata.get(key))
        if value is not None:
            return value
    return None


def _strategy_digest_from_metadata(
    metadata: Mapping[str, JsonValue],
) -> str | None:
    if metadata.get("strategy_digest") is not None:
        return str(metadata["strategy_digest"])
    strategy = metadata.get("strategy")
    if isinstance(strategy, Mapping):
        return compute_strategy_digest(strategy)
    return None


def _sorted_piece_classes(
    piece_classes: Mapping[str, PieceClass] | Sequence[PieceClass],
) -> tuple[PieceClass, ...]:
    if isinstance(piece_classes, Mapping):
        values = piece_classes.values()
    else:
        values = piece_classes
    return tuple(sorted(values, key=lambda piece_class: piece_class.class_id))


def _coerce_piece_metadata(
    value: JsonValue | Mapping[str, JsonValue] | None,
) -> dict[str, dict[str, JsonValue]]:
    if not isinstance(value, Mapping):
        return {}
    normalized: dict[str, dict[str, JsonValue]] = {}
    for piece_id, piece_metadata in value.items():
        if isinstance(piece_metadata, Mapping):
            normalized[str(piece_id)] = {
                str(key): piece_metadata[key]
                for key in sorted(piece_metadata)
            }
    return normalized


def _coerce_int(value: JsonValue | int | None) -> int | None:
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


def _sorted_counter_dict(counter: Counter[str]) -> dict[str, int]:
    return {
        key: counter[key]
        for key in sorted(counter)
    }


def _numeric_summary(values: Sequence[int]) -> dict[str, JsonValue] | None:
    if not values:
        return None
    total = sum(values)
    return {
        "min": min(values),
        "max": max(values),
        "average": total / len(values),
    }
