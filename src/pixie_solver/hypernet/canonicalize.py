from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
from typing import Any

from pixie_solver.hypernet.schema import (
    AdapterBundle,
    AttentionBias,
    GatingValues,
    LayerModulation,
)
from pixie_solver.hypernet.validator import validate_adapter_bundle
from pixie_solver.utils.serialization import JsonValue, canonical_json

FLOAT_PRECISION = 10


def canonicalize_adapter_bundle(
    bundle: AdapterBundle | Mapping[str, Any],
) -> dict[str, JsonValue]:
    candidate = _coerce_bundle(bundle)
    validate_adapter_bundle(candidate)
    return {
        "bundle_id": candidate.bundle_id,
        "world_digest": candidate.world_digest,
        "strategy_digest": candidate.strategy_digest,
        "layer_modulations": [
            _canonicalize_layer_modulation(component)
            for component in sorted(
                candidate.layer_modulations,
                key=lambda component: component.layer_name,
            )
        ],
        "attention_biases": [
            _canonicalize_attention_bias(component)
            for component in sorted(
                candidate.attention_biases,
                key=lambda component: component.layer_name,
            )
        ],
        "gating_values": [
            _canonicalize_gating_values(component)
            for component in sorted(
                candidate.gating_values,
                key=lambda component: component.layer_name,
            )
        ],
        "metadata": deepcopy(dict(candidate.metadata)),
    }


def adapter_bundle_digest(bundle: AdapterBundle | Mapping[str, Any]) -> str:
    payload = canonical_json(canonicalize_adapter_bundle(bundle))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _coerce_bundle(bundle: AdapterBundle | Mapping[str, Any]) -> AdapterBundle:
    if isinstance(bundle, AdapterBundle):
        return bundle
    if not isinstance(bundle, Mapping):
        raise TypeError(
            "adapter bundle must be an AdapterBundle or a mapping compatible with AdapterBundle"
        )
    return AdapterBundle(
        bundle_id=str(bundle.get("bundle_id", "")),
        world_digest=str(bundle.get("world_digest", "")),
        strategy_digest=bundle.get("strategy_digest"),
        layer_modulations=tuple(
            LayerModulation(
                layer_name=str(item.get("layer_name", "")),
                scale=tuple(item.get("scale", ())),
                shift=tuple(item.get("shift", ())),
            )
            for item in bundle.get("layer_modulations", ())
        ),
        attention_biases=tuple(
            AttentionBias(
                layer_name=str(item.get("layer_name", "")),
                values=tuple(item.get("values", ())),
            )
            for item in bundle.get("attention_biases", ())
        ),
        gating_values=tuple(
            GatingValues(
                layer_name=str(item.get("layer_name", "")),
                values=tuple(item.get("values", ())),
            )
            for item in bundle.get("gating_values", ())
        ),
        metadata=dict(bundle.get("metadata", {})),
    )


def _canonicalize_layer_modulation(
    component: LayerModulation,
) -> dict[str, JsonValue]:
    return {
        "layer_name": component.layer_name,
        "scale": [_canonicalize_float(value) for value in component.scale],
        "shift": [_canonicalize_float(value) for value in component.shift],
    }


def _canonicalize_attention_bias(component: AttentionBias) -> dict[str, JsonValue]:
    return {
        "layer_name": component.layer_name,
        "values": [_canonicalize_float(value) for value in component.values],
    }


def _canonicalize_gating_values(component: GatingValues) -> dict[str, JsonValue]:
    return {
        "layer_name": component.layer_name,
        "values": [_canonicalize_float(value) for value in component.values],
    }


def _canonicalize_float(value: float) -> float:
    normalized = round(float(value), FLOAT_PRECISION)
    if normalized == 0.0:
        return 0.0
    return normalized
