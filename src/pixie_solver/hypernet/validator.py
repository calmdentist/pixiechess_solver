from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from pixie_solver.hypernet.schema import (
    AdapterBundle,
    AttentionBias,
    GatingValues,
    LayerModulation,
)

IDENTIFIER_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
MAX_BUNDLE_COMPONENTS = 256
MAX_ADAPTER_VECTOR_LENGTH = 4096


class AdapterBundleValidationError(ValueError):
    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("\n".join(errors))


def collect_adapter_bundle_validation_errors(
    bundle: AdapterBundle | Mapping[str, Any],
) -> list[str]:
    errors: list[str] = []
    candidate = _coerce_bundle(bundle)

    if not candidate.bundle_id or not IDENTIFIER_PATTERN.fullmatch(candidate.bundle_id):
        errors.append("bundle_id must be a lowercase identifier")
    if not candidate.world_digest.strip():
        errors.append("world_digest must be a non-empty string")
    if candidate.strategy_digest is not None and not candidate.strategy_digest.strip():
        errors.append("strategy_digest must be null or a non-empty string")
    if len(candidate.layer_modulations) > MAX_BUNDLE_COMPONENTS:
        errors.append(
            f"layer_modulations may contain at most {MAX_BUNDLE_COMPONENTS} items"
        )
    if len(candidate.attention_biases) > MAX_BUNDLE_COMPONENTS:
        errors.append(
            f"attention_biases may contain at most {MAX_BUNDLE_COMPONENTS} items"
        )
    if len(candidate.gating_values) > MAX_BUNDLE_COMPONENTS:
        errors.append(
            f"gating_values may contain at most {MAX_BUNDLE_COMPONENTS} items"
        )

    _validate_layer_names(
        errors,
        candidate.layer_modulations,
        field_name="layer_modulations",
    )
    _validate_layer_names(
        errors,
        candidate.attention_biases,
        field_name="attention_biases",
    )
    _validate_layer_names(
        errors,
        candidate.gating_values,
        field_name="gating_values",
    )

    for index, modulation in enumerate(candidate.layer_modulations):
        if not modulation.layer_name.strip():
            errors.append(
                f"layer_modulations[{index}].layer_name must be a non-empty string"
            )
        if (
            modulation.scale
            and modulation.shift
            and len(modulation.scale) != len(modulation.shift)
        ):
            errors.append(
                f"layer_modulations[{index}] scale and shift lengths must match when both are present"
            )
        _validate_vector(
            errors,
            modulation.scale,
            field_name=f"layer_modulations[{index}].scale",
        )
        _validate_vector(
            errors,
            modulation.shift,
            field_name=f"layer_modulations[{index}].shift",
        )

    for index, bias in enumerate(candidate.attention_biases):
        if not bias.layer_name.strip():
            errors.append(
                f"attention_biases[{index}].layer_name must be a non-empty string"
            )
        _validate_vector(
            errors,
            bias.values,
            field_name=f"attention_biases[{index}].values",
        )

    for index, gate in enumerate(candidate.gating_values):
        if not gate.layer_name.strip():
            errors.append(
                f"gating_values[{index}].layer_name must be a non-empty string"
            )
        _validate_vector(
            errors,
            gate.values,
            field_name=f"gating_values[{index}].values",
        )

    if not isinstance(candidate.metadata, Mapping):
        errors.append("metadata must be a mapping")

    return errors


def validate_adapter_bundle(bundle: AdapterBundle | Mapping[str, Any]) -> None:
    errors = collect_adapter_bundle_validation_errors(bundle)
    if errors:
        raise AdapterBundleValidationError(errors)


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
            for item in _mapping_sequence(bundle.get("layer_modulations", ()))
        ),
        attention_biases=tuple(
            AttentionBias(
                layer_name=str(item.get("layer_name", "")),
                values=tuple(item.get("values", ())),
            )
            for item in _mapping_sequence(bundle.get("attention_biases", ()))
        ),
        gating_values=tuple(
            GatingValues(
                layer_name=str(item.get("layer_name", "")),
                values=tuple(item.get("values", ())),
            )
            for item in _mapping_sequence(bundle.get("gating_values", ()))
        ),
        metadata=dict(bundle.get("metadata", {})),
    )


def _mapping_sequence(value: Any) -> Sequence[Mapping[str, Any]]:
    if value is None:
        return ()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and all(
        isinstance(item, Mapping) for item in value
    ):
        return value
    raise TypeError("adapter bundle component collections must be sequences of mappings")


def _validate_layer_names(
    errors: list[str],
    components: Sequence[LayerModulation | AttentionBias | GatingValues],
    *,
    field_name: str,
) -> None:
    layer_names = [component.layer_name for component in components if component.layer_name]
    duplicates = sorted({name for name in layer_names if layer_names.count(name) > 1})
    if duplicates:
        errors.append(
            f"{field_name} layer_name values must be unique: " + ", ".join(duplicates)
        )


def _validate_vector(
    errors: list[str],
    values: Sequence[float],
    *,
    field_name: str,
) -> None:
    if len(values) > MAX_ADAPTER_VECTOR_LENGTH:
        errors.append(
            f"{field_name} may contain at most {MAX_ADAPTER_VECTOR_LENGTH} values"
        )
