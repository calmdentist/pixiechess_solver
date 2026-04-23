from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from pixie_solver.hypernet.schema import AdapterBundle


@dataclass(slots=True)
class PreparedAdapterBundle:
    bundle_id: str
    layer_modulations: dict[str, tuple[Tensor, Tensor]]
    attention_biases: dict[str, Tensor]
    gating_values: dict[str, Tensor]


def prepare_adapter_bundle(
    bundle: AdapterBundle | None,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> PreparedAdapterBundle | None:
    if bundle is None:
        return None
    return PreparedAdapterBundle(
        bundle_id=bundle.bundle_id,
        layer_modulations={
            component.layer_name: (
                _to_tensor(component.scale, device=device, dtype=dtype, default=1.0),
                _to_tensor(component.shift, device=device, dtype=dtype, default=0.0),
            )
            for component in bundle.layer_modulations
        },
        attention_biases={
            component.layer_name: _to_tensor(
                component.values,
                device=device,
                dtype=dtype,
                default=0.0,
            )
            for component in bundle.attention_biases
        },
        gating_values={
            component.layer_name: _to_tensor(
                component.values,
                device=device,
                dtype=dtype,
                default=1.0,
            )
            for component in bundle.gating_values
        },
    )


def apply_layer_adapter(
    tokens: Tensor,
    bundle: PreparedAdapterBundle | None,
    *,
    layer_name: str,
) -> Tensor:
    if bundle is None or tokens.numel() == 0:
        return tokens
    output = tokens
    modulation = bundle.layer_modulations.get(layer_name)
    if modulation is not None:
        scale, shift = modulation
        output = output * _broadcast_vector(scale, output) + _broadcast_vector(shift, output)
    gate = bundle.gating_values.get(layer_name)
    if gate is not None:
        output = output * _broadcast_vector(gate, output)
    return output


def attention_bias_for_layer(
    bundle: PreparedAdapterBundle | None,
    *,
    layer_name: str,
) -> Tensor | None:
    if bundle is None:
        return None
    return bundle.attention_biases.get(layer_name)


def _to_tensor(
    values: tuple[float, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
    default: float,
) -> Tensor:
    if not values:
        return torch.tensor([default], dtype=dtype, device=device)
    return torch.tensor(values, dtype=dtype, device=device)


def _broadcast_vector(vector: Tensor, reference: Tensor) -> Tensor:
    if vector.numel() not in (1, reference.shape[-1]):
        raise ValueError(
            "adapter vector length must be 1 or match the final token dimension"
        )
    view_shape = [1] * reference.dim()
    view_shape[-1] = vector.numel()
    return vector.view(*view_shape)
