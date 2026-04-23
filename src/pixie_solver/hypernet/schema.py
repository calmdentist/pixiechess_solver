from __future__ import annotations

from dataclasses import dataclass, field

from pixie_solver.utils.serialization import JsonValue


@dataclass(frozen=True, slots=True)
class LayerModulation:
    layer_name: str
    scale: tuple[float, ...] = ()
    shift: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "layer_name", str(self.layer_name))
        object.__setattr__(self, "scale", tuple(float(value) for value in self.scale))
        object.__setattr__(self, "shift", tuple(float(value) for value in self.shift))

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "layer_name": self.layer_name,
            "scale": list(self.scale),
            "shift": list(self.shift),
        }


@dataclass(frozen=True, slots=True)
class AttentionBias:
    layer_name: str
    values: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "layer_name", str(self.layer_name))
        object.__setattr__(self, "values", tuple(float(value) for value in self.values))

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "layer_name": self.layer_name,
            "values": list(self.values),
        }


@dataclass(frozen=True, slots=True)
class GatingValues:
    layer_name: str
    values: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "layer_name", str(self.layer_name))
        object.__setattr__(self, "values", tuple(float(value) for value in self.values))

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "layer_name": self.layer_name,
            "values": list(self.values),
        }


@dataclass(frozen=True, slots=True)
class AdapterBundle:
    bundle_id: str
    world_digest: str
    strategy_digest: str | None = None
    layer_modulations: tuple[LayerModulation, ...] = ()
    attention_biases: tuple[AttentionBias, ...] = ()
    gating_values: tuple[GatingValues, ...] = ()
    metadata: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "bundle_id", str(self.bundle_id))
        object.__setattr__(self, "world_digest", str(self.world_digest))
        object.__setattr__(
            self,
            "strategy_digest",
            None if self.strategy_digest is None else str(self.strategy_digest),
        )
        object.__setattr__(
            self,
            "layer_modulations",
            tuple(self.layer_modulations),
        )
        object.__setattr__(
            self,
            "attention_biases",
            tuple(self.attention_biases),
        )
        object.__setattr__(
            self,
            "gating_values",
            tuple(self.gating_values),
        )
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "bundle_id": self.bundle_id,
            "world_digest": self.world_digest,
            "strategy_digest": self.strategy_digest,
            "layer_modulations": [
                component.to_dict()
                for component in self.layer_modulations
            ],
            "attention_biases": [
                component.to_dict()
                for component in self.attention_biases
            ],
            "gating_values": [
                component.to_dict()
                for component in self.gating_values
            ],
            "metadata": dict(self.metadata),
        }
