from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class SearchNode:
    state_hash: str
    prior: float = 0.0
    visit_count: int = 0
    value_sum: float = 0.0

    @property
    def mean_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
