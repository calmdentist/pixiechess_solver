from __future__ import annotations

import math


def puct_score(*, parent_visits: int, child_visits: int, prior: float, value: float, c_puct: float = 1.5) -> float:
    exploration = c_puct * prior * math.sqrt(max(parent_visits, 1)) / (1 + child_visits)
    return value + exploration
