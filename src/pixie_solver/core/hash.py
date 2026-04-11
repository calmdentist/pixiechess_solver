from __future__ import annotations

import hashlib
from typing import Any

from pixie_solver.core.state import GameState
from pixie_solver.utils.serialization import canonical_json


def stable_digest(value: Any) -> str:
    serialized = canonical_json(value)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def stable_state_hash(state: GameState) -> str:
    return stable_digest(state.to_dict())


def stable_position_hash(state: GameState) -> str:
    payload = dict(state.to_dict())
    payload.pop("halfmove_clock", None)
    payload.pop("fullmove_number", None)
    payload.pop("repetition_counts", None)
    payload.pop("pending_events", None)
    payload.pop("metadata", None)
    return stable_digest(payload)
