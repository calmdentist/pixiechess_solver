from __future__ import annotations

from pixie_solver.core.event import Event
from pixie_solver.core.state import GameState
from pixie_solver.simulator.hooks import resolve_hooks


def resolve_event_cascade(
    state: GameState, seed_events: tuple[Event, ...]
) -> tuple[GameState, tuple[Event, ...]]:
    return resolve_hooks(state, seed_events)
