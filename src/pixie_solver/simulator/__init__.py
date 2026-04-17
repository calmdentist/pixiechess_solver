from pixie_solver.simulator.actiongen import (
    enumerate_shadow_legal_actions,
    enumerate_shadow_pseudo_legal_actions_for_piece,
)
from pixie_solver.simulator.engine import apply_move, is_terminal, result
from pixie_solver.simulator.invariants import assert_state_invariants
from pixie_solver.simulator.movegen import (
    is_in_check,
    is_square_attacked,
    king_for_color,
    legal_moves,
    pseudo_legal_moves,
)
from pixie_solver.simulator.query import (
    enumerate_query_facts,
    enumerate_query_facts_for_piece,
    is_king_capturable,
    is_square_capturable,
    query_fact_exists,
)
from pixie_solver.simulator.resolution import (
    apply_action_shadow,
    apply_action_shadow_unchecked,
    resolve_program_event_cascade,
)

__all__ = [
    "apply_move",
    "apply_action_shadow",
    "apply_action_shadow_unchecked",
    "assert_state_invariants",
    "enumerate_shadow_legal_actions",
    "enumerate_shadow_pseudo_legal_actions_for_piece",
    "enumerate_query_facts",
    "enumerate_query_facts_for_piece",
    "is_in_check",
    "is_king_capturable",
    "is_square_attacked",
    "is_square_capturable",
    "is_terminal",
    "king_for_color",
    "legal_moves",
    "pseudo_legal_moves",
    "query_fact_exists",
    "resolve_program_event_cascade",
    "result",
]
