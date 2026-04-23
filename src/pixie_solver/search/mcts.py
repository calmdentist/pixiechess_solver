from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Protocol

from pixie_solver.core.move import Move
from pixie_solver.core.piece import BasePieceType, Color, PieceClass
from pixie_solver.core.state import GameState
from pixie_solver.core import stable_move_id
from pixie_solver.model.policy_value import PolicyValueModel
from pixie_solver.search.node import SearchEdge, SearchNode
from pixie_solver.search.puct import puct_score
from pixie_solver.simulator.engine import apply_move, result
from pixie_solver.simulator.movegen import is_in_check, legal_moves
from pixie_solver.simulator.transition import other_color
from pixie_solver.utils.serialization import JsonValue

DEFAULT_SIMULATIONS = 64


@dataclass(frozen=True, slots=True)
class DirichletRootNoise:
    alpha: float = 0.3
    exploration_fraction: float = 0.25

    def __post_init__(self) -> None:
        if self.alpha <= 0.0:
            raise ValueError("Dirichlet alpha must be positive")
        if self.exploration_fraction < 0.0 or self.exploration_fraction > 1.0:
            raise ValueError("root exploration fraction must be in [0, 1]")


@dataclass(slots=True)
class SearchResult:
    selected_move: Move | None
    selected_move_id: str | None = None
    legal_moves: tuple[Move, ...] = ()
    visit_distribution: dict[str, float] = field(default_factory=dict)
    visit_counts: dict[str, int] = field(default_factory=dict)
    root_value: float = 0.0
    policy_logits: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, JsonValue] = field(default_factory=dict)


class StateEvaluator(Protocol):
    def evaluate(self, state: GameState) -> float:
        """Return a value in [-1, 1] from the perspective of state.side_to_move."""


@dataclass(slots=True)
class HeuristicEvaluator:
    mobility_weight: float = 0.05
    check_weight: float = 0.2
    scale: float = 12.0

    def evaluate(self, state: GameState) -> float:
        terminal = result(state)
        if terminal is not None:
            return _outcome_value(terminal, state.side_to_move)

        active_color = state.side_to_move
        opponent_color = other_color(active_color)
        raw_score = self._material_score(state, active_color) - self._material_score(
            state, opponent_color
        )

        raw_score += self.mobility_weight * (
            len(legal_moves(state))
            - len(legal_moves(_with_side_to_move(state, opponent_color)))
        )
        if is_in_check(state, opponent_color):
            raw_score += self.check_weight
        if is_in_check(state, active_color):
            raw_score -= self.check_weight
        return math.tanh(raw_score / self.scale)

    def _material_score(self, state: GameState, color: Color) -> float:
        score = 0.0
        for piece in state.active_pieces():
            if piece.color != color:
                continue
            piece_class = state.piece_classes[piece.piece_class_id]
            score += _piece_class_value(piece_class)
        return score


def run_mcts(
    state: GameState,
    *,
    simulations: int = DEFAULT_SIMULATIONS,
    policy_value_model: PolicyValueModel | None = None,
    evaluator: StateEvaluator | None = None,
    c_puct: float = 1.5,
    root_noise: DirichletRootNoise | None = None,
    rng: random.Random | None = None,
    strategy: object | None = None,
    adaptive_search: bool = False,
    adaptive_min_simulations: int | None = None,
    adaptive_max_simulations: int | None = None,
) -> SearchResult:
    if simulations < 1:
        raise ValueError("simulations must be at least 1")
    if adaptive_min_simulations is not None and adaptive_min_simulations < 1:
        raise ValueError("adaptive_min_simulations must be at least 1")
    if adaptive_max_simulations is not None and adaptive_max_simulations < 1:
        raise ValueError("adaptive_max_simulations must be at least 1")

    total_start = time.perf_counter()
    evaluator_impl = HeuristicEvaluator() if evaluator is None else evaluator
    root = SearchNode.from_state(state)
    diagnostics = {
        "expanded_nodes": 0,
        "terminal_expansions": 0,
        "heuristic_evaluations": 0,
        "model_inference_calls": 0,
        "applied_moves": 0,
        "terminal_checks": 0,
        "legal_move_generations": 0,
        "child_selection_calls": 0,
    }
    timings = {
        "search_expand_ms": 0.0,
        "search_terminal_check_ms": 0.0,
        "search_legal_moves_ms": 0.0,
        "search_model_inference_ms": 0.0,
        "search_evaluator_ms": 0.0,
        "search_apply_move_ms": 0.0,
        "search_child_selection_ms": 0.0,
    }
    noise_metadata: dict[str, JsonValue] = {
        "root_noise_applied": False,
        "root_dirichlet_alpha": None,
        "root_exploration_fraction": 0.0,
    }
    adaptive_enabled = adaptive_search and policy_value_model is not None
    effective_simulations = simulations
    root_uncertainty: float | None = None
    remaining_simulations = simulations
    if adaptive_enabled:
        value = _expand_node(
            node=root,
            policy_value_model=policy_value_model,
            evaluator=evaluator_impl,
            diagnostics=diagnostics,
            timings=timings,
            strategy=strategy,
        )
        root.visit_count += 1
        root.value_sum += value
        root_uncertainty = root.model_uncertainty
        effective_simulations = _adaptive_simulation_budget(
            requested_simulations=simulations,
            uncertainty=root_uncertainty,
            minimum_simulations=adaptive_min_simulations,
            maximum_simulations=adaptive_max_simulations,
        )
        remaining_simulations = max(0, effective_simulations - 1)
        if root_noise is not None:
            noise_metadata = _apply_root_dirichlet_noise(
                root=root,
                noise=root_noise,
                rng=rng if rng is not None else random.Random(),
            )
    elif root_noise is not None:
        value = _expand_node(
            node=root,
            policy_value_model=policy_value_model,
            evaluator=evaluator_impl,
            diagnostics=diagnostics,
            timings=timings,
            strategy=strategy,
        )
        root.visit_count += 1
        root.value_sum += value
        remaining_simulations -= 1
        noise_metadata = _apply_root_dirichlet_noise(
            root=root,
            noise=root_noise,
            rng=rng if rng is not None else random.Random(),
        )

    for _ in range(remaining_simulations):
        _simulate(
            node=root,
            policy_value_model=policy_value_model,
            evaluator=evaluator_impl,
            c_puct=c_puct,
            diagnostics=diagnostics,
            timings=timings,
            strategy=strategy,
        )

    visit_counts = {
        move_id: edge.visit_count
        for move_id, edge in sorted(root.children_by_move_id.items())
    }
    total_visits = sum(visit_counts.values())
    if total_visits > 0:
        visit_distribution = {
            move_id: count / total_visits
            for move_id, count in visit_counts.items()
        }
    else:
        visit_distribution = _normalized_priors(root)

    selected_move_id = _select_root_move_id(root)
    selected_move = None
    if selected_move_id is not None:
        selected_move = root.children_by_move_id[selected_move_id].move

    return SearchResult(
        selected_move=selected_move,
        selected_move_id=selected_move_id,
        legal_moves=root.legal_moves,
        visit_distribution=visit_distribution,
        visit_counts=visit_counts,
        root_value=root.mean_value,
        policy_logits=dict(sorted(root.policy_logits.items())),
        metadata={
            "simulations": simulations,
            "simulations_requested": simulations,
            "simulations_used": effective_simulations,
            "root_state_hash": root.state_hash,
            "expanded_children": len(root.children_by_move_id),
            "evaluator": type(evaluator_impl).__name__,
            "used_model": policy_value_model is not None,
            "c_puct": c_puct,
            "simulations_completed": root.visit_count,
            "root_legal_move_count": len(root.legal_moves),
            "root_uncertainty": root_uncertainty,
            "adaptive_search_enabled": adaptive_enabled,
            "adaptive_search_requested": adaptive_search,
            "adaptive_min_simulations": adaptive_min_simulations,
            "adaptive_max_simulations": adaptive_max_simulations,
            "search_total_ms": (time.perf_counter() - total_start) * 1000.0,
            **noise_metadata,
            **diagnostics,
            **timings,
        },
    )


def _simulate(
    *,
    node: SearchNode,
    policy_value_model: PolicyValueModel | None,
    evaluator: StateEvaluator,
    c_puct: float,
    diagnostics: dict[str, int],
    timings: dict[str, float],
    strategy: object | None,
) -> float:
    if node.is_terminal:
        value = 0.0 if node.terminal_value is None else node.terminal_value
        node.visit_count += 1
        node.value_sum += value
        return value

    if not node.is_expanded:
        value = _expand_node(
            node=node,
            policy_value_model=policy_value_model,
            evaluator=evaluator,
            diagnostics=diagnostics,
            timings=timings,
            strategy=strategy,
        )
        node.visit_count += 1
        node.value_sum += value
        return value

    diagnostics["child_selection_calls"] += 1
    selection_start = time.perf_counter()
    edge = _select_child(node=node, c_puct=c_puct)
    timings["search_child_selection_ms"] += (time.perf_counter() - selection_start) * 1000.0
    if edge is None:
        diagnostics["heuristic_evaluations"] += 1
        eval_start = time.perf_counter()
        value = evaluator.evaluate(node.state)
        timings["search_evaluator_ms"] += (time.perf_counter() - eval_start) * 1000.0
        node.visit_count += 1
        node.value_sum += value
        return value

    child = edge.child
    if child is None:
        diagnostics["applied_moves"] += 1
        apply_start = time.perf_counter()
        child_state, delta = apply_move(node.state, edge.move)
        timings["search_apply_move_ms"] += (time.perf_counter() - apply_start) * 1000.0
        child = SearchNode.from_state(child_state)
        edge.child = child
        edge.child_state_hash = child.state_hash
        edge.state_delta = delta

    child_value = _simulate(
        node=child,
        policy_value_model=policy_value_model,
        evaluator=evaluator,
        c_puct=c_puct,
        diagnostics=diagnostics,
        timings=timings,
        strategy=strategy,
    )
    value = -child_value
    edge.visit_count += 1
    edge.value_sum += value
    node.visit_count += 1
    node.value_sum += value
    return value


def _expand_node(
    *,
    node: SearchNode,
    policy_value_model: PolicyValueModel | None,
    evaluator: StateEvaluator,
    diagnostics: dict[str, int],
    timings: dict[str, float],
    strategy: object | None,
) -> float:
    expand_start = time.perf_counter()
    diagnostics["expanded_nodes"] += 1
    diagnostics["terminal_checks"] += 1
    terminal_start = time.perf_counter()
    terminal = result(node.state)
    timings["search_terminal_check_ms"] += (time.perf_counter() - terminal_start) * 1000.0
    if terminal is not None:
        diagnostics["terminal_expansions"] += 1
        node.is_terminal = True
        node.terminal_value = _outcome_value(terminal, node.to_play)
        node.is_expanded = True
        timings["search_expand_ms"] += (time.perf_counter() - expand_start) * 1000.0
        return node.terminal_value

    diagnostics["legal_move_generations"] += 1
    movegen_start = time.perf_counter()
    moves = tuple(legal_moves(node.state))
    timings["search_legal_moves_ms"] += (time.perf_counter() - movegen_start) * 1000.0
    move_ids = tuple(stable_move_id(move) for move in moves)
    node.legal_moves = moves
    node.move_ids = move_ids
    node.is_expanded = True

    if not moves:
        node.is_terminal = True
        node.terminal_value = 0.0
        timings["search_expand_ms"] += (time.perf_counter() - expand_start) * 1000.0
        return 0.0

    policy_logits: dict[str, float] = {}
    if policy_value_model is not None:
        diagnostics["model_inference_calls"] += 1
        model_start = time.perf_counter()
        if strategy is None:
            model_output = policy_value_model.infer(node.state, moves)
        else:
            model_output = policy_value_model.infer(
                node.state,
                moves,
                strategy=strategy,
            )
        timings["search_model_inference_ms"] += (time.perf_counter() - model_start) * 1000.0
        policy_logits = {
            str(move_id): float(logit)
            for move_id, logit in model_output.policy_logits.items()
            if move_id in set(move_ids)
        }
        value = _clamp_value(model_output.value)
        node.model_uncertainty = _clamp_uncertainty(model_output.uncertainty)
    else:
        diagnostics["heuristic_evaluations"] += 1
        eval_start = time.perf_counter()
        value = _clamp_value(evaluator.evaluate(node.state))
        timings["search_evaluator_ms"] += (time.perf_counter() - eval_start) * 1000.0
        node.model_uncertainty = None
    node.policy_logits = dict(sorted(policy_logits.items()))

    priors = _priors_for_moves(move_ids=move_ids, policy_logits=policy_logits)
    node.children_by_move_id = {
        move_id: SearchEdge(move=move, move_id=move_id, prior=priors[move_id])
        for move_id, move in zip(move_ids, moves, strict=True)
    }
    timings["search_expand_ms"] += (time.perf_counter() - expand_start) * 1000.0
    return value


def _adaptive_simulation_budget(
    *,
    requested_simulations: int,
    uncertainty: float | None,
    minimum_simulations: int | None,
    maximum_simulations: int | None,
) -> int:
    max_simulations = (
        requested_simulations if maximum_simulations is None else maximum_simulations
    )
    min_default = max(1, min(requested_simulations, requested_simulations // 2))
    min_simulations = min_default if minimum_simulations is None else minimum_simulations
    if max_simulations < min_simulations:
        raise ValueError(
            "adaptive_max_simulations must be greater than or equal to adaptive_min_simulations"
        )
    if uncertainty is None:
        return max_simulations
    scaled = min_simulations + int(
        round(_clamp_uncertainty(uncertainty) * (max_simulations - min_simulations))
    )
    return max(min_simulations, min(max_simulations, scaled))


def _clamp_uncertainty(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _select_child(*, node: SearchNode, c_puct: float) -> SearchEdge | None:
    best_edge: SearchEdge | None = None
    best_score: float | None = None
    for move_id in sorted(node.children_by_move_id):
        edge = node.children_by_move_id[move_id]
        score = puct_score(
            parent_visits=max(node.visit_count, 1),
            child_visits=edge.visit_count,
            prior=edge.prior,
            value=edge.mean_value,
            c_puct=c_puct,
        )
        if best_score is None or score > best_score or (
            math.isclose(score, best_score) and move_id < best_edge.move_id
        ):
            best_score = score
            best_edge = edge
    return best_edge


def _select_root_move_id(root: SearchNode) -> str | None:
    if not root.children_by_move_id:
        return None

    best_move_id: str | None = None
    best_visits = -1
    best_prior = -1.0
    for move_id in sorted(root.children_by_move_id):
        edge = root.children_by_move_id[move_id]
        if edge.visit_count > best_visits or (
            edge.visit_count == best_visits and edge.prior > best_prior
        ):
            best_move_id = move_id
            best_visits = edge.visit_count
            best_prior = edge.prior
    return best_move_id


def _normalized_priors(root: SearchNode) -> dict[str, float]:
    if not root.children_by_move_id:
        return {}
    priors = {
        move_id: edge.prior
        for move_id, edge in sorted(root.children_by_move_id.items())
    }
    total = sum(priors.values())
    if total <= 0:
        uniform = 1.0 / len(priors)
        return {move_id: uniform for move_id in priors}
    return {move_id: prior / total for move_id, prior in priors.items()}


def _priors_for_moves(
    *, move_ids: tuple[str, ...], policy_logits: dict[str, float]
) -> dict[str, float]:
    if not move_ids:
        return {}
    if not policy_logits:
        uniform = 1.0 / len(move_ids)
        return {move_id: uniform for move_id in move_ids}

    logits = [policy_logits.get(move_id, 0.0) for move_id in move_ids]
    max_logit = max(logits)
    exp_logits = [math.exp(logit - max_logit) for logit in logits]
    total = sum(exp_logits)
    if total <= 0:
        uniform = 1.0 / len(move_ids)
        return {move_id: uniform for move_id in move_ids}
    return {
        move_id: exp_value / total
        for move_id, exp_value in zip(move_ids, exp_logits, strict=True)
    }


def _apply_root_dirichlet_noise(
    *,
    root: SearchNode,
    noise: DirichletRootNoise,
    rng: random.Random,
) -> dict[str, JsonValue]:
    if not root.children_by_move_id or noise.exploration_fraction <= 0.0:
        return {
            "root_noise_applied": False,
            "root_dirichlet_alpha": noise.alpha,
            "root_exploration_fraction": noise.exploration_fraction,
        }

    move_ids = tuple(sorted(root.children_by_move_id))
    samples = _sample_dirichlet(
        count=len(move_ids),
        alpha=noise.alpha,
        rng=rng,
    )
    sample_by_move_id = {
        move_id: sample
        for move_id, sample in zip(move_ids, samples, strict=True)
    }
    mixed_priors: dict[str, float] = {}
    for move_id in move_ids:
        edge = root.children_by_move_id[move_id]
        mixed_prior = (
            (1.0 - noise.exploration_fraction) * edge.prior
            + noise.exploration_fraction * sample_by_move_id[move_id]
        )
        edge.prior = mixed_prior
        mixed_priors[move_id] = mixed_prior

    return {
        "root_noise_applied": True,
        "root_dirichlet_alpha": noise.alpha,
        "root_exploration_fraction": noise.exploration_fraction,
        "root_noise": dict(sorted(sample_by_move_id.items())),
        "root_priors_after_noise": dict(sorted(mixed_priors.items())),
    }


def _sample_dirichlet(
    *,
    count: int,
    alpha: float,
    rng: random.Random,
) -> tuple[float, ...]:
    if count < 1:
        return ()
    samples = [rng.gammavariate(alpha, 1.0) for _ in range(count)]
    total = sum(samples)
    if total <= 0.0:
        uniform = 1.0 / count
        return tuple(uniform for _ in samples)
    return tuple(sample / total for sample in samples)


def _outcome_value(outcome: str, color: Color) -> float:
    if outcome == "draw":
        return 0.0
    if outcome == color.value:
        return 1.0
    return -1.0


def _clamp_value(value: float) -> float:
    return max(-1.0, min(1.0, float(value)))


def _with_side_to_move(state: GameState, side_to_move: Color) -> GameState:
    return GameState(
        piece_classes=state.piece_classes,
        piece_instances=state.piece_instances,
        side_to_move=side_to_move,
        castling_rights=state.castling_rights,
        en_passant_square=state.en_passant_square,
        halfmove_clock=state.halfmove_clock,
        fullmove_number=state.fullmove_number,
        repetition_counts=state.repetition_counts,
        pending_events=state.pending_events,
        metadata=state.metadata,
    )


def _piece_class_value(piece_class: PieceClass) -> float:
    base_value = {
        BasePieceType.PAWN: 1.0,
        BasePieceType.KNIGHT: 3.0,
        BasePieceType.BISHOP: 3.25,
        BasePieceType.ROOK: 5.0,
        BasePieceType.QUEEN: 9.0,
        BasePieceType.KING: 0.0,
    }[piece_class.base_piece_type]

    for modifier in piece_class.movement_modifiers:
        if modifier.op == "extend_range":
            base_value += 0.15 * int(modifier.args["extra_steps"])
        elif modifier.op == "limit_range":
            base_value -= 0.1 * max(0, 7 - int(modifier.args["max_steps"]))
        elif modifier.op == "phase_through_allies":
            base_value += 0.35
        elif modifier.op != "inherit_base":
            base_value += 0.1

    for modifier in piece_class.capture_modifiers:
        if modifier.op == "replace_capture_with_push":
            base_value += 0.2 + 0.1 * int(modifier.args["distance"])
        elif modifier.op != "inherit_base":
            base_value += 0.1

    hook_count = len(piece_class.hooks)
    effect_count = sum(len(hook.effects) for hook in piece_class.hooks)
    condition_count = sum(len(hook.conditions) for hook in piece_class.hooks)
    base_value += 0.12 * hook_count
    base_value += 0.04 * effect_count
    base_value += 0.03 * condition_count
    base_value += 0.03 * len(piece_class.instance_state_schema)
    return max(base_value, 0.0)
