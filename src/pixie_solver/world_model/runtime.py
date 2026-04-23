from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from pixie_solver.core import GameState, Move
from pixie_solver.model.policy_value import PolicyValueModel
from pixie_solver.rules import (
    ReplayCase,
    RepairResult,
    StateMismatch,
    build_state_mismatch,
    diff_states,
    repair_and_verify_piece_candidates,
)
from pixie_solver.rules.mismatch import behavioral_state_signature, replace_piece_program
from pixie_solver.rules.providers import PieceProgramRepairProvider
from pixie_solver.search import SearchResult, StateEvaluator, run_mcts
from pixie_solver.simulator.engine import apply_move
from pixie_solver.strategy import (
    StrategyProvider,
    StrategyRequest,
    strategy_digest as compute_strategy_digest,
)
from pixie_solver.utils.serialization import JsonValue, to_primitive


@dataclass(frozen=True, slots=True)
class OnlineRuntimeConfig:
    simulations: int = 64
    c_puct: float = 1.5
    adaptive_search: bool = True
    adaptive_min_simulations: int | None = None
    adaptive_max_simulations: int | None = None
    repair_candidates_per_class: int = 3
    max_repair_passes: int = 3
    joint_repair_beam_width: int = 32
    joint_repair_max_nodes: int = 256
    strategy_refresh_on_uncertainty: bool = True
    strategy_refresh_uncertainty_threshold: float = 0.75

    def __post_init__(self) -> None:
        if self.simulations < 1:
            raise ValueError("simulations must be at least 1")
        if self.c_puct <= 0.0:
            raise ValueError("c_puct must be positive")
        if self.adaptive_min_simulations is not None and self.adaptive_min_simulations < 1:
            raise ValueError("adaptive_min_simulations must be at least 1")
        if self.adaptive_max_simulations is not None and self.adaptive_max_simulations < 1:
            raise ValueError("adaptive_max_simulations must be at least 1")
        if self.repair_candidates_per_class < 1:
            raise ValueError("repair_candidates_per_class must be at least 1")
        if self.max_repair_passes < 1:
            raise ValueError("max_repair_passes must be at least 1")
        if self.joint_repair_beam_width < 1:
            raise ValueError("joint_repair_beam_width must be at least 1")
        if self.joint_repair_max_nodes < 1:
            raise ValueError("joint_repair_max_nodes must be at least 1")
        if (
            self.adaptive_min_simulations is not None
            and self.adaptive_max_simulations is not None
            and self.adaptive_max_simulations < self.adaptive_min_simulations
        ):
            raise ValueError(
                "adaptive_max_simulations must be greater than or equal to adaptive_min_simulations"
            )
        if (
            self.strategy_refresh_uncertainty_threshold < 0.0
            or self.strategy_refresh_uncertainty_threshold > 1.0
        ):
            raise ValueError("strategy_refresh_uncertainty_threshold must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class OnlineDecisionResult:
    search_result: SearchResult
    strategy: dict[str, JsonValue] | None = None
    strategy_metadata: dict[str, JsonValue] = field(default_factory=dict)
    strategy_refreshed: bool = False


@dataclass(frozen=True, slots=True)
class OnlineObservationResult:
    move: Move
    predicted_state: GameState
    observed_state: GameState
    mismatch: StateMismatch | None = None
    resolved: bool = True
    repaired: bool = False
    repaired_class_id: str | None = None
    repaired_class_ids: tuple[str, ...] = ()
    repair_result: RepairResult | None = None
    repair_attempts: tuple[RepairResult, ...] = ()
    strategy: dict[str, JsonValue] | None = None
    strategy_metadata: dict[str, JsonValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class _RepairProposal:
    class_id: str
    candidate_index: int
    mismatch: StateMismatch
    repair_result: RepairResult


@dataclass(frozen=True, slots=True)
class _JointRepairPlan:
    proposals: tuple[_RepairProposal, ...]
    before_state: GameState
    predicted_state: GameState


@dataclass(frozen=True, slots=True)
class _JointRepairBeamState:
    proposals: tuple[_RepairProposal, ...]
    before_state: GameState
    predicted_state: GameState
    diff_size: int


@dataclass(slots=True)
class OnlineWorldModelRuntime:
    state: GameState
    editable_programs: dict[str, dict[str, JsonValue]]
    editable_descriptions: dict[str, str]
    repair_provider: PieceProgramRepairProvider | None = None
    policy_value_model: PolicyValueModel | None = None
    evaluator: StateEvaluator | None = None
    strategy_provider: StrategyProvider | None = None
    config: OnlineRuntimeConfig = field(default_factory=OnlineRuntimeConfig)
    current_strategy: dict[str, JsonValue] | None = None
    current_strategy_metadata: dict[str, JsonValue] = field(default_factory=dict)
    regression_cases_by_class_id: dict[str, list[ReplayCase]] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.editable_programs = {
            str(class_id): dict(program)
            for class_id, program in self.editable_programs.items()
        }
        self.editable_descriptions = {
            str(class_id): str(description)
            for class_id, description in self.editable_descriptions.items()
        }
        normalized_cases: dict[str, list[ReplayCase]] = {}
        for class_id, cases in dict(self.regression_cases_by_class_id).items():
            normalized_cases[str(class_id)] = list(cases)
        self.regression_cases_by_class_id = normalized_cases
        if self.current_strategy is not None:
            self.current_strategy = dict(to_primitive(self.current_strategy))
        self.current_strategy_metadata = dict(self.current_strategy_metadata)

    def select_action(self) -> OnlineDecisionResult:
        strategy_refreshed = False
        if self.current_strategy is None and self.strategy_provider is not None:
            self.current_strategy, self.current_strategy_metadata = self._request_strategy(
                phase="game_start",
                prior_strategy=None,
                metadata={"state_hash": self.state.state_hash()},
            )

        search_result = self._search_with_current_strategy()
        root_uncertainty = search_result.metadata.get("root_uncertainty")
        if (
            self.strategy_provider is not None
            and self.config.strategy_refresh_on_uncertainty
            and root_uncertainty is not None
            and float(root_uncertainty)
            >= self.config.strategy_refresh_uncertainty_threshold
        ):
            refreshed_strategy, refreshed_metadata = self._request_strategy(
                phase="high_uncertainty",
                prior_strategy=self.current_strategy,
                metadata={
                    "state_hash": self.state.state_hash(),
                    "root_uncertainty": search_result.metadata.get("root_uncertainty"),
                },
            )
            if _strategy_digest(refreshed_strategy) != _strategy_digest(self.current_strategy):
                self.current_strategy = refreshed_strategy
                self.current_strategy_metadata = refreshed_metadata
                strategy_refreshed = True
                search_result = self._search_with_current_strategy()

        return OnlineDecisionResult(
            search_result=search_result,
            strategy=self.current_strategy,
            strategy_metadata=dict(self.current_strategy_metadata),
            strategy_refreshed=strategy_refreshed,
        )

    def observe_transition(
        self,
        *,
        move: Move,
        observed_state: GameState,
    ) -> OnlineObservationResult:
        before_state = self.state
        predicted_state, _ = apply_move(before_state, move)
        normalized_observed = _replace_piece_classes(observed_state, self.state.piece_classes)
        if behavioral_state_signature(predicted_state) == behavioral_state_signature(
            normalized_observed
        ):
            self.state = normalized_observed
            self._append_regression_cases_for_transition(
                before_state=before_state,
                move=move,
                observed_state=normalized_observed,
            )
            return OnlineObservationResult(
                move=move,
                predicted_state=predicted_state,
                observed_state=normalized_observed,
                strategy=self.current_strategy,
                strategy_metadata=dict(self.current_strategy_metadata),
            )

        current_before_state = before_state
        current_predicted_state = predicted_state
        repair_attempts: list[RepairResult] = []
        selected_mismatch: StateMismatch | None = None
        selected_class_id: str | None = None
        repaired_class_ids: list[str] = []
        last_repair_result: RepairResult | None = None
        repair_passes_used = 0

        while repair_passes_used < self.config.max_repair_passes:
            if behavioral_state_signature(current_predicted_state) == behavioral_state_signature(
                normalized_observed
            ):
                break
            candidate_class_ids = self._repair_candidate_class_ids(
                before_state=current_before_state,
                move=move,
                predicted_state=current_predicted_state,
                observed_state=normalized_observed,
                exclude_class_ids=tuple(repaired_class_ids),
            )
            if not candidate_class_ids:
                break
            if self.repair_provider is None:
                class_id = candidate_class_ids[0]
                current_program = self.editable_programs.get(class_id)
                if current_program is not None:
                    selected_mismatch = build_state_mismatch(
                        before_state=current_before_state,
                        move=move,
                        observed_state=normalized_observed,
                        current_program=current_program,
                        )
                    selected_class_id = class_id
                break

            proposals = self._repair_proposals(
                before_state=current_before_state,
                move=move,
                observed_state=normalized_observed,
                candidate_class_ids=candidate_class_ids,
            )
            if proposals:
                selected_mismatch = proposals[0].mismatch
                selected_class_id = proposals[0].class_id
                repair_attempts.extend(
                    proposal.repair_result for proposal in proposals
                )
                last_repair_result = proposals[-1].repair_result

            joint_repair_plan = _select_joint_repair_plan(
                proposals=proposals,
                before_state=current_before_state,
                move=move,
                observed_state=normalized_observed,
                beam_width=self.config.joint_repair_beam_width,
                max_nodes=self.config.joint_repair_max_nodes,
            )
            if joint_repair_plan is not None:
                for proposal in joint_repair_plan.proposals:
                    assert proposal.repair_result.patched_program is not None
                    self.editable_programs[proposal.class_id] = dict(
                        proposal.repair_result.patched_program
                    )
                    repaired_class_ids.append(proposal.class_id)
                current_before_state = joint_repair_plan.before_state
                current_predicted_state = joint_repair_plan.predicted_state
                last_repair_result = joint_repair_plan.proposals[-1].repair_result
                repair_passes_used += 1
                _clear_model_adapter_cache(self.policy_value_model)
                continue

            repaired_this_pass = False
            for proposal in proposals:
                class_id = proposal.class_id
                repair_result = proposal.repair_result
                next_transition = None
                if repair_result.accepted and repair_result.patched_program is not None:
                    next_before_state = replace_piece_program(
                        current_before_state,
                        repair_result.patched_program,
                    )
                    next_predicted_state, _ = apply_move(next_before_state, move)
                    next_transition = (next_before_state, next_predicted_state)
                else:
                    next_transition = _partial_repair_transition(
                        class_id=class_id,
                        move=move,
                        current_before_state=current_before_state,
                        current_predicted_state=current_predicted_state,
                        observed_state=normalized_observed,
                        repair_result=repair_result,
                    )
                if next_transition is None or repair_result.patched_program is None:
                    continue

                self.editable_programs[class_id] = dict(repair_result.patched_program)
                current_before_state, current_predicted_state = next_transition
                repaired_class_ids.append(class_id)
                last_repair_result = repair_result
                repaired_this_pass = True
                repair_passes_used += 1
                _clear_model_adapter_cache(self.policy_value_model)
                break

            if not repaired_this_pass:
                break

        resolved = behavioral_state_signature(current_predicted_state) == behavioral_state_signature(
            normalized_observed
        )
        final_observed_state = _replace_piece_classes(
            normalized_observed,
            current_before_state.piece_classes,
        )
        self.state = final_observed_state
        if repaired_class_ids:
            if resolved:
                for class_id in repaired_class_ids:
                    self._append_regression_case(
                        class_id=class_id,
                        before_state=current_before_state,
                        move=move,
                        observed_state=final_observed_state,
                    )
            if self.strategy_provider is not None:
                self.current_strategy, self.current_strategy_metadata = self._request_strategy(
                    phase="world_repair",
                    prior_strategy=self.current_strategy,
                    metadata={
                        "state_hash": self.state.state_hash(),
                        "repaired_class_id": repaired_class_ids[-1],
                        "repaired_class_ids": list(repaired_class_ids),
                        "repair_passes": repair_passes_used,
                        "resolved": resolved,
                        "strategy_digest": _strategy_digest(self.current_strategy),
                    },
                )
        elif resolved:
            self._append_regression_cases_for_transition(
                before_state=before_state,
                move=move,
                observed_state=final_observed_state,
            )

        return OnlineObservationResult(
            move=move,
            predicted_state=current_predicted_state,
            observed_state=final_observed_state,
            mismatch=selected_mismatch,
            resolved=resolved,
            repaired=bool(repaired_class_ids),
            repaired_class_id=(
                repaired_class_ids[-1] if repaired_class_ids else selected_class_id
            ),
            repaired_class_ids=tuple(repaired_class_ids),
            repair_result=last_repair_result,
            repair_attempts=tuple(repair_attempts),
            strategy=self.current_strategy,
            strategy_metadata=dict(self.current_strategy_metadata),
        )

    def _search_with_current_strategy(self) -> SearchResult:
        return run_mcts(
            self.state,
            simulations=self.config.simulations,
            policy_value_model=self.policy_value_model,
            evaluator=self.evaluator,
            c_puct=self.config.c_puct,
            strategy=self.current_strategy,
            adaptive_search=self.config.adaptive_search,
            adaptive_min_simulations=self.config.adaptive_min_simulations,
            adaptive_max_simulations=self.config.adaptive_max_simulations,
        )

    def _repair_proposals(
        self,
        *,
        before_state: GameState,
        move: Move,
        observed_state: GameState,
        candidate_class_ids: tuple[str, ...],
    ) -> tuple[_RepairProposal, ...]:
        if self.repair_provider is None:
            return ()
        proposals: list[_RepairProposal] = []
        for class_id in candidate_class_ids:
            current_program = self.editable_programs.get(class_id)
            if current_program is None:
                continue
            mismatch = build_state_mismatch(
                before_state=before_state,
                move=move,
                observed_state=observed_state,
                current_program=current_program,
            )
            repair_results = repair_and_verify_piece_candidates(
                mismatch,
                description=self.editable_descriptions[class_id],
                current_program=current_program,
                provider=self.repair_provider,
                regression_cases=tuple(self.regression_cases_by_class_id.get(class_id, ())),
                candidate_count=self.config.repair_candidates_per_class,
            )
            for candidate_index, repair_result in enumerate(repair_results):
                proposals.append(
                    _RepairProposal(
                        class_id=class_id,
                        candidate_index=candidate_index,
                        mismatch=mismatch,
                        repair_result=repair_result,
                    )
                )
        return tuple(proposals)

    def _request_strategy(
        self,
        *,
        phase: str,
        prior_strategy: dict[str, JsonValue] | None,
        metadata: dict[str, JsonValue],
    ) -> tuple[dict[str, JsonValue], dict[str, JsonValue]]:
        assert self.strategy_provider is not None
        response = self.strategy_provider.propose_strategy(
            StrategyRequest(
                state=self.state.to_dict(),
                world_summary={
                    "piece_classes": {
                        class_id: dict(to_primitive(piece_class))
                        for class_id, piece_class in sorted(self.state.piece_classes.items())
                    },
                    "editable_program_ids": sorted(self.editable_programs),
                    "side_to_move": self.state.side_to_move.value,
                },
                phase=phase,
                prior_strategy=prior_strategy,
                metadata=metadata,
            )
        )
        return dict(response.strategy), {
            "provider": type(self.strategy_provider).__name__,
            "phase": phase,
            "explanation": response.explanation,
            **dict(response.metadata),
        }

    def _repair_candidate_class_ids(
        self,
        *,
        before_state: GameState,
        move: Move,
        predicted_state: GameState,
        observed_state: GameState,
        exclude_class_ids: tuple[str, ...] = (),
    ) -> tuple[str, ...]:
        candidate_ids: list[str] = []
        moving_piece = before_state.piece_instances.get(move.piece_id)
        if moving_piece is not None:
            candidate_ids.append(moving_piece.piece_class_id)
        changed_piece_ids = {
            piece_id
            for piece_id in set(predicted_state.piece_instances) | set(observed_state.piece_instances)
            if predicted_state.piece_instances.get(piece_id) != observed_state.piece_instances.get(piece_id)
        }
        for piece_id in sorted(changed_piece_ids):
            for state in (predicted_state, observed_state):
                piece = state.piece_instances.get(piece_id)
                if piece is not None:
                    candidate_ids.append(piece.piece_class_id)
        deduped: list[str] = []
        for class_id in candidate_ids:
            if (
                class_id in self.editable_programs
                and class_id not in exclude_class_ids
                and class_id not in deduped
            ):
                deduped.append(class_id)
        return tuple(deduped)

    def _append_regression_cases_for_transition(
        self,
        *,
        before_state: GameState,
        move: Move,
        observed_state: GameState,
    ) -> None:
        moving_piece = before_state.piece_instances.get(move.piece_id)
        if moving_piece is None:
            return
        if moving_piece.piece_class_id not in self.editable_programs:
            return
        self._append_regression_case(
            class_id=moving_piece.piece_class_id,
            before_state=before_state,
            move=move,
            observed_state=observed_state,
        )

    def _append_regression_case(
        self,
        *,
        class_id: str,
        before_state: GameState,
        move: Move,
        observed_state: GameState,
    ) -> None:
        cases = self.regression_cases_by_class_id.setdefault(class_id, [])
        cases.append(
            ReplayCase(
                before_state=before_state,
                move=move,
                observed_state=observed_state,
                label=f"{class_id}_observed_transition",
            )
        )


def _replace_piece_classes(
    state: GameState,
    piece_classes: Mapping[str, object],
) -> GameState:
    return GameState(
        piece_classes=dict(piece_classes),
        piece_instances=state.piece_instances,
        side_to_move=state.side_to_move,
        castling_rights=state.castling_rights,
        en_passant_square=state.en_passant_square,
        halfmove_clock=state.halfmove_clock,
        fullmove_number=state.fullmove_number,
        repetition_counts=state.repetition_counts,
        pending_events=state.pending_events,
        metadata=state.metadata,
    )


def _strategy_digest(strategy: Mapping[str, JsonValue] | None) -> str | None:
    if strategy is None:
        return None
    return compute_strategy_digest(strategy)


def _clear_model_adapter_cache(model: PolicyValueModel | None) -> None:
    if model is None:
        return
    clear_cache = getattr(model, "clear_adapter_cache", None)
    if callable(clear_cache):
        clear_cache()


def _partial_repair_transition(
    *,
    class_id: str,
    move: Move,
    current_before_state: GameState,
    current_predicted_state: GameState,
    observed_state: GameState,
    repair_result: RepairResult,
) -> tuple[GameState, GameState] | None:
    if repair_result.patched_program is None:
        return None
    if any(
        error != "primary_mismatch: patched transition still differs from observed state"
        for error in repair_result.verification_errors
    ):
        return None
    patched_before_state = replace_piece_program(
        current_before_state,
        repair_result.patched_program,
    )
    try:
        patched_predicted_state, _ = apply_move(patched_before_state, move)
    except Exception:
        return None

    current_diff = diff_states(current_predicted_state, observed_state)
    patched_diff = diff_states(patched_predicted_state, observed_state)
    candidate_piece_ids = _piece_ids_for_class(
        class_id,
        current_predicted_state,
        observed_state,
    )
    current_candidate_diffs = {
        piece_id
        for piece_id in current_diff.changed_piece_ids
        if piece_id in candidate_piece_ids
    }
    patched_candidate_diffs = {
        piece_id
        for piece_id in patched_diff.changed_piece_ids
        if piece_id in candidate_piece_ids
    }
    if not current_candidate_diffs:
        return None
    if len(patched_candidate_diffs) >= len(current_candidate_diffs):
        return None
    if _diff_size(patched_diff) >= _diff_size(current_diff):
        return None
    return patched_before_state, patched_predicted_state


def _select_joint_repair_plan(
    *,
    proposals: tuple[_RepairProposal, ...],
    before_state: GameState,
    move: Move,
    observed_state: GameState,
    beam_width: int,
    max_nodes: int,
) -> _JointRepairPlan | None:
    eligible_by_class: dict[str, list[_RepairProposal]] = {}
    class_order: list[str] = []
    for proposal in proposals:
        if not _is_joint_repair_candidate(proposal.repair_result):
            continue
        if proposal.class_id not in eligible_by_class:
            eligible_by_class[proposal.class_id] = []
            class_order.append(proposal.class_id)
        eligible_by_class[proposal.class_id].append(proposal)
    if not class_order:
        return None

    initial_predicted_state, _ = apply_move(before_state, move)
    beam = [
        _JointRepairBeamState(
            proposals=(),
            before_state=before_state,
            predicted_state=initial_predicted_state,
            diff_size=_transition_diff_size(initial_predicted_state, observed_state),
        )
    ]
    expanded_nodes = 0
    for class_id in class_order:
        next_beam: list[_JointRepairBeamState] = list(beam)
        for state in beam:
            for proposal in eligible_by_class[class_id]:
                assert proposal.repair_result.patched_program is not None
                patched_before_state = replace_piece_program(
                    state.before_state,
                    proposal.repair_result.patched_program,
                )
                try:
                    patched_predicted_state, _ = apply_move(patched_before_state, move)
                except Exception:
                    continue
                expanded_nodes += 1
                next_state = _JointRepairBeamState(
                    proposals=state.proposals + (proposal,),
                    before_state=patched_before_state,
                    predicted_state=patched_predicted_state,
                    diff_size=_transition_diff_size(patched_predicted_state, observed_state),
                )
                if next_state.diff_size == 0:
                    return _JointRepairPlan(
                        proposals=next_state.proposals,
                        before_state=next_state.before_state,
                        predicted_state=next_state.predicted_state,
                    )
                next_beam.append(next_state)
                if expanded_nodes >= max_nodes:
                    break
            if expanded_nodes >= max_nodes:
                break
        beam = _prune_joint_repair_beam(next_beam, beam_width)
        if expanded_nodes >= max_nodes:
            break
    return None


def _is_joint_repair_candidate(repair_result: RepairResult) -> bool:
    if repair_result.patched_program is None:
        return False
    if repair_result.accepted:
        return True
    return all(
        error == "primary_mismatch: patched transition still differs from observed state"
        for error in repair_result.verification_errors
    )


def _piece_ids_for_class(class_id: str, *states: GameState) -> set[str]:
    piece_ids: set[str] = set()
    for state in states:
        for piece_id, piece in state.piece_instances.items():
            if piece.piece_class_id == class_id:
                piece_ids.add(piece_id)
    return piece_ids


def _diff_size(diff: object) -> int:
    piece_diffs = getattr(diff, "piece_diffs", ())
    global_diffs = getattr(diff, "global_diffs", {})
    return len(piece_diffs) + len(global_diffs)


def _transition_diff_size(predicted_state: GameState, observed_state: GameState) -> int:
    return _diff_size(diff_states(predicted_state, observed_state))


def _prune_joint_repair_beam(
    states: list[_JointRepairBeamState],
    beam_width: int,
) -> list[_JointRepairBeamState]:
    deduped: dict[str, _JointRepairBeamState] = {}
    for state in states:
        key = state.before_state.state_hash()
        incumbent = deduped.get(key)
        if incumbent is None or _joint_repair_state_key(state) < _joint_repair_state_key(incumbent):
            deduped[key] = state
    return sorted(deduped.values(), key=_joint_repair_state_key)[:beam_width]


def _joint_repair_state_key(state: _JointRepairBeamState) -> tuple[int, int, str]:
    proposal_signature = ",".join(
        f"{proposal.class_id}:{proposal.candidate_index}" for proposal in state.proposals
    )
    return (state.diff_size, len(state.proposals), proposal_signature)
