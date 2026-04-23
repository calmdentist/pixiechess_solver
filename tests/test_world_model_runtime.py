from __future__ import annotations

from copy import deepcopy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pixie_solver.core import BasePieceType, Color, PieceClass, PieceInstance, stable_move_id
from pixie_solver.core.move import Move
from pixie_solver.dsl import compile_piece_program, load_piece_program
from pixie_solver.model.policy_value import PolicyValueOutput
from pixie_solver.rules import StaticRepairProvider
from pixie_solver.rules.providers import RepairRequest, RepairResponse
from pixie_solver.rules.mismatch import replace_piece_program
from pixie_solver.simulator.engine import apply_move
from pixie_solver.simulator.movegen import legal_moves
from pixie_solver.core.state import GameState
from pixie_solver.strategy.providers import StrategyRequest, StrategyResponse
from pixie_solver.world_model import OnlineRuntimeConfig, OnlineWorldModelRuntime


class _UncertainPolicyValueModel:
    def __init__(self, *, uncertainty: float) -> None:
        self.uncertainty = uncertainty

    def infer(self, state: GameState, legal_moves: tuple[Move, ...], *, strategy=None) -> PolicyValueOutput:
        del state, strategy
        return PolicyValueOutput(
            policy_logits={stable_move_id(move): 0.0 for move in legal_moves},
            value=0.0,
            uncertainty=self.uncertainty,
        )


class _SequenceStrategyProvider:
    def __init__(self, phases_to_strategy: dict[str, dict[str, object]]) -> None:
        self.phases_to_strategy = phases_to_strategy
        self.phases: list[str] = []

    def propose_strategy(self, request: StrategyRequest) -> StrategyResponse:
        self.phases.append(request.phase)
        strategy = self.phases_to_strategy[request.phase]
        return StrategyResponse(
            strategy=strategy,
            explanation=f"strategy for {request.phase}",
            metadata={"provider": "test"},
        )


class _PerPieceRepairProvider:
    def __init__(self, patched_programs: dict[str, dict[str, object]]) -> None:
        self.patched_programs = patched_programs
        self.requests: list[str] = []

    def repair_piece(self, request: RepairRequest) -> RepairResponse:
        piece_id = str(request.current_program["piece_id"])
        self.requests.append(piece_id)
        return RepairResponse(
            patched_program=self.patched_programs[piece_id],
            explanation=f"patched {piece_id}",
            metadata={"provider": "per_piece_test"},
        )


class _PerPieceCandidateRepairProvider:
    def __init__(self, patched_program_candidates: dict[str, list[dict[str, object]]]) -> None:
        self.patched_program_candidates = patched_program_candidates
        self.requests: list[tuple[str, int]] = []

    def repair_piece(self, request: RepairRequest) -> RepairResponse:
        piece_id = str(request.current_program["piece_id"])
        return RepairResponse(
            patched_program=self.patched_program_candidates[piece_id][0],
            explanation=f"patched {piece_id} candidate 0",
            metadata={"provider": "per_piece_candidate_test"},
        )

    def repair_piece_candidates(
        self,
        request: RepairRequest,
        candidate_count: int,
    ) -> tuple[RepairResponse, ...]:
        piece_id = str(request.current_program["piece_id"])
        self.requests.append((piece_id, candidate_count))
        return tuple(
            RepairResponse(
                patched_program=program,
                explanation=f"patched {piece_id} candidate {index}",
                metadata={"provider": "per_piece_candidate_test", "candidate_index": index},
            )
            for index, program in enumerate(
                self.patched_program_candidates[piece_id][:candidate_count]
            )
        )


class WorldModelRuntimeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.phasing_rook_program = load_piece_program(
            ROOT / "data/pieces/handauthored/phasing_rook.json"
        )
        self.current_program = load_piece_program(
            ROOT / "data/pieces/handauthored/war_automaton.json"
        )
        self.patched_program = _war_automaton_with_forward_offset(2)
        self.phasing_rook = compile_piece_program(self.phasing_rook_program)
        self.current_war_automaton = compile_piece_program(self.current_program)
        self.before_state = GameState(
            piece_classes={
                self.phasing_rook.class_id: self.phasing_rook,
                self.current_war_automaton.class_id: self.current_war_automaton,
            },
            piece_instances={
                "white_rook": PieceInstance(
                    instance_id="white_rook",
                    piece_class_id=self.phasing_rook.class_id,
                    color=Color.WHITE,
                    square="h1",
                ),
                "white_war_auto": PieceInstance(
                    instance_id="white_war_auto",
                    piece_class_id=self.current_war_automaton.class_id,
                    color=Color.WHITE,
                    square="a2",
                ),
                "black_target": PieceInstance(
                    instance_id="black_target",
                    piece_class_id=self.current_war_automaton.class_id,
                    color=Color.BLACK,
                    square="h3",
                ),
            },
            side_to_move=Color.WHITE,
        )
        self.move = next(
            move
            for move in legal_moves(self.before_state)
            if move.piece_id == "white_rook" and move.to_square == "h3"
        )
        teacher_before = replace_piece_program(self.before_state, self.patched_program)
        self.observed_state, _ = apply_move(teacher_before, self.move)

    def test_select_action_refreshes_strategy_on_high_uncertainty(self) -> None:
        search_state = GameState(
            piece_classes={
                self.phasing_rook.class_id: self.phasing_rook,
                "baseline_king": PieceClass(
                    class_id="baseline_king",
                    name="Baseline King",
                    base_piece_type=BasePieceType.KING,
                ),
            },
            piece_instances={
                "white_king": PieceInstance(
                    instance_id="white_king",
                    piece_class_id="baseline_king",
                    color=Color.WHITE,
                    square="a1",
                ),
                "white_rook": PieceInstance(
                    instance_id="white_rook",
                    piece_class_id=self.phasing_rook.class_id,
                    color=Color.WHITE,
                    square="e7",
                ),
                "black_king": PieceInstance(
                    instance_id="black_king",
                    piece_class_id="baseline_king",
                    color=Color.BLACK,
                    square="e8",
                ),
            },
            side_to_move=Color.WHITE,
        )
        provider = _SequenceStrategyProvider(
            {
                "game_start": {
                    "strategy_id": "activate_rook",
                    "summary": "activate the rook quickly",
                    "confidence": 0.8,
                    "scope": "game_start",
                },
                "high_uncertainty": {
                    "strategy_id": "simplify_pressure",
                    "summary": "simplify into direct pressure",
                    "confidence": 0.9,
                    "scope": "refresh",
                },
            }
        )
        runtime = OnlineWorldModelRuntime(
            state=search_state,
            editable_programs={},
            editable_descriptions={},
            policy_value_model=_UncertainPolicyValueModel(uncertainty=1.0),
            strategy_provider=provider,
            config=OnlineRuntimeConfig(
                simulations=4,
                adaptive_search=True,
                adaptive_min_simulations=1,
                adaptive_max_simulations=3,
                strategy_refresh_on_uncertainty=True,
                strategy_refresh_uncertainty_threshold=0.5,
            ),
        )

        decision = runtime.select_action()

        self.assertEqual(["game_start", "high_uncertainty"], provider.phases)
        self.assertTrue(decision.strategy_refreshed)
        self.assertEqual("simplify_pressure", decision.strategy["strategy_id"])

    def test_observe_transition_repairs_editable_program_and_refreshes_strategy(self) -> None:
        provider = _SequenceStrategyProvider(
            {
                "world_repair": {
                    "strategy_id": "post_repair_plan",
                    "summary": "recompute pressure after repair",
                    "confidence": 0.85,
                    "scope": "refresh",
                }
            }
        )
        runtime = OnlineWorldModelRuntime(
            state=self.before_state,
            editable_programs={"war_automaton": self.current_program},
            editable_descriptions={"war_automaton": "surges after a capture"},
            repair_provider=StaticRepairProvider(self.patched_program),
            strategy_provider=provider,
            current_strategy={
                "strategy_id": "initial_plan",
                "summary": "initial pressure",
                "confidence": 0.6,
                "scope": "game_start",
            },
        )

        observation = runtime.observe_transition(
            move=self.move,
            observed_state=self.observed_state,
        )

        self.assertTrue(observation.repaired)
        self.assertTrue(observation.resolved)
        self.assertEqual("war_automaton", observation.repaired_class_id)
        self.assertEqual(("war_automaton",), observation.repaired_class_ids)
        self.assertTrue(observation.repair_result.accepted)
        self.assertEqual("a4", runtime.state.piece_instances["white_war_auto"].square)
        self.assertEqual("post_repair_plan", runtime.current_strategy["strategy_id"])
        self.assertEqual(["world_repair"], provider.phases)

    def test_observe_transition_can_repair_multiple_editable_programs(self) -> None:
        secondary_current_program = _war_automaton_variant(
            piece_id="war_automaton_b",
            name="War Automaton B",
            offset=1,
        )
        secondary_patched_program = _war_automaton_variant(
            piece_id="war_automaton_b",
            name="War Automaton B",
            offset=2,
        )
        secondary_piece_class = compile_piece_program(secondary_current_program)
        before_state = GameState(
            piece_classes={
                self.phasing_rook.class_id: self.phasing_rook,
                self.current_war_automaton.class_id: self.current_war_automaton,
                secondary_piece_class.class_id: secondary_piece_class,
            },
            piece_instances={
                "white_rook": PieceInstance(
                    instance_id="white_rook",
                    piece_class_id=self.phasing_rook.class_id,
                    color=Color.WHITE,
                    square="h1",
                ),
                "white_war_auto": PieceInstance(
                    instance_id="white_war_auto",
                    piece_class_id=self.current_war_automaton.class_id,
                    color=Color.WHITE,
                    square="a2",
                ),
                "white_war_auto_b": PieceInstance(
                    instance_id="white_war_auto_b",
                    piece_class_id=secondary_piece_class.class_id,
                    color=Color.WHITE,
                    square="b2",
                ),
                "black_target": PieceInstance(
                    instance_id="black_target",
                    piece_class_id=secondary_piece_class.class_id,
                    color=Color.BLACK,
                    square="h3",
                ),
            },
            side_to_move=Color.WHITE,
        )
        move = next(
            candidate
            for candidate in legal_moves(before_state)
            if candidate.piece_id == "white_rook" and candidate.to_square == "h3"
        )
        teacher_before = replace_piece_program(before_state, self.patched_program)
        teacher_before = replace_piece_program(teacher_before, secondary_patched_program)
        observed_state, _ = apply_move(teacher_before, move)
        strategy_provider = _SequenceStrategyProvider(
            {
                "world_repair": {
                    "strategy_id": "double_repair_plan",
                    "summary": "recompute after repairing both automata",
                    "confidence": 0.9,
                    "scope": "refresh",
                }
            }
        )
        repair_provider = _PerPieceRepairProvider(
            {
                "war_automaton": self.patched_program,
                "war_automaton_b": secondary_patched_program,
            }
        )
        runtime = OnlineWorldModelRuntime(
            state=before_state,
            editable_programs={
                "war_automaton": self.current_program,
                "war_automaton_b": secondary_current_program,
            },
            editable_descriptions={
                "war_automaton": "surges after a capture",
                "war_automaton_b": "surges after a capture",
            },
            repair_provider=repair_provider,
            strategy_provider=strategy_provider,
            current_strategy={
                "strategy_id": "initial_plan",
                "summary": "pressure the target",
                "confidence": 0.7,
                "scope": "game_start",
            },
            config=OnlineRuntimeConfig(max_repair_passes=1),
        )

        observation = runtime.observe_transition(move=move, observed_state=observed_state)

        self.assertTrue(observation.repaired)
        self.assertTrue(observation.resolved)
        self.assertEqual(
            ("war_automaton", "war_automaton_b"),
            observation.repaired_class_ids,
        )
        self.assertEqual("war_automaton_b", observation.repaired_class_id)
        self.assertEqual(
            ["war_automaton", "war_automaton_b"],
            repair_provider.requests,
        )
        self.assertEqual("a4", runtime.state.piece_instances["white_war_auto"].square)
        self.assertEqual("b4", runtime.state.piece_instances["white_war_auto_b"].square)
        self.assertEqual("double_repair_plan", runtime.current_strategy["strategy_id"])
        self.assertEqual(["world_repair"], strategy_provider.phases)
        self.assertEqual(
            1,
            len(runtime.regression_cases_by_class_id["war_automaton"]),
        )
        self.assertEqual(
            1,
            len(runtime.regression_cases_by_class_id["war_automaton_b"]),
        )

    def test_observe_transition_can_choose_nonfirst_candidate_in_joint_repair(self) -> None:
        secondary_current_program = _war_automaton_variant(
            piece_id="war_automaton_b",
            name="War Automaton B",
            offset=1,
        )
        secondary_wrong_program = _war_automaton_variant(
            piece_id="war_automaton_b",
            name="War Automaton B",
            offset=3,
        )
        secondary_patched_program = _war_automaton_variant(
            piece_id="war_automaton_b",
            name="War Automaton B",
            offset=2,
        )
        secondary_piece_class = compile_piece_program(secondary_current_program)
        before_state = GameState(
            piece_classes={
                self.phasing_rook.class_id: self.phasing_rook,
                self.current_war_automaton.class_id: self.current_war_automaton,
                secondary_piece_class.class_id: secondary_piece_class,
            },
            piece_instances={
                "white_rook": PieceInstance(
                    instance_id="white_rook",
                    piece_class_id=self.phasing_rook.class_id,
                    color=Color.WHITE,
                    square="h1",
                ),
                "white_war_auto": PieceInstance(
                    instance_id="white_war_auto",
                    piece_class_id=self.current_war_automaton.class_id,
                    color=Color.WHITE,
                    square="a2",
                ),
                "white_war_auto_b": PieceInstance(
                    instance_id="white_war_auto_b",
                    piece_class_id=secondary_piece_class.class_id,
                    color=Color.WHITE,
                    square="b2",
                ),
                "black_target": PieceInstance(
                    instance_id="black_target",
                    piece_class_id=secondary_piece_class.class_id,
                    color=Color.BLACK,
                    square="h3",
                ),
            },
            side_to_move=Color.WHITE,
        )
        move = next(
            candidate
            for candidate in legal_moves(before_state)
            if candidate.piece_id == "white_rook" and candidate.to_square == "h3"
        )
        teacher_before = replace_piece_program(before_state, self.patched_program)
        teacher_before = replace_piece_program(teacher_before, secondary_patched_program)
        observed_state, _ = apply_move(teacher_before, move)
        repair_provider = _PerPieceCandidateRepairProvider(
            {
                "war_automaton": [self.patched_program],
                "war_automaton_b": [secondary_wrong_program, secondary_patched_program],
            }
        )
        runtime = OnlineWorldModelRuntime(
            state=before_state,
            editable_programs={
                "war_automaton": self.current_program,
                "war_automaton_b": secondary_current_program,
            },
            editable_descriptions={
                "war_automaton": "surges after a capture",
                "war_automaton_b": "surges after a capture",
            },
            repair_provider=repair_provider,
            config=OnlineRuntimeConfig(
                max_repair_passes=1,
                repair_candidates_per_class=2,
            ),
        )

        observation = runtime.observe_transition(move=move, observed_state=observed_state)

        self.assertTrue(observation.repaired)
        self.assertTrue(observation.resolved)
        self.assertEqual(
            ("war_automaton", "war_automaton_b"),
            observation.repaired_class_ids,
        )
        self.assertEqual("b4", runtime.state.piece_instances["white_war_auto_b"].square)
        self.assertEqual(
            [("war_automaton", 2), ("war_automaton_b", 2)],
            repair_provider.requests,
        )


def _war_automaton_with_forward_offset(offset: int) -> dict[str, object]:
    return _war_automaton_variant(
        piece_id="war_automaton",
        name="War Automaton",
        offset=offset,
    )


def _war_automaton_variant(*, piece_id: str, name: str, offset: int) -> dict[str, object]:
    program = load_piece_program(ROOT / "data/pieces/handauthored/war_automaton.json")
    hook = deepcopy(program["hooks"][0])
    hook["conditions"][1]["args"]["square"]["offset"] = [0, offset]
    hook["effects"][0]["args"]["to"]["offset"] = [0, offset]
    patched = deepcopy(program)
    patched["piece_id"] = piece_id
    patched["name"] = name
    patched["hooks"][0] = hook
    return patched


if __name__ == "__main__":
    unittest.main()
