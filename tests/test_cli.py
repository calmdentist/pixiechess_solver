from __future__ import annotations

import contextlib
import importlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pixie_solver.core import BasePieceType, Color, Modifier, PieceClass, PieceInstance
from pixie_solver.core.state import GameState
from pixie_solver.curriculum import generate_teacher_piece
from pixie_solver.dsl import load_piece_program
from pixie_solver.rules import (
    StaticCompileProvider,
    append_verified_piece_version,
    load_verified_piece_records,
)
from pixie_solver.rules.mismatch import replace_piece_program
from pixie_solver.simulator.engine import apply_move
from pixie_solver.simulator.movegen import legal_moves
from pixie_solver.training import (
    SelfPlayConfig,
    generate_selfplay_games,
    load_training_checkpoint,
    save_training_checkpoint,
    write_selfplay_examples_jsonl,
)
from pixie_solver.model import PolicyValueConfig, PolicyValueModel
from pixie_solver.utils import canonical_json

cli_main = importlib.import_module("pixie_solver.cli.main")


class CLITest(unittest.TestCase):
    def setUp(self) -> None:
        king = PieceClass(
            class_id="baseline_king",
            name="Baseline King",
            base_piece_type=BasePieceType.KING,
            movement_modifiers=(Modifier(op="inherit_base"),),
            capture_modifiers=(Modifier(op="inherit_base"),),
        )
        rook = PieceClass(
            class_id="baseline_rook",
            name="Baseline Rook",
            base_piece_type=BasePieceType.ROOK,
            movement_modifiers=(Modifier(op="inherit_base"),),
            capture_modifiers=(Modifier(op="inherit_base"),),
        )
        self.initial_state = GameState(
            piece_classes={
                king.class_id: king,
                rook.class_id: rook,
            },
            piece_instances={
                "white_king": PieceInstance(
                    instance_id="white_king",
                    piece_class_id=king.class_id,
                    color=Color.WHITE,
                    square="a1",
                ),
                "white_rook": PieceInstance(
                    instance_id="white_rook",
                    piece_class_id=rook.class_id,
                    color=Color.WHITE,
                    square="e7",
                ),
                "black_king": PieceInstance(
                    instance_id="black_king",
                    piece_class_id=king.class_id,
                    color=Color.BLACK,
                    square="e8",
                ),
            },
            side_to_move=Color.WHITE,
        )

    def _run(self, *args: str) -> subprocess.CompletedProcess[str]:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(ROOT / "src")
        return subprocess.run(
            [sys.executable, "-m", "pixie_solver", *args],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_compile_piece_command(self) -> None:
        result = self._run(
            "compile-piece",
            "--file",
            "data/pieces/handauthored/phasing_rook.json",
        )
        self.assertEqual(0, result.returncode, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual("phasing_rook", payload["class_id"])

    def test_verify_piece_command(self) -> None:
        result = self._run(
            "verify-piece",
            "--file",
            "data/pieces/handauthored/war_automaton.json",
        )
        self.assertEqual(0, result.returncode, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual("ok", payload["status"])
        self.assertEqual("war_automaton", payload["piece_id"])

    def test_repair_piece_command_writes_registry_record(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            current_program = load_piece_program(
                ROOT / "data/pieces/handauthored/war_automaton.json"
            )
            patched_program = _war_automaton_with_forward_offset(2)
            before_state, move, observed_state = _repair_fixture_states(
                current_program,
                patched_program,
            )
            current_path = temp_path / "current.json"
            patched_response_path = temp_path / "provider_response.json"
            before_path = temp_path / "before.json"
            move_path = temp_path / "move.json"
            observed_path = temp_path / "observed.json"
            registry_path = temp_path / "registry.json"
            repaired_dir = temp_path / "repaired"
            current_path.write_text(canonical_json(current_program, indent=2), encoding="utf-8")
            patched_response_path.write_text(
                canonical_json({"patched_program": patched_program}, indent=2),
                encoding="utf-8",
            )
            before_path.write_text(canonical_json(before_state.to_dict(), indent=2), encoding="utf-8")
            move_path.write_text(canonical_json(move.to_dict(), indent=2), encoding="utf-8")
            observed_path.write_text(canonical_json(observed_state.to_dict(), indent=2), encoding="utf-8")

            result = self._run(
                "repair-piece",
                "--description",
                "A pawn that surges forward after a capture.",
                "--current-program",
                str(current_path),
                "--before-state",
                str(before_path),
                "--move",
                str(move_path),
                "--observed-state",
                str(observed_path),
                "--provider-response",
                str(patched_response_path),
                "--registry",
                str(registry_path),
                "--out-dir",
                str(repaired_dir),
            )

            self.assertEqual(0, result.returncode, result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual("accepted", payload["status"])
            self.assertTrue(payload["accepted"])
            self.assertEqual(1, payload["verified_cases"])
            self.assertTrue(registry_path.exists())
            self.assertTrue(Path(payload["registry_record"]["dsl_path"]).exists())

    def test_selfplay_can_sample_verified_registry_pieces(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            registry_path = temp_path / "registry.json"
            out_dir = temp_path / "repaired"
            games_path = temp_path / "games.jsonl"
            examples_path = temp_path / "examples.jsonl"
            patched_program = _war_automaton_with_forward_offset(2)
            record = append_verified_piece_version(
                registry_path=registry_path,
                out_dir=out_dir,
                program=patched_program,
                description="Verified repaired automaton.",
                source="test",
                verified_cases=1,
            )

            result = self._run(
                "selfplay",
                "--standard-initial-state",
                "--randomize-handauthored-specials",
                "--use-verified-pieces",
                "--piece-registry",
                str(registry_path),
                "--special-piece-inclusion-probability",
                "1.0",
                "--games",
                "1",
                "--games-out",
                str(games_path),
                "--examples-out",
                str(examples_path),
                "--simulations",
                "1",
                "--max-plies",
                "1",
                "--opening-temperature",
                "0",
                "--final-temperature",
                "0",
                "--temperature-drop-after-ply",
                "0",
                "--seed",
                "41",
                "--quiet",
            )

            self.assertEqual(0, result.returncode, result.stderr)
            payload = json.loads(result.stdout)
            self.assertTrue(payload["used_verified_pieces"])
            self.assertEqual(1, payload["verified_piece_count"])
            self.assertEqual(record.dsl_digest, payload["verified_piece_digests"]["war_automaton"]["dsl_digest"])
            game_payload = json.loads(games_path.read_text(encoding="utf-8").splitlines()[0])
            self.assertIn("piece_registry", game_payload["replay_trace"]["metadata"])
            self.assertIn(
                "active_verified_piece_digests",
                game_payload["replay_trace"]["metadata"],
            )
            example_payload = json.loads(
                examples_path.read_text(encoding="utf-8").splitlines()[0]
            )
            self.assertIn("verified_piece_digests", example_payload["metadata"])
            self.assertIn("verified_piece_training_metadata", example_payload["metadata"])
            verified_training_metadata = example_payload["metadata"]["verified_piece_training_metadata"]
            self.assertNotIn("compile_response", verified_training_metadata["war_automaton"])

    def test_selfplay_does_not_mark_foundation_examples_as_verified(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            registry_path = temp_path / "registry.json"
            out_dir = temp_path / "repaired"
            examples_path = temp_path / "examples.jsonl"
            patched_program = _war_automaton_with_forward_offset(2)
            append_verified_piece_version(
                registry_path=registry_path,
                out_dir=out_dir,
                program=patched_program,
                description="Verified repaired automaton.",
                source="test",
                verified_cases=1,
            )

            result = self._run(
                "selfplay",
                "--standard-initial-state",
                "--randomize-handauthored-specials",
                "--use-verified-pieces",
                "--piece-registry",
                str(registry_path),
                "--special-piece-inclusion-probability",
                "0.0",
                "--games",
                "1",
                "--examples-out",
                str(examples_path),
                "--simulations",
                "1",
                "--max-plies",
                "1",
                "--opening-temperature",
                "0",
                "--final-temperature",
                "0",
                "--temperature-drop-after-ply",
                "0",
                "--seed",
                "41",
                "--quiet",
            )

            self.assertEqual(0, result.returncode, result.stderr)
            example_payload = json.loads(
                examples_path.read_text(encoding="utf-8").splitlines()[0]
            )
            self.assertNotIn("verified_piece_digests", example_payload["metadata"])
            self.assertNotIn("verified_piece_training_metadata", example_payload["metadata"])
            self.assertEqual(str(registry_path), example_payload["metadata"]["piece_registry"])

    def test_piece_curriculum_command_repairs_and_writes_registry(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            teacher = generate_teacher_piece(seed=1, recipe="capture_sprint")
            candidate = _program_with_forward_offset(teacher.teacher_program, offset=1)
            compile_response_path = temp_path / "compile_response.json"
            repair_response_path = temp_path / "repair_response.json"
            registry_path = temp_path / "registry.json"
            repaired_dir = temp_path / "repaired"
            artifact_dir = temp_path / "artifacts"
            compile_response_path.write_text(
                canonical_json({"candidate_program": candidate}, indent=2),
                encoding="utf-8",
            )
            repair_response_path.write_text(
                canonical_json({"patched_program": teacher.teacher_program}, indent=2),
                encoding="utf-8",
            )

            result = self._run(
                "piece-curriculum",
                "--seed",
                "1",
                "--recipe",
                "capture_sprint",
                "--compile-response",
                str(compile_response_path),
                "--repair-response",
                str(repair_response_path),
                "--registry",
                str(registry_path),
                "--out-dir",
                str(repaired_dir),
                "--artifact-dir",
                str(artifact_dir),
            )

            self.assertEqual(0, result.returncode, result.stderr)
            payload = json.loads(result.stdout)
            self.assertTrue(payload["accepted"])
            self.assertEqual(1, payload["metadata"]["repairs"])
            self.assertEqual("synthetic_capture_sprint_1", payload["registry_record"]["piece_id"])
            self.assertTrue(registry_path.exists())
            self.assertTrue((artifact_dir / "teacher_program.json").exists())
            self.assertTrue((artifact_dir / "summary.json").exists())

    def test_piece_curriculum_can_use_live_compile_provider(self) -> None:
        teacher = generate_teacher_piece(seed=1, recipe="capture_sprint")
        parser = cli_main.build_parser()
        args = parser.parse_args(
            [
                "piece-curriculum",
                "--seed",
                "1",
                "--recipe",
                "capture_sprint",
                "--no-registry-write",
            ]
        )
        output = io.StringIO()

        with patch.object(
            cli_main,
            "_build_llm_piece_provider",
            return_value=StaticCompileProvider(teacher.teacher_program),
        ):
            with contextlib.redirect_stdout(output):
                returncode = args.handler(args)

        self.assertEqual(0, returncode)
        payload = json.loads(output.getvalue())
        self.assertTrue(payload["accepted"])
        self.assertEqual("synthetic_capture_sprint_1", payload["synthetic_piece"]["piece_id"])

    def test_selfplay_command_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            games_path = Path(temp_dir) / "games.jsonl"
            examples_path = Path(temp_dir) / "examples.jsonl"
            manifest_path = Path(temp_dir) / "manifest.json"

            result = self._run(
                "selfplay",
                "--standard-initial-state",
                "--randomize-handauthored-specials",
                "--special-piece-inclusion-probability",
                "1.0",
                "--games",
                "1",
                "--games-out",
                str(games_path),
                "--examples-out",
                str(examples_path),
                "--manifest-out",
                str(manifest_path),
                "--simulations",
                "8",
                "--max-plies",
                "4",
                "--opening-temperature",
                "0",
                "--final-temperature",
                "0",
                "--temperature-drop-after-ply",
                "0",
                "--seed",
                "3",
            )

            self.assertEqual(0, result.returncode, result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual("ok", payload["status"])
            self.assertEqual(1, payload["games_generated"])
            self.assertEqual(1, payload["seed_state_count"])
            self.assertTrue(payload["randomized_special_pieces"])
            self.assertFalse(payload["used_model"])
            self.assertEqual(payload["examples_generated"], payload["search_metrics"]["plies"])
            self.assertGreaterEqual(payload["search_metrics"]["avg_search_ms"], 0.0)
            self.assertIn("selfplay game 1/1 started", result.stderr)
            self.assertIn("selfplay game 1/1 completed", result.stderr)
            self.assertTrue(games_path.exists())
            self.assertTrue(examples_path.exists())
            self.assertTrue(manifest_path.exists())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual("pixie-lite-v0", manifest["ruleset"])
            self.assertEqual("selfplay", manifest["command"])
            games_lines = games_path.read_text(encoding="utf-8").strip().splitlines()
            game_payload = json.loads(games_lines[0])
            self.assertIn("phasing_rook", game_payload["replay_trace"]["initial_state"]["piece_classes"])

    def test_selfplay_command_supports_parallel_workers(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            games_path = Path(temp_dir) / "games.jsonl"
            examples_path = Path(temp_dir) / "examples.jsonl"

            result = self._run(
                "selfplay",
                "--standard-initial-state",
                "--games",
                "2",
                "--workers",
                "2",
                "--games-out",
                str(games_path),
                "--examples-out",
                str(examples_path),
                "--simulations",
                "2",
                "--max-plies",
                "1",
                "--opening-temperature",
                "0",
                "--final-temperature",
                "0",
                "--temperature-drop-after-ply",
                "0",
                "--root-exploration-fraction",
                "0",
                "--seed",
                "13",
                "--quiet",
            )

            self.assertEqual(0, result.returncode, result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual(2, payload["games_generated"])
            self.assertEqual(2, payload["workers"])
            self.assertEqual(payload["examples_generated"], payload["search_metrics"]["plies"])
            self.assertGreaterEqual(payload["search_metrics"]["plies_per_search_second"], 0.0)
            self.assertTrue(games_path.exists())
            self.assertEqual(2, len(games_path.read_text(encoding="utf-8").splitlines()))

    def test_stress_simulator_command_reports_ok_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "stress.json"
            manifest_path = Path(temp_dir) / "stress_manifest.json"

            result = self._run(
                "stress-simulator",
                "--standard-initial-state",
                "--no-randomize-handauthored-specials",
                "--games",
                "1",
                "--max-plies",
                "1",
                "--seed",
                "17",
                "--output",
                str(output_path),
                "--manifest-out",
                str(manifest_path),
            )

            self.assertEqual(0, result.returncode, result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual("ok", payload["status"])
            self.assertTrue(payload["stress_summary"]["ok"])
            self.assertTrue(output_path.exists())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual("pixie-lite-v0", manifest["ruleset"])

    def test_bench_throughput_command_reports_wall_clock_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "bench.json"
            manifest_path = Path(temp_dir) / "bench_manifest.json"

            result = self._run(
                "bench-throughput",
                "--standard-initial-state",
                "--games",
                "2",
                "--repeats",
                "1",
                "--output",
                str(output_path),
                "--manifest-out",
                str(manifest_path),
                "--simulations",
                "1",
                "--max-plies",
                "1",
                "--seed",
                "23",
                "--quiet",
            )

            self.assertEqual(0, result.returncode, result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual("ok", payload["status"])
            self.assertEqual(1, payload["benchmark_summary"]["repeats"])
            self.assertEqual(2, payload["benchmark_summary"]["games_generated_total"])
            self.assertEqual(2, payload["benchmark_summary"]["examples_generated_total"])
            self.assertGreaterEqual(payload["benchmark_summary"]["wall_ms_average"], 0.0)
            self.assertGreaterEqual(
                payload["benchmark_summary"]["examples_per_second_average"],
                0.0,
            )
            self.assertIsNotNone(payload["average_search_metrics"])
            self.assertIsNone(payload["average_inference_stats"])
            self.assertEqual(1, len(payload["trials"]))
            self.assertEqual(2, payload["trials"][0]["examples_generated"])
            self.assertTrue(output_path.exists())
            self.assertTrue(manifest_path.exists())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual("bench-throughput", manifest["command"])

    def test_bench_throughput_command_supports_batched_inference(self) -> None:
        model = PolicyValueModel(
            PolicyValueConfig(
                d_model=32,
                num_heads=4,
                num_layers=1,
                dropout=0.0,
                feedforward_multiplier=2,
            ),
            device="cpu",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "model.pt"
            save_training_checkpoint(checkpoint_path, model=model)

            result = self._run(
                "bench-throughput",
                "--standard-initial-state",
                "--games",
                "2",
                "--workers",
                "2",
                "--repeats",
                "1",
                "--checkpoint",
                str(checkpoint_path),
                "--batched-inference",
                "--device",
                "cpu",
                "--selfplay-device",
                "cpu",
                "--inference-device",
                "cpu",
                "--simulations",
                "1",
                "--max-plies",
                "1",
                "--seed",
                "29",
                "--quiet",
            )

            self.assertEqual(0, result.returncode, result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual("ok", payload["status"])
            self.assertTrue(payload["batched_inference"])
            self.assertIsNotNone(payload["average_inference_stats"])
            self.assertGreaterEqual(
                payload["average_inference_stats"]["average_batch_size"],
                1.0,
            )
            self.assertIsNotNone(payload["average_search_metrics"])

    def test_train_command_writes_checkpoint_and_supports_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            examples_path = Path(temp_dir) / "examples.jsonl"
            checkpoint_path = Path(temp_dir) / "model.pt"
            resumed_checkpoint_path = Path(temp_dir) / "model_resumed.pt"

            games = generate_selfplay_games(
                [self.initial_state],
                games=1,
                config=SelfPlayConfig(
                    simulations=8,
                    max_plies=4,
                    opening_temperature=0.0,
                    final_temperature=0.0,
                    temperature_drop_after_ply=0,
                    seed=19,
                ),
            )
            write_selfplay_examples_jsonl(
                examples_path,
                games[0].examples,
            )

            first_run = self._run(
                "train",
                "--examples",
                str(examples_path),
                "--checkpoint-out",
                str(checkpoint_path),
                "--epochs",
                "1",
                "--batch-size",
                "1",
                "--device",
                "cpu",
                "--seed",
                "19",
                "--no-shuffle",
                "--model-architecture",
                "baseline_v1",
                "--d-model",
                "32",
                "--num-heads",
                "4",
                "--num-layers",
                "1",
                "--dropout",
                "0",
                "--feedforward-multiplier",
                "2",
            )

            self.assertEqual(0, first_run.returncode, first_run.stderr)
            first_payload = json.loads(first_run.stdout)
            self.assertEqual("ok", first_payload["status"])
            self.assertIn("train started", first_run.stderr)
            self.assertIn("train completed", first_run.stderr)
            self.assertTrue(checkpoint_path.exists())
            self.assertEqual("baseline_v1", first_payload["model_config"]["architecture"])

            checkpoint = load_training_checkpoint(checkpoint_path, device="cpu")
            self.assertEqual("baseline_v1", checkpoint.model_config.architecture)
            self.assertEqual(32, checkpoint.model_config.d_model)
            self.assertIsNotNone(checkpoint.training_metrics)
            self.assertIsNotNone(checkpoint.optimizer_state_dict)

            resumed_run = self._run(
                "train",
                "--examples",
                str(examples_path),
                "--checkpoint-out",
                str(resumed_checkpoint_path),
                "--resume-checkpoint",
                str(checkpoint_path),
                "--epochs",
                "1",
                "--batch-size",
                "1",
                "--device",
                "cpu",
                "--seed",
                "19",
                "--no-shuffle",
            )

            self.assertEqual(0, resumed_run.returncode, resumed_run.stderr)
            resumed_payload = json.loads(resumed_run.stdout)
            self.assertEqual(str(checkpoint_path), resumed_payload["resumed_from"])
            self.assertEqual("baseline_v1", resumed_payload["model_config"]["architecture"])
            self.assertIn("train started", resumed_run.stderr)
            self.assertTrue(resumed_checkpoint_path.exists())

    def test_train_command_supports_world_conditioned_v2(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            examples_path = Path(temp_dir) / "examples.jsonl"
            checkpoint_path = Path(temp_dir) / "world_conditioned.pt"

            games = generate_selfplay_games(
                [self.initial_state],
                games=1,
                config=SelfPlayConfig(
                    simulations=4,
                    max_plies=4,
                    opening_temperature=0.0,
                    final_temperature=0.0,
                    temperature_drop_after_ply=0,
                    seed=41,
                ),
            )
            write_selfplay_examples_jsonl(examples_path, games[0].examples)

            train_run = self._run(
                "train",
                "--examples",
                str(examples_path),
                "--checkpoint-out",
                str(checkpoint_path),
                "--epochs",
                "1",
                "--batch-size",
                "1",
                "--device",
                "cpu",
                "--seed",
                "41",
                "--no-shuffle",
                "--model-architecture",
                "world_conditioned_v2",
                "--d-model",
                "32",
                "--num-heads",
                "4",
                "--num-layers",
                "1",
                "--dropout",
                "0",
                "--feedforward-multiplier",
                "2",
                "--quiet",
            )

            self.assertEqual(0, train_run.returncode, train_run.stderr)
            payload = json.loads(train_run.stdout)
            self.assertEqual("world_conditioned_v2", payload["model_config"]["architecture"])
            checkpoint = load_training_checkpoint(checkpoint_path, device="cpu")
            self.assertEqual("world_conditioned_v2", checkpoint.model_config.architecture)
            self.assertEqual("world_conditioned_v2", checkpoint.model.config.architecture)

    def test_eval_model_command_reports_learning_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            examples_path = Path(temp_dir) / "examples.jsonl"
            checkpoint_path = Path(temp_dir) / "model.pt"

            games = generate_selfplay_games(
                [self.initial_state],
                games=1,
                config=SelfPlayConfig(
                    simulations=8,
                    max_plies=4,
                    opening_temperature=0.0,
                    final_temperature=0.0,
                    temperature_drop_after_ply=0,
                    seed=23,
                ),
            )
            write_selfplay_examples_jsonl(examples_path, games[0].examples)

            train_run = self._run(
                "train",
                "--examples",
                str(examples_path),
                "--checkpoint-out",
                str(checkpoint_path),
                "--epochs",
                "1",
                "--batch-size",
                "1",
                "--device",
                "cpu",
                "--seed",
                "23",
                "--no-shuffle",
                "--quiet",
                "--d-model",
                "32",
                "--num-heads",
                "4",
                "--num-layers",
                "1",
                "--dropout",
                "0",
                "--feedforward-multiplier",
                "2",
            )
            self.assertEqual(0, train_run.returncode, train_run.stderr)

            eval_run = self._run(
                "eval-model",
                "--checkpoint",
                str(checkpoint_path),
                "--examples",
                str(examples_path),
                "--device",
                "cpu",
                "--log-every-examples",
                "1",
            )

            self.assertEqual(0, eval_run.returncode, eval_run.stderr)
            self.assertIn("eval-model started", eval_run.stderr)
            self.assertIn("eval-model completed", eval_run.stderr)
            payload = json.loads(eval_run.stdout)
            self.assertEqual("ok", payload["status"])
            self.assertEqual(len(games[0].examples), payload["metrics"]["examples"])
            self.assertGreater(payload["metrics"]["average_policy_cross_entropy"], 0.0)
            self.assertGreaterEqual(payload["metrics"]["top1_agreement"], 0.0)
            self.assertLessEqual(payload["metrics"]["top1_agreement"], 1.0)
            self.assertGreaterEqual(payload["metrics"]["value_mse"], 0.0)

    def test_arena_command_compares_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            candidate_path = temp_path / "candidate.pt"
            baseline_path = temp_path / "baseline.pt"
            output_path = temp_path / "arena.json"
            games_path = temp_path / "arena_games.jsonl"
            model_config = PolicyValueConfig(
                d_model=32,
                num_heads=4,
                num_layers=1,
                dropout=0.0,
                feedforward_multiplier=2,
            )
            save_training_checkpoint(
                candidate_path,
                model=PolicyValueModel(model_config, device="cpu"),
            )
            save_training_checkpoint(
                baseline_path,
                model=PolicyValueModel(model_config, device="cpu"),
            )

            result = self._run(
                "arena",
                "--candidate",
                str(candidate_path),
                "--baseline",
                str(baseline_path),
                "--output",
                str(output_path),
                "--games-out",
                str(games_path),
                "--games",
                "2",
                "--simulations",
                "1",
                "--max-plies",
                "1",
                "--device",
                "cpu",
                "--no-randomize-handauthored-specials",
                "--quiet",
            )

            self.assertEqual(0, result.returncode, result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual("ok", payload["status"])
            self.assertEqual(2, payload["arena_summary"]["games_played"])
            self.assertIn("candidate_score_rate", payload["arena_summary"])
            self.assertIn("promotion_decision", payload)
            self.assertTrue(output_path.exists())
            self.assertTrue(games_path.exists())
            self.assertEqual(2, len(games_path.read_text(encoding="utf-8").splitlines()))

    def test_train_loop_command_runs_one_cycle(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "run"
            result = self._run(
                "train-loop",
                "--output-dir",
                str(output_dir),
                "--cycles",
                "1",
                "--train-games",
                "1",
                "--val-games",
                "1",
                "--simulations",
                "2",
                "--max-plies",
                "2",
                "--epochs-per-cycle",
                "1",
                "--batch-size",
                "1",
                "--device",
                "cpu",
                "--seed",
                "31",
                "--model-architecture",
                "baseline_v1",
                "--d-model",
                "32",
                "--num-heads",
                "4",
                "--num-layers",
                "1",
                "--dropout",
                "0",
                "--feedforward-multiplier",
                "2",
                "--quiet",
            )

            self.assertEqual(0, result.returncode, result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual("ok", payload["status"])
            self.assertEqual("baseline_v1", payload["model_config"]["architecture"])
            self.assertEqual(1, len(payload["cycles"]))
            self.assertTrue(Path(payload["latest_checkpoint"]).exists())
            self.assertTrue((output_dir / "summary.json").exists())
            self.assertTrue((output_dir / "manifest.json").exists())
            self.assertTrue((output_dir / "metrics/cycle_001.json").exists())
            cycle = payload["cycles"][0]
            self.assertEqual("baseline_v1", cycle["model_config"]["architecture"])
            self.assertGreater(cycle["train_examples"], 0)
            self.assertEqual(cycle["train_examples"], cycle["replay_examples"])
            self.assertEqual("bucket_balanced", cycle["replay_sampling_strategy"])
            self.assertEqual(cycle["replay_examples"], sum(cycle["replay_bucket_counts"].values()))
            self.assertGreater(cycle["val_examples"], 0)
            self.assertEqual(cycle["train_games"], cycle["train_termination_summary"]["games"])
            self.assertEqual(cycle["val_games"], cycle["val_termination_summary"]["games"])
            self.assertEqual(
                cycle["train_games"],
                sum(cycle["train_termination_summary"]["termination_reasons"].values()),
            )
            self.assertEqual(
                cycle["val_games"],
                sum(cycle["val_termination_summary"]["termination_reasons"].values()),
            )
            self.assertGreaterEqual(cycle["train_termination_summary"]["max_plies_rate"], 0.0)
            self.assertLessEqual(cycle["train_termination_summary"]["max_plies_rate"], 1.0)
            self.assertGreaterEqual(cycle["val_termination_summary"]["max_plies_rate"], 0.0)
            self.assertLessEqual(cycle["val_termination_summary"]["max_plies_rate"], 1.0)
            self.assertIn("average_policy_cross_entropy", cycle["train_eval_metrics"])
            self.assertIn("average_policy_cross_entropy", cycle["val_eval_metrics"])

    def test_train_loop_command_supports_world_conditioned_v2(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "run"
            result = self._run(
                "train-loop",
                "--output-dir",
                str(output_dir),
                "--cycles",
                "1",
                "--train-games",
                "1",
                "--val-games",
                "1",
                "--simulations",
                "1",
                "--max-plies",
                "2",
                "--epochs-per-cycle",
                "1",
                "--batch-size",
                "1",
                "--device",
                "cpu",
                "--seed",
                "43",
                "--model-architecture",
                "world_conditioned_v2",
                "--d-model",
                "32",
                "--num-heads",
                "4",
                "--num-layers",
                "1",
                "--dropout",
                "0",
                "--feedforward-multiplier",
                "2",
                "--quiet",
            )

            self.assertEqual(0, result.returncode, result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual("ok", payload["status"])
            self.assertEqual("world_conditioned_v2", payload["model_config"]["architecture"])
            self.assertEqual("world_conditioned_v2", payload["cycles"][0]["model_config"]["architecture"])
            checkpoint = load_training_checkpoint(payload["latest_checkpoint"], device="cpu")
            self.assertEqual("world_conditioned_v2", checkpoint.model_config.architecture)

    def test_train_loop_runs_scheduled_curriculum_and_reloads_training_pool(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output_dir = temp_path / "run"
            registry_path = temp_path / "registry.json"
            special_piece_dir = temp_path / "specials"
            special_piece_dir.mkdir(parents=True, exist_ok=True)
            special_piece_path = ROOT / "data/pieces/handauthored/phasing_rook.json"
            special_piece_copy = special_piece_dir / special_piece_path.name
            special_piece_copy.write_text(special_piece_path.read_text(encoding="utf-8"), encoding="utf-8")
            first_teacher = generate_teacher_piece(seed=101, recipe="capture_sprint")
            second_teacher = generate_teacher_piece(seed=202, recipe="turn_charge")

            parser = cli_main.build_parser()
            args = parser.parse_args(
                [
                    "train-loop",
                    "--output-dir",
                    str(output_dir),
                    "--cycles",
                    "2",
                    "--train-games",
                    "1",
                    "--val-games",
                    "1",
                    "--simulations",
                    "1",
                    "--max-plies",
                    "1",
                    "--epochs-per-cycle",
                    "1",
                    "--batch-size",
                    "1",
                    "--device",
                    "cpu",
                    "--seed",
                    "59",
                    "--model-architecture",
                    "baseline_v1",
                    "--d-model",
                    "32",
                    "--num-heads",
                    "4",
                    "--num-layers",
                    "1",
                    "--dropout",
                    "0",
                    "--feedforward-multiplier",
                    "2",
                    "--special-piece-dir",
                    str(special_piece_dir),
                    "--special-piece-inclusion-probability",
                    "1.0",
                    "--piece-registry",
                    str(registry_path),
                    "--use-verified-pieces",
                    "--curriculum-task",
                    "1:101:capture_sprint:train:introduced",
                    "--curriculum-task",
                    "2:202:turn_charge:train:introduced",
                    "--curriculum-oracle",
                    "--quiet",
                ]
            )
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                returncode = args.handler(args)

            self.assertEqual(0, returncode)
            payload = json.loads(output.getvalue())
            self.assertEqual("ok", payload["status"])
            self.assertEqual(2, len(payload["cycles"]))
            first_cycle = payload["cycles"][0]
            second_cycle = payload["cycles"][1]
            self.assertEqual(1, first_cycle["curriculum_tasks_run"])
            self.assertEqual(1, first_cycle["curriculum_tasks_accepted"])
            self.assertEqual(1, second_cycle["curriculum_tasks_run"])
            self.assertEqual(1, second_cycle["curriculum_tasks_accepted"])
            self.assertEqual(1, first_cycle["verified_piece_count"])
            self.assertEqual(2, second_cycle["verified_piece_count"])
            self.assertEqual(1.0, first_cycle["train_termination_summary"]["max_plies_rate"])
            self.assertEqual(1.0, first_cycle["val_termination_summary"]["max_plies_rate"])
            self.assertEqual(1.0, second_cycle["train_termination_summary"]["max_plies_rate"])
            self.assertEqual(1.0, second_cycle["val_termination_summary"]["max_plies_rate"])
            self.assertEqual(first_cycle["train_examples"], first_cycle["replay_examples"])
            self.assertEqual(
                first_cycle["train_examples"] + second_cycle["train_examples"],
                second_cycle["replay_examples"],
            )
            self.assertGreater(second_cycle["replay_bucket_counts"]["recent"], 0)
            self.assertGreater(second_cycle["replay_bucket_counts"]["verified"], 0)
            self.assertEqual(
                first_teacher.teacher_program["piece_id"],
                first_cycle["curriculum_results"][0]["synthetic_piece"]["piece_id"],
            )
            self.assertEqual(
                second_teacher.teacher_program["piece_id"],
                second_cycle["curriculum_results"][0]["synthetic_piece"]["piece_id"],
            )
            self.assertEqual(
                "capture_sprint",
                first_cycle["curriculum_results"][0]["synthetic_piece"]["recipe"],
            )
            self.assertEqual(
                "turn_charge",
                second_cycle["curriculum_results"][0]["synthetic_piece"]["recipe"],
            )
            records = load_verified_piece_records(registry_path)
            self.assertEqual(2, len(records))
            digests_by_piece = {
                record.piece_id: record.dsl_digest
                for record in records
            }
            self.assertEqual(
                digests_by_piece[first_teacher.piece_id],
                first_cycle["verified_piece_digests"][first_teacher.piece_id]["dsl_digest"],
            )
            self.assertIn(second_teacher.piece_id, second_cycle["verified_piece_digests"])
            record_metadata = {
                record.piece_id: dict(record.metadata)
                for record in records
            }
            self.assertEqual("capture_sprint", record_metadata[first_teacher.piece_id]["family_id"])
            self.assertEqual("train", record_metadata[first_teacher.piece_id]["split"])
            self.assertEqual("introduced", record_metadata[first_teacher.piece_id]["novelty_tier"])
            self.assertEqual(1, record_metadata[first_teacher.piece_id]["admission_cycle"])
            self.assertEqual("turn_charge", record_metadata[second_teacher.piece_id]["family_id"])
            self.assertEqual(2, record_metadata[second_teacher.piece_id]["admission_cycle"])

            cycle_two_examples_path = Path(second_cycle["train_examples_path"])
            example_payload = json.loads(cycle_two_examples_path.read_text(encoding="utf-8").splitlines()[0])
            self.assertIn("verified_piece_digests", example_payload["metadata"])
            self.assertEqual(
                str(registry_path),
                example_payload["metadata"]["piece_registry"],
            )
            example_registry_metadata = example_payload["metadata"]["verified_piece_training_metadata"]
            self.assertLessEqual(len(example_registry_metadata), 2)
            self.assertTrue(
                set(example_registry_metadata).issubset(
                    {first_teacher.piece_id, second_teacher.piece_id}
                )
            )
            for training_metadata in example_registry_metadata.values():
                self.assertNotIn("compile_response", training_metadata)
                self.assertIn("version", training_metadata)
                self.assertIn("source", training_metadata)

    def test_train_loop_promotion_gate_records_best_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "run"
            result = self._run(
                "train-loop",
                "--output-dir",
                str(output_dir),
                "--cycles",
                "2",
                "--train-games",
                "1",
                "--val-games",
                "1",
                "--simulations",
                "1",
                "--max-plies",
                "1",
                "--epochs-per-cycle",
                "1",
                "--batch-size",
                "1",
                "--device",
                "cpu",
                "--seed",
                "37",
                "--d-model",
                "32",
                "--num-heads",
                "4",
                "--num-layers",
                "1",
                "--dropout",
                "0",
                "--feedforward-multiplier",
                "2",
                "--promotion-gate",
                "--arena-games",
                "1",
                "--arena-simulations",
                "1",
                "--arena-max-plies",
                "1",
                "--promotion-score-threshold",
                "1.0",
                "--no-randomize-handauthored-specials",
                "--quiet",
            )

            self.assertEqual(0, result.returncode, result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual("ok", payload["status"])
            self.assertEqual(2, len(payload["cycles"]))
            self.assertTrue(Path(payload["best_checkpoint"]).exists())
            self.assertTrue(Path(payload["latest_candidate_checkpoint"]).exists())
            first_cycle = payload["cycles"][0]
            second_cycle = payload["cycles"][1]
            self.assertTrue(first_cycle["promotion_decision"]["promoted"])
            self.assertEqual(
                "initial_candidate",
                first_cycle["promotion_decision"]["reason"],
            )
            self.assertIsNotNone(second_cycle["promotion_decision"])
            self.assertIsNotNone(second_cycle["arena_metrics"])
            self.assertTrue(Path(second_cycle["arena_summary_path"]).exists())
            self.assertTrue(Path(second_cycle["arena_games_path"]).exists())
            metrics_payload = json.loads(
                (output_dir / "metrics/cycle_002.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                second_cycle["promotion_decision"],
                metrics_payload["promotion_decision"],
            )
            analyzer = subprocess.run(
                [
                    sys.executable,
                    "scripts/analyze_training_run.py",
                    str(output_dir),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(0, analyzer.returncode, analyzer.stderr)
            self.assertIn("Promotion Gate", analyzer.stdout)
            self.assertIn("latest_arena", analyzer.stdout)

def _war_automaton_with_forward_offset(offset: int) -> dict[str, object]:
    program = load_piece_program(ROOT / "data/pieces/handauthored/war_automaton.json")
    hook = program["hooks"][0]
    hook["conditions"][1]["args"]["square"]["offset"] = [0, offset]
    hook["effects"][0]["args"]["to"]["offset"] = [0, offset]
    return program


def _program_with_forward_offset(program: dict[str, object], *, offset: int) -> dict[str, object]:
    from copy import deepcopy

    candidate = deepcopy(program)
    hook = candidate["hooks"][0]
    hook["conditions"][1]["args"]["square"]["offset"] = [0, offset]
    hook["effects"][0]["args"]["to"]["offset"] = [0, offset]
    return candidate


def _repair_fixture_states(
    current_program: dict[str, object],
    patched_program: dict[str, object],
):
    from pixie_solver.dsl import compile_piece_program

    phasing_rook = compile_piece_program(
        load_piece_program(ROOT / "data/pieces/handauthored/phasing_rook.json")
    )
    current_war_automaton = compile_piece_program(current_program)
    before_state = GameState(
        piece_classes={
            phasing_rook.class_id: phasing_rook,
            current_war_automaton.class_id: current_war_automaton,
        },
        piece_instances={
            "white_rook": PieceInstance(
                instance_id="white_rook",
                piece_class_id=phasing_rook.class_id,
                color=Color.WHITE,
                square="h1",
            ),
            "white_war_auto": PieceInstance(
                instance_id="white_war_auto",
                piece_class_id=current_war_automaton.class_id,
                color=Color.WHITE,
                square="a2",
            ),
            "black_target": PieceInstance(
                instance_id="black_target",
                piece_class_id=current_war_automaton.class_id,
                color=Color.BLACK,
                square="h3",
            ),
        },
        side_to_move=Color.WHITE,
    )
    move = next(
        move
        for move in legal_moves(before_state)
        if move.piece_id == "white_rook" and move.to_square == "h3"
    )
    teacher_before = replace_piece_program(before_state, patched_program)
    observed_state, _ = apply_move(teacher_before, move)
    return before_state, move, observed_state


if __name__ == "__main__":
    unittest.main()
