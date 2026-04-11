from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pixie_solver.core import BasePieceType, Color, Modifier, PieceClass, PieceInstance
from pixie_solver.core.state import GameState
from pixie_solver.training import (
    SelfPlayConfig,
    generate_selfplay_games,
    load_training_checkpoint,
    write_selfplay_examples_jsonl,
)


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

    def test_selfplay_command_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            games_path = Path(temp_dir) / "games.jsonl"
            examples_path = Path(temp_dir) / "examples.jsonl"

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
            self.assertIn("selfplay game 1/1 started", result.stderr)
            self.assertIn("selfplay game 1/1 completed", result.stderr)
            self.assertTrue(games_path.exists())
            self.assertTrue(examples_path.exists())
            games_lines = games_path.read_text(encoding="utf-8").strip().splitlines()
            game_payload = json.loads(games_lines[0])
            self.assertIn("phasing_rook", game_payload["replay_trace"]["initial_state"]["piece_classes"])

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

            checkpoint = load_training_checkpoint(checkpoint_path, device="cpu")
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
            self.assertIn("train started", resumed_run.stderr)
            self.assertTrue(resumed_checkpoint_path.exists())

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
            self.assertEqual(1, len(payload["cycles"]))
            self.assertTrue(Path(payload["latest_checkpoint"]).exists())
            self.assertTrue((output_dir / "summary.json").exists())
            self.assertTrue((output_dir / "metrics/cycle_001.json").exists())
            cycle = payload["cycles"][0]
            self.assertGreater(cycle["train_examples"], 0)
            self.assertGreater(cycle["val_examples"], 0)
            self.assertIn("average_policy_cross_entropy", cycle["train_eval_metrics"])
            self.assertIn("average_policy_cross_entropy", cycle["val_eval_metrics"])


if __name__ == "__main__":
    unittest.main()
