from __future__ import annotations

import sys
import unittest
from urllib.request import urlopen
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pixie_solver.core import Color, PieceInstance, standard_initial_state, standard_piece_classes
from pixie_solver.core.state import GameState
from pixie_solver.dsl import compile_piece_file
from pixie_solver.gui.view_model import (
    board_snapshot_from_state,
    replay_frames_from_game,
    selfplay_trace_event_to_frame,
)
from pixie_solver.gui.server import ViewerServer
from pixie_solver.training import SelfPlayConfig, generate_selfplay_games


class GuiViewModelTest(unittest.TestCase):
    def test_board_snapshot_marks_magical_piece_with_base_symbol_and_letter(self) -> None:
        piece_classes = standard_piece_classes()
        phasing_rook = compile_piece_file(
            ROOT / "data/pieces/handauthored/phasing_rook.json"
        )
        piece_classes[phasing_rook.class_id] = phasing_rook
        state = GameState(
            piece_classes=piece_classes,
            piece_instances={
                "white_phasing_rook_a": PieceInstance(
                    instance_id="white_phasing_rook_a",
                    piece_class_id=phasing_rook.class_id,
                    color=Color.WHITE,
                    square="a1",
                ),
                "black_rook_h": PieceInstance(
                    instance_id="black_rook_h",
                    piece_class_id="baseline_rook",
                    color=Color.BLACK,
                    square="h8",
                ),
            },
        )

        snapshot = board_snapshot_from_state(state)
        pieces = {piece["id"]: piece for piece in snapshot["pieces"]}

        magical_piece = pieces["white_phasing_rook_a"]
        self.assertEqual("rook", magical_piece["base_piece_type"])
        self.assertEqual("P", magical_piece["display_letter"])
        self.assertTrue(magical_piece["magical"])

        baseline_piece = pieces["black_rook_h"]
        self.assertEqual("rook", baseline_piece["base_piece_type"])
        self.assertIsNone(baseline_piece["display_letter"])
        self.assertFalse(baseline_piece["magical"])

    def test_replay_frames_reconstruct_game_trace(self) -> None:
        config = SelfPlayConfig(
            simulations=1,
            max_plies=1,
            opening_temperature=0.0,
            final_temperature=0.0,
            temperature_drop_after_ply=0,
            root_exploration_fraction=0.0,
            seed=3,
        )
        game = generate_selfplay_games([standard_initial_state()], games=1, config=config)[0]

        frames = replay_frames_from_game(game)

        self.assertEqual("game_started", frames[0]["event"])
        self.assertEqual("game_completed", frames[-1]["event"])
        self.assertEqual(game.final_state_hash, frames[-1]["after"]["state_hash"])
        self.assertEqual(1, len([frame for frame in frames if frame["event"] == "ply_completed"]))

    def test_selfplay_trace_event_converts_live_search_payload(self) -> None:
        trace_events = []
        config = SelfPlayConfig(
            simulations=1,
            max_plies=1,
            opening_temperature=0.0,
            final_temperature=0.0,
            temperature_drop_after_ply=0,
            root_exploration_fraction=0.0,
            seed=5,
        )

        generate_selfplay_games(
            [standard_initial_state()],
            games=1,
            config=config,
            trace_callback=trace_events.append,
        )

        ply_event = next(event for event in trace_events if event.event == "ply_completed")
        frame = selfplay_trace_event_to_frame(ply_event, cycle=2, phase="train_selfplay")

        self.assertEqual("ply_completed", frame["event"])
        self.assertEqual(2, frame["cycle"])
        self.assertEqual("train_selfplay", frame["phase"])
        self.assertIsNotNone(frame["before"])
        self.assertIsNotNone(frame["after"])
        self.assertIsNotNone(frame["move"])
        self.assertIsNotNone(frame["delta"])
        self.assertGreater(frame["search"]["legal_move_count"], 0)

    def test_viewer_server_serves_static_shell_and_replay_payload(self) -> None:
        server = ViewerServer(
            replay_payload={
                "mode": "replay",
                "source": "test",
                "frames": [],
                "game_count": 0,
            }
        )
        url = server.start()
        try:
            html = urlopen(url, timeout=2).read().decode("utf-8")
            payload = urlopen(f"{url}api/replay", timeout=2).read().decode("utf-8")
        finally:
            server.stop()

        self.assertIn("PixieChess Viewer", html)
        self.assertIn('"source":"test"', payload)


if __name__ == "__main__":
    unittest.main()
