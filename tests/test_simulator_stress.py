from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pixie_solver.core import standard_initial_state
from pixie_solver.eval import SimulatorStressConfig, run_simulator_stress


class SimulatorStressTest(unittest.TestCase):
    def test_stress_run_checks_replay_and_legal_move_application(self) -> None:
        summary = run_simulator_stress(
            [standard_initial_state()],
            config=SimulatorStressConfig(
                games=2,
                max_plies=2,
                seed=5,
                verify_all_legal_moves=True,
            ),
        )

        self.assertTrue(summary.ok, summary.to_dict())
        self.assertEqual(2, summary.games_started)
        self.assertEqual(2, summary.games_completed)
        self.assertGreater(summary.legal_moves_checked, 0)
        self.assertEqual(0, summary.replay_hash_failures)


if __name__ == "__main__":
    unittest.main()
