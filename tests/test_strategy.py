from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pixie_solver.strategy import (
    StaticStrategyProvider,
    StrategyHypothesis,
    StrategyRequest,
    StrategyResponse,
    StrategyValidationError,
    canonicalize_strategy_hypothesis,
    strategy_digest,
    validate_strategy_hypothesis,
)


class StrategyTest(unittest.TestCase):
    def test_strategy_canonicalization_is_deterministic(self) -> None:
        strategy = StrategyHypothesis(
            strategy_id="activate_rook",
            summary="  Activate the rook early. ",
            confidence=0.73456789,
            scope="opening",
            subgoals=("open the file", "lift the rook"),
            action_biases=("clear file",),
            avoid_biases=("trade rook",),
            success_predicates=("rook active",),
            failure_triggers=("rook trapped",),
            metadata={"tier": 2},
        )

        first = canonicalize_strategy_hypothesis(strategy)
        second = canonicalize_strategy_hypothesis(strategy)

        self.assertEqual(first, second)
        self.assertEqual("Activate the rook early.", first["summary"])
        self.assertEqual(0.734568, first["confidence"])

    def test_strategy_digest_is_stable(self) -> None:
        strategy = StrategyHypothesis(
            strategy_id="avoid_reflection",
            summary="avoid opening reflected lines",
            confidence=0.6,
            avoid_biases=("open reflected diagonal",),
        )

        self.assertEqual(strategy_digest(strategy), strategy_digest(strategy))

    def test_validate_strategy_rejects_duplicate_biases(self) -> None:
        strategy = StrategyHypothesis(
            strategy_id="bad_strategy",
            summary="duplicate biases",
            confidence=0.5,
            action_biases=("same move", "same move"),
        )

        with self.assertRaisesRegex(
            StrategyValidationError,
            "action_biases values must be unique",
        ):
            validate_strategy_hypothesis(strategy)

    def test_static_strategy_provider_returns_canonical_strategy(self) -> None:
        provider = StaticStrategyProvider(
            StrategyHypothesis(
                strategy_id="activate_rook",
                summary="activate the rook early",
                confidence=0.8,
                scope="game_start",
            )
        )

        response = provider.propose_strategy(
            StrategyRequest(
                state={"side_to_move": "white"},
                world_summary={"active_piece_count": 3},
            )
        )

        self.assertEqual("activate_rook", response.strategy["strategy_id"])
        self.assertEqual("static", response.metadata["provider"])
        self.assertEqual("game_start", response.metadata["phase"])

    def test_strategy_response_accepts_raw_strategy_shape(self) -> None:
        response = StrategyResponse.from_dict(
            {
                "strategy_id": "pressure_king",
                "summary": "pressure the king immediately",
                "confidence": 0.7,
                "scope": "refresh",
            }
        )

        self.assertEqual("pressure_king", response.strategy["strategy_id"])
        self.assertEqual("raw_strategy", response.metadata["response_shape"])


if __name__ == "__main__":
    unittest.main()
