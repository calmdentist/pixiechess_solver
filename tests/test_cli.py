from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class CLITest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
