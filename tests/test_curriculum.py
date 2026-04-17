from __future__ import annotations

from copy import deepcopy
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pixie_solver.curriculum import (
    generate_diagnostic_probes,
    generate_teacher_piece,
    run_synthetic_piece_curriculum,
)
from pixie_solver.rules import (
    RepairRequest,
    RepairResponse,
    StaticCompileProvider,
    StaticRepairProvider,
    load_verified_piece_records,
)


class CurriculumTest(unittest.TestCase):
    def test_teacher_generation_is_deterministic(self) -> None:
        first = generate_teacher_piece(seed=1, recipe="capture_sprint")
        second = generate_teacher_piece(seed=1, recipe="capture_sprint")

        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual("synthetic_capture_sprint_1", first.piece_id)
        self.assertEqual(2, first.metadata["hidden_forward_distance"])

    def test_diagnostic_probe_exercises_teacher_behavior(self) -> None:
        teacher = generate_teacher_piece(seed=1, recipe="capture_sprint")
        probes = generate_diagnostic_probes(teacher.teacher_program)

        self.assertEqual(1, len(probes))
        self.assertEqual("piece_captured_hook", probes[0].label)
        self.assertEqual(
            "a4",
            probes[0].observed_state.piece_instances["white_teacher"].square,
        )

    def test_curriculum_repairs_candidate_and_writes_registry(self) -> None:
        teacher = generate_teacher_piece(seed=1, recipe="capture_sprint")
        candidate = _with_forward_offset(teacher.teacher_program, offset=1)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            registry_path = temp_path / "registry.json"
            out_dir = temp_path / "repaired"
            result = run_synthetic_piece_curriculum(
                seed=1,
                recipe="capture_sprint",
                compile_provider=StaticCompileProvider(candidate),
                repair_provider=StaticRepairProvider(teacher.teacher_program),
                registry_path=str(registry_path),
                out_dir=str(out_dir),
            )

            self.assertTrue(result.accepted, result.verification_errors)
            self.assertEqual(1, result.metadata["repairs"])
            self.assertIsNotNone(result.registry_record)
            self.assertTrue(Path(result.registry_record.dsl_path).exists())
            records = load_verified_piece_records(registry_path)
            self.assertEqual(1, len(records))
            self.assertEqual("synthetic_capture_sprint_1", records[0].piece_id)
            self.assertEqual(1, records[0].verified_cases)

    def test_curriculum_writes_registry_metadata(self) -> None:
        teacher = generate_teacher_piece(seed=1, recipe="capture_sprint")
        candidate = _with_forward_offset(teacher.teacher_program, offset=1)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            registry_path = temp_path / "registry.json"
            out_dir = temp_path / "repaired"
            result = run_synthetic_piece_curriculum(
                seed=1,
                recipe="capture_sprint",
                compile_provider=StaticCompileProvider(candidate),
                repair_provider=StaticRepairProvider(teacher.teacher_program),
                registry_path=str(registry_path),
                out_dir=str(out_dir),
                registry_metadata={
                    "family_id": "capture_sprint",
                    "split": "train",
                    "novelty_tier": "introduced",
                    "admission_cycle": 1,
                    "task_id": "cycle_001:seed_101:capture_sprint:train:introduced",
                },
            )

            self.assertTrue(result.accepted, result.verification_errors)
            self.assertIsNotNone(result.registry_record)
            metadata = dict(result.registry_record.metadata)
            self.assertEqual("capture_sprint", metadata["family_id"])
            self.assertEqual("train", metadata["split"])
            self.assertEqual("introduced", metadata["novelty_tier"])
            self.assertEqual(1, metadata["admission_cycle"])
            self.assertEqual(
                "cycle_001:seed_101:capture_sprint:train:introduced",
                metadata["task_id"],
            )

    def test_curriculum_rejects_mismatch_without_repair_provider(self) -> None:
        teacher = generate_teacher_piece(seed=1, recipe="capture_sprint")
        candidate = _with_forward_offset(teacher.teacher_program, offset=1)

        result = run_synthetic_piece_curriculum(
            seed=1,
            recipe="capture_sprint",
            compile_provider=StaticCompileProvider(candidate),
        )

        self.assertFalse(result.accepted)
        self.assertIn(
            "piece_captured_hook: mismatch without repair provider",
            result.verification_errors,
        )
        self.assertIsNone(result.final_program)

    def test_curriculum_repairs_missing_legal_move(self) -> None:
        teacher = generate_teacher_piece(seed=3, recipe="phase_rook")
        candidate = deepcopy(teacher.teacher_program)
        candidate["movement_modifiers"] = [{"op": "inherit_base", "args": {}}]
        repair_provider = _RecordingRepairProvider(teacher.teacher_program)

        result = run_synthetic_piece_curriculum(
            seed=3,
            recipe="phase_rook",
            compile_provider=StaticCompileProvider(candidate),
            repair_provider=repair_provider,
        )

        self.assertTrue(result.accepted, result.verification_errors)
        self.assertEqual("phase_through_allies", result.probe_results[0].label)
        self.assertIsNotNone(result.probe_results[0].mismatch["predicted_error"])
        self.assertIsNotNone(repair_provider.requests[0].predicted_error)
        self.assertEqual(1, result.metadata["repairs"])


def _with_forward_offset(program: dict[str, object], *, offset: int) -> dict[str, object]:
    candidate = deepcopy(program)
    hook = candidate["hooks"][0]
    hook["conditions"][1]["args"]["square"]["offset"] = [0, offset]
    hook["effects"][0]["args"]["to"]["offset"] = [0, offset]
    return candidate


class _RecordingRepairProvider:
    def __init__(self, patched_program: dict[str, object]) -> None:
        self.patched_program = patched_program
        self.requests: list[RepairRequest] = []

    def repair_piece(self, request: RepairRequest) -> RepairResponse:
        self.requests.append(request)
        return RepairResponse(patched_program=self.patched_program)


if __name__ == "__main__":
    unittest.main()
