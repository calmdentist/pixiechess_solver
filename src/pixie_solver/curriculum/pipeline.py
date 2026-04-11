from __future__ import annotations

from dataclasses import dataclass, field

from pixie_solver.core.hash import stable_digest
from pixie_solver.dsl.canonicalize import canonicalize_piece_program
from pixie_solver.rules import (
    CompileRequest,
    PieceRegistryRecord,
    RepairResult,
    append_verified_piece_version,
    build_state_mismatch,
    repair_and_verify_piece,
)
from pixie_solver.rules.providers import (
    PieceProgramCompileProvider,
    PieceProgramRepairProvider,
)
from pixie_solver.rules.repair import default_dsl_reference
from pixie_solver.curriculum.probes import DiagnosticProbe, generate_diagnostic_probes
from pixie_solver.curriculum.teacher import SyntheticPiece, generate_teacher_piece
from pixie_solver.utils.serialization import JsonValue


@dataclass(frozen=True, slots=True)
class CurriculumProbeResult:
    label: str
    matched: bool
    repaired: bool = False
    repair_result: RepairResult | None = None
    mismatch: dict[str, JsonValue] | None = None

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "label": self.label,
            "matched": self.matched,
            "repaired": self.repaired,
            "repair_result": (
                self.repair_result.to_dict()
                if self.repair_result is not None
                else None
            ),
            "mismatch": self.mismatch,
        }


@dataclass(frozen=True, slots=True)
class CurriculumRunResult:
    accepted: bool
    synthetic_piece: SyntheticPiece
    initial_candidate_program: dict[str, JsonValue]
    final_program: dict[str, JsonValue] | None
    probe_results: tuple[CurriculumProbeResult, ...] = ()
    verification_errors: tuple[str, ...] = ()
    registry_record: PieceRegistryRecord | None = None
    metadata: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "initial_candidate_program",
            dict(self.initial_candidate_program),
        )
        if self.final_program is not None:
            object.__setattr__(self, "final_program", dict(self.final_program))
        object.__setattr__(self, "probe_results", tuple(self.probe_results))
        object.__setattr__(
            self,
            "verification_errors",
            tuple(str(error) for error in self.verification_errors),
        )
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "accepted": self.accepted,
            "synthetic_piece": self.synthetic_piece.to_dict(),
            "initial_candidate_program": dict(self.initial_candidate_program),
            "final_program": dict(self.final_program) if self.final_program is not None else None,
            "probe_results": [result.to_dict() for result in self.probe_results],
            "verification_errors": list(self.verification_errors),
            "registry_record": (
                self.registry_record.to_dict()
                if self.registry_record is not None
                else None
            ),
            "metadata": dict(self.metadata),
        }


def run_synthetic_piece_curriculum(
    *,
    seed: int,
    compile_provider: PieceProgramCompileProvider,
    repair_provider: PieceProgramRepairProvider | None = None,
    recipe: str | None = None,
    piece_id: str | None = None,
    registry_path: str | None = None,
    out_dir: str | None = None,
) -> CurriculumRunResult:
    synthetic_piece = generate_teacher_piece(
        seed=seed,
        recipe=recipe,
        piece_id=piece_id,
    )
    dsl_reference = default_dsl_reference()
    compile_response = compile_provider.compile_piece(
        CompileRequest(
            description=synthetic_piece.description,
            dsl_reference=dsl_reference,
            metadata={
                "seed": seed,
                "recipe": synthetic_piece.recipe,
                "piece_id": synthetic_piece.piece_id,
            },
        )
    )
    try:
        current_program = canonicalize_piece_program(compile_response.candidate_program)
    except Exception as exc:
        return CurriculumRunResult(
            accepted=False,
            synthetic_piece=synthetic_piece,
            initial_candidate_program=compile_response.candidate_program,
            final_program=None,
            verification_errors=(f"candidate DSL failed validation: {exc}",),
            metadata={"compile_response": compile_response.to_dict()},
        )

    probes = generate_diagnostic_probes(synthetic_piece.teacher_program)
    probe_results: list[CurriculumProbeResult] = []
    errors: list[str] = []
    repairs = 0

    for probe in probes:
        mismatch = build_state_mismatch(
            before_state=probe.before_state,
            move=probe.move,
            observed_state=probe.observed_state,
            current_program=current_program,
        )
        if mismatch.diff.is_empty and mismatch.predicted_error is None:
            probe_results.append(
                CurriculumProbeResult(
                    label=probe.label,
                    matched=True,
                )
            )
            continue

        if repair_provider is None:
            errors.append(f"{probe.label}: mismatch without repair provider")
            probe_results.append(
                CurriculumProbeResult(
                    label=probe.label,
                    matched=False,
                    mismatch=mismatch.to_dict(),
                )
            )
            continue

        repair_result = repair_and_verify_piece(
            mismatch,
            description=synthetic_piece.description,
            current_program=current_program,
            provider=repair_provider,
        )
        if not repair_result.accepted:
            errors.extend(f"{probe.label}: {error}" for error in repair_result.verification_errors)
            probe_results.append(
                CurriculumProbeResult(
                    label=probe.label,
                    matched=False,
                    repaired=False,
                    repair_result=repair_result,
                    mismatch=mismatch.to_dict(),
                )
            )
            continue

        repairs += 1
        current_program = dict(repair_result.patched_program)
        probe_results.append(
            CurriculumProbeResult(
                label=probe.label,
                matched=True,
                repaired=True,
                repair_result=repair_result,
                mismatch=mismatch.to_dict(),
            )
        )

    final_errors = _verify_all_probes(
        probes=probes,
        current_program=current_program,
        description=synthetic_piece.description,
    )
    errors.extend(final_errors)
    accepted = not errors
    registry_record = None
    if accepted and registry_path is not None and out_dir is not None:
        registry_record = append_verified_piece_version(
            registry_path=registry_path,
            out_dir=out_dir,
            program=current_program,
            description=synthetic_piece.description,
            source="synthetic_curriculum",
            parent_digest=stable_digest(canonicalize_piece_program(compile_response.candidate_program)),
            verified_cases=len(probes),
            repair_attempts=max(repairs, 1),
            metadata={
                "seed": seed,
                "recipe": synthetic_piece.recipe,
                "teacher_digest": stable_digest(synthetic_piece.teacher_program),
                "compile_response": compile_response.to_dict(),
            },
        )

    return CurriculumRunResult(
        accepted=accepted,
        synthetic_piece=synthetic_piece,
        initial_candidate_program=canonicalize_piece_program(compile_response.candidate_program),
        final_program=current_program if accepted else None,
        probe_results=tuple(probe_results),
        verification_errors=tuple(errors),
        registry_record=registry_record,
        metadata={
            "probes": len(probes),
            "repairs": repairs,
            "compile_response": compile_response.to_dict(),
        },
    )


def _verify_all_probes(
    *,
    probes: tuple[DiagnosticProbe, ...],
    current_program: dict[str, JsonValue],
    description: str,
) -> list[str]:
    errors: list[str] = []
    from pixie_solver.rules.providers import StaticRepairProvider

    for probe in probes:
        mismatch = build_state_mismatch(
            before_state=probe.before_state,
            move=probe.move,
            observed_state=probe.observed_state,
            current_program=current_program,
        )
        if mismatch.diff.is_empty and mismatch.predicted_error is None:
            continue
        repair_result = repair_and_verify_piece(
            mismatch,
            description=description,
            current_program=current_program,
            provider=StaticRepairProvider(current_program),
        )
        errors.extend(f"{probe.label}: {error}" for error in repair_result.verification_errors)
    return errors
