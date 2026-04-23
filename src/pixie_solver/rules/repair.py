from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any

from pixie_solver.core import GameState, Move
from pixie_solver.dsl.canonicalize import canonicalize_piece_program
from pixie_solver.dsl.compiler import compile_piece_program
from pixie_solver.rules.mismatch import (
    StateMismatch,
    behavioral_state_signature,
    diff_states,
    replace_piece_program,
)
from pixie_solver.rules.providers import (
    PieceProgramRepairProvider,
    RepairRequest,
    RepairResponse,
)
from pixie_solver.simulator.engine import apply_move
from pixie_solver.utils.serialization import JsonValue


@dataclass(frozen=True, slots=True)
class ReplayCase:
    before_state: GameState
    move: Move
    observed_state: GameState
    label: str = ""

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "before_state": self.before_state.to_dict(),
            "move": self.move.to_dict(),
            "observed_state": self.observed_state.to_dict(),
            "label": self.label,
        }


@dataclass(frozen=True, slots=True)
class RepairResult:
    accepted: bool
    current_program: dict[str, JsonValue]
    patched_program: dict[str, JsonValue] | None = None
    response: RepairResponse | None = None
    verification_errors: tuple[str, ...] = ()
    primary_diff_after_repair: dict[str, JsonValue] | None = None
    verified_cases: int = 0
    metadata: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "current_program", dict(self.current_program))
        if self.patched_program is not None:
            object.__setattr__(self, "patched_program", dict(self.patched_program))
        object.__setattr__(
            self,
            "verification_errors",
            tuple(str(error) for error in self.verification_errors),
        )
        if self.primary_diff_after_repair is not None:
            object.__setattr__(
                self,
                "primary_diff_after_repair",
                dict(self.primary_diff_after_repair),
            )
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "accepted": self.accepted,
            "current_program": dict(self.current_program),
            "patched_program": (
                dict(self.patched_program)
                if self.patched_program is not None
                else None
            ),
            "response": self.response.to_dict() if self.response is not None else None,
            "verification_errors": list(self.verification_errors),
            "primary_diff_after_repair": self.primary_diff_after_repair,
            "verified_cases": self.verified_cases,
            "metadata": dict(self.metadata),
        }


def repair_and_verify_piece(
    mismatch: StateMismatch,
    *,
    description: str,
    current_program: dict[str, Any],
    provider: PieceProgramRepairProvider,
    regression_cases: tuple[ReplayCase, ...] = (),
    dsl_reference: dict[str, JsonValue] | None = None,
    allow_piece_id_change: bool = False,
) -> RepairResult:
    return repair_and_verify_piece_candidates(
        mismatch,
        description=description,
        current_program=current_program,
        provider=provider,
        regression_cases=regression_cases,
        dsl_reference=dsl_reference,
        allow_piece_id_change=allow_piece_id_change,
        candidate_count=1,
    )[0]


def repair_and_verify_piece_candidates(
    mismatch: StateMismatch,
    *,
    description: str,
    current_program: dict[str, Any],
    provider: PieceProgramRepairProvider,
    regression_cases: tuple[ReplayCase, ...] = (),
    dsl_reference: dict[str, JsonValue] | None = None,
    allow_piece_id_change: bool = False,
    candidate_count: int = 1,
) -> tuple[RepairResult, ...]:
    canonical_current = canonicalize_piece_program(current_program)
    request = RepairRequest(
        description=description,
        current_program=canonical_current,
        before_state=mismatch.before_state.to_dict(),
        move=mismatch.move.to_dict(),
        predicted_state=mismatch.predicted_state.to_dict(),
        observed_state=mismatch.observed_state.to_dict(),
        diff=mismatch.diff.to_dict(),
        predicted_error=mismatch.predicted_error,
        implicated_piece_ids=mismatch.implicated_piece_ids,
        implicated_piece_class_ids=mismatch.implicated_piece_class_ids,
        dsl_reference=dsl_reference or default_dsl_reference(),
    )
    responses = _repair_candidate_responses(provider, request, candidate_count)
    if not responses:
        return (
            RepairResult(
                accepted=False,
                current_program=canonical_current,
                verification_errors=("repair provider returned no candidates",),
                metadata={"stage": "provider"},
            ),
        )

    results: list[RepairResult] = []
    seen_signatures: set[str] = set()
    for candidate_index, response in enumerate(responses):
        result = verify_repair_response(
            mismatch,
            current_program=canonical_current,
            response=response,
            regression_cases=regression_cases,
            allow_piece_id_change=allow_piece_id_change,
        )
        if result.patched_program is not None:
            signature = json.dumps(result.patched_program, sort_keys=True)
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
        results.append(
            RepairResult(
                accepted=result.accepted,
                current_program=result.current_program,
                patched_program=result.patched_program,
                response=result.response,
                verification_errors=result.verification_errors,
                primary_diff_after_repair=result.primary_diff_after_repair,
                verified_cases=result.verified_cases,
                metadata={
                    **dict(result.metadata),
                    "candidate_index": candidate_index,
                    "candidate_count": len(responses),
                },
            )
        )
    return tuple(results)


def verify_repair_response(
    mismatch: StateMismatch,
    *,
    current_program: dict[str, Any],
    response: RepairResponse,
    regression_cases: tuple[ReplayCase, ...] = (),
    allow_piece_id_change: bool = False,
) -> RepairResult:
    canonical_current = canonicalize_piece_program(current_program)
    errors: list[str] = []
    try:
        patched_program = canonicalize_piece_program(response.patched_program)
        patched_piece_class = compile_piece_program(patched_program)
    except Exception as exc:
        return RepairResult(
            accepted=False,
            current_program=canonical_current,
            response=response,
            verification_errors=(f"patched DSL failed validation: {exc}",),
            metadata={"stage": "validation"},
        )

    current_piece_id = str(canonical_current["piece_id"])
    patched_piece_id = patched_piece_class.class_id
    if current_piece_id != patched_piece_id and not allow_piece_id_change:
        errors.append(
            "patched DSL changed piece_id "
            f"from {current_piece_id!r} to {patched_piece_id!r}"
        )

    primary_diff_after_repair = _verify_case(
        before_state=mismatch.before_state,
        move=mismatch.move,
        observed_state=mismatch.observed_state,
        patched_program=patched_program,
        label="primary_mismatch",
        errors=errors,
    )
    verified_cases = 0 if primary_diff_after_repair is not None else 1

    for case in regression_cases:
        case_diff = _verify_case(
            before_state=case.before_state,
            move=case.move,
            observed_state=case.observed_state,
            patched_program=patched_program,
            label=case.label or "regression_case",
            errors=errors,
        )
        if case_diff is None:
            verified_cases += 1

    return RepairResult(
        accepted=not errors,
        current_program=canonical_current,
        patched_program=patched_program,
        response=response,
        verification_errors=tuple(errors),
        primary_diff_after_repair=primary_diff_after_repair,
        verified_cases=verified_cases,
        metadata={
            "regression_cases": len(regression_cases),
            "current_piece_id": current_piece_id,
            "patched_piece_id": patched_piece_id,
        },
    )


def default_dsl_reference() -> dict[str, JsonValue]:
    from pixie_solver.dsl import schema

    return {
        "base_piece_types": sorted(schema.BASE_PIECE_TYPES),
        "movement_modifiers": sorted(schema.MOVEMENT_MODIFIERS),
        "capture_modifiers": sorted(schema.CAPTURE_MODIFIERS),
        "hook_events": sorted(schema.HOOK_EVENTS),
        "condition_operators": sorted(schema.CONDITION_OPERATORS),
        "effect_operators": sorted(schema.EFFECT_OPERATORS),
        "piece_refs": sorted(schema.PIECE_REFS),
        "edge_behaviors": sorted(schema.EDGE_BEHAVIORS),
        "state_field_types": sorted(schema.STATE_FIELD_TYPES),
    }


def _repair_candidate_responses(
    provider: PieceProgramRepairProvider,
    request: RepairRequest,
    candidate_count: int,
) -> tuple[RepairResponse, ...]:
    if candidate_count < 1:
        raise ValueError("candidate_count must be at least 1")
    repair_piece_candidates = getattr(provider, "repair_piece_candidates", None)
    if callable(repair_piece_candidates):
        responses = tuple(repair_piece_candidates(request, candidate_count))
        if responses:
            return responses[:candidate_count]
    return (provider.repair_piece(request),)


def _verify_case(
    *,
    before_state: GameState,
    move: Move,
    observed_state: GameState,
    patched_program: dict[str, Any],
    label: str,
    errors: list[str],
) -> dict[str, JsonValue] | None:
    try:
        patched_before = replace_piece_program(before_state, patched_program)
        patched_state, _ = apply_move(patched_before, move)
    except Exception as exc:
        errors.append(f"{label}: patched transition failed: {exc}")
        return {"transition_error": str(exc)}

    if behavioral_state_signature(patched_state) == behavioral_state_signature(observed_state):
        return None

    diff = diff_states(patched_state, observed_state).to_dict()
    errors.append(f"{label}: patched transition still differs from observed state")
    return diff
