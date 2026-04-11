from __future__ import annotations

from typing import Any

from pixie_solver.llm.frontier import FrontierLLMClient, FrontierLLMPieceProgramProvider, LLMConfig
from pixie_solver.rules.providers import RepairRequest
from pixie_solver.rules.repair import default_dsl_reference


def repair_piece_program(
    *,
    description: str,
    current_program: dict[str, Any],
    diff: dict[str, Any],
    config: LLMConfig | None = None,
) -> dict[str, Any]:
    provider = FrontierLLMPieceProgramProvider(
        FrontierLLMClient(config or LLMConfig())
    )
    response = provider.repair_piece(
        RepairRequest(
            description=description,
            current_program=current_program,
            before_state={},
            move={},
            predicted_state={},
            observed_state={},
            diff=diff,
            dsl_reference=default_dsl_reference(),
        )
    )
    return dict(response.patched_program)
