from __future__ import annotations

from pixie_solver.llm.frontier import FrontierLLMClient, FrontierLLMPieceProgramProvider, LLMConfig
from pixie_solver.rules.providers import CompileRequest
from pixie_solver.rules.repair import default_dsl_reference


def compile_piece_from_text(
    description: str,
    *,
    config: LLMConfig | None = None,
) -> dict[str, object]:
    provider = FrontierLLMPieceProgramProvider(
        FrontierLLMClient(config or LLMConfig())
    )
    response = provider.compile_piece(
        CompileRequest(
            description=description,
            dsl_reference=default_dsl_reference(),
        )
    )
    return dict(response.candidate_program)
