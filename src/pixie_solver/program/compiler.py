from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from pixie_solver.dsl.parser import load_piece_program
from pixie_solver.program.canonicalize import canonicalize_program_ir
from pixie_solver.program.lower_legacy_dsl import lower_legacy_piece_program


def compile_program_ir(program: Mapping[str, Any]) -> dict[str, Any]:
    if "program_id" in program:
        return canonicalize_program_ir(program)
    if "piece_id" in program:
        return lower_legacy_piece_program(program)
    raise ValueError(
        "program source must be canonical ProgramIR or legacy piece DSL"
    )


def compile_program_file(path: str | Path) -> dict[str, Any]:
    return compile_program_ir(load_piece_program(path))
