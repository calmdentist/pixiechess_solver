from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "CompiledPieceArtifacts": ("pixie_solver.dsl.compiler", "CompiledPieceArtifacts"),
    "PieceValidationError": ("pixie_solver.dsl.validator", "PieceValidationError"),
    "canonicalize_piece_program": (
        "pixie_solver.dsl.canonicalize",
        "canonicalize_piece_program",
    ),
    "compile_piece_artifacts": ("pixie_solver.dsl.compiler", "compile_piece_artifacts"),
    "compile_piece_file": ("pixie_solver.dsl.compiler", "compile_piece_file"),
    "compile_piece_file_ir": ("pixie_solver.dsl.compiler", "compile_piece_file_ir"),
    "compile_piece_program": ("pixie_solver.dsl.compiler", "compile_piece_program"),
    "compile_piece_program_ir": (
        "pixie_solver.dsl.compiler",
        "compile_piece_program_ir",
    ),
    "load_piece_program": ("pixie_solver.dsl.parser", "load_piece_program"),
    "parse_piece_program_text": (
        "pixie_solver.dsl.parser",
        "parse_piece_program_text",
    ),
    "validate_piece_program": ("pixie_solver.dsl.validator", "validate_piece_program"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> object:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
