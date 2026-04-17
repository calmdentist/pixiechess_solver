from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "ActionContext": ("pixie_solver.program.contexts", "ActionContext"),
    "EventContext": ("pixie_solver.program.contexts", "EventContext"),
    "QueryContext": ("pixie_solver.program.contexts", "QueryContext"),
    "ThreatContext": ("pixie_solver.program.contexts", "ThreatContext"),
    "ProgramValidationError": (
        "pixie_solver.program.validator",
        "ProgramValidationError",
    ),
    "build_legacy_piece_class_for_actions": (
        "pixie_solver.program.stdlib",
        "build_legacy_piece_class_for_actions",
    ),
    "canonicalize_program_ir": (
        "pixie_solver.program.canonicalize",
        "canonicalize_program_ir",
    ),
    "compile_program_file": ("pixie_solver.program.compiler", "compile_program_file"),
    "compile_program_ir": ("pixie_solver.program.compiler", "compile_program_ir"),
    "lower_legacy_piece_class": (
        "pixie_solver.program.lower_legacy_dsl",
        "lower_legacy_piece_class",
    ),
    "lower_legacy_piece_program": (
        "pixie_solver.program.lower_legacy_dsl",
        "lower_legacy_piece_program",
    ),
    "resolve_piece_ref": ("pixie_solver.program.stdlib", "resolve_piece_ref"),
    "resolve_square_ref": ("pixie_solver.program.stdlib", "resolve_square_ref"),
    "validate_program_ir": ("pixie_solver.program.validator", "validate_program_ir"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> object:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
