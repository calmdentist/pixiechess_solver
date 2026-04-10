from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pixie_solver.core.hash import stable_digest
from pixie_solver.dsl.canonicalize import canonicalize_piece_program
from pixie_solver.dsl.compiler import compile_piece_file
from pixie_solver.dsl.parser import load_piece_program
from pixie_solver.dsl.validator import PieceValidationError, validate_piece_program
from pixie_solver.llm.compile_piece import compile_piece_from_text
from pixie_solver.utils.serialization import canonical_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pixie")
    subparsers = parser.add_subparsers(dest="command", required=True)

    compile_parser = subparsers.add_parser("compile-piece", help="Compile a piece program into the runtime PieceClass form.")
    compile_group = compile_parser.add_mutually_exclusive_group(required=True)
    compile_group.add_argument("--file", type=Path, help="Path to a JSON or YAML piece program.")
    compile_group.add_argument("--text", help="Natural-language piece description for future LLM compilation.")
    compile_parser.add_argument("--pretty", action="store_true", help="Pretty-print the compiled JSON.")
    compile_parser.set_defaults(handler=_handle_compile_piece)

    verify_parser = subparsers.add_parser("verify-piece", help="Validate a piece DSL program and print a stable digest.")
    verify_parser.add_argument("--file", type=Path, required=True, help="Path to a JSON or YAML piece program.")
    verify_parser.set_defaults(handler=_handle_verify_piece)

    for command_name in ("repair-piece", "run-match", "selfplay", "train", "eval-tactics"):
        placeholder_parser = subparsers.add_parser(command_name, help=f"{command_name} is planned but not implemented yet.")
        placeholder_parser.set_defaults(handler=_handle_placeholder)

    return parser


def _handle_compile_piece(args: argparse.Namespace) -> int:
    if args.text is not None:
        try:
            compile_piece_from_text(args.text)
        except NotImplementedError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        return 0

    piece_class = compile_piece_file(args.file)
    print(canonical_json(piece_class.to_dict(), indent=2 if args.pretty else None))
    return 0


def _handle_verify_piece(args: argparse.Namespace) -> int:
    program = load_piece_program(args.file)
    validate_piece_program(program)
    canonical_program = canonicalize_piece_program(program)
    compiled = compile_piece_file(args.file)
    print(
        canonical_json(
            {
                "piece_id": compiled.class_id,
                "status": "ok",
                "dsl_digest": stable_digest(program),
                "canonical_dsl_digest": stable_digest(canonical_program),
                "compiled_digest": stable_digest(compiled.to_dict()),
            },
            indent=2,
        )
    )
    return 0


def _handle_placeholder(args: argparse.Namespace) -> int:
    print(
        f"{args.command} is part of a later milestone and is not implemented in the foundation build.",
        file=sys.stderr,
    )
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except PieceValidationError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
