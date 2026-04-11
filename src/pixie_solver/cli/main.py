from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

from pixie_solver.core.hash import stable_digest

from pixie_solver.core import (
    GameState,
    Move,
    PieceClass,
    sample_standard_initial_state,
    standard_initial_state,
)
from pixie_solver.curriculum import run_synthetic_piece_curriculum
from pixie_solver.dsl.canonicalize import canonicalize_piece_program
from pixie_solver.dsl.compiler import compile_piece_file
from pixie_solver.dsl.parser import load_piece_program
from pixie_solver.dsl.validator import PieceValidationError, validate_piece_program
from pixie_solver.eval import ModelEvalProgress, evaluate_policy_value_model
from pixie_solver.llm import FrontierLLMClient, FrontierLLMPieceProgramProvider, LLMConfig
from pixie_solver.model import PolicyValueConfig
from pixie_solver.rules import (
    CompileRequest,
    JsonFileCompileProvider,
    JsonFileRepairProvider,
    append_verified_piece_version,
    build_state_mismatch,
    load_verified_piece_classes,
    load_verified_piece_records,
    registry_piece_digest_metadata,
    repair_and_verify_piece,
)
from pixie_solver.rules.repair import default_dsl_reference
from pixie_solver.training import (
    SelfPlayConfig,
    SelfPlayProgress,
    TrainingConfig,
    TrainingProgress,
    flatten_selfplay_examples,
    generate_selfplay_games,
    load_training_checkpoint,
    save_training_checkpoint,
    read_selfplay_examples_jsonl,
    write_selfplay_examples_jsonl,
    write_selfplay_games_jsonl,
    train_from_replays,
)
from pixie_solver.utils.serialization import canonical_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pixie")
    subparsers = parser.add_subparsers(dest="command", required=True)

    compile_parser = subparsers.add_parser("compile-piece", help="Compile a piece program into the runtime PieceClass form.")
    compile_group = compile_parser.add_mutually_exclusive_group(required=True)
    compile_group.add_argument("--file", type=Path, help="Path to a JSON or YAML piece program.")
    compile_group.add_argument("--text", help="Natural-language piece description for future LLM compilation.")
    compile_parser.add_argument("--pretty", action="store_true", help="Pretty-print the compiled JSON.")
    _add_llm_args(compile_parser)
    compile_parser.set_defaults(handler=_handle_compile_piece)

    verify_parser = subparsers.add_parser("verify-piece", help="Validate a piece DSL program and print a stable digest.")
    verify_parser.add_argument("--file", type=Path, required=True, help="Path to a JSON or YAML piece program.")
    verify_parser.set_defaults(handler=_handle_verify_piece)

    repair_parser = subparsers.add_parser(
        "repair-piece",
        help="Patch a piece DSL program from an observed transition mismatch.",
    )
    repair_parser.add_argument("--description", required=True, help="Natural-language piece description.")
    repair_parser.add_argument("--current-program", type=Path, required=True, help="Current JSON/YAML DSL program.")
    repair_parser.add_argument("--before-state", type=Path, required=True, help="JSON GameState before the observed move.")
    repair_parser.add_argument("--move", type=Path, required=True, help="JSON Move that produced the mismatch.")
    repair_parser.add_argument("--observed-state", type=Path, required=True, help="JSON GameState observed from the oracle/live game.")
    repair_parser.add_argument(
        "--provider-response",
        type=Path,
        help="JSON file containing the provider response or raw patched DSL program. If omitted, the live LLM provider is used.",
    )
    _add_llm_args(repair_parser)
    repair_parser.add_argument(
        "--registry",
        type=Path,
        default=Path("data/pieces/registry.json"),
        help="Piece registry manifest to append accepted repaired versions to.",
    )
    repair_parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/pieces/repaired"),
        help="Directory where accepted repaired DSL programs are written.",
    )
    repair_parser.add_argument(
        "--repair-attempts",
        type=int,
        default=1,
        help="Repair attempt count to record in the registry metadata.",
    )
    repair_parser.set_defaults(handler=_handle_repair_piece)

    curriculum_parser = subparsers.add_parser(
        "piece-curriculum",
        help="Generate a synthetic teacher piece, compile/repair a candidate, and optionally admit it to the verified registry.",
    )
    curriculum_parser.add_argument("--seed", type=int, required=True, help="Deterministic synthetic piece seed.")
    curriculum_parser.add_argument(
        "--recipe",
        choices=("capture_sprint", "edge_sumo", "phase_rook", "turn_charge"),
        help="Optional synthetic piece recipe. Defaults to seed-based selection.",
    )
    curriculum_parser.add_argument("--piece-id", help="Optional deterministic piece id override.")
    curriculum_parser.add_argument(
        "--compile-response",
        type=Path,
        help="JSON file containing a candidate_program response or raw candidate DSL. If omitted, the live LLM provider is used.",
    )
    curriculum_parser.add_argument(
        "--repair-response",
        type=Path,
        help="Optional JSON file containing a patched_program response or raw patched DSL.",
    )
    curriculum_parser.add_argument(
        "--llm-repair",
        action="store_true",
        help="Use the configured live LLM for mismatch repair when --repair-response is omitted.",
    )
    _add_llm_args(curriculum_parser)
    curriculum_parser.add_argument(
        "--registry",
        type=Path,
        default=Path("data/pieces/registry.json"),
        help="Piece registry manifest to append accepted curriculum pieces to.",
    )
    curriculum_parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/pieces/repaired"),
        help="Directory where accepted curriculum DSL programs are written.",
    )
    curriculum_parser.add_argument(
        "--no-registry-write",
        action="store_true",
        help="Run verification only without writing accepted pieces to the registry.",
    )
    curriculum_parser.add_argument(
        "--artifact-dir",
        type=Path,
        help="Optional directory for teacher, description, and probe artifacts.",
    )
    curriculum_parser.set_defaults(handler=_handle_piece_curriculum)

    selfplay_parser = subparsers.add_parser("selfplay", help="Generate self-play games/examples from serialized GameState seeds.")
    selfplay_source_group = selfplay_parser.add_mutually_exclusive_group(required=True)
    selfplay_source_group.add_argument(
        "--state-file",
        type=Path,
        action="append",
        help="Path to a JSON file containing one GameState object or a list of GameState objects.",
    )
    selfplay_source_group.add_argument(
        "--standard-initial-state",
        action="store_true",
        help="Use the built-in orthodox chess initial position as the sole seed state.",
    )
    selfplay_parser.add_argument(
        "--randomize-handauthored-specials",
        action="store_true",
        help="When using --standard-initial-state, randomly replace eligible starting slots with hand-authored special pieces.",
    )
    selfplay_parser.add_argument(
        "--special-piece-dir",
        type=Path,
        default=Path("data/pieces/handauthored"),
        help="Directory of hand-authored piece JSON/YAML files used for randomized special-piece openings.",
    )
    selfplay_parser.add_argument(
        "--piece-registry",
        type=Path,
        default=Path("data/pieces/registry.json"),
        help="Verified piece registry used when --use-verified-pieces is enabled.",
    )
    selfplay_parser.add_argument(
        "--use-verified-pieces",
        action="store_true",
        help="Include latest verified repaired pieces from --piece-registry in randomized openings.",
    )
    selfplay_parser.add_argument(
        "--special-piece-inclusion-probability",
        type=float,
        default=0.5,
        help="Independent inclusion probability for each special piece when randomizing openings.",
    )
    selfplay_parser.add_argument("--games", type=int, default=1, help="Number of games to generate.")
    selfplay_parser.add_argument("--games-out", type=Path, help="Optional JSONL output path for self-play game records.")
    selfplay_parser.add_argument("--examples-out", type=Path, help="Optional JSONL output path for flattened self-play examples.")
    selfplay_parser.add_argument("--checkpoint", type=Path, help="Optional checkpoint for model-guided self-play.")
    selfplay_parser.add_argument("--device", help="Device to load the checkpoint onto, e.g. cpu, mps, cuda.")
    selfplay_parser.add_argument("--simulations", type=int, default=64, help="MCTS simulations per move.")
    selfplay_parser.add_argument("--max-plies", type=int, default=256, help="Maximum plies per generated game.")
    selfplay_parser.add_argument("--opening-temperature", type=float, default=1.0, help="Sampling temperature before the drop ply.")
    selfplay_parser.add_argument("--final-temperature", type=float, default=0.0, help="Sampling temperature after the drop ply.")
    selfplay_parser.add_argument("--temperature-drop-after-ply", type=int, default=12, help="Ply after which the final temperature is used.")
    selfplay_parser.add_argument("--c-puct", type=float, default=1.5, help="PUCT exploration constant.")
    selfplay_parser.add_argument("--root-dirichlet-alpha", type=float, default=0.3, help="Symmetric Dirichlet alpha for root exploration noise during self-play.")
    selfplay_parser.add_argument("--root-exploration-fraction", type=float, default=0.25, help="Fraction of root priors replaced by Dirichlet noise during self-play. Set to 0 to disable.")
    selfplay_parser.add_argument("--seed", type=int, default=0, help="Deterministic seed for self-play sampling.")
    selfplay_parser.add_argument(
        "--adjudicate-max-plies",
        dest="adjudicate_max_plies",
        action="store_true",
        default=True,
        help="Use deterministic cutoff adjudication when a game reaches --max-plies.",
    )
    selfplay_parser.add_argument(
        "--no-adjudicate-max-plies",
        dest="adjudicate_max_plies",
        action="store_false",
        help="Treat non-terminal max-ply games as draws.",
    )
    selfplay_parser.add_argument(
        "--adjudication-threshold",
        type=float,
        default=0.2,
        help="White-perspective heuristic cutoff threshold in [0, 1].",
    )
    selfplay_parser.add_argument(
        "--log-every-plies",
        type=int,
        default=8,
        help="Emit a self-play progress line every N plies within a game. Set to 1 for per-ply logging.",
    )
    selfplay_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress logs on stderr and only print the final JSON summary.",
    )
    selfplay_parser.set_defaults(handler=_handle_selfplay)

    train_parser = subparsers.add_parser("train", help="Train or resume the policy/value model from self-play examples.")
    train_parser.add_argument("--examples", type=Path, required=True, help="Input JSONL path of SelfPlayExample rows.")
    train_parser.add_argument("--checkpoint-out", type=Path, required=True, help="Output checkpoint path.")
    train_parser.add_argument("--resume-checkpoint", type=Path, help="Optional checkpoint to resume model and optimizer state from.")
    train_parser.add_argument("--device", help="Training device, e.g. cpu, mps, cuda.")
    train_parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    train_parser.add_argument("--batch-size", type=int, default=8, help="Training batch size.")
    train_parser.add_argument("--learning-rate", type=float, default=3e-4, help="AdamW learning rate.")
    train_parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    train_parser.add_argument("--policy-weight", type=float, default=1.0, help="Weight on the policy loss term.")
    train_parser.add_argument("--value-weight", type=float, default=1.0, help="Weight on the value loss term.")
    train_parser.add_argument("--seed", type=int, default=0, help="Deterministic seed for DataLoader shuffling and initialization.")
    train_parser.add_argument("--shuffle", dest="shuffle", action="store_true", help="Shuffle training examples before each epoch.")
    train_parser.add_argument("--no-shuffle", dest="shuffle", action="store_false", help="Disable example shuffling.")
    train_parser.set_defaults(shuffle=True)
    train_parser.add_argument("--d-model", type=int, help="Transformer model width. Only valid when not resuming.")
    train_parser.add_argument("--num-heads", type=int, help="Transformer attention heads. Only valid when not resuming.")
    train_parser.add_argument("--num-layers", type=int, help="Transformer encoder layers. Only valid when not resuming.")
    train_parser.add_argument("--dropout", type=float, help="Transformer dropout. Only valid when not resuming.")
    train_parser.add_argument("--feedforward-multiplier", type=int, help="Feedforward hidden-size multiplier. Only valid when not resuming.")
    train_parser.add_argument(
        "--log-every-batches",
        type=int,
        default=10,
        help="Emit a training progress line every N completed batches.",
    )
    train_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress logs on stderr and only print the final JSON summary.",
    )
    train_parser.set_defaults(handler=_handle_train)

    eval_model_parser = subparsers.add_parser(
        "eval-model",
        help="Evaluate a policy/value checkpoint against self-play examples.",
    )
    eval_model_parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint to evaluate.")
    eval_model_parser.add_argument("--examples", type=Path, required=True, help="SelfPlayExample JSONL file to score.")
    eval_model_parser.add_argument("--device", help="Evaluation device, e.g. cpu, mps, cuda.")
    eval_model_parser.add_argument(
        "--log-every-examples",
        type=int,
        default=25,
        help="Emit an evaluation progress line every N examples.",
    )
    eval_model_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress logs on stderr and only print the final JSON summary.",
    )
    eval_model_parser.set_defaults(handler=_handle_eval_model)

    train_loop_parser = subparsers.add_parser(
        "train-loop",
        help="Run self-play, training, and train/validation evaluation for multiple cycles.",
    )
    train_loop_parser.add_argument("--output-dir", type=Path, required=True, help="Directory for generated data, checkpoints, and metrics.")
    train_loop_parser.add_argument("--cycles", type=int, default=2, help="Number of self-play/train/eval cycles to run.")
    train_loop_parser.add_argument("--train-games", type=int, default=20, help="Training self-play games per cycle.")
    train_loop_parser.add_argument("--val-games", type=int, default=6, help="Validation self-play games per cycle.")
    train_loop_parser.add_argument("--simulations", type=int, default=8, help="MCTS simulations per self-play move.")
    train_loop_parser.add_argument("--max-plies", type=int, default=48, help="Maximum plies per self-play game.")
    train_loop_parser.add_argument("--opening-temperature", type=float, default=1.0, help="Self-play opening temperature.")
    train_loop_parser.add_argument("--final-temperature", type=float, default=0.0, help="Self-play final temperature.")
    train_loop_parser.add_argument("--temperature-drop-after-ply", type=int, default=12, help="Ply after which final temperature is used.")
    train_loop_parser.add_argument("--c-puct", type=float, default=1.5, help="PUCT exploration constant.")
    train_loop_parser.add_argument("--root-dirichlet-alpha", type=float, default=0.3, help="Symmetric Dirichlet alpha for root exploration noise during self-play.")
    train_loop_parser.add_argument("--root-exploration-fraction", type=float, default=0.25, help="Fraction of root priors replaced by Dirichlet noise during self-play. Set to 0 to disable.")
    train_loop_parser.add_argument("--seed", type=int, default=0, help="Base seed for self-play and training.")
    train_loop_parser.add_argument("--adjudicate-max-plies", dest="adjudicate_max_plies", action="store_true", default=True, help="Use deterministic cutoff adjudication when self-play reaches --max-plies.")
    train_loop_parser.add_argument("--no-adjudicate-max-plies", dest="adjudicate_max_plies", action="store_false", help="Treat non-terminal max-ply games as draws.")
    train_loop_parser.add_argument("--adjudication-threshold", type=float, default=0.2, help="White-perspective heuristic cutoff threshold in [0, 1].")
    train_loop_parser.add_argument("--device", help="Device for training/eval/model-guided self-play, e.g. cpu, mps, cuda.")
    train_loop_parser.add_argument("--epochs-per-cycle", type=int, default=2, help="Training epochs per cycle.")
    train_loop_parser.add_argument("--batch-size", type=int, default=4, help="Training batch size.")
    train_loop_parser.add_argument("--learning-rate", type=float, default=3e-4, help="AdamW learning rate.")
    train_loop_parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    train_loop_parser.add_argument("--policy-weight", type=float, default=1.0, help="Policy loss weight.")
    train_loop_parser.add_argument("--value-weight", type=float, default=1.0, help="Value loss weight.")
    train_loop_parser.add_argument("--d-model", type=int, default=64, help="Transformer model width for a fresh run.")
    train_loop_parser.add_argument("--num-heads", type=int, default=4, help="Transformer attention heads for a fresh run.")
    train_loop_parser.add_argument("--num-layers", type=int, default=2, help="Transformer encoder layers for a fresh run.")
    train_loop_parser.add_argument("--dropout", type=float, default=0.1, help="Transformer dropout for a fresh run.")
    train_loop_parser.add_argument("--feedforward-multiplier", type=int, default=2, help="Feedforward hidden-size multiplier for a fresh run.")
    train_loop_parser.add_argument("--resume-checkpoint", type=Path, help="Optional checkpoint to resume the loop from.")
    train_loop_parser.add_argument("--no-guided-selfplay", action="store_true", help="Always generate self-play with search-only MCTS instead of the latest model.")
    train_loop_parser.add_argument("--randomize-handauthored-specials", action="store_true", default=True, help="Randomize hand-authored special pieces in standard openings.")
    train_loop_parser.add_argument("--no-randomize-handauthored-specials", dest="randomize_handauthored_specials", action="store_false", help="Use plain orthodox standard openings.")
    train_loop_parser.add_argument("--special-piece-dir", type=Path, default=Path("data/pieces/handauthored"), help="Directory of hand-authored piece JSON/YAML files.")
    train_loop_parser.add_argument("--special-piece-inclusion-probability", type=float, default=0.5, help="Special-piece inclusion probability.")
    train_loop_parser.add_argument("--log-every-plies", type=int, default=8, help="Self-play progress frequency.")
    train_loop_parser.add_argument("--log-every-batches", type=int, default=10, help="Training progress frequency.")
    train_loop_parser.add_argument("--log-every-examples", type=int, default=25, help="Evaluation progress frequency.")
    train_loop_parser.add_argument("--quiet", action="store_true", help="Suppress progress logs on stderr and only print final JSON.")
    train_loop_parser.set_defaults(handler=_handle_train_loop)

    train_loop_parser.add_argument("--piece-registry", type=Path, default=Path("data/pieces/registry.json"), help="Verified piece registry used when --use-verified-pieces is enabled.")
    train_loop_parser.add_argument("--use-verified-pieces", action="store_true", help="Include latest verified repaired pieces from --piece-registry in randomized openings.")

    for command_name in ("run-match", "eval-tactics"):
        placeholder_parser = subparsers.add_parser(command_name, help=f"{command_name} is planned but not implemented yet.")
        placeholder_parser.set_defaults(handler=_handle_placeholder)

    return parser


def _add_llm_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--llm-provider",
        choices=("anthropic", "openai"),
        help="Live LLM provider. Defaults to PIXIE_LLM_PROVIDER or anthropic.",
    )
    parser.add_argument(
        "--llm-model",
        help="Live LLM model id. Defaults to PIXIE_LLM_MODEL, claude-opus-4-6, or gpt-5.4.",
    )
    parser.add_argument(
        "--llm-api-key-env",
        help="Environment variable containing the provider API key.",
    )
    parser.add_argument(
        "--llm-max-tokens",
        type=int,
        help="Maximum output tokens for live LLM calls.",
    )
    parser.add_argument(
        "--llm-timeout-seconds",
        type=float,
        help="HTTP timeout for live LLM calls.",
    )
    parser.add_argument(
        "--llm-temperature",
        type=float,
        help="Sampling temperature for live LLM calls.",
    )
    parser.add_argument(
        "--llm-effort",
        choices=("none", "low", "medium", "high", "max", "xhigh"),
        help="Reasoning/thinking effort for live LLM calls.",
    )
    parser.add_argument(
        "--no-llm-thinking",
        action="store_true",
        help="Disable Claude adaptive thinking for live Anthropic calls.",
    )


def _build_llm_piece_provider(args: argparse.Namespace) -> FrontierLLMPieceProgramProvider:
    config_kwargs = {
        "provider": args.llm_provider,
        "model": args.llm_model,
        "api_key_env": args.llm_api_key_env,
        "thinking": not args.no_llm_thinking,
    }
    if args.llm_max_tokens is not None:
        config_kwargs["max_tokens"] = args.llm_max_tokens
    if args.llm_timeout_seconds is not None:
        config_kwargs["timeout_seconds"] = args.llm_timeout_seconds
    if args.llm_temperature is not None:
        config_kwargs["temperature"] = args.llm_temperature
    if args.llm_effort is not None:
        config_kwargs["effort"] = args.llm_effort
    config = LLMConfig(**config_kwargs)
    return FrontierLLMPieceProgramProvider(FrontierLLMClient(config))


def _handle_compile_piece(args: argparse.Namespace) -> int:
    if args.text is not None:
        provider = _build_llm_piece_provider(args)
        response = provider.compile_piece(
            CompileRequest(
                description=args.text,
                dsl_reference=default_dsl_reference(),
            )
        )
        program = canonicalize_piece_program(response.candidate_program)
        print(
            canonical_json(
                {
                    "candidate_program": program,
                    "explanation": response.explanation,
                    "metadata": response.metadata,
                },
                indent=2 if args.pretty else None,
            )
        )
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


def _handle_repair_piece(args: argparse.Namespace) -> int:
    if args.repair_attempts < 1:
        raise ValueError("--repair-attempts must be at least 1")

    current_program = load_piece_program(args.current_program)
    before_state = GameState.from_dict(_load_json_object(args.before_state))
    observed_state = GameState.from_dict(_load_json_object(args.observed_state))
    move = Move.from_dict(_load_json_object(args.move))
    mismatch = build_state_mismatch(
        before_state=before_state,
        move=move,
        observed_state=observed_state,
        current_program=current_program,
    )
    provider = (
        JsonFileRepairProvider(args.provider_response)
        if args.provider_response is not None
        else _build_llm_piece_provider(args)
    )
    repair_result = repair_and_verify_piece(
        mismatch,
        description=args.description,
        current_program=current_program,
        provider=provider,
    )

    registry_record = None
    if repair_result.accepted:
        registry_record = append_verified_piece_version(
            registry_path=args.registry,
            out_dir=args.out_dir,
            program=repair_result.patched_program,
            description=args.description,
            source="repair",
            parent_digest=stable_digest(canonicalize_piece_program(current_program)),
            verified_cases=repair_result.verified_cases,
            repair_attempts=args.repair_attempts,
            metadata={
                "current_program": str(args.current_program),
                "provider_response": (
                    str(args.provider_response)
                    if args.provider_response is not None
                    else None
                ),
                "mismatch_diff": mismatch.diff.to_dict(),
            },
        )

    print(
        canonical_json(
            {
                "status": "accepted" if repair_result.accepted else "rejected",
                "accepted": repair_result.accepted,
                "piece_id": canonicalize_piece_program(current_program)["piece_id"],
                "verification_errors": list(repair_result.verification_errors),
                "verified_cases": repair_result.verified_cases,
                "mismatch_diff": mismatch.diff.to_dict(),
                "primary_diff_after_repair": repair_result.primary_diff_after_repair,
                "registry_record": (
                    registry_record.to_dict() if registry_record is not None else None
                ),
            },
            indent=2,
        )
    )
    return 0 if repair_result.accepted else 2


def _handle_piece_curriculum(args: argparse.Namespace) -> int:
    live_provider = None
    if args.compile_response is None or (
        args.repair_response is None and args.llm_repair
    ):
        live_provider = _build_llm_piece_provider(args)
    compile_provider = (
        JsonFileCompileProvider(args.compile_response)
        if args.compile_response is not None
        else live_provider
    )
    repair_provider = None
    if args.repair_response is not None:
        repair_provider = JsonFileRepairProvider(args.repair_response)
    elif args.compile_response is None or args.llm_repair:
        repair_provider = live_provider
    if compile_provider is None:
        raise ValueError("piece-curriculum requires --compile-response or live LLM config")
    result = run_synthetic_piece_curriculum(
        seed=args.seed,
        recipe=args.recipe,
        piece_id=args.piece_id,
        compile_provider=compile_provider,
        repair_provider=repair_provider,
        registry_path=None if args.no_registry_write else str(args.registry),
        out_dir=None if args.no_registry_write else str(args.out_dir),
    )
    if args.artifact_dir is not None:
        _write_curriculum_artifacts(args.artifact_dir, result)
    print(canonical_json(result.to_dict(), indent=2))
    return 0 if result.accepted else 2


def _handle_selfplay(args: argparse.Namespace) -> int:
    if args.games_out is None and args.examples_out is None:
        raise ValueError("selfplay requires --games-out, --examples-out, or both")
    if args.use_verified_pieces and not args.standard_initial_state:
        raise ValueError("--use-verified-pieces requires --standard-initial-state")
    if args.use_verified_pieces and not args.randomize_handauthored_specials:
        raise ValueError("--use-verified-pieces requires --randomize-handauthored-specials")

    verified_records = (
        _load_verified_registry_records(args.piece_registry)
        if args.use_verified_pieces
        else []
    )

    if args.standard_initial_state:
        if args.randomize_handauthored_specials:
            opening_rng = random.Random(args.seed)
            handauthored_piece_classes = _load_piece_classes_from_directory(args.special_piece_dir)
            verified_piece_classes = (
                load_verified_piece_classes(args.piece_registry)
                if args.use_verified_pieces
                else []
            )
            special_piece_classes = _merge_piece_classes(
                handauthored_piece_classes,
                verified_piece_classes,
            )
            initial_states = [
                sample_standard_initial_state(
                    opening_rng,
                    special_piece_classes=special_piece_classes,
                    inclusion_probability=args.special_piece_inclusion_probability,
                )
                for _ in range(args.games)
            ]
        else:
            initial_states = [standard_initial_state()]
    else:
        if args.randomize_handauthored_specials:
            raise ValueError(
                "--randomize-handauthored-specials requires --standard-initial-state"
            )
        initial_states = _load_state_files(args.state_file)
    model = None
    checkpoint_metadata: dict[str, object] = {}
    if args.checkpoint is not None:
        checkpoint = load_training_checkpoint(args.checkpoint, device=args.device)
        model = checkpoint.model
        checkpoint_metadata = {
            "checkpoint_path": str(args.checkpoint),
            "checkpoint_metadata": dict(checkpoint.metadata),
        }

    config = SelfPlayConfig(
        simulations=args.simulations,
        max_plies=args.max_plies,
        opening_temperature=args.opening_temperature,
        final_temperature=args.final_temperature,
        temperature_drop_after_ply=args.temperature_drop_after_ply,
        c_puct=args.c_puct,
        root_dirichlet_alpha=args.root_dirichlet_alpha,
        root_exploration_fraction=args.root_exploration_fraction,
        seed=args.seed,
        adjudicate_max_plies=args.adjudicate_max_plies,
        adjudication_threshold=args.adjudication_threshold,
    )
    games = generate_selfplay_games(
        initial_states,
        games=args.games,
        config=config,
        policy_value_model=model,
        progress_callback=(
            None
            if args.quiet
            else _build_selfplay_progress_logger(log_every_plies=args.log_every_plies)
        ),
    )
    _annotate_games_with_registry(
        games,
        registry_path=args.piece_registry,
        verified_records=verified_records,
    )
    examples = flatten_selfplay_examples(games)

    if args.games_out is not None:
        write_selfplay_games_jsonl(args.games_out, games)
    if args.examples_out is not None:
        write_selfplay_examples_jsonl(args.examples_out, examples)

    print(
        canonical_json(
            {
                "status": "ok",
                "games_generated": len(games),
                "examples_generated": len(examples),
                "used_model": args.checkpoint is not None,
                "games_out": str(args.games_out) if args.games_out is not None else None,
                "examples_out": (
                    str(args.examples_out) if args.examples_out is not None else None
                ),
                "outcomes": _outcome_counts(games),
                "seed_state_count": len(initial_states),
                "randomized_special_pieces": args.randomize_handauthored_specials,
                "special_piece_dir": (
                    str(args.special_piece_dir)
                    if args.randomize_handauthored_specials
                    else None
                ),
                "special_piece_inclusion_probability": (
                    args.special_piece_inclusion_probability
                    if args.randomize_handauthored_specials
                    else None
                ),
                "used_verified_pieces": args.use_verified_pieces,
                "piece_registry": (
                    str(args.piece_registry) if args.use_verified_pieces else None
                ),
                "verified_piece_count": len(verified_records),
                "verified_piece_digests": registry_piece_digest_metadata(verified_records),
                "config": {
                    "simulations": config.simulations,
                    "max_plies": config.max_plies,
                    "opening_temperature": config.opening_temperature,
                    "final_temperature": config.final_temperature,
                    "temperature_drop_after_ply": config.temperature_drop_after_ply,
                    "c_puct": config.c_puct,
                    "root_dirichlet_alpha": config.root_dirichlet_alpha,
                    "root_exploration_fraction": config.root_exploration_fraction,
                    "seed": config.seed,
                    "adjudicate_max_plies": config.adjudicate_max_plies,
                    "adjudication_threshold": config.adjudication_threshold,
                },
                **checkpoint_metadata,
            },
            indent=2,
        )
    )
    return 0


def _handle_train(args: argparse.Namespace) -> int:
    examples = read_selfplay_examples_jsonl(args.examples)
    resume_checkpoint = None
    model = None
    optimizer_state_dict = None
    if args.resume_checkpoint is not None:
        _validate_resume_flags(args)
        resume_checkpoint = load_training_checkpoint(args.resume_checkpoint, device=args.device)
        model = resume_checkpoint.model
        optimizer_state_dict = resume_checkpoint.optimizer_state_dict
        model_config = resume_checkpoint.model_config
    else:
        model_config = PolicyValueConfig(
            d_model=args.d_model if args.d_model is not None else 192,
            num_heads=args.num_heads if args.num_heads is not None else 8,
            num_layers=args.num_layers if args.num_layers is not None else 4,
            dropout=args.dropout if args.dropout is not None else 0.1,
            feedforward_multiplier=(
                args.feedforward_multiplier
                if args.feedforward_multiplier is not None
                else 4
            ),
        )

    training_config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        policy_weight=args.policy_weight,
        value_weight=args.value_weight,
        device=args.device,
        shuffle=args.shuffle,
        seed=args.seed,
        model_config=model_config,
    )
    run_result = train_from_replays(
        examples,
        model=model,
        optimizer_state_dict=optimizer_state_dict,
        config=training_config,
        progress_callback=(
            None
            if args.quiet
            else _build_training_progress_logger(log_every_batches=args.log_every_batches)
        ),
    )
    save_training_checkpoint(
        args.checkpoint_out,
        model=run_result.model,
        training_config=training_config,
        training_metrics=run_result.metrics,
        optimizer_state_dict=run_result.optimizer_state_dict,
        metadata={
            "examples_path": str(args.examples),
            "example_count": len(examples),
            "resumed_from": (
                str(args.resume_checkpoint)
                if args.resume_checkpoint is not None
                else None
            ),
        },
    )
    print(
        canonical_json(
            {
                "status": "ok",
                "checkpoint_out": str(args.checkpoint_out),
                "examples_path": str(args.examples),
                "examples_loaded": len(examples),
                "resumed_from": (
                    str(args.resume_checkpoint)
                    if args.resume_checkpoint is not None
                    else None
                ),
                "training_metrics": {
                    "epochs_completed": run_result.metrics.epochs_completed,
                    "examples_seen": run_result.metrics.examples_seen,
                    "batches_completed": run_result.metrics.batches_completed,
                    "average_policy_loss": run_result.metrics.average_policy_loss,
                    "average_value_loss": run_result.metrics.average_value_loss,
                    "average_total_loss": run_result.metrics.average_total_loss,
                    "device": run_result.metrics.device,
                },
                "model_config": {
                    "d_model": model_config.d_model,
                    "num_heads": model_config.num_heads,
                    "num_layers": model_config.num_layers,
                    "dropout": model_config.dropout,
                    "feedforward_multiplier": model_config.feedforward_multiplier,
                },
            },
            indent=2,
        )
    )
    return 0


def _handle_eval_model(args: argparse.Namespace) -> int:
    checkpoint = load_training_checkpoint(args.checkpoint, device=args.device)
    examples = read_selfplay_examples_jsonl(args.examples)
    metrics = evaluate_policy_value_model(
        model=checkpoint.model,
        examples=examples,
        progress_callback=(
            None
            if args.quiet
            else _build_model_eval_progress_logger(
                log_every_examples=args.log_every_examples
            )
        ),
    )
    print(
        canonical_json(
            {
                "status": "ok",
                "checkpoint": str(args.checkpoint),
                "examples_path": str(args.examples),
                "examples_loaded": len(examples),
                "metrics": metrics.to_dict(),
                "model_config": {
                    "d_model": checkpoint.model_config.d_model,
                    "num_heads": checkpoint.model_config.num_heads,
                    "num_layers": checkpoint.model_config.num_layers,
                    "dropout": checkpoint.model_config.dropout,
                    "feedforward_multiplier": checkpoint.model_config.feedforward_multiplier,
                },
            },
            indent=2,
        )
    )
    return 0


def _handle_train_loop(args: argparse.Namespace) -> int:
    if args.cycles < 1:
        raise ValueError("--cycles must be at least 1")
    if args.train_games < 1:
        raise ValueError("--train-games must be at least 1")
    if args.val_games < 1:
        raise ValueError("--val-games must be at least 1")
    if args.use_verified_pieces and not args.randomize_handauthored_specials:
        raise ValueError("--use-verified-pieces requires randomized special-piece openings")

    data_dir = args.output_dir / "selfplay"
    checkpoint_dir = args.output_dir / "checkpoints"
    metrics_dir = args.output_dir / "metrics"
    data_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    verified_records = (
        _load_verified_registry_records(args.piece_registry)
        if args.use_verified_pieces
        else []
    )
    handauthored_piece_classes = (
        _load_piece_classes_from_directory(args.special_piece_dir)
        if args.randomize_handauthored_specials
        else ()
    )
    verified_piece_classes = (
        load_verified_piece_classes(args.piece_registry)
        if args.use_verified_pieces
        else ()
    )
    special_piece_classes = _merge_piece_classes(
        handauthored_piece_classes,
        verified_piece_classes,
    )
    model = None
    optimizer_state_dict = None
    model_config = PolicyValueConfig(
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        feedforward_multiplier=args.feedforward_multiplier,
    )
    previous_checkpoint_path: Path | None = None
    if args.resume_checkpoint is not None:
        checkpoint = load_training_checkpoint(args.resume_checkpoint, device=args.device)
        model = checkpoint.model
        optimizer_state_dict = checkpoint.optimizer_state_dict
        model_config = checkpoint.model_config
        previous_checkpoint_path = args.resume_checkpoint

    cycle_summaries: list[dict[str, object]] = []
    for cycle_index in range(1, args.cycles + 1):
        if not args.quiet:
            _stderr_print(f"=== train-loop cycle {cycle_index}/{args.cycles} ===")
        train_seed = args.seed + cycle_index * 1000
        val_seed = args.seed + cycle_index * 1000 + 999

        selfplay_config = SelfPlayConfig(
            simulations=args.simulations,
            max_plies=args.max_plies,
            opening_temperature=args.opening_temperature,
            final_temperature=args.final_temperature,
            temperature_drop_after_ply=args.temperature_drop_after_ply,
            c_puct=args.c_puct,
            root_dirichlet_alpha=args.root_dirichlet_alpha,
            root_exploration_fraction=args.root_exploration_fraction,
            seed=train_seed,
            adjudicate_max_plies=args.adjudicate_max_plies,
            adjudication_threshold=args.adjudication_threshold,
        )
        train_states = _sample_loop_initial_states(
            games=args.train_games,
            seed=train_seed,
            randomize_specials=args.randomize_handauthored_specials,
            special_piece_classes=special_piece_classes,
            inclusion_probability=args.special_piece_inclusion_probability,
        )
        train_games = generate_selfplay_games(
            train_states,
            games=args.train_games,
            config=selfplay_config,
            policy_value_model=None if args.no_guided_selfplay else model,
            progress_callback=(
                None
                if args.quiet
                else _build_selfplay_progress_logger(log_every_plies=args.log_every_plies)
            ),
        )
        _annotate_games_with_registry(
            train_games,
            registry_path=args.piece_registry,
            verified_records=verified_records,
        )
        train_examples = flatten_selfplay_examples(train_games)
        train_games_path = data_dir / f"cycle_{cycle_index:03d}_train_games.jsonl"
        train_examples_path = data_dir / f"cycle_{cycle_index:03d}_train_examples.jsonl"
        write_selfplay_games_jsonl(train_games_path, train_games)
        write_selfplay_examples_jsonl(train_examples_path, train_examples)

        training_config = TrainingConfig(
            epochs=args.epochs_per_cycle,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            policy_weight=args.policy_weight,
            value_weight=args.value_weight,
            device=args.device,
            shuffle=True,
            seed=train_seed,
            model_config=model_config,
        )
        training_run = train_from_replays(
            train_examples,
            model=model,
            optimizer_state_dict=optimizer_state_dict,
            config=training_config,
            progress_callback=(
                None
                if args.quiet
                else _build_training_progress_logger(
                    log_every_batches=args.log_every_batches
                )
            ),
        )
        model = training_run.model
        optimizer_state_dict = training_run.optimizer_state_dict
        checkpoint_path = checkpoint_dir / f"model_{cycle_index:03d}.pt"
        save_training_checkpoint(
            checkpoint_path,
            model=model,
            training_config=training_config,
            training_metrics=training_run.metrics,
            optimizer_state_dict=optimizer_state_dict,
            metadata={
                "cycle": cycle_index,
                "train_examples_path": str(train_examples_path),
                "previous_checkpoint": (
                    str(previous_checkpoint_path)
                    if previous_checkpoint_path is not None
                    else None
                ),
            },
        )
        previous_checkpoint_path = checkpoint_path

        val_config = SelfPlayConfig(
            simulations=args.simulations,
            max_plies=args.max_plies,
            opening_temperature=args.opening_temperature,
            final_temperature=args.final_temperature,
            temperature_drop_after_ply=args.temperature_drop_after_ply,
            c_puct=args.c_puct,
            root_dirichlet_alpha=args.root_dirichlet_alpha,
            root_exploration_fraction=args.root_exploration_fraction,
            seed=val_seed,
            adjudicate_max_plies=args.adjudicate_max_plies,
            adjudication_threshold=args.adjudication_threshold,
        )
        val_states = _sample_loop_initial_states(
            games=args.val_games,
            seed=val_seed,
            randomize_specials=args.randomize_handauthored_specials,
            special_piece_classes=special_piece_classes,
            inclusion_probability=args.special_piece_inclusion_probability,
        )
        val_games = generate_selfplay_games(
            val_states,
            games=args.val_games,
            config=val_config,
            policy_value_model=None if args.no_guided_selfplay else model,
            progress_callback=(
                None
                if args.quiet
                else _build_selfplay_progress_logger(log_every_plies=args.log_every_plies)
            ),
        )
        _annotate_games_with_registry(
            val_games,
            registry_path=args.piece_registry,
            verified_records=verified_records,
        )
        val_examples = flatten_selfplay_examples(val_games)
        val_games_path = data_dir / f"cycle_{cycle_index:03d}_val_games.jsonl"
        val_examples_path = data_dir / f"cycle_{cycle_index:03d}_val_examples.jsonl"
        write_selfplay_games_jsonl(val_games_path, val_games)
        write_selfplay_examples_jsonl(val_examples_path, val_examples)

        train_eval_metrics = evaluate_policy_value_model(
            model=model,
            examples=train_examples,
            progress_callback=(
                None
                if args.quiet
                else _build_model_eval_progress_logger(
                    log_every_examples=args.log_every_examples
                )
            ),
        )
        val_eval_metrics = evaluate_policy_value_model(
            model=model,
            examples=val_examples,
            progress_callback=(
                None
                if args.quiet
                else _build_model_eval_progress_logger(
                    log_every_examples=args.log_every_examples
                )
            ),
        )
        cycle_summary = {
            "cycle": cycle_index,
            "checkpoint": str(checkpoint_path),
            "train_games": len(train_games),
            "train_examples": len(train_examples),
            "train_games_path": str(train_games_path),
            "train_examples_path": str(train_examples_path),
            "val_games": len(val_games),
            "val_examples": len(val_examples),
            "val_games_path": str(val_games_path),
            "val_examples_path": str(val_examples_path),
            "training_metrics": _training_metrics_dict(training_run.metrics),
            "train_eval_metrics": train_eval_metrics.to_dict(),
            "val_eval_metrics": val_eval_metrics.to_dict(),
            "root_dirichlet_alpha": selfplay_config.root_dirichlet_alpha,
            "root_exploration_fraction": selfplay_config.root_exploration_fraction,
            "used_verified_pieces": args.use_verified_pieces,
            "piece_registry": (
                str(args.piece_registry) if args.use_verified_pieces else None
            ),
            "verified_piece_count": len(verified_records),
            "verified_piece_digests": registry_piece_digest_metadata(verified_records),
        }
        cycle_summaries.append(cycle_summary)
        metrics_path = metrics_dir / f"cycle_{cycle_index:03d}.json"
        metrics_path.write_text(canonical_json(cycle_summary, indent=2), encoding="utf-8")
        if not args.quiet:
            _stderr_print(
                "cycle "
                f"{cycle_index} summary: "
                f"train_ce={train_eval_metrics.average_policy_cross_entropy:.4f} "
                f"val_ce={val_eval_metrics.average_policy_cross_entropy:.4f} "
                f"train_top1={train_eval_metrics.top1_agreement:.3f} "
                f"val_top1={val_eval_metrics.top1_agreement:.3f}"
            )

    summary = {
        "status": "ok",
        "output_dir": str(args.output_dir),
        "cycles": cycle_summaries,
        "latest_checkpoint": cycle_summaries[-1]["checkpoint"],
    }
    (args.output_dir / "summary.json").write_text(
        canonical_json(summary, indent=2),
        encoding="utf-8",
    )
    print(canonical_json(summary, indent=2))
    return 0


def _handle_placeholder(args: argparse.Namespace) -> int:
    print(
        f"{args.command} is part of a later milestone and is not implemented in the foundation build.",
        file=sys.stderr,
    )
    return 2


def _load_state_files(paths: list[Path]) -> list[GameState]:
    states: list[GameState] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            states.extend(GameState.from_dict(dict(item)) for item in payload)
        elif isinstance(payload, dict):
            states.append(GameState.from_dict(dict(payload)))
        else:
            raise ValueError(f"State file {path!s} must contain an object or list of objects")
    return states


def _load_json_object(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path!s} must contain a JSON object")
    return dict(payload)


def _load_piece_classes_from_directory(directory: Path) -> list[PieceClass]:
    if not directory.exists():
        raise ValueError(f"Special-piece directory does not exist: {directory!s}")
    piece_files = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in {".json", ".yaml", ".yml"}
    )
    if not piece_files:
        raise ValueError(f"No piece files found in {directory!s}")
    return [compile_piece_file(path) for path in piece_files]


def _load_verified_registry_records(registry_path: Path):
    if not registry_path.exists():
        raise ValueError(f"Piece registry does not exist: {registry_path!s}")
    records = load_verified_piece_records(registry_path)
    if not records:
        raise ValueError(f"Piece registry has no verified records: {registry_path!s}")
    return records


def _merge_piece_classes(*groups) -> list[PieceClass]:
    merged: dict[str, PieceClass] = {}
    for group in groups:
        for piece_class in group:
            merged[piece_class.class_id] = piece_class
    return [
        merged[class_id]
        for class_id in sorted(merged)
    ]


def _annotate_games_with_registry(
    games,
    *,
    registry_path: Path,
    verified_records,
) -> None:
    if not verified_records:
        return
    metadata = {
        "piece_registry": str(registry_path),
        "verified_piece_digests": registry_piece_digest_metadata(verified_records),
    }
    for game in games:
        game.metadata.update(metadata)
        game.replay_trace.metadata.update(metadata)


def _write_curriculum_artifacts(artifact_dir: Path, result) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "teacher_program.json").write_text(
        canonical_json(result.synthetic_piece.teacher_program, indent=2),
        encoding="utf-8",
    )
    (artifact_dir / "description.txt").write_text(
        result.synthetic_piece.description,
        encoding="utf-8",
    )
    (artifact_dir / "summary.json").write_text(
        canonical_json(result.to_dict(), indent=2),
        encoding="utf-8",
    )
    for index, probe_result in enumerate(result.probe_results, start=1):
        mismatch = probe_result.mismatch
        if mismatch is None:
            continue
        (artifact_dir / f"probe_{index:03d}_{probe_result.label}.json").write_text(
            canonical_json(mismatch, indent=2),
            encoding="utf-8",
        )


def _sample_loop_initial_states(
    *,
    games: int,
    seed: int,
    randomize_specials: bool,
    special_piece_classes: tuple[PieceClass, ...] | list[PieceClass],
    inclusion_probability: float,
) -> list[GameState]:
    rng = random.Random(seed)
    if not randomize_specials:
        return [standard_initial_state() for _ in range(games)]
    return [
        sample_standard_initial_state(
            rng,
            special_piece_classes=special_piece_classes,
            inclusion_probability=inclusion_probability,
        )
        for _ in range(games)
    ]


def _training_metrics_dict(metrics) -> dict[str, object]:
    return {
        "epochs_completed": metrics.epochs_completed,
        "examples_seen": metrics.examples_seen,
        "batches_completed": metrics.batches_completed,
        "average_policy_loss": metrics.average_policy_loss,
        "average_value_loss": metrics.average_value_loss,
        "average_total_loss": metrics.average_total_loss,
        "device": metrics.device,
    }


def _outcome_counts(games: list[object]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for game in games:
        outcome = str(game.outcome)
        counts[outcome] = counts.get(outcome, 0) + 1
    return dict(sorted(counts.items()))


def _validate_resume_flags(args: argparse.Namespace) -> None:
    for field_name in (
        "d_model",
        "num_heads",
        "num_layers",
        "dropout",
        "feedforward_multiplier",
    ):
        if getattr(args, field_name) is not None:
            raise ValueError(
                f"--{field_name.replace('_', '-')} cannot be used with --resume-checkpoint"
            )


def _build_selfplay_progress_logger(
    *,
    log_every_plies: int,
):
    start_time = time.monotonic()

    def callback(progress: SelfPlayProgress) -> None:
        elapsed = time.monotonic() - start_time
        if progress.event == "game_started":
            _stderr_print(
                f"[{elapsed:7.1f}s] selfplay game {progress.game_index + 1}/{progress.games_total} started"
                f" model={progress.used_model}"
            )
            return
        if progress.event == "ply_completed":
            if log_every_plies > 0 and progress.ply is not None and progress.ply % log_every_plies == 0:
                _stderr_print(
                    f"[{elapsed:7.1f}s] selfplay game {progress.game_index + 1}/{progress.games_total}"
                    f" ply {progress.ply}"
                    f" legal={progress.legal_move_count}"
                    f" move={progress.selected_move_id}"
                )
            return
        if progress.event == "game_completed":
            _stderr_print(
                f"[{elapsed:7.1f}s] selfplay game {progress.game_index + 1}/{progress.games_total} completed"
                f" plies={progress.plies_played}"
                f" outcome={progress.outcome}"
                f" reason={progress.termination_reason}"
            )

    return callback


def _build_training_progress_logger(
    *,
    log_every_batches: int,
):
    start_time = time.monotonic()

    def callback(progress: TrainingProgress) -> None:
        elapsed = time.monotonic() - start_time
        if progress.event == "training_started":
            _stderr_print(
                f"[{elapsed:7.1f}s] train started"
                f" epochs={progress.epochs_total}"
                f" batches_per_epoch={progress.batches_total}"
                f" device={progress.device}"
            )
            return
        if progress.event == "epoch_started":
            _stderr_print(
                f"[{elapsed:7.1f}s] train epoch {progress.epoch}/{progress.epochs_total} started"
            )
            return
        if progress.event == "batch_completed":
            if log_every_batches > 0 and progress.batch_index is not None and progress.batch_index % log_every_batches == 0:
                _stderr_print(
                    f"[{elapsed:7.1f}s] train epoch {progress.epoch}/{progress.epochs_total}"
                    f" batch {progress.batch_index}/{progress.batches_total}"
                    f" seen={progress.examples_seen}"
                    f" loss={progress.total_loss:.4f}"
                    f" policy={progress.policy_loss:.4f}"
                    f" value={progress.value_loss:.4f}"
                )
            return
        if progress.event == "epoch_completed":
            _stderr_print(
                f"[{elapsed:7.1f}s] train epoch {progress.epoch}/{progress.epochs_total} completed"
                f" seen={progress.examples_seen}"
            )
            return
        if progress.event == "training_completed":
            _stderr_print(
                f"[{elapsed:7.1f}s] train completed"
                f" seen={progress.examples_seen}"
                f" avg_loss={progress.total_loss:.4f}"
                f" avg_policy={progress.policy_loss:.4f}"
                f" avg_value={progress.value_loss:.4f}"
            )

    return callback


def _build_model_eval_progress_logger(
    *,
    log_every_examples: int,
):
    start_time = time.monotonic()

    def callback(progress: ModelEvalProgress) -> None:
        elapsed = time.monotonic() - start_time
        if progress.event == "eval_started":
            _stderr_print(
                f"[{elapsed:7.1f}s] eval-model started examples={progress.examples_total}"
            )
            return
        if progress.event == "example_evaluated":
            if (
                log_every_examples > 0
                and progress.examples_seen % log_every_examples == 0
            ):
                _stderr_print(
                    f"[{elapsed:7.1f}s] eval-model examples"
                    f" {progress.examples_seen}/{progress.examples_total}"
                )
            return
        if progress.event == "eval_completed":
            _stderr_print(
                f"[{elapsed:7.1f}s] eval-model completed examples={progress.examples_total}"
            )

    return callback


def _stderr_print(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


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
