from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

from pixie_solver.core import GameState, sample_standard_initial_state, standard_initial_state
from pixie_solver.curriculum import generate_teacher_piece
from pixie_solver.dsl import compile_piece_program
from pixie_solver.training import (
    SelfPlayConfig,
    benchmark_metadata_for_state,
    flatten_selfplay_examples,
    generate_selfplay_games,
    summarize_benchmark_metadata_records,
    write_selfplay_examples_jsonl,
    write_selfplay_games_jsonl,
)
from pixie_solver.utils.serialization import JsonValue, canonical_json


@dataclass(frozen=True, slots=True)
class BenchmarkCorpusProgress:
    event: str
    suite_id: str | None = None
    suite_index: int | None = None
    suites_total: int | None = None
    games: int | None = None


@dataclass(frozen=True, slots=True)
class BenchmarkWorldSpec:
    world_id: str
    family_id: str
    split: str
    novelty_tier: str
    admission_cycle: int | None
    recipes: tuple[str, ...] = ()
    piece_seeds: tuple[int, ...] = ()
    placement_seed: int = 0
    tags: tuple[str, ...] = ()
    world_family_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class BenchmarkSuiteSpec:
    suite_id: str
    description: str
    tags: tuple[str, ...]
    worlds: tuple[BenchmarkWorldSpec, ...]


@dataclass(frozen=True, slots=True)
class BenchmarkCorpusConfig:
    output_dir: Path
    manifest_name: str = "phase0_serious_run_v0"
    manifest_description: str = (
        "Deterministic PixieChess benchmark corpus for serious adaptation runs."
    )
    games_per_world: int = 4
    simulations: int = 16
    max_plies: int = 48
    seed_offset: int = 0
    opening_temperature: float = 0.0
    final_temperature: float = 0.0
    temperature_drop_after_ply: int = 0
    c_puct: float = 1.5
    adjudicate_max_plies: bool = True
    adjudication_threshold: float = 0.2


def build_benchmark_corpus(
    *,
    config: BenchmarkCorpusConfig,
    progress_callback: Callable[[BenchmarkCorpusProgress], None] | None = None,
) -> dict[str, JsonValue]:
    if config.games_per_world < 1:
        raise ValueError("games_per_world must be at least 1")
    if config.simulations < 1:
        raise ValueError("simulations must be at least 1")
    if config.max_plies < 1:
        raise ValueError("max_plies must be at least 1")

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    suite_specs = _default_suite_specs(seed_offset=config.seed_offset)
    manifest_suites: list[dict[str, JsonValue]] = []
    suite_summaries: list[dict[str, JsonValue]] = []

    base_selfplay_config = SelfPlayConfig(
        simulations=config.simulations,
        max_plies=config.max_plies,
        opening_temperature=config.opening_temperature,
        final_temperature=config.final_temperature,
        temperature_drop_after_ply=config.temperature_drop_after_ply,
        c_puct=config.c_puct,
        root_exploration_fraction=0.0,
        seed=config.seed_offset,
        adjudicate_max_plies=config.adjudicate_max_plies,
        adjudication_threshold=config.adjudication_threshold,
    )

    for suite_index, suite_spec in enumerate(suite_specs, start=1):
        total_games = len(suite_spec.worlds) * config.games_per_world
        if progress_callback is not None:
            progress_callback(
                BenchmarkCorpusProgress(
                    event="suite_started",
                    suite_id=suite_spec.suite_id,
                    suite_index=suite_index,
                    suites_total=len(suite_specs),
                    games=total_games,
                )
            )

        games = _generate_suite_games(
            suite_spec,
            suite_index=suite_index,
            games_per_world=config.games_per_world,
            base_selfplay_config=base_selfplay_config,
            seed_offset=config.seed_offset,
        )
        examples = flatten_selfplay_examples(games)

        games_filename = f"{suite_spec.suite_id}_games.jsonl"
        examples_filename = f"{suite_spec.suite_id}_examples.jsonl"
        games_path = output_dir / games_filename
        examples_path = output_dir / examples_filename
        write_selfplay_games_jsonl(games_path, games)
        write_selfplay_examples_jsonl(examples_path, examples)

        suite_summary = {
            "suite_id": suite_spec.suite_id,
            "description": suite_spec.description,
            "tags": list(suite_spec.tags),
            "games": len(games),
            "examples": len(examples),
            "worlds": len(suite_spec.worlds),
            "benchmark_metadata_summary": summarize_benchmark_metadata_records(
                example.metadata for example in examples
            ),
            "games_path": str(games_path),
            "examples_path": str(examples_path),
        }
        suite_summaries.append(suite_summary)
        manifest_suites.append(
            {
                "suite_id": suite_spec.suite_id,
                "description": suite_spec.description,
                "games": games_filename,
                "examples": examples_filename,
                "tags": list(suite_spec.tags),
            }
        )

        if progress_callback is not None:
            progress_callback(
                BenchmarkCorpusProgress(
                    event="suite_completed",
                    suite_id=suite_spec.suite_id,
                    suite_index=suite_index,
                    suites_total=len(suite_specs),
                    games=len(games),
                )
            )

    manifest = {
        "format_version": 1,
        "name": config.manifest_name,
        "description": config.manifest_description,
        "suites": manifest_suites,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(canonical_json(manifest, indent=2), encoding="utf-8")

    summary = {
        "status": "ok",
        "output_dir": str(output_dir),
        "manifest_path": str(manifest_path),
        "manifest_name": config.manifest_name,
        "games_per_world": config.games_per_world,
        "simulations": config.simulations,
        "max_plies": config.max_plies,
        "suite_count": len(suite_summaries),
        "suites": suite_summaries,
    }
    (output_dir / "summary.json").write_text(
        canonical_json(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def _generate_suite_games(
    suite_spec: BenchmarkSuiteSpec,
    *,
    suite_index: int,
    games_per_world: int,
    base_selfplay_config: SelfPlayConfig,
    seed_offset: int,
) -> list:
    games = []
    suite_game_index = 0
    for world_index, world_spec in enumerate(suite_spec.worlds):
        state = _build_world_state(world_spec)
        strategy = _benchmark_strategy_for_world(world_spec)
        world_config = replace(
            base_selfplay_config,
            seed=seed_offset + suite_index * 10_000 + world_index * 1_000,
            strategy=strategy,
        )
        world_games = generate_selfplay_games(
            [state],
            games=games_per_world,
            config=world_config,
            policy_value_model=None,
        )
        per_game_metadata = []
        for local_game_index in range(games_per_world):
            per_game_metadata.append(
                {
                    "family_id": world_spec.family_id,
                    "split": world_spec.split,
                    "novelty_tier": world_spec.novelty_tier,
                    "admission_cycle": world_spec.admission_cycle,
                    "benchmark_suite_id": suite_spec.suite_id,
                    "benchmark_world_id": world_spec.world_id,
                    "benchmark_world_index": world_index,
                    "benchmark_suite_game_index": suite_game_index + local_game_index,
                    "benchmark_world_game_index": local_game_index,
                    "benchmark_world_recipes": list(world_spec.recipes),
                    "benchmark_world_piece_seeds": list(world_spec.piece_seeds),
                    "benchmark_world_tags": list(world_spec.tags),
                    "benchmark_strategy_id": strategy["strategy_id"],
                    "benchmark_strategy_source": "deterministic_world_strategy",
                    "strategy": dict(strategy),
                    "world_family_ids": list(world_spec.world_family_ids),
                }
            )
        _annotate_suite_games(
            world_games,
            suite_spec=suite_spec,
            per_game_metadata=per_game_metadata,
        )
        games.extend(world_games)
        suite_game_index += len(world_games)
    return games


def _build_world_state(world_spec: BenchmarkWorldSpec) -> GameState:
    if not world_spec.recipes:
        return standard_initial_state()
    special_piece_classes = []
    for recipe_index, (recipe, seed) in enumerate(
        zip(world_spec.recipes, world_spec.piece_seeds, strict=True)
    ):
        synthetic_piece = generate_teacher_piece(
            seed=seed,
            recipe=recipe,
            piece_id=f"{world_spec.world_id}_{recipe_index}_{recipe}_{seed}",
        )
        special_piece_classes.append(compile_piece_program(synthetic_piece.teacher_program))
    return sample_standard_initial_state(
        random.Random(world_spec.placement_seed),
        special_piece_classes=tuple(special_piece_classes),
        inclusion_probability=1.0,
    )


def _annotate_suite_games(
    games,
    *,
    suite_spec: BenchmarkSuiteSpec,
    per_game_metadata: Sequence[dict[str, JsonValue]],
) -> None:
    for game, extra_metadata in zip(games, per_game_metadata, strict=True):
        shared_metadata = {
            **extra_metadata,
            "suite_tags": list(suite_spec.tags),
        }
        game.metadata.update(shared_metadata)
        game.metadata.update(
            benchmark_metadata_for_state(
                game.replay_trace.initial_state,
                game.metadata,
                search_budget=game.metadata.get("search_budget"),
                model_architecture=game.metadata.get("model_architecture"),
            )
        )
        game.replay_trace.metadata.update(shared_metadata)
        game.replay_trace.metadata.update(
            benchmark_metadata_for_state(
                game.replay_trace.initial_state,
                game.replay_trace.metadata,
                search_budget=game.replay_trace.metadata.get("search_budget"),
                model_architecture=game.replay_trace.metadata.get("model_architecture"),
            )
        )
        for example in game.examples:
            example.metadata.update(shared_metadata)
            example.metadata.update(
                benchmark_metadata_for_state(
                    example.state,
                    example.metadata,
                    search_budget=example.metadata.get("search_budget"),
                    model_architecture=example.metadata.get("model_architecture"),
                )
            )


def _benchmark_strategy_for_world(
    world_spec: BenchmarkWorldSpec,
) -> dict[str, JsonValue]:
    family_ids = (
        tuple(world_spec.world_family_ids)
        if world_spec.world_family_ids
        else tuple(world_spec.recipes)
    )
    if world_spec.family_id == "foundation":
        return {
            "strategy_id": "foundation_development",
            "summary": "Develop orthodox pieces, secure the king, and improve central control before forcing tactics.",
            "confidence": 0.72,
            "scope": "game_start",
            "subgoals": [
                "complete development",
                "protect king safety",
                "contest the center",
            ],
            "action_biases": [
                "develop toward active squares",
                "connect major pieces",
            ],
            "avoid_biases": [
                "premature queen sorties",
                "unforced king exposure",
            ],
            "metadata": {
                "family_id": world_spec.family_id,
                "benchmark_world_id": world_spec.world_id,
            },
        }
    if world_spec.family_id == "composition":
        family_summary = " + ".join(family_ids)
        return {
            "strategy_id": f"compose_{'_'.join(family_ids)}",
            "summary": (
                f"Exploit the interaction between {family_summary} while preserving king safety "
                "and converting tactical windows quickly."
            ),
            "confidence": 0.78,
            "scope": "game_start",
            "subgoals": [
                "activate interacting mechanics",
                "force favorable tactical sequences",
                "avoid coordination breakdowns",
            ],
            "action_biases": [
                f"sequence {family_ids[0]} pressure into {family_ids[-1]} follow-through",
                "prioritize multi-piece coordination",
            ],
            "avoid_biases": [
                "isolated one-piece attacks",
                "tempo loss before interaction triggers",
            ],
            "metadata": {
                "family_id": world_spec.family_id,
                "world_family_ids": list(family_ids),
                "benchmark_world_id": world_spec.world_id,
            },
        }
    if world_spec.family_id == "capture_sprint":
        summary = "Trade into immediate follow-up captures and keep initiative after every material swing."
        action_biases = [
            "force capture races on favorable terms",
            "keep pieces ready for consecutive captures",
        ]
    elif world_spec.family_id == "phase_rook":
        summary = "Use phasing lanes to pressure blocked files and convert access into direct rook activity."
        action_biases = [
            "open phasing lanes",
            "occupy contested files through blockers",
        ]
    elif world_spec.family_id == "turn_charge":
        summary = "Build charge tempo deliberately, then convert stored momentum into forcing attacks."
        action_biases = [
            "prepare charge turns safely",
            "spend stored momentum on forcing lines",
        ]
    elif world_spec.family_id == "edge_sumo":
        summary = "Dominate edge files, create displacement threats, and convert spatial control into material wins."
        action_biases = [
            "press along the board edge",
            "use displacement threats to win tempo",
        ]
    else:
        summary = "Probe the new mechanic carefully, keep the king safe, and cash in concrete tactical edges."
        action_biases = [
            "test the mechanic in forcing lines",
            "prefer moves with reversible risk",
        ]
    return {
        "strategy_id": f"{world_spec.family_id}_benchmark_plan",
        "summary": summary,
        "confidence": 0.75,
        "scope": "game_start",
        "subgoals": [
            "identify the mechanic's strongest tactical window",
            "convert initiative without overextending",
        ],
        "action_biases": action_biases,
        "avoid_biases": [
            "drifting without activating the signature mechanic",
            "king exposure before initiative is secured",
        ],
        "metadata": {
            "family_id": world_spec.family_id,
            "benchmark_world_id": world_spec.world_id,
            "novelty_tier": world_spec.novelty_tier,
        },
    }


def _default_suite_specs(*, seed_offset: int) -> tuple[BenchmarkSuiteSpec, ...]:
    return (
        BenchmarkSuiteSpec(
            suite_id="foundation",
            description="Foundation-world regression suite.",
            tags=("foundation", "strategy_conditioned"),
            worlds=(
                BenchmarkWorldSpec(
                    world_id="foundation_standard",
                    family_id="foundation",
                    split="foundation",
                    novelty_tier="foundation",
                    admission_cycle=None,
                    tags=("standard",),
                ),
            ),
        ),
        BenchmarkSuiteSpec(
            suite_id="known_mechanic",
            description="Seen mechanic families under stable admitted semantics.",
            tags=("known_mechanic", "seen_family", "strategy_conditioned"),
            worlds=(
                _single_piece_world(
                    world_id="known_capture_sprint",
                    recipe="capture_sprint",
                    piece_seed=101 + seed_offset,
                    placement_seed=1001 + seed_offset,
                    split="train",
                    novelty_tier="introduced",
                    admission_cycle=1,
                ),
                _single_piece_world(
                    world_id="known_phase_rook",
                    recipe="phase_rook",
                    piece_seed=202 + seed_offset,
                    placement_seed=1002 + seed_offset,
                    split="train",
                    novelty_tier="introduced",
                    admission_cycle=2,
                ),
                _single_piece_world(
                    world_id="known_turn_charge",
                    recipe="turn_charge",
                    piece_seed=303 + seed_offset,
                    placement_seed=1003 + seed_offset,
                    split="train",
                    novelty_tier="introduced",
                    admission_cycle=3,
                ),
            ),
        ),
        BenchmarkSuiteSpec(
            suite_id="recent_admission",
            description="Recently admitted worlds that should measure fast adaptation.",
            tags=("recent_admission", "seen_family", "strategy_conditioned"),
            worlds=(
                _single_piece_world(
                    world_id="recent_capture_sprint",
                    recipe="capture_sprint",
                    piece_seed=1101 + seed_offset,
                    placement_seed=2001 + seed_offset,
                    split="train",
                    novelty_tier="recent_admission",
                    admission_cycle=21,
                ),
                _single_piece_world(
                    world_id="recent_phase_rook",
                    recipe="phase_rook",
                    piece_seed=1202 + seed_offset,
                    placement_seed=2002 + seed_offset,
                    split="train",
                    novelty_tier="recent_admission",
                    admission_cycle=22,
                ),
                _single_piece_world(
                    world_id="recent_turn_charge",
                    recipe="turn_charge",
                    piece_seed=1303 + seed_offset,
                    placement_seed=2003 + seed_offset,
                    split="train",
                    novelty_tier="recent_admission",
                    admission_cycle=23,
                ),
            ),
        ),
        BenchmarkSuiteSpec(
            suite_id="composition",
            description="Worlds containing interacting mechanics from seen families.",
            tags=("composition", "seen_family", "strategy_conditioned"),
            worlds=(
                BenchmarkWorldSpec(
                    world_id="composition_capture_phase",
                    family_id="composition",
                    split="mixed",
                    novelty_tier="composition",
                    admission_cycle=31,
                    recipes=("capture_sprint", "phase_rook"),
                    piece_seeds=(2101 + seed_offset, 2202 + seed_offset),
                    placement_seed=3001 + seed_offset,
                    tags=("composition",),
                    world_family_ids=("capture_sprint", "phase_rook"),
                ),
                BenchmarkWorldSpec(
                    world_id="composition_phase_charge",
                    family_id="composition",
                    split="mixed",
                    novelty_tier="composition",
                    admission_cycle=32,
                    recipes=("phase_rook", "turn_charge"),
                    piece_seeds=(2302 + seed_offset, 2303 + seed_offset),
                    placement_seed=3002 + seed_offset,
                    tags=("composition",),
                    world_family_ids=("phase_rook", "turn_charge"),
                ),
                BenchmarkWorldSpec(
                    world_id="composition_capture_charge",
                    family_id="composition",
                    split="mixed",
                    novelty_tier="composition",
                    admission_cycle=33,
                    recipes=("capture_sprint", "turn_charge"),
                    piece_seeds=(2401 + seed_offset, 2403 + seed_offset),
                    placement_seed=3003 + seed_offset,
                    tags=("composition",),
                    world_family_ids=("capture_sprint", "turn_charge"),
                ),
            ),
        ),
        BenchmarkSuiteSpec(
            suite_id="heldout_seen_family",
            description="Held-out parameterizations within seen mechanic families.",
            tags=("heldout_seen_family", "seen_family", "strategy_conditioned"),
            worlds=(
                _single_piece_world(
                    world_id="heldout_seen_capture_sprint",
                    recipe="capture_sprint",
                    piece_seed=3101 + seed_offset,
                    placement_seed=4001 + seed_offset,
                    split="test_seen_family",
                    novelty_tier="heldout_seen_family",
                    admission_cycle=None,
                ),
                _single_piece_world(
                    world_id="heldout_seen_phase_rook",
                    recipe="phase_rook",
                    piece_seed=3202 + seed_offset,
                    placement_seed=4002 + seed_offset,
                    split="test_seen_family",
                    novelty_tier="heldout_seen_family",
                    admission_cycle=None,
                ),
                _single_piece_world(
                    world_id="heldout_seen_turn_charge",
                    recipe="turn_charge",
                    piece_seed=3303 + seed_offset,
                    placement_seed=4003 + seed_offset,
                    split="test_seen_family",
                    novelty_tier="heldout_seen_family",
                    admission_cycle=None,
                ),
            ),
        ),
        BenchmarkSuiteSpec(
            suite_id="heldout_family",
            description="Mechanic families excluded from training.",
            tags=("heldout_family", "strategy_conditioned"),
            worlds=(
                _single_piece_world(
                    world_id="heldout_family_edge_sumo_a",
                    recipe="edge_sumo",
                    piece_seed=4104 + seed_offset,
                    placement_seed=5001 + seed_offset,
                    split="test_heldout_family",
                    novelty_tier="heldout_family",
                    admission_cycle=None,
                ),
                _single_piece_world(
                    world_id="heldout_family_edge_sumo_b",
                    recipe="edge_sumo",
                    piece_seed=4204 + seed_offset,
                    placement_seed=5002 + seed_offset,
                    split="test_heldout_family",
                    novelty_tier="heldout_family",
                    admission_cycle=None,
                ),
            ),
        ),
    )


def _single_piece_world(
    *,
    world_id: str,
    recipe: str,
    piece_seed: int,
    placement_seed: int,
    split: str,
    novelty_tier: str,
    admission_cycle: int | None,
) -> BenchmarkWorldSpec:
    return BenchmarkWorldSpec(
        world_id=world_id,
        family_id=recipe,
        split=split,
        novelty_tier=novelty_tier,
        admission_cycle=admission_cycle,
        recipes=(recipe,),
        piece_seeds=(piece_seed,),
        placement_seed=placement_seed,
        tags=(recipe,),
        world_family_ids=(recipe,),
    )
