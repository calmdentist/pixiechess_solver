# PixieChess Solver

PixieChess is a chess variant with frequently changing magical pieces. This repository builds a solver around one core bet:

1. special-piece rules should compile into one crisp executable DSL,
2. the simulator should stay exact and deterministic,
3. the learned model should focus on strategy under the current rules rather than legality.

The current foundation covers:

- a Python package layout aligned to the implementation plan,
- frozen core runtime types for pieces, moves, events, and game state,
- a minimal DSL parser/validator/compiler path,
- starter hand-authored piece programs,
- a deterministic simulator with orthodox movement, legal-move filtering, hooks, and replay traces,
- a search-only MCTS baseline with stable move ids and a deterministic fallback evaluator,
- deterministic self-play records with replay traces and JSONL export helpers,
- structured DSL, board, and move encoders for learned strategy,
- a small PyTorch policy/value network with legal-candidate scoring,
- a minimal self-play training loop that runs on CPU, MPS, or CUDA,
- a bootstrap path from search-only self-play into model-guided self-play,
- deterministic search comparison utilities for search-only vs model-guided runs,
- AlphaZero-style Dirichlet root noise for self-play exploration,
- checkpoint save/load for model + optimizer state,
- real `pixie selfplay`, `pixie train`, `pixie eval-model`, and `pixie train-loop` CLI commands,
- a local browser viewer for live self-play/training games and completed replay files,
- a CLI boundary for compile and verification flows,
- live Anthropic/OpenAI LLM providers for English-to-DSL compile and mismatch repair,
- synthetic piece curriculum runs that can compile, repair, verify, and admit DSL programs,
- importable interfaces for simulator, search, model, training, and evaluation work.

## Layout

- `implementation_plan.md`: execution plan provided for this workspace
- `pixiechess_solver_design.md`: architecture and modeling rationale
- `docs/`: working reference docs for the initial implementation
- `data/pieces/handauthored/`: hand-authored starter piece programs
- `src/pixie_solver/`: package source
- `tests/`: foundation tests

## Quick Start

```bash
python3 -m unittest discover -s tests
PYTHONPATH=src python3 -m pixie_solver compile-piece --file data/pieces/handauthored/phasing_rook.json --pretty
PYTHONPATH=src python3 -m pixie_solver verify-piece --file data/pieces/handauthored/war_automaton.json
PYTHONPATH=src python3 -m pixie_solver selfplay --standard-initial-state --randomize-handauthored-specials --examples-out data/selfplay/examples.jsonl --games 4
PYTHONPATH=src python3 -m pixie_solver selfplay --standard-initial-state --randomize-handauthored-specials --games 1 --viewer --viewer-keep-open
PYTHONPATH=src python3 -m pixie_solver train --examples data/selfplay/examples.jsonl --checkpoint-out checkpoints/model.pt --device mps
PYTHONPATH=src python3 -m pixie_solver eval-model --checkpoint checkpoints/model.pt --examples data/selfplay/examples.jsonl --device mps
PYTHONPATH=src python3 -m pixie_solver train-loop --output-dir runs/local_smoke --cycles 2 --train-games 20 --val-games 6 --device mps
PYTHONPATH=src python3 -m pixie_solver view-replay --games runs/local_smoke/selfplay/cycle_001_train_games.jsonl --viewer-open-browser
```

`pixie train-loop` is the ergonomic local learning check: each cycle generates fresh self-play,
trains or resumes the policy/value checkpoint, evaluates train and validation examples, and writes
`selfplay/`, `checkpoints/`, `metrics/`, and `summary.json` under the output directory. Watch
`average_policy_cross_entropy`, `average_policy_kl`, `top1_agreement`, and `value_mse` in the
cycle metrics to confirm whether the model is learning instead of only fitting noise.
Max-ply games are cutoff-adjudicated by default with a deterministic material/mobility/check
heuristic, so clearly favorable non-terminal games still produce nonzero value targets.
Self-play also mixes Dirichlet noise into root priors by default
(`--root-dirichlet-alpha 0.3`, `--root-exploration-fraction 0.25`) so early
checkpoints do not collapse into a narrow opening sample. Set
`--root-exploration-fraction 0` for deterministic no-noise data generation.

`pixie selfplay`, `pixie train`, `pixie eval-model`, and `pixie train-loop` emit progress logs to `stderr` during long runs.
Use `--quiet` if you only want the final JSON summary on `stdout`.

Add `--viewer` to `pixie selfplay` or `pixie train-loop` to start a local browser board at
`127.0.0.1` and stream games as they are generated. The viewer renders magical pieces as their
base chess piece with a letter badge, for example `P` for Phasing Rook, `S` for Sumo Rook, and
`W` for War Automaton. Use `--viewer-open-browser` to launch the browser automatically, and
`--viewer-keep-open` when you want the process to keep serving the board after a short run ends.

## Live LLM Providers

Live English-to-DSL compilation and repair use `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`.
Anthropic is the default provider and uses `claude-opus-4-6` with adaptive thinking enabled.

```bash
export ANTHROPIC_API_KEY=...
PYTHONPATH=src python3 -m pixie_solver compile-piece --text "A rook that can slide through friendly pieces." --pretty

PYTHONPATH=src python3 -m pixie_solver piece-curriculum \
  --seed 1 \
  --recipe capture_sprint \
  --llm-repair \
  --artifact-dir runs/piece_curriculum/seed_1
```

Use OpenAI with:

```bash
export OPENAI_API_KEY=...
PYTHONPATH=src python3 -m pixie_solver compile-piece \
  --text "A pawn that gathers charge as its turn begins." \
  --llm-provider openai \
  --llm-model gpt-5.4 \
  --pretty
```

## Current Milestone

This repo is in early M5:

- repository skeleton and core interfaces are in place,
- DSL compilation works for hand-authored starter pieces,
- the simulator covers orthodox movement plus the minimal v1 magical modifiers/effects,
- deterministic replay is available for move/event verification,
- search-only MCTS and deterministic self-play are available,
- the policy/value model and minimal training loop are now implemented,
- model-guided bootstrap/evaluation utilities are now implemented,
- checkpointing and training/self-play CLI workflows are now implemented,
- tactical regression harness and broader match evaluation are still ahead.
