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
- a world-conditioned transformer policy/value path (`world_conditioned_v2`) and
  a hypernetwork-specialized successor (`hypernetwork_conditioned_v4`),
- structured `ProgramIR`, `StrategyIR`, and action encoders for learned strategy,
- strategy providers that can inject static, file-backed, or live frontier-LLM
  strategy hypotheses into self-play and runtime search,
- a hypernetwork compiler/cache that turns world + strategy context into frozen
  executor adapters for the current search episode,
- an uncertainty-aware search path with adaptive root simulation budgets,
- an online executable-world runtime that can detect state contradictions,
  refresh strategy, repair editable programs, and patch the live world model,
- a self-play training loop that runs on CPU, MPS, or CUDA,
- a bootstrap path from search-only self-play into model-guided self-play,
- deterministic search comparison utilities for search-only vs model-guided runs,
- AlphaZero-style Dirichlet root noise for self-play exploration,
- checkpoint save/load for model + optimizer state,
- checkpoint arena evaluation and optional train-loop promotion gates,
- deterministic per-game seeds and parallel self-play workers for cloud shakedowns,
- PixieChess-Lite simulator stress checks and run manifests for reproducible scale runs,
- real `pixie selfplay`, `pixie train`, `pixie eval-model`, and `pixie train-loop` CLI commands,
- a local browser viewer for live self-play/training games and completed replay files,
- a read-only training-run analyzer for monitoring train-loop artifacts and arena gates,
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
- `src/pixie_solver/strategy/`: canonical strategy schema and providers
- `src/pixie_solver/hypernet/`: world-compiler hypernetwork contracts/cache/layers
- `src/pixie_solver/world_model/`: executable world-model interfaces and runtime
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
PYTHONPATH=src python3 -m pixie_solver stress-simulator --standard-initial-state --randomize-handauthored-specials --games 64 --max-plies 48 --output runs/local_stress/stress.json --manifest-out runs/local_stress/manifest.json
PYTHONPATH=src python3 -m pixie_solver bench-throughput --standard-initial-state --games 32 --workers 8 --checkpoint checkpoints/model.pt --batched-inference --device cuda --selfplay-device cpu --inference-device cuda --simulations 16 --max-plies 8 --output runs/benchmarks/throughput.json
PYTHONPATH=src python3 -m pixie_solver train-loop --output-dir runs/local_parallel --cycles 2 --train-games 40 --val-games 8 --workers 4 --promotion-gate --device mps
PYTHONPATH=src python3 -m pixie_solver arena --candidate runs/local_smoke/checkpoints/model_002.pt --baseline runs/local_smoke/checkpoints/model_001.pt --games 20 --simulations 16 --device mps --output runs/local_smoke/arena/model_002_vs_001.json
PYTHONPATH=src python3 -m pixie_solver view-replay --games runs/local_smoke/selfplay/cycle_001_train_games.jsonl --viewer-open-browser
python3 scripts/analyze_training_run.py runs/local_smoke
OUTPUT_DIR=runs/runpod_001 WORKERS=16 DEVICE=cuda SELFPLAY_DEVICE=cpu scripts/runpod_train_loop.sh
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

`pixie train-loop` now samples from verified-world curriculum buckets instead of
one undifferentiated random special-piece pool. Use `--curriculum-foundation-weight`,
`--curriculum-known-weight`, `--curriculum-recent-weight`, and
`--curriculum-composition-weight` to control the mixture, and
`--strategy-provider llm` or `--strategy-provider json_file --strategy-file ...`
to inject a strategy hypothesis at game start for each sampled world. Replay
sampling now preserves those same world families through `foundation`,
`known_mechanic`, `recent`, and `composition` buckets instead of collapsing
everything into one coarse verified bucket.

For decision-grade runs, add `--benchmark-manifest data/benchmarks/phase0_suite_template.json`
(after replacing the template paths with your frozen benchmark corpus). `pixie train-loop`
will snapshot the manifest into the run directory and write one candidate benchmark report per
cycle under `benchmarks/`.

An initial frozen strategy-conditioned corpus now lives at
`data/benchmarks/frozen/phase0_serious_v0/manifest.json`. Regenerate or scale it with:

```bash
PYTHONPATH=src python3 -m pixie_solver build-benchmark-corpus \
  --output-dir data/benchmarks/frozen/phase0_serious_v0 \
  --games-per-world 2 \
  --simulations 1 \
  --max-plies 8
```

For single-GPU AWS validation runs, `scripts/aws_run_proof.sh` now defaults to a
serious workload, honors `WORKERS=auto`, runs a short throughput preflight to
pick a better worker count on 64-vCPU hosts, writes the preflight summary under
`preflight/`, and periodically syncs artifacts to S3 during the run.

The train/train-loop stack now supports multiple executor architectures through
`--model-architecture`, including:

- `baseline_v1`
- `world_conditioned_v2`
- `hypernetwork_conditioned_v4`

`hypernetwork_conditioned_v4` is the current research path: it conditions on the
current executable world and active strategy, compiles small adapter bundles for
the executor, emits uncertainty, and can drive adaptive search budgets.

Use `pixie arena` when you need strength evidence instead of loss-only evidence. It plays
a candidate checkpoint against a baseline checkpoint with deterministic model-guided MCTS,
alternates colors by default, disables root noise, and reports candidate win/draw/loss,
score rate, confidence interval, termination reasons, and a promotion decision. Add
`--promotion-gate` to `pixie train-loop` to maintain `checkpoints/best.pt`; after each
cycle the latest candidate plays the current champion and only replaces it when
`--promotion-score-threshold` is met.

Use `--workers` on `pixie selfplay` or `pixie train-loop` for deterministic process-level
parallel self-play. Each game gets a stable per-game seed derived from the base seed, so
serial and parallel search-only runs are reproducible by game index. Viewer streaming stays
serial and requires `--workers 1`. Add `--batched-inference --inference-device cuda`
for model-guided parallel workers so CPU actors share one model-owning GPU process
instead of loading one CUDA model per worker.

`pixie stress-simulator` is the PixieChess-Lite volume check before cloud runs. It plays
random legal games, verifies that generated legal moves can be applied, and replay-checks
the final state hash. This is intentionally about deterministic simulator stability under
volume, not exact FIDE edge-case coverage.

`pixie bench-throughput` is the fixed throughput harness for config comparisons. It runs
deterministic self-play workloads, reports end-to-end wall-clock games/examples per second,
and also preserves the lower-level search and batched-inference timing breakdowns so you can
tell whether a change helped move generation, model inference, queueing, or overall throughput.

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

## Handoff Snapshot

This repo is now in a full executable-world-model research shape:

- The deterministic world layer is in place: hand-authored magical pieces compile through
  the DSL and canonical `ProgramIR`, query logic and legality sit on the program path,
  replay traces are stable, and the simulator is exact under the current inferred world.
- The learned executor stack is in place: `world_conditioned_v2` and
  `hypernetwork_conditioned_v4` can train end-to-end, consume structured world/program
  semantics, and score legal actions under the current rule set.
- The strategy-conditioning path is in place: canonical `StrategyIR`, strategy providers,
  strategy-aware self-play, strategy-aware replay training, and uncertainty-triggered
  strategy refreshes are all wired through the training/search path.
- The hypernetwork path is in place: world + strategy context can compile into frozen
  adapter bundles, and the `v4` executor can specialize per world without retraining from
  scratch for every rules change.
- The online runtime path is in place: `OnlineWorldModelRuntime` can act under the current
  inferred world, compare predicted vs observed transitions, refresh strategy, repair
  editable programs, and commit joint multi-program repairs when a contradiction spans
  multiple implicated classes.
- The operational loop is in place: `pixie selfplay`, `pixie train`, `pixie eval-model`,
  `pixie arena`, and `pixie train-loop` generate artifacts, checkpoints, metrics, and
  progress logs; `scripts/analyze_training_run.py` summarizes run health and arena gates
  without touching an active process.
- The main remaining gaps are experimental, not structural: larger held-out evaluations,
  live LLM-driven strategy experiments, and proof-quality A/B runs against simpler baselines.

## Next High-Leverage Improvements

1. Run real A/B evaluations. Compare `baseline_v1`, `world_conditioned_v2`, and
   `hypernetwork_conditioned_v4` on held-out rule families and adaptation speed after
   world changes, not just on training loss or in-distribution self-play.
2. Connect the live LLM loop. The runtime is ready for strategy refresh and repair,
   but the decisive experiment is still missing: let a live frontier provider maintain
   the world model and emit strategies under real contradiction-triggered updates.
3. Tighten held-out evaluation. Build fixed suites for unseen mechanic families,
   composition splits, and world-repair episodes so claims about generalization and
   adaptation are not resting on one curriculum slice.
4. Improve self-play throughput. The loop is now semantically richer but still slow.
   Profile adaptive-search behavior, batch model inference more aggressively, and keep
   pushing CPU worker throughput before scaling game counts.
5. Expand strategy learning. Right now strategy is injected and consumed structurally.
   The next leap is proving that strategy-conditioned executors actually adapt faster
   than pure RL/search baselines under frequent rule changes.
6. Scale the executor only after the evaluation story is tight. The repo now has the
   right interfaces for bigger strategy/world-conditioned models, but wider/deeper
   networks are only worth it once held-out evaluations can prove the gains.
