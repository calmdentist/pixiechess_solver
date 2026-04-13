#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-src}"

OUTPUT_DIR="${OUTPUT_DIR:-runs/runpod_$(date -u +%Y%m%d_%H%M%S)}"
DEVICE="${DEVICE:-cuda}"
SELFPLAY_DEVICE="${SELFPLAY_DEVICE:-cpu}"
WORKERS="${WORKERS:-8}"
SEED="${SEED:-1000}"

STRESS_GAMES="${STRESS_GAMES:-64}"
STRESS_MAX_PLIES="${STRESS_MAX_PLIES:-48}"

CYCLES="${CYCLES:-8}"
TRAIN_GAMES="${TRAIN_GAMES:-512}"
VAL_GAMES="${VAL_GAMES:-64}"
SIMULATIONS="${SIMULATIONS:-64}"
MAX_PLIES="${MAX_PLIES:-96}"
EPOCHS_PER_CYCLE="${EPOCHS_PER_CYCLE:-2}"
BATCH_SIZE="${BATCH_SIZE:-16}"

ARENA_GAMES="${ARENA_GAMES:-64}"
ARENA_SIMULATIONS="${ARENA_SIMULATIONS:-64}"
ARENA_MAX_PLIES="${ARENA_MAX_PLIES:-96}"
PROMOTION_SCORE_THRESHOLD="${PROMOTION_SCORE_THRESHOLD:-0.55}"

mkdir -p "$OUTPUT_DIR"

python3 -m pixie_solver stress-simulator \
  --standard-initial-state \
  --randomize-handauthored-specials \
  --games "$STRESS_GAMES" \
  --max-plies "$STRESS_MAX_PLIES" \
  --seed "$SEED" \
  --output "$OUTPUT_DIR/stress.json" \
  --manifest-out "$OUTPUT_DIR/stress_manifest.json"

python3 -m pixie_solver train-loop \
  --output-dir "$OUTPUT_DIR" \
  --cycles "$CYCLES" \
  --train-games "$TRAIN_GAMES" \
  --val-games "$VAL_GAMES" \
  --simulations "$SIMULATIONS" \
  --max-plies "$MAX_PLIES" \
  --epochs-per-cycle "$EPOCHS_PER_CYCLE" \
  --batch-size "$BATCH_SIZE" \
  --device "$DEVICE" \
  --selfplay-device "$SELFPLAY_DEVICE" \
  --workers "$WORKERS" \
  --batched-inference \
  --inference-device "$DEVICE" \
  --seed "$SEED" \
  --promotion-gate \
  --arena-games "$ARENA_GAMES" \
  --arena-simulations "$ARENA_SIMULATIONS" \
  --arena-max-plies "$ARENA_MAX_PLIES" \
  --promotion-score-threshold "$PROMOTION_SCORE_THRESHOLD" \
  --randomize-handauthored-specials
