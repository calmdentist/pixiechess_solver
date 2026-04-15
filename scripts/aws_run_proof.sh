#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
RUNS_ROOT="${RUNS_ROOT:-$ROOT_DIR/runs}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUNS_ROOT/aws_proof_$(date -u +%Y%m%d_%H%M%S)}"

export PYTHONPATH="${PYTHONPATH:-src}"

DEVICE="${DEVICE:-cuda}"
SELFPLAY_DEVICE="${SELFPLAY_DEVICE:-cpu}"
INFERENCE_DEVICE="${INFERENCE_DEVICE:-$DEVICE}"
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

BATCHED_INFERENCE="${BATCHED_INFERENCE:-1}"
PROMOTION_GATE="${PROMOTION_GATE:-1}"
ANALYZE_AFTER_RUN="${ANALYZE_AFTER_RUN:-1}"
PIXIE_S3_OUTPUT_URI="${PIXIE_S3_OUTPUT_URI:-}"

mkdir -p "$OUTPUT_DIR"

sync_outputs() {
  if [[ -z "$PIXIE_S3_OUTPUT_URI" ]]; then
    return
  fi
  if ! command -v aws >/dev/null 2>&1; then
    echo "aws CLI not found; skipping sync to $PIXIE_S3_OUTPUT_URI" >&2
    return
  fi
  if ! aws s3 sync "$OUTPUT_DIR" "$PIXIE_S3_OUTPUT_URI" --only-show-errors; then
    echo "artifact sync failed for $PIXIE_S3_OUTPUT_URI" >&2
  fi
}

trap sync_outputs EXIT

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
fi

"$PYTHON_BIN" - <<'PY'
import sys

if sys.version_info < (3, 11):
    raise SystemExit(f"Python 3.11+ required, found {sys.version}")
print(sys.version)
PY

"$PYTHON_BIN" - <<'PY'
import torch

print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"cuda_device_count={torch.cuda.device_count()}")
PY

"$PYTHON_BIN" -m pixie_solver stress-simulator \
  --standard-initial-state \
  --randomize-handauthored-specials \
  --games "$STRESS_GAMES" \
  --max-plies "$STRESS_MAX_PLIES" \
  --seed "$SEED" \
  --output "$OUTPUT_DIR/stress.json" \
  --manifest-out "$OUTPUT_DIR/stress_manifest.json"

train_loop_cmd=(
  "$PYTHON_BIN" -m pixie_solver train-loop
  --output-dir "$OUTPUT_DIR"
  --cycles "$CYCLES"
  --train-games "$TRAIN_GAMES"
  --val-games "$VAL_GAMES"
  --simulations "$SIMULATIONS"
  --max-plies "$MAX_PLIES"
  --epochs-per-cycle "$EPOCHS_PER_CYCLE"
  --batch-size "$BATCH_SIZE"
  --device "$DEVICE"
  --selfplay-device "$SELFPLAY_DEVICE"
  --workers "$WORKERS"
  --inference-device "$INFERENCE_DEVICE"
  --seed "$SEED"
  --arena-games "$ARENA_GAMES"
  --arena-simulations "$ARENA_SIMULATIONS"
  --arena-max-plies "$ARENA_MAX_PLIES"
  --promotion-score-threshold "$PROMOTION_SCORE_THRESHOLD"
  --randomize-handauthored-specials
)

if [[ "$BATCHED_INFERENCE" == "1" ]]; then
  train_loop_cmd+=(--batched-inference)
fi

if [[ "$PROMOTION_GATE" == "1" ]]; then
  train_loop_cmd+=(--promotion-gate)
fi

if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  train_loop_cmd+=(--resume-checkpoint "$RESUME_CHECKPOINT")
fi

if [[ -n "${BEST_CHECKPOINT:-}" ]]; then
  train_loop_cmd+=(--best-checkpoint "$BEST_CHECKPOINT")
fi

if [[ -n "${BEST_CHECKPOINT_OUT:-}" ]]; then
  train_loop_cmd+=(--best-checkpoint-out "$BEST_CHECKPOINT_OUT")
fi

if [[ -n "${D_MODEL:-}" ]]; then
  train_loop_cmd+=(--d-model "$D_MODEL")
fi

if [[ -n "${NUM_HEADS:-}" ]]; then
  train_loop_cmd+=(--num-heads "$NUM_HEADS")
fi

if [[ -n "${NUM_LAYERS:-}" ]]; then
  train_loop_cmd+=(--num-layers "$NUM_LAYERS")
fi

if [[ -n "${DROPOUT:-}" ]]; then
  train_loop_cmd+=(--dropout "$DROPOUT")
fi

if [[ -n "${FEEDFORWARD_MULTIPLIER:-}" ]]; then
  train_loop_cmd+=(--feedforward-multiplier "$FEEDFORWARD_MULTIPLIER")
fi

if [[ -n "${TRAIN_LOOP_EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  extra_args=( $TRAIN_LOOP_EXTRA_ARGS )
  train_loop_cmd+=("${extra_args[@]}")
fi

"${train_loop_cmd[@]}"

if [[ "$ANALYZE_AFTER_RUN" == "1" ]]; then
  "$PYTHON_BIN" scripts/analyze_training_run.py "$OUTPUT_DIR" | tee "$OUTPUT_DIR/analyzer.txt"
fi

sync_outputs
