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
WORKERS="${WORKERS:-16}"
SEED="${SEED:-1000}"

STRESS_GAMES="${STRESS_GAMES:-64}"
STRESS_MAX_PLIES="${STRESS_MAX_PLIES:-48}"

CYCLES="${CYCLES:-6}"
TRAIN_GAMES="${TRAIN_GAMES:-256}"
VAL_GAMES="${VAL_GAMES:-32}"
SIMULATIONS="${SIMULATIONS:-32}"
MAX_PLIES="${MAX_PLIES:-96}"
EPOCHS_PER_CYCLE="${EPOCHS_PER_CYCLE:-2}"
BATCH_SIZE="${BATCH_SIZE:-16}"

ARENA_GAMES="${ARENA_GAMES:-32}"
ARENA_SIMULATIONS="${ARENA_SIMULATIONS:-32}"
ARENA_MAX_PLIES="${ARENA_MAX_PLIES:-96}"
PROMOTION_SCORE_THRESHOLD="${PROMOTION_SCORE_THRESHOLD:-0.55}"

BATCHED_INFERENCE="${BATCHED_INFERENCE:-1}"
PROMOTION_GATE="${PROMOTION_GATE:-1}"
ANALYZE_AFTER_RUN="${ANALYZE_AFTER_RUN:-1}"
PIXIE_S3_OUTPUT_URI="${PIXIE_S3_OUTPUT_URI:-}"
MODEL_ARCHITECTURE="${MODEL_ARCHITECTURE:-world_conditioned_v2}"
PIECE_REGISTRY="${PIECE_REGISTRY:-$OUTPUT_DIR/pieces/registry.json}"
USE_VERIFIED_PIECES="${USE_VERIFIED_PIECES:-1}"
SPECIAL_PIECE_INCLUSION_PROBABILITY="${SPECIAL_PIECE_INCLUSION_PROBABILITY:-1.0}"
REPLAY_WINDOW_CYCLES="${REPLAY_WINDOW_CYCLES:-4}"
ENABLE_SCHEDULED_CURRICULUM="${ENABLE_SCHEDULED_CURRICULUM:-1}"
CURRICULUM_PROVIDER_MODE="${CURRICULUM_PROVIDER_MODE:-oracle}"
CURRICULUM_TASKS="${CURRICULUM_TASKS:-1:101:capture_sprint:train:introduced;2:202:phase_rook:train:introduced;3:303:turn_charge:train:introduced;4:404:edge_sumo:train:introduced;5:505:capture_sprint:train:introduced;6:606:phase_rook:train:introduced}"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$(dirname "$PIECE_REGISTRY")"

if [[ "$BATCHED_INFERENCE" == "1" && "$WORKERS" -le 1 ]]; then
  echo "BATCHED_INFERENCE=1 requires WORKERS greater than 1" >&2
  exit 1
fi

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

cat <<EOF
[aws-run-proof] output_dir=$OUTPUT_DIR
[aws-run-proof] model_architecture=$MODEL_ARCHITECTURE
[aws-run-proof] cycles=$CYCLES train_games=$TRAIN_GAMES val_games=$VAL_GAMES simulations=$SIMULATIONS max_plies=$MAX_PLIES
[aws-run-proof] workers=$WORKERS device=$DEVICE selfplay_device=$SELFPLAY_DEVICE inference_device=$INFERENCE_DEVICE
[aws-run-proof] piece_registry=$PIECE_REGISTRY use_verified_pieces=$USE_VERIFIED_PIECES curriculum_mode=$CURRICULUM_PROVIDER_MODE
EOF

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
  --model-architecture "$MODEL_ARCHITECTURE"
  --piece-registry "$PIECE_REGISTRY"
  --special-piece-inclusion-probability "$SPECIAL_PIECE_INCLUSION_PROBABILITY"
  --replay-window-cycles "$REPLAY_WINDOW_CYCLES"
  --randomize-handauthored-specials
)

if [[ "$BATCHED_INFERENCE" == "1" ]]; then
  train_loop_cmd+=(--batched-inference)
fi

if [[ "$PROMOTION_GATE" == "1" ]]; then
  train_loop_cmd+=(--promotion-gate)
fi

if [[ "$USE_VERIFIED_PIECES" == "1" ]]; then
  train_loop_cmd+=(--use-verified-pieces)
fi

if [[ "$ENABLE_SCHEDULED_CURRICULUM" == "1" ]]; then
  IFS=';' read -r -a curriculum_tasks <<<"$CURRICULUM_TASKS"
  for task in "${curriculum_tasks[@]}"; do
    if [[ -n "$task" ]]; then
      train_loop_cmd+=(--curriculum-task "$task")
    fi
  done
  case "$CURRICULUM_PROVIDER_MODE" in
    oracle)
      train_loop_cmd+=(--curriculum-oracle)
      ;;
    live_llm)
      ;;
    *)
      echo "unsupported CURRICULUM_PROVIDER_MODE=$CURRICULUM_PROVIDER_MODE" >&2
      exit 1
      ;;
  esac
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
