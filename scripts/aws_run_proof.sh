#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
RUNS_ROOT="${RUNS_ROOT:-$ROOT_DIR/runs}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUNS_ROOT/aws_proof_$(date -u +%Y%m%d_%H%M%S)}"
DEFAULT_BENCHMARK_MANIFEST="$ROOT_DIR/data/benchmarks/frozen/phase0_serious_v0/manifest.json"

export PYTHONPATH="${PYTHONPATH:-src}"

detect_host_vcpus() {
  if command -v nproc >/dev/null 2>&1; then
    nproc --all
    return
  fi
  if command -v sysctl >/dev/null 2>&1; then
    sysctl -n hw.logicalcpu
    return
  fi
  echo "1"
}

suggest_worker_count() {
  local host_vcpus="$1"
  local max_workers=$(( host_vcpus / 2 ))
  if (( max_workers < 1 )); then
    max_workers=1
  fi
  if (( max_workers > 32 )); then
    max_workers=32
  fi
  local suggested=$(( max_workers - 4 ))
  if (( suggested < 8 )); then
    suggested=8
  fi
  if (( suggested > max_workers )); then
    suggested="$max_workers"
  fi
  if (( suggested < 1 )); then
    suggested=1
  fi
  echo "$suggested"
}

build_worker_candidates() {
  local host_vcpus="$1"
  local default_workers="$2"
  local max_workers=$(( host_vcpus / 2 ))
  if (( max_workers < 1 )); then
    max_workers=1
  fi
  if (( max_workers > 32 )); then
    max_workers=32
  fi

  local -A seen=()
  local raw_candidates=(
    8
    "$(( default_workers - 8 ))"
    "$default_workers"
    "$(( default_workers + 4 ))"
    "$max_workers"
  )
  local candidate
  for candidate in "${raw_candidates[@]}"; do
    if (( candidate < 1 )); then
      continue
    fi
    if (( candidate > max_workers )); then
      candidate="$max_workers"
    fi
    if (( candidate < 8 && max_workers >= 8 )); then
      candidate=8
    fi
    if [[ -z "${seen[$candidate]+x}" ]]; then
      seen[$candidate]=1
      printf '%s\n' "$candidate"
    fi
  done
}

throughput_examples_per_second() {
  local benchmark_path="$1"
  "$PYTHON_BIN" - <<'PY' "$benchmark_path"
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(payload["benchmark_summary"]["examples_per_second_average"])
PY
}

DEVICE="${DEVICE:-cuda}"
SELFPLAY_DEVICE="${SELFPLAY_DEVICE:-cpu}"
INFERENCE_DEVICE="${INFERENCE_DEVICE:-$DEVICE}"
HOST_VCPUS="$(detect_host_vcpus)"
WORKERS_INPUT="${WORKERS:-auto}"
AUTO_SELECTED_WORKERS="0"
if [[ -z "$WORKERS_INPUT" || "$WORKERS_INPUT" == "auto" ]]; then
  WORKERS="$(suggest_worker_count "$HOST_VCPUS")"
  AUTO_SELECTED_WORKERS="1"
else
  WORKERS="$WORKERS_INPUT"
fi
SEED="${SEED:-1000}"
RUN_PROFILE="${RUN_PROFILE:-serious}"

STRESS_GAMES="${STRESS_GAMES:-64}"
STRESS_MAX_PLIES="${STRESS_MAX_PLIES:-48}"

CYCLES="${CYCLES:-8}"
TRAIN_GAMES="${TRAIN_GAMES:-384}"
VAL_GAMES="${VAL_GAMES:-64}"
SIMULATIONS="${SIMULATIONS:-32}"
MAX_PLIES="${MAX_PLIES:-96}"
EPOCHS_PER_CYCLE="${EPOCHS_PER_CYCLE:-2}"
BATCH_SIZE="${BATCH_SIZE:-16}"
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-64}"
INFERENCE_MAX_WAIT_MS="${INFERENCE_MAX_WAIT_MS:-5.0}"

ARENA_GAMES="${ARENA_GAMES:-64}"
ARENA_SIMULATIONS="${ARENA_SIMULATIONS:-32}"
ARENA_MAX_PLIES="${ARENA_MAX_PLIES:-96}"
PROMOTION_SCORE_THRESHOLD="${PROMOTION_SCORE_THRESHOLD:-0.55}"

BATCHED_INFERENCE="${BATCHED_INFERENCE:-1}"
PROMOTION_GATE="${PROMOTION_GATE:-1}"
ANALYZE_AFTER_RUN="${ANALYZE_AFTER_RUN:-1}"
PIXIE_S3_OUTPUT_URI="${PIXIE_S3_OUTPUT_URI:-}"
MODEL_ARCHITECTURE="${MODEL_ARCHITECTURE:-hypernetwork_conditioned_v4}"
ROOT_VALUE_TARGET_WEIGHT="${ROOT_VALUE_TARGET_WEIGHT:-0.5}"
OUTCOME_TARGET_WEIGHT="${OUTCOME_TARGET_WEIGHT:-0.5}"
UNCERTAINTY_WEIGHT="${UNCERTAINTY_WEIGHT:-auto}"
if [[ "$UNCERTAINTY_WEIGHT" == "auto" ]]; then
  case "$MODEL_ARCHITECTURE" in
    hypernetwork_conditioned_v4)
      UNCERTAINTY_WEIGHT="0.25"
      ;;
    *)
      UNCERTAINTY_WEIGHT="0.0"
      ;;
  esac
fi
ADAPTIVE_SEARCH="${ADAPTIVE_SEARCH:-0}"
ADAPTIVE_MIN_SIMULATIONS="${ADAPTIVE_MIN_SIMULATIONS:-}"
ADAPTIVE_MAX_SIMULATIONS="${ADAPTIVE_MAX_SIMULATIONS:-}"
CURRICULUM_FOUNDATION_WEIGHT="${CURRICULUM_FOUNDATION_WEIGHT:-0.6}"
CURRICULUM_KNOWN_WEIGHT="${CURRICULUM_KNOWN_WEIGHT:-0.2}"
CURRICULUM_RECENT_WEIGHT="${CURRICULUM_RECENT_WEIGHT:-0.2}"
CURRICULUM_COMPOSITION_WEIGHT="${CURRICULUM_COMPOSITION_WEIGHT:-0.0}"
CURRICULUM_RECENT_WINDOW="${CURRICULUM_RECENT_WINDOW:-2}"
STRATEGY_PROVIDER="${STRATEGY_PROVIDER:-llm}"
STRATEGY_FILE="${STRATEGY_FILE:-}"
STRATEGY_CACHE_SCOPE="${STRATEGY_CACHE_SCOPE:-world_phase}"
STRATEGY_REFRESH_ON_UNCERTAINTY="${STRATEGY_REFRESH_ON_UNCERTAINTY:-auto}"
if [[ "$STRATEGY_REFRESH_ON_UNCERTAINTY" == "auto" ]]; then
  if [[ "$STRATEGY_PROVIDER" == "none" ]]; then
    STRATEGY_REFRESH_ON_UNCERTAINTY="0"
  else
    STRATEGY_REFRESH_ON_UNCERTAINTY="1"
  fi
fi
STRATEGY_REFRESH_UNCERTAINTY_THRESHOLD="${STRATEGY_REFRESH_UNCERTAINTY_THRESHOLD:-0.75}"
BENCHMARK_MANIFEST="${BENCHMARK_MANIFEST:-}"
if [[ -z "$BENCHMARK_MANIFEST" && -f "$DEFAULT_BENCHMARK_MANIFEST" ]]; then
  BENCHMARK_MANIFEST="$DEFAULT_BENCHMARK_MANIFEST"
fi
BENCHMARK_DIR="${BENCHMARK_DIR:-$OUTPUT_DIR/benchmarks}"
PIECE_REGISTRY="${PIECE_REGISTRY:-$OUTPUT_DIR/pieces/registry.json}"
USE_VERIFIED_PIECES="${USE_VERIFIED_PIECES:-1}"
SPECIAL_PIECE_INCLUSION_PROBABILITY="${SPECIAL_PIECE_INCLUSION_PROBABILITY:-1.0}"
RANDOMIZE_HANDAUTHORED_SPECIALS="${RANDOMIZE_HANDAUTHORED_SPECIALS:-0}"
REPLAY_WINDOW_CYCLES="${REPLAY_WINDOW_CYCLES:-6}"
ENABLE_SCHEDULED_CURRICULUM="${ENABLE_SCHEDULED_CURRICULUM:-1}"
CURRICULUM_PROVIDER_MODE="${CURRICULUM_PROVIDER_MODE:-live_llm}"
CURRICULUM_TASKS="${CURRICULUM_TASKS:-1:101:capture_sprint:train:introduced;2:202:phase_rook:train:introduced;3:303:turn_charge:train:introduced;4:404:edge_sumo:train:introduced;5:505:capture_sprint:train:introduced;6:606:phase_rook:train:introduced}"
RUN_THROUGHPUT_PREFLIGHT="${RUN_THROUGHPUT_PREFLIGHT:-1}"
AUTO_TUNE_WORKERS="${AUTO_TUNE_WORKERS:-1}"
THROUGHPUT_DIR="${THROUGHPUT_DIR:-$OUTPUT_DIR/preflight}"
THROUGHPUT_GAMES="${THROUGHPUT_GAMES:-24}"
THROUGHPUT_REPEATS="${THROUGHPUT_REPEATS:-1}"
THROUGHPUT_WARMUP_RUNS="${THROUGHPUT_WARMUP_RUNS:-1}"
THROUGHPUT_SIMULATIONS="${THROUGHPUT_SIMULATIONS:-12}"
THROUGHPUT_MAX_PLIES="${THROUGHPUT_MAX_PLIES:-12}"
THROUGHPUT_CHECKPOINT="${THROUGHPUT_CHECKPOINT:-}"
PERIODIC_SYNC="${PERIODIC_SYNC:-1}"
SYNC_INTERVAL_SECONDS="${SYNC_INTERVAL_SECONDS:-900}"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$(dirname "$PIECE_REGISTRY")"

if ! [[ "$WORKERS" =~ ^[0-9]+$ ]] || (( WORKERS < 1 )); then
  echo "WORKERS must be a positive integer or auto, found: $WORKERS" >&2
  exit 1
fi

if [[ "$BATCHED_INFERENCE" == "1" && "$WORKERS" -le 1 ]]; then
  echo "BATCHED_INFERENCE=1 requires WORKERS greater than 1" >&2
  exit 1
fi

if [[ "$ADAPTIVE_SEARCH" != "0" && "$ADAPTIVE_SEARCH" != "1" ]]; then
  echo "ADAPTIVE_SEARCH must be 0 or 1, found: $ADAPTIVE_SEARCH" >&2
  exit 1
fi

if [[ "$STRATEGY_REFRESH_ON_UNCERTAINTY" != "0" && "$STRATEGY_REFRESH_ON_UNCERTAINTY" != "1" ]]; then
  echo "STRATEGY_REFRESH_ON_UNCERTAINTY must resolve to 0 or 1, found: $STRATEGY_REFRESH_ON_UNCERTAINTY" >&2
  exit 1
fi

if [[ "$STRATEGY_REFRESH_ON_UNCERTAINTY" == "1" && "$STRATEGY_PROVIDER" == "none" ]]; then
  echo "STRATEGY_REFRESH_ON_UNCERTAINTY=1 requires STRATEGY_PROVIDER to be non-none" >&2
  exit 1
fi

if [[ -n "$BENCHMARK_MANIFEST" && ! -f "$BENCHMARK_MANIFEST" ]]; then
  echo "BENCHMARK_MANIFEST does not exist: $BENCHMARK_MANIFEST" >&2
  exit 1
fi

if [[ -n "$THROUGHPUT_CHECKPOINT" && ! -f "$THROUGHPUT_CHECKPOINT" ]]; then
  echo "THROUGHPUT_CHECKPOINT does not exist: $THROUGHPUT_CHECKPOINT" >&2
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

PERIODIC_SYNC_PID=""

start_periodic_sync() {
  if [[ "$PERIODIC_SYNC" != "1" || -z "$PIXIE_S3_OUTPUT_URI" ]]; then
    return
  fi
  (
    while true; do
      sleep "$SYNC_INTERVAL_SECONDS"
      sync_outputs
    done
  ) &
  PERIODIC_SYNC_PID="$!"
}

cleanup() {
  local status="$?"
  trap - EXIT
  if [[ -n "$PERIODIC_SYNC_PID" ]]; then
    kill "$PERIODIC_SYNC_PID" >/dev/null 2>&1 || true
    wait "$PERIODIC_SYNC_PID" 2>/dev/null || true
  fi
  sync_outputs
  exit "$status"
}

run_throughput_benchmark() {
  local workers="$1"
  local output_path="$THROUGHPUT_DIR/throughput_workers_${workers}.json"
  local manifest_path="$THROUGHPUT_DIR/throughput_workers_${workers}_manifest.json"
  local checkpoint_path="$THROUGHPUT_CHECKPOINT"
  if [[ -z "$checkpoint_path" && -n "${RESUME_CHECKPOINT:-}" && -f "${RESUME_CHECKPOINT:-}" ]]; then
    checkpoint_path="$RESUME_CHECKPOINT"
  fi

  local benchmark_cmd=(
    "$PYTHON_BIN" -m pixie_solver bench-throughput
    --standard-initial-state
    --randomize-handauthored-specials
    --games "$THROUGHPUT_GAMES"
    --workers "$workers"
    --repeats "$THROUGHPUT_REPEATS"
    --warmup-runs "$THROUGHPUT_WARMUP_RUNS"
    --simulations "$THROUGHPUT_SIMULATIONS"
    --max-plies "$THROUGHPUT_MAX_PLIES"
    --seed "$SEED"
    --output "$output_path"
    --manifest-out "$manifest_path"
    --quiet
  )

  if [[ -n "$checkpoint_path" ]]; then
    benchmark_cmd+=(
      --checkpoint "$checkpoint_path"
      --device "$DEVICE"
      --selfplay-device "$SELFPLAY_DEVICE"
      --inference-device "$INFERENCE_DEVICE"
    )
    if [[ "$BATCHED_INFERENCE" == "1" && "$workers" -gt 1 ]]; then
      benchmark_cmd+=(
        --batched-inference
        --inference-batch-size "$INFERENCE_BATCH_SIZE"
        --inference-max-wait-ms "$INFERENCE_MAX_WAIT_MS"
      )
    fi
  fi

  "${benchmark_cmd[@]}" >/dev/null
  echo "$output_path"
}

run_preflight_benchmark() {
  if [[ "$RUN_THROUGHPUT_PREFLIGHT" != "1" ]]; then
    return
  fi

  mkdir -p "$THROUGHPUT_DIR"
  local summary_path="$THROUGHPUT_DIR/summary.txt"
  : >"$summary_path"

  if [[ "$AUTO_TUNE_WORKERS" == "1" && "$AUTO_SELECTED_WORKERS" == "1" ]]; then
    local best_workers="$WORKERS"
    local best_score="-1"
    local candidate=""
    while IFS= read -r candidate; do
      [[ -n "$candidate" ]] || continue
      local output_path
      output_path="$(run_throughput_benchmark "$candidate")"
      local score
      score="$(throughput_examples_per_second "$output_path")"
      printf 'workers=%s examples_per_second=%s output=%s\n' \
        "$candidate" \
        "$score" \
        "$output_path" \
        >>"$summary_path"
      if awk "BEGIN { exit !($score > $best_score) }"; then
        best_workers="$candidate"
        best_score="$score"
      fi
    done < <(build_worker_candidates "$HOST_VCPUS" "$WORKERS")
    WORKERS="$best_workers"
    printf 'selected_workers=%s selected_examples_per_second=%s\n' \
      "$WORKERS" \
      "$best_score" \
      >>"$summary_path"
  else
    local output_path
    output_path="$(run_throughput_benchmark "$WORKERS")"
    local score
    score="$(throughput_examples_per_second "$output_path")"
    printf 'workers=%s examples_per_second=%s output=%s\n' \
      "$WORKERS" \
      "$score" \
      "$output_path" \
      >>"$summary_path"
  fi
}

trap cleanup EXIT
trap 'exit 130' INT TERM

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
fi

cat <<EOF
[aws-run-proof] output_dir=$OUTPUT_DIR
[aws-run-proof] run_profile=$RUN_PROFILE host_vcpus=$HOST_VCPUS workers_input=$WORKERS_INPUT selected_workers=$WORKERS auto_selected=$AUTO_SELECTED_WORKERS
[aws-run-proof] model_architecture=$MODEL_ARCHITECTURE
[aws-run-proof] cycles=$CYCLES train_games=$TRAIN_GAMES val_games=$VAL_GAMES simulations=$SIMULATIONS max_plies=$MAX_PLIES
[aws-run-proof] workers=$WORKERS device=$DEVICE selfplay_device=$SELFPLAY_DEVICE inference_device=$INFERENCE_DEVICE inference_batch_size=$INFERENCE_BATCH_SIZE inference_max_wait_ms=$INFERENCE_MAX_WAIT_MS
[aws-run-proof] value_targets=root:$ROOT_VALUE_TARGET_WEIGHT outcome:$OUTCOME_TARGET_WEIGHT uncertainty_weight=$UNCERTAINTY_WEIGHT
[aws-run-proof] adaptive_search=$ADAPTIVE_SEARCH adaptive_min=${ADAPTIVE_MIN_SIMULATIONS:-<none>} adaptive_max=${ADAPTIVE_MAX_SIMULATIONS:-<none>}
[aws-run-proof] curriculum_weights=foundation:$CURRICULUM_FOUNDATION_WEIGHT known:$CURRICULUM_KNOWN_WEIGHT recent:$CURRICULUM_RECENT_WEIGHT composition:$CURRICULUM_COMPOSITION_WEIGHT recent_window=$CURRICULUM_RECENT_WINDOW
[aws-run-proof] strategy_provider=$STRATEGY_PROVIDER strategy_file=${STRATEGY_FILE:-<none>} strategy_cache_scope=$STRATEGY_CACHE_SCOPE strategy_refresh=$STRATEGY_REFRESH_ON_UNCERTAINTY strategy_refresh_threshold=$STRATEGY_REFRESH_UNCERTAINTY_THRESHOLD
[aws-run-proof] randomize_handauthored_specials=$RANDOMIZE_HANDAUTHORED_SPECIALS
[aws-run-proof] throughput_preflight=$RUN_THROUGHPUT_PREFLIGHT auto_tune_workers=$AUTO_TUNE_WORKERS periodic_sync=$PERIODIC_SYNC sync_interval_seconds=$SYNC_INTERVAL_SECONDS
[aws-run-proof] benchmark_manifest=${BENCHMARK_MANIFEST:-<none>}
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

run_preflight_benchmark
start_periodic_sync

if [[ "$RUN_THROUGHPUT_PREFLIGHT" == "1" ]]; then
  echo "[aws-run-proof] preflight_summary=$THROUGHPUT_DIR/summary.txt" >&2
  echo "[aws-run-proof] workers_after_preflight=$WORKERS" >&2
fi

if [[ "$BATCHED_INFERENCE" == "1" && "$WORKERS" -le 1 ]]; then
  echo "BATCHED_INFERENCE=1 requires WORKERS greater than 1 after preflight" >&2
  exit 1
fi

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
  --inference-batch-size "$INFERENCE_BATCH_SIZE"
  --inference-max-wait-ms "$INFERENCE_MAX_WAIT_MS"
  --seed "$SEED"
  --arena-games "$ARENA_GAMES"
  --arena-simulations "$ARENA_SIMULATIONS"
  --arena-max-plies "$ARENA_MAX_PLIES"
  --promotion-score-threshold "$PROMOTION_SCORE_THRESHOLD"
  --model-architecture "$MODEL_ARCHITECTURE"
  --root-value-target-weight "$ROOT_VALUE_TARGET_WEIGHT"
  --outcome-target-weight "$OUTCOME_TARGET_WEIGHT"
  --uncertainty-weight "$UNCERTAINTY_WEIGHT"
  --curriculum-foundation-weight "$CURRICULUM_FOUNDATION_WEIGHT"
  --curriculum-known-weight "$CURRICULUM_KNOWN_WEIGHT"
  --curriculum-recent-weight "$CURRICULUM_RECENT_WEIGHT"
  --curriculum-composition-weight "$CURRICULUM_COMPOSITION_WEIGHT"
  --curriculum-recent-window "$CURRICULUM_RECENT_WINDOW"
  --piece-registry "$PIECE_REGISTRY"
  --special-piece-inclusion-probability "$SPECIAL_PIECE_INCLUSION_PROBABILITY"
  --replay-window-cycles "$REPLAY_WINDOW_CYCLES"
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

if [[ "$RANDOMIZE_HANDAUTHORED_SPECIALS" == "1" ]]; then
  train_loop_cmd+=(--randomize-handauthored-specials)
else
  train_loop_cmd+=(--no-randomize-handauthored-specials)
fi

if [[ "$ADAPTIVE_SEARCH" == "1" ]]; then
  train_loop_cmd+=(--adaptive-search)
fi

if [[ -n "$ADAPTIVE_MIN_SIMULATIONS" ]]; then
  train_loop_cmd+=(--adaptive-min-simulations "$ADAPTIVE_MIN_SIMULATIONS")
fi

if [[ -n "$ADAPTIVE_MAX_SIMULATIONS" ]]; then
  train_loop_cmd+=(--adaptive-max-simulations "$ADAPTIVE_MAX_SIMULATIONS")
fi

if [[ "$STRATEGY_PROVIDER" != "none" ]]; then
  train_loop_cmd+=(--strategy-provider "$STRATEGY_PROVIDER")
fi

if [[ -n "$STRATEGY_FILE" ]]; then
  train_loop_cmd+=(--strategy-file "$STRATEGY_FILE")
fi

train_loop_cmd+=(--strategy-cache-scope "$STRATEGY_CACHE_SCOPE")

if [[ "$STRATEGY_REFRESH_ON_UNCERTAINTY" == "1" ]]; then
  train_loop_cmd+=(
    --strategy-refresh-on-uncertainty
    --strategy-refresh-uncertainty-threshold "$STRATEGY_REFRESH_UNCERTAINTY_THRESHOLD"
  )
fi

if [[ -n "$BENCHMARK_MANIFEST" ]]; then
  train_loop_cmd+=(--benchmark-manifest "$BENCHMARK_MANIFEST" --benchmark-dir "$BENCHMARK_DIR")
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
