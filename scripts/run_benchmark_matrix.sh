#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CONFIG="${1:-configs/mdl_perf.yaml}"
OUTPUT_DIR="${2:-artifacts/benchmarks/$(basename "${CONFIG%.yaml}")}"
GPU_COUNTS="${GPU_COUNTS:-1 2 4 8}"
MODES="${MODES:-data embedding compute end-to-end}"
EMBEDDING_ID_DISTRIBUTIONS="${EMBEDDING_ID_DISTRIBUTIONS:-uniform zipf}"
WARMUP_STEPS="${WARMUP_STEPS:-20}"
MEASURED_STEPS="${MEASURED_STEPS:-100}"
PROFILE_STEPS="${PROFILE_STEPS:-3}"
BASE_MASTER_PORT="${BASE_MASTER_PORT:-29500}"

mkdir -p "${OUTPUT_DIR}"
python src/main.py validate-config --config "${CONFIG}"

read -r -a gpu_count_values <<< "${GPU_COUNTS}"
read -r -a mode_values <<< "${MODES}"

for world_size in "${gpu_count_values[@]}"; do
  if (( world_size < 1 || world_size > 8 )); then
    echo "GPU_COUNTS entries must be inside [1, 8], got ${world_size}" >&2
    exit 2
  fi
  master_port=$((BASE_MASTER_PORT + world_size))

  for mode in "${mode_values[@]}"; do
    distributions=(uniform)
    if [[ "${mode}" == "embedding" ]]; then
      read -r -a distributions <<< "${EMBEDDING_ID_DISTRIBUTIONS}"
    fi

    for distribution in "${distributions[@]}"; do
      output_path="${OUTPUT_DIR}/${mode}_${distribution}_${world_size}gpu.json"
      extra_args=()
      if [[ -n "${PEAK_TFLOPS:-}" ]]; then
        extra_args+=(--peak-tflops "${PEAK_TFLOPS}")
      fi
      if [[ "${mode}" == "compute" && -n "${SEQUENCE_LENGTH:-}" ]]; then
        extra_args+=(--sequence-length "${SEQUENCE_LENGTH}")
      fi
      if [[ "${mode}" == "compute" && -n "${FIXED_GLOBAL_BATCH_SIZE:-}" ]]; then
        if (( FIXED_GLOBAL_BATCH_SIZE % world_size != 0 )); then
          echo "FIXED_GLOBAL_BATCH_SIZE must be divisible by ${world_size}" >&2
          exit 2
        fi
        extra_args+=(--batch-size "$((FIXED_GLOBAL_BATCH_SIZE / world_size))")
      fi

      python src/main.py benchmark \
        --config "${CONFIG}" \
        --mode "${mode}" \
        --warmup-steps "${WARMUP_STEPS}" \
        --steps "${MEASURED_STEPS}" \
        --profile-steps "${PROFILE_STEPS}" \
        --id-distribution "${distribution}" \
        --distributed ddp \
        --nproc-per-node "${world_size}" \
        --master-port "${master_port}" \
        --output "${output_path}" \
        "${extra_args[@]}"
    done
  done
done
