#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR"

mkdir -p output proof

make

LOG_FILE="proof/terminal_run.log"
{
  echo "===== nvcc --version ====="
  nvcc --version
  echo
  echo "===== nvidia-smi ====="
  nvidia-smi
  echo
  echo "===== timed execution ====="
  /usr/bin/time -f "Elapsed wall time: %E" ./batch_image_pipeline \
    --num-images 256 \
    --width 2048 \
    --height 2048 \
    --threshold 110 \
    --output-dir output
  echo
  echo "===== output files ====="
  ls -lh output
} | tee "$LOG_FILE"
