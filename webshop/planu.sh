#!/usr/bin/env bash
set -euo pipefail

python -m planu_core.webshop.runner \
  --temperature 0.8 \
  --prompt-mode cot \
  --n-generate-sample 5 \
  --n-evaluate-sample 1 \
  --iterations 10 \
  --depth 10 \
  --task-start-index 1 \
  --task-end-index 50 \
  "$@"
