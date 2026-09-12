#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
PYTHON="${PYTHON:-python}"
export CUDA_VISIBLE_DEVICES

for seed in 1 10 20 30 40; do
  "${PYTHON}" mcts/overcooked/PlanU_inference.py \
    --exp-name "tomato_salad_llm" \
    --num-envs 1 \
    --depth 15 \
    --env-reward 0.2 1 0.1 0.001 \
    --task 0 \
    --env-id "Overcooked-LLMA-v4" \
    --record-path "workdir" \
    --normalization-mode "token" \
    --maxiterations 1000 \
    --stochastic 0.5 \
    --seed "${seed}" \
    --rnd "True"
done
