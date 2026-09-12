#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
PYTHON="${PYTHON:-python}"
export CUDA_VISIBLE_DEVICES

for seed in 1 10 20 30 40; do
  "${PYTHON}" mcts/virtualhome/PlanU_inference_food.py \
    --depth 15 \
    --stochastic 0.2 \
    --maxiterations 1000 \
    --rnd "True" \
    --seed "$seed"

  "${PYTHON}" mcts/virtualhome/PlanU_entertainment.py \
    --depth 15 \
    --stochastic 0.2 \
    --maxiterations 1000 \
    --rnd "True" \
    --seed "$seed"
done
