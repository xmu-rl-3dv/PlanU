#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
SMOKE_MODEL="${SMOKE_MODEL:-hf-internal-testing/tiny-random-LlamaForCausalLM}"
RUN_ROOT="${RUN_ROOT:-$(mktemp -d "${TMPDIR:-/tmp}/planu-smoke.XXXXXX")}"
PLANBENCH_PATH="${PLANBENCH_PATH:-${RUN_ROOT}/LLMs-Planning}"
PLANBENCH_COMMIT="34e6841f81ca7708f2f8b8241504bfe8a908e40b"

mkdir -p "${RUN_ROOT}"
if [[ ! -d "${PLANBENCH_PATH}/.git" ]]; then
  git clone --filter=blob:none \
    https://github.com/karthikv792/LLMs-Planning.git \
    "${PLANBENCH_PATH}"
  git -C "${PLANBENCH_PATH}" checkout "${PLANBENCH_COMMIT}"
fi

actual_planbench_commit="$(git -C "${PLANBENCH_PATH}" rev-parse HEAD)"
if [[ "${actual_planbench_commit}" != "${PLANBENCH_COMMIT}" ]]; then
  printf 'PLANBENCH_PATH must be at commit %s; found %s\n' \
    "${PLANBENCH_COMMIT}" "${actual_planbench_commit}" >&2
  exit 1
fi

export HF_HUB_DISABLE_TELEMETRY=1
export PLANBENCH_PATH
export PYGAME_HIDE_SUPPORT_PROMPT=1

cd "${RUN_ROOT}"

"${PYTHON}" "${ROOT}/mcts/overcooked/PlanU_inference.py" \
  --exp-name smoke \
  --seed 1 \
  --cuda False \
  --capture-video False \
  --num-envs 1 \
  --depth 3 \
  --env-id Overcooked-LLMA-v4 \
  --n-agent 1 \
  --grid-dim 7 7 \
  --task 0 \
  --map-type A \
  --obs-radius 2 \
  --mode vector \
  --env-reward 0.2 1 0.1 0.001 \
  --normalization-mode token \
  --stochastic 0.5 \
  --maxiterations 1 \
  --rnd True \
  --init_dist False \
  --temperature 1 \
  --base-model "${SMOKE_MODEL}"

"${PYTHON}" "${ROOT}/mcts/virtualhome/PlanU_inference_food.py" \
  --maxiterations 1 \
  --depth 3 \
  --stochastic 0.2 \
  --seed 100 \
  --rnd True

"${PYTHON}" "${ROOT}/mcts/virtualhome/PlanU_entertainment.py" \
  --maxiterations 1 \
  --depth 3 \
  --stochastic 0.2 \
  --seed 100 \
  --rnd True \
  --temperature 1 \
  --base-model "${SMOKE_MODEL}"

"${PYTHON}" "${ROOT}/blockworld/evaluate_stochastic.py" \
  --algorithm planu \
  --version 2 \
  --steps 2 \
  --model "${SMOKE_MODEL}" \
  --device cpu \
  --max-examples 1 \
  --output-dir "${RUN_ROOT}/blockworld" \
  --n-iters 1 \
  --seed 100 \
  --success-probability 0.8

printf 'Phase-one smoke outputs: %s\n' "${RUN_ROOT}"
