# NeurIPS 2025 PlanU: Large Language Model Reasoning through Planning under Uncertainty

![Python 3.9](https://img.shields.io/badge/Python-3.9-blue)
![Code style](https://img.shields.io/badge/code%20style-black-000000.svg)
![MIT](https://img.shields.io/badge/license-MIT-blue)

This is the official implementation of **"PlanU: Large Language Model
Reasoning through Planning under Uncertainty"**, accepted at **NeurIPS 2025**.
PlanU combines language-model action proposals with planning under uncertainty
for long-horizon decisions in stochastic environments.

## Unified PlanU Algorithm

The migrated phase-one benchmarks (Overcooked, VirtualHome, and BlockWorld) use
one shared PlanU algorithm implementation in `planu_core`; within this migrated
scope, there is no published/canonical split. The shared search follows these
steps:

1. A **State Node** contains **Action Node** children, and each executed action
   leads to an **outcome State Node**. Distinct stochastic outcomes under the
   same Action Node are retained as separate children.
2. Each action owns a **Quantile Distribution**. Candidate initialization uses
   the action scorer plus an optional preview reward.
3. These migrated benchmarks use **Upper Confidence Bounds with Curiosity (UCC)**
   as the shared selection score: distorted quantile value plus normalized
   optional RND curiosity, scaled by the configured curiosity weight.
4. Search keeps a persistent tree across trajectories instead of rebuilding
   the tree after every rollout.
5. Completed trajectories use a Monte Carlo suffix-return quantile pinball
   backup to update every selected action distribution.

These three migrated phase-one benchmarks share one
tree/selection/quantile/backup search core. Benchmark-specific adapters, action
scorers, runner/evaluator orchestration, and numeric configuration remain
outside the core.

## BlockWorld Migration Notes

BlockWorld now uses an alternating **State Node** / **Action Node** / outcome
State Node tree. The transition is re-sampled on each action visit, preserving
arbitrary outcomes instead of caching the first result.

The risk-neutral quantile value is the arithmetic mean, and the default
quantile range is `-10` to `100` so it covers the goal reward. The fast/action
prior is separate from the executed transition reward, and the first-visit
terminal goal reward is retained.

BlockWorld uses the shared suffix-return update and selection. Unsupported
legacy output strategies and non-default options are rejected rather than
silently accepted. These corrections mean BlockWorld results should be reported
as unified implementation results, not assumed bit-identical to the previous
broken entrypoint.

## Architecture

- `planu_core/nodes.py` and `planu_core/distribution.py` define tree ownership
  and quantile values.
- `planu_core/selection.py`, `planu_core/backup.py`, and
  `planu_core/search.py` implement selection, trajectory updates, and the
  persistent search loop.
- `planu_core/curiosity.py` and `planu_core/scorers.py` provide optional RND
  novelty and action-scoring policies.
- `planu_core/provenance.py` records reproducibility metadata.
- `planu_core/adapters` is the environment boundary: adapters translate
  benchmark state, action, preview, and transition APIs while the core remains
  environment-agnostic.

## Unified-core migration status

| Benchmark | Delivery phase | Status |
| --- | --- | --- |
| Overcooked | Phase one | Supported |
| VirtualHome | Phase one | Supported |
| BlockWorld | Phase one | Supported |
| WebShop | Phase two | Planned |
| TravelPlanner | Phase two | Planned |

TravelPlanner is tracked from
[OSU-NLP-Group/TravelPlanner](https://github.com/OSU-NLP-Group/TravelPlanner.git).
WebShop and TravelPlanner remain planned for phase two; their unified-core
adapters are not implemented. The existing `webshop/` legacy code is not yet
migrated to `planu_core` and does not indicate unified-core support.

## Installation

Prerequisites are Python 3.9, a model checkpoint compatible with the selected
runner, and the system dependencies required by that benchmark. From the
repository root, create an isolated environment and install the shared package
in editable mode:

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install easydict DI-engine
python -m pip install -e .
python -m pip install -e gym-macro-overcooked
```

The Overcooked example script enables RND, so DI-engine and easydict are required
runtime dependencies.

The benchmark environments remain in `gym-macro-overcooked` and
`virtual-home`; `planu_core` does not vendor or replace them. BlockWorld
datasets, prompts, PDDL files, and planner binaries under `blockworld/examples`
are external benchmark data and are not included in the Python distribution.

## Configuration Scope

Overcooked and VirtualHome model and device choices use runner arguments or
environment variables. BlockWorld exposes GPU, seed, iterations, and success
probability settings, but its current HF model identifier remains in
`blockworld/evaluate_stochastic.py`; changing that identifier requires a source
edit.

## Run Overcooked

The legacy source ignored both the `--base-model` and
`--normalization-mode` flags. Its effective published settings were
`Neko-Institute-of-Science/LLaMA-7B-HF` with `token` normalization. The unified
runner now honors both flags, while the reference launcher pins those effective
published settings. It defaults to CUDA devices `0,1,2,3`; `BASE_MODEL`,
`CUDA_VISIBLE_DEVICES`, and `PYTHON` remain available as environment overrides.
Such overrides are new configurations and should not be compared as an exact
reproduction of the published Overcooked results.

```bash
bash scripts/PlanU_overcooked.sh
```

VirtualHome and BlockWorld use their benchmark-specific environment setup
through the [VirtualHome adapter](planu_core/adapters/virtualhome.py) and
[BlockWorld adapter](planu_core/adapters/blockworld.py), respectively, with the
same shared PlanU core. WebShop and TravelPlanner remain phase-two plans and do
not have unified-core execution instructions yet.

## Outputs And Provenance

The Overcooked and VirtualHome runners use config-hashed result paths.
Their TensorBoard logs record the effective configuration and run metadata,
including the git commit, Python version, and dependency versions. BlockWorld
writes JSON provenance to a config-hashed evaluator path containing the seed.
Each run stores `effective_config.json` and `run_metadata.json`, including the
same git commit, Python version, and dependency version metadata.

## Citation

```bibtex
@inproceedings{planu2025,
  title={PlanU: Large Language Model Reasoning through Planning under Uncertainty},
  author={Ziwei Deng, Mian Deng, Chenjing Liang, Zeming Gao, Chennan Ma, Chenxing Lin, Haipeng Zhang, Songzhu Mei, Cheng Wang, Siqi Shen},
  booktitle={NeurIPS},
  year={2025}
}
```

## Acknowledgements

Code adapted from [LATS](https://github.com/lapisrocks/LanguageAgentTreeSearch) and [Overcooked-AI](https://github.com/HumanCompatibleAI/overcooked_ai).
