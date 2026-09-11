# NeurIPS 2025 PlanU: Large Language Model Reasoning through Planning under Uncertainty

![Python 3.9](https://img.shields.io/badge/Python-3.9-blue)
![Code style](https://img.shields.io/badge/code%20style-black-000000.svg)
![MIT](https://img.shields.io/badge/license-MIT-blue)

This is the official implementation of **"PlanU: Large Language Model
Reasoning through Planning under Uncertainty"**, accepted at **NeurIPS 2025**.
PlanU combines language-model action proposals with planning under uncertainty
for long-horizon decisions in stochastic environments.

## Unified PlanU Algorithm

The repository has one shared PlanU algorithm implementation in `planu_core`;
there is no published/canonical split. The shared search follows these steps:

1. A **State Node** contains **Action Node** children, and each executed action
   leads to an **outcome State Node**. Distinct stochastic outcomes under the
   same Action Node are retained as separate children.
2. Each action owns a **Quantile Distribution**. Candidate initialization uses
   the action scorer plus an optional preview reward.
3. Every benchmark uses the shared selection score: distorted quantile value
   plus normalized optional RND curiosity.
4. Search keeps a persistent tree across trajectories instead of rebuilding
   the tree after every rollout.
5. Completed trajectories use a Monte Carlo suffix-return quantile pinball
   backup to update every selected action distribution.

The algorithm and its update rules are benchmark-independent; benchmark
differences are limited to adapters and configuration.

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

## Benchmark Status

| Benchmark | Delivery phase | Status |
| --- | --- | --- |
| Overcooked | Phase one | Supported |
| VirtualHome | Phase one | Supported |
| BlockWorld | Phase one | Supported |
| WebShop | Phase two | Planned |
| TravelPlanner | Phase two | Planned |

TravelPlanner is tracked from
[OSU-NLP-Group/TravelPlanner](https://github.com/OSU-NLP-Group/TravelPlanner.git).
Phase-two adapters are not implemented.

## Installation

Prerequisites are Python 3.9, a model checkpoint compatible with the selected
runner, and the system dependencies required by that benchmark. From the
repository root, create an isolated environment and install the shared package
in editable mode:

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

The benchmark environments remain in `gym-macro-overcooked` and
`virtual-home`; `planu_core` does not vendor or replace them. BlockWorld
datasets, prompts, PDDL files, and planner binaries under `blockworld/examples`
are external benchmark data and are not included in the Python distribution.
Configure model and device choices through runner arguments or environment
variables, without editing hardcoded source locations.

## Run Overcooked

The launcher defaults to CUDA devices `0,1,2,3` and
`meta-llama/Meta-Llama-3.1-8B-Instruct`. Override `CUDA_VISIBLE_DEVICES`,
`BASE_MODEL`, or `PYTHON` in the environment when needed.

```bash
bash scripts/PlanU_overcooked.sh
```

VirtualHome and BlockWorld use their benchmark-specific environment setup and
adapters with the same shared PlanU core. WebShop and TravelPlanner remain
phase-two plans and do not have unified-core execution instructions yet.

## Outputs And Provenance

Run output directories include the effective config hash, so configurations do
not silently share a result path. TensorBoard logs record the effective
configuration and run metadata, including the git commit, Python version, and
dependency versions.

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
