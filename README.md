# PlanU: Large Language Model Reasoning through Planning under Uncertainty

This repository contains the implementation of **"PlanU: Large Language Model
Reasoning through Planning under Uncertainty"**, accepted at NeurIPS 2025.

All experiment runners share the same PlanU search, quantile selection, and
backup implementation. New tasks can reuse the algorithm by implementing an
adapter for their state, action, reward, and model interfaces.

## Core Modules

The shared algorithm lives in `planu_core/`:

| Path | Responsibility |
| --- | --- |
| `planu_core/search.py` | `PlanUSearch` orchestration |
| `planu_core/nodes.py` | State, action, and stochastic outcome nodes |
| `planu_core/distribution.py` | Quantile distributions |
| `planu_core/selection.py` | UCC action selection |
| `planu_core/backup.py` | Suffix-return quantile backup |
| `planu_core/interfaces.py` | Environment, action provider, and scorer protocols |
| `planu_core/adapters/` | Environment-specific adapters |

Experiment runners construct an adapter, action provider, and scorer before
calling the shared `PlanUSearch`.

## Installation

Create the PlanU Python 3.9 environment from the repository root:

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  -r requirements-experiments.txt \
  -c requirements-experiments-lock.txt
python -m pip install --no-deps -e .
python -m pip install --no-deps -e gym-macro-overcooked
python -m pip install --no-deps -e virtual-home
```

BlockWorld also requires the pinned PlanBench checkout:

```bash
git clone \
  https://github.com/karthikv792/LLMs-Planning.git \
  external/LLMs-Planning
git -C external/LLMs-Planning checkout \
  34e6841f81ca7708f2f8b8241504bfe8a908e40b
export PLANBENCH_PATH="$PWD/external/LLMs-Planning"
```

## Core Experiments

Run the Overcooked reference configuration:

```bash
source .venv/bin/activate
bash scripts/PlanU_overcooked.sh
```

Run the VirtualHome reference configurations:

```bash
source .venv/bin/activate
bash scripts/PlanU_Virtualhome.sh
```

Run a 2-step BlockWorld evaluation:

```bash
source .venv/bin/activate
export PLANBENCH_PATH="$PWD/external/LLMs-Planning"
python blockworld/evaluate_stochastic.py \
  --algorithm planu \
  --version 2 \
  --steps 2 \
  --n-iters 10 \
  --seed 100 \
  --success-probability 0.8
```

### Model Configuration

All runners use `PLANU_MODEL` as the common model configuration entry:

```bash
export PLANU_MODEL="<model-id-or-local-path>"
```

Overcooked, VirtualHome, and BlockWorld interpret the value as a Hugging Face
model ID or local checkpoint directory. These experiments require token-level
likelihoods and do not directly support an OpenAI-compatible API.

WebShop interprets `PLANU_MODEL` as an OpenAI-compatible model name. The same
configuration works with hosted providers and local compatible servers:

```bash
export PLANU_MODEL="<model-name>"
export OPENAI_BASE_URL="<openai-compatible-base-url>"
export OPENAI_API_KEY="<provider-api-key>"
```

When `PLANU_MODEL` is unset, each Python runner uses its paper-reference
default. The `--base-model`, `--model`, and `--backend` arguments remain
available for one-off overrides.

## WebShop Setup

WebShop runs as a separate service pinned to:

```text
princeton-nlp/WebShop@64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd
```

Prepare the service:

```bash
export WEBSHOP_ROOT="$PWD/external/WebShop"
export WEBSHOP_ENV_PREFIX="$WEBSHOP_ROOT/.conda-planu"
bash scripts/bootstrap_webshop.sh
```

The bootstrap creates an isolated Python 3.8.13 environment, installs the
pinned dependencies, downloads the small dataset, builds the Lucene index, and
checks that the server starts.

## Run WebShop

Start the official service in terminal 1:

```bash
export WEBSHOP_ROOT="$PWD/external/WebShop"
export WEBSHOP_ENV_PREFIX="$WEBSHOP_ROOT/.conda-planu"
cd "$WEBSHOP_ROOT"
"$WEBSHOP_ENV_PREFIX/bin/python" -m web_agent_site.app --log --attrs
```

Run PlanU from the repository root in terminal 2:

```bash
source .venv/bin/activate
export OPENAI_API_KEY="<provider-api-key>"
export OPENAI_BASE_URL="<openai-compatible-base-url>"
bash webshop/planu.sh \
  --webshop-url http://127.0.0.1:3000 \
  --output-dir logs/webshop-reference
```

The launcher calls `python -m planu_core.webshop.runner`. Model credentials are
read from environment variables and are not written to result artifacts.

## Outputs

WebShop writes:

```text
<output-dir>/
  run_manifest.json
  effective_config.json
  run_metadata.json
  results.jsonl
  tasks/fixed_<index>.json
```

Other experiment runners record config-hashed results, effective
configuration, source revision, and dependency versions.

## Citation

```bibtex
@inproceedings{planu2025,
  title={PlanU: Large Language Model Reasoning through Planning under Uncertainty},
  author={Ziwei Deng, Mian Deng, Chenjing Liang, Zeming Gao, Chennan Ma, Chenxing Lin, Haipeng Zhang, Songzhu Mei, Cheng Wang, Siqi Shen},
  booktitle={NeurIPS},
  year={2025}
}
```
