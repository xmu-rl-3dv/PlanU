# NeurIPS 2025 PlanU: Large Language Model Reasoning through Planning under Uncertainty

![Python 3.9](https://img.shields.io/badge/Python-3.9-blue)
![WebShop Python 3.8.13](https://img.shields.io/badge/WebShop-Python%203.8.13-blue)
![MIT](https://img.shields.io/badge/license-MIT-blue)

This repository contains the implementation of **"PlanU: Large Language Model
Reasoning through Planning under Uncertainty"**, accepted at **NeurIPS 2025**.
PlanU combines language-model action proposals with distributional planning for
long-horizon decisions in stochastic environments.

The current unified implementation supports Overcooked, VirtualHome,
BlockWorld, and WebShop. The authoritative WebShop smoke has exercised
`fixed_1` against the pinned official Flask/Lucene server; broader model-backed
WebShop results are not implied.

## Unified-core migration status

| Benchmark | Delivery phase | Status | Executable entry point |
| --- | --- | --- | --- |
| Overcooked | Phase one | Supported | `mcts/overcooked/PlanU_inference.py` |
| VirtualHome | Phase one | Supported | `mcts/virtualhome/PlanU_inference_food.py`, `PlanU_entertainment.py` |
| BlockWorld | Phase one | Supported | `blockworld/evaluate_stochastic.py` |
| WebShop | Phase two | Supported | `python -m planu_core.webshop.runner` |
| TravelPlanner | Phase two | Planned | Not implemented |

TravelPlanner is tracked from
[OSU-NLP-Group/TravelPlanner](https://github.com/OSU-NLP-Group/TravelPlanner.git).
The files under `webshop/` are compatibility entry points; WebShop search is
implemented in `planu_core`, not in a separate legacy tree.

## Unified PlanU Algorithm

The migrated phase-one benchmarks (Overcooked, VirtualHome, and BlockWorld) and
the phase-two WebShop benchmark use one shared PlanU search implementation:

1. A **State Node** contains **Action Node** children. Executing an action leads
   to an **outcome State Node**: distinct stochastic outcomes remain separate,
   while repeated equal `state_key` outcomes merge under the same action.
2. Every action owns a **Quantile Distribution** initialized by an action
   scorer and, when enabled, a preview reward.
3. **Upper Confidence Bounds with Curiosity (UCC)** selects actions from a
   distorted quantile value plus normalized optional **RND curiosity**.
4. `PlanUSearch` keeps a **persistent tree** across trajectories.
5. Completed trajectories update selected actions with Monte Carlo
   **suffix-return** targets and quantile **pinball** loss.

All supported benchmarks share the tree, selection, quantile, backup, and
search core. Benchmark-specific adapters, action providers, action scorers,
runner/evaluator orchestration, and numeric configuration remain outside the
core.

## Architecture

The algorithm body is under `planu_core/`:

| Location | Responsibility |
| --- | --- |
| `planu_core/search.py` | `PlanUSearch`, expansion, transition sampling, persistent-tree iteration |
| `planu_core/nodes.py` | State Node and Action Node ownership |
| `planu_core/distribution.py` | Quantile Distribution representation and updates |
| `planu_core/selection.py` | UCC selection and risk distortion |
| `planu_core/backup.py` | Suffix-return quantile backup |
| `planu_core/curiosity.py`, `planu_core/rnd.py` | Optional RND curiosity |
| `planu_core/interfaces.py` | Environment, action-provider, scorer, and transition contracts |
| `planu_core/scorers.py` | Shared Hugging Face action scoring |
| `planu_core/provenance.py` | Effective config hashing and runtime provenance |
| `planu_core/adapters/` | Environment boundary for Overcooked, VirtualHome, BlockWorld, and WebShop |

The concrete boundaries are `planu_core/adapters/overcooked.py`,
`planu_core/adapters/virtualhome.py`, `planu_core/adapters/blockworld.py`, and
`planu_core/adapters/webshop.py`.

WebShop adds model-generated actions and HTTP transport without duplicating the
search algorithm:

```text
PlanUSearch
  -> ActionProvider / ActionScorer
  -> WebShopAdapter
  -> WebShopHttpClient
  -> official princeton-nlp/WebShop server
```

Its implementation is split across:

- `planu_core/adapters/webshop.py`: state, legal transitions, reward, terminal
  handling, and stochastic latency.
- `planu_core/webshop/actions.py`: typed WebShop actions and strict parsing.
- `planu_core/webshop/providers.py`: scripted and model-backed action providers
  and scorers.
- `planu_core/webshop/client.py`: HTTP route construction and HTML parsing.
- `planu_core/webshop/runner.py`: CLI, orchestration, resume-safe persistence,
  and provenance.

## Prerequisites

Install these system tools before creating Python environments:

- Git and `curl`.
- Python 3.9 for PlanU and the phase-one environments.
- Conda and Python 3.8.13 for the official WebShop server.
- Java 11 for WebShop's Lucene/Pyserini index.
- Internet access to GitHub and Hugging Face.
- Access to the requested Hugging Face checkpoints. Paper-scale 7B/8B runs
  normally require a CUDA GPU; the tiny smoke model can run on CPU.

Keep the PlanU and WebShop Python environments separate. The official WebShop
pin uses old Flask, spaCy, Pyserini, and Torch packages that conflict with the
PlanU Python 3.9 dependency set.

## Installation

Run all commands from the repository root. Create the PlanU Python 3.9
environment:

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-experiments.txt -c requirements-experiments-lock.txt
python -m pip install --no-deps -e .
python -m pip install --no-deps -e gym-macro-overcooked
python -m pip install --no-deps -e virtual-home
```

`requirements-experiments-lock.txt` freezes the resolved Python 3.9 runtime,
including the compatible Gym/NumPy/DI-engine/Werkzeug combination, the tested
OpenAI SDK used by model-backed WebShop, and `easydict` required by RND. The
benchmark packages are installed with `--no-deps` so they cannot replace these
pins.

Prepare the external PlanBench checkout used by BlockWorld:

```bash
git clone https://github.com/karthikv792/LLMs-Planning.git external/LLMs-Planning
git -C external/LLMs-Planning checkout 34e6841f81ca7708f2f8b8241504bfe8a908e40b
export PLANBENCH_PATH="$PWD/external/LLMs-Planning"
```

The Overcooked and VirtualHome source environments are already present in
`gym-macro-overcooked` and `virtual-home`. BlockWorld benchmark data stays under
`blockworld/examples/`; this external benchmark data is not bundled into the
Python package.

## Quick Validation

Run all phase-one environments with one small real trajectory/example and a
public tiny Llama checkpoint:

```bash
source .venv/bin/activate
export PLANBENCH_PATH="$PWD/external/LLMs-Planning"
RUN_ROOT=/tmp/planu-phase-one-smoke bash scripts/smoke_phase_one.sh
```

This executes Overcooked, VirtualHome food, VirtualHome entertainment, and
BlockWorld through their real environments and the shared search core. It
validates integration, not paper metrics.

Run the unit suite:

```bash
python -m pytest tests/planu_core -q
```

## Run Overcooked

A short single-seed run:

```bash
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 python mcts/overcooked/PlanU_inference.py \
  --exp-name tomato_salad_check \
  --env-id Overcooked-LLMA-v4 \
  --task 0 \
  --depth 15 \
  --maxiterations 10 \
  --stochastic 0.5 \
  --env-reward 0.2 1 0.1 0.001 \
  --base-model Neko-Institute-of-Science/LLaMA-7B-HF \
  --normalization-mode token \
  --seed 1 \
  --rnd True
```

The five-seed, 1000-trajectory reference launcher is:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/PlanU_overcooked.sh
```

The legacy source ignored `--base-model` and `--normalization-mode`; its
effective published settings were `Neko-Institute-of-Science/LLaMA-7B-HF` and
`token`. The unified runner now honors both options, and the reference launcher
pins those effective settings. Model or device overrides are new
configurations and are not exact reproductions.

## Run VirtualHome

Food uses a constant action prior and does not load an LLM scorer:

```bash
source .venv/bin/activate
python mcts/virtualhome/PlanU_inference_food.py \
  --depth 15 \
  --maxiterations 1000 \
  --stochastic 0.2 \
  --seed 1 \
  --rnd True
```

Entertainment uses a Hugging Face action scorer:

```bash
CUDA_VISIBLE_DEVICES=0 python mcts/virtualhome/PlanU_entertainment.py \
  --depth 15 \
  --maxiterations 1000 \
  --stochastic 0.2 \
  --base-model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --temperature 1.0 \
  --seed 1 \
  --rnd True
```

The checkpoint is gated; authenticate with `huggingface-cli login` or substitute
an accessible compatible model. Run both tasks over the five reference seeds
with:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/PlanU_Virtualhome.sh
```

## Run BlockWorld

Run one 2-step example as a fast integration check:

```bash
source .venv/bin/activate
export PLANBENCH_PATH="$PWD/external/LLMs-Planning"
python blockworld/evaluate_stochastic.py \
  --algorithm planu \
  --version 2 \
  --steps 2 \
  --model hf-internal-testing/tiny-random-LlamaForCausalLM \
  --device cpu \
  --max-examples 1 \
  --n-iters 1 \
  --seed 100 \
  --success-probability 0.8 \
  --output-dir /tmp/planu-blockworld-smoke
```

Remove `--max-examples`, use the intended model, and restore `--n-iters 10` for
a full split:

```bash
CUDA_VISIBLE_DEVICES=0 python blockworld/evaluate_stochastic.py \
  --algorithm planu \
  --version 2 \
  --steps 2 \
  --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --device cuda \
  --n-iters 10 \
  --seed 100 \
  --success-probability 0.8
```

## BlockWorld Migration Notes

BlockWorld uses an alternating **State Node** / **Action Node** / outcome State
Node tree. Every action visit is re-sampled, preserving arbitrary outcomes
instead of caching the first result. Its risk-neutral quantile value is the
arithmetic mean, and the `[-10, 100]` support includes the goal reward. The
fast/action prior remains separate from executed transition reward, including
the first-visit terminal goal reward.

The runner uses shared suffix-return backup and selection. Unsupported legacy
output strategies and non-default options are rejected. Report results as
unified implementation results, not as bit-identical output from the previous
broken entrypoint.

## Run WebShop

WebShop has two isolated processes:

- PlanU client: the Python 3.9 `.venv`.
- Official WebShop server: Python 3.8.13, Java 11, Flask, spaCy, and Pyserini.

The server is pinned to
`princeton-nlp/WebShop@64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd`.
Do not install its requirements into `.venv`.

### Prepare The Official Server

With Conda available on `PATH`, use `scripts/bootstrap_webshop.sh` for the
standard bootstrap:

```bash
export WEBSHOP_ROOT="$PWD/external/WebShop"
export WEBSHOP_ENV_PREFIX="$WEBSHOP_ROOT/.conda-planu"
bash scripts/bootstrap_webshop.sh
```

The bootstrap clones and pins the official repository, creates Python 3.8.13,
installs and verifies `Werkzeug==2.1.2` before the upstream Flask 2.1.2 setup,
downloads the small 1000-product dataset, builds the Lucene index, and checks
that the server starts. The checkout must remain clean; generated data,
indexes, and the isolated environment are excluded from source validation.

On macOS arm64, export the Conda OpenJDK paths before bootstrap and smoke:
The relevant variables are `JAVA_HOME`, `JVM_PATH`, `PATH`,
`WEBSHOP_ENV_PREFIX`, and `WEBSHOP_ROOT`.

```bash
export JAVA_HOME="$WEBSHOP_ENV_PREFIX/lib/jvm"
export JVM_PATH="$JAVA_HOME/lib/server/libjvm.dylib"
export PATH="$WEBSHOP_ENV_PREFIX/bin:$PATH"
```

The pinned upstream requirements predate Apple Silicon wheels for several
packages. If `setup.sh` fails on PyYAML, `tokenizers`, or `nmslib`, the following
minimal server environment is the configuration exercised on macOS arm64:

```bash
conda create -y -p "$WEBSHOP_ENV_PREFIX" python=3.8.13
conda install -y -p "$WEBSHOP_ENV_PREFIX" -c conda-forge openjdk=11 faiss-cpu

WEBSHOP_PYTHON="$WEBSHOP_ENV_PREFIX/bin/python"
"$WEBSHOP_PYTHON" -m pip install "Cython<3" "setuptools<69" wheel
"$WEBSHOP_PYTHON" -m pip install --no-build-isolation PyYAML==6.0
"$WEBSHOP_PYTHON" -m pip install \
  beautifulsoup4==4.11.1 cleantext==1.1.4 Flask==2.1.2 Werkzeug==2.1.2 \
  gdown==5.2.0 numpy==1.22.4 pandas==1.4.2 \
  rank-bm25==0.2.2 requests==2.27.1 rich==12.4.4 \
  scipy==1.10.1 spacy==3.3.0 thefuzz==0.19.0 \
  torch==1.11.0 tqdm==4.64.0 pydantic==1.8.2 \
  typing_extensions==4.5.0 pyjnius==1.7.0
"$WEBSHOP_PYTHON" -m pip install --no-deps pyserini==0.17.0
"$WEBSHOP_PYTHON" -m pip install \
  transformers==4.30.2 tokenizers==0.13.3 \
  huggingface-hub==0.17.3 onnxruntime==1.16.3 \
  sentencepiece==0.1.99 lightgbm==4.5.0
"$WEBSHOP_PYTHON" -m spacy download en_core_web_sm
```

`nmslib` is not needed by the Lucene search path used by the Flask server. On
machines where LightGBM cannot load its OpenMP runtime, install `libomp` with
the platform package manager.

If the upstream Google Drive links reject anonymous `gdown` access, download
the same small files from the cross-checked Hugging Face mirror:

```bash
mkdir -p "$WEBSHOP_ROOT/data"
curl -L --fail \
  https://huggingface.co/datasets/loongyy/webshop_catalog/resolve/main/items_shuffle_1000.json \
  -o "$WEBSHOP_ROOT/data/items_shuffle_1000.json"
curl -L --fail \
  https://huggingface.co/datasets/loongyy/webshop_catalog/resolve/main/items_ins_v2_1000.json \
  -o "$WEBSHOP_ROOT/data/items_ins_v2_1000.json"
curl -L --fail \
  https://huggingface.co/datasets/loongyy/webshop_catalog/resolve/main/items_human_ins.json \
  -o "$WEBSHOP_ROOT/data/items_human_ins.json"
```

Expected SHA-256 values:

```text
30a4765c3a327af72d9a9a95a6b2486d516f0fa1d3ecd83681901ce82a21b269  items_shuffle_1000.json
f88a36314a397b53b3d9c3fa5878e5f7b26d35019a51ec83fbedeca61a948f6f  items_ins_v2_1000.json
cf78667548a71786e1d9049c24b802e48e1084ad4bb021cae56ce1f6d96954a3  items_human_ins.json
```

After a fallback download, build the official indexes from the WebShop
environment:

```bash
cd "$WEBSHOP_ROOT/search_engine"
mkdir -p resources resources_100 resources_1k resources_100k
mkdir -p indexes indexes_100 indexes_1k indexes_100k
"$WEBSHOP_ENV_PREFIX/bin/python" convert_product_file_format.py
PATH="$WEBSHOP_ENV_PREFIX/bin:$PATH" ./run_indexing.sh
cd -
```

### Credential-free Real Smoke

The `scripts/smoke_webshop.sh` command starts the pinned official server, runs
one PlanU task, validates the real HTTP trajectory and one quantile backup,
checks output provenance, and stops the server. Set `PLANU_PYTHON` to the
Python 3.9 executable used by the PlanU client:

```bash
source .venv/bin/activate
export PLANU_PYTHON="$PWD/.venv/bin/python"
export WEBSHOP_ROOT="$PWD/external/WebShop"
export WEBSHOP_ENV_PREFIX="$WEBSHOP_ROOT/.conda-planu"
export JAVA_HOME="$WEBSHOP_ENV_PREFIX/lib/jvm"
export JVM_PATH="$JAVA_HOME/lib/server/libjvm.dylib"
export PATH="$WEBSHOP_ENV_PREFIX/bin:$PATH"
bash scripts/smoke_webshop.sh
```

`WEBSHOP_ROOT` locates the official checkout, `WEBSHOP_ENV_PREFIX` locates its
Conda environment, and `WEBSHOP_PYTHON` can override the server interpreter.
`PLANU_PYTHON` selects the Python 3.9 client. `WEBSHOP_URL` defaults to the
local server; authoritative smoke rejects non-local endpoints. `RUN_ROOT` is
optional and otherwise a unique temporary output directory is created.

The authoritative smoke evidence currently covers only `fixed_1`, using the
scripted provider against the pinned local official server. It exits `0` and
prints a JSON summary with `http_transition_count: 3` and
`quantile_backup_count: 1`. The scripted path is:

```text
search[product] -> click[ASIN] -> click[Buy Now]
```

Its purchase reward may be `0.0` because the generic query is intentionally
credential-free and does not optimize the task. The smoke validates the real
server and algorithm path; it does not reproduce paper metrics. The
model-backed reference profile has not been validated end to end, so this
evidence makes no claim about its success rate or reward.

### Model-backed WebShop Run

Start the official server in terminal 1:

```bash
export WEBSHOP_ROOT="$PWD/external/WebShop"
export WEBSHOP_ENV_PREFIX="$WEBSHOP_ROOT/.conda-planu"
export JAVA_HOME="$WEBSHOP_ENV_PREFIX/lib/jvm"
export JVM_PATH="$JAVA_HOME/lib/server/libjvm.dylib"
export PATH="$WEBSHOP_ENV_PREFIX/bin:$PATH"
cd "$WEBSHOP_ROOT"
"$WEBSHOP_ENV_PREFIX/bin/python" -m web_agent_site.app --log --attrs
```

Run the PlanU client from the repository root in terminal 2. Task bounds are
half-open, so `1 50` means `fixed_1` through `fixed_49`:

```bash
source .venv/bin/activate
export OPENAI_API_KEY="<provider-api-key>"
export OPENAI_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
python -m planu_core.webshop.runner \
  --backend qwen-plus \
  --temperature 0.8 \
  --prompt-mode cot \
  --n-generate-sample 5 \
  --n-evaluate-sample 1 \
  --iterations 10 \
  --depth 10 \
  --task-start-index 1 \
  --task-end-index 50 \
  --webshop-url http://127.0.0.1:3000 \
  --output-dir logs/webshop-reference
```

`bash webshop/planu.sh` is the compatibility launcher for the same reference
profile. `OPENAI_API_KEY` supplies credentials and `OPENAI_BASE_URL` selects
the compatible model endpoint. Credentials are read only from environment
variables and are not written to logs or provenance.

## Configuration Scope

Overcooked and VirtualHome expose model and device choices through runner
arguments or environment variables. BlockWorld exposes GPU, seed, iterations,
success probability, `--model`, `--device`, and `--max-examples` through
`blockworld/evaluate_stochastic.py`, so the HF model identifier and smoke budget
can be changed without editing hardcoded source locations.

WebShop exposes the backend, base URL, server URL, task range, seed, iterations,
depth, timeout, and output directory through
`planu_core.webshop.runner`. `--smoke` selects the deterministic,
credential-free provider.

## Outputs And Provenance

Overcooked and VirtualHome write config-hashed TensorBoard result paths. Their
text summaries contain the effective configuration, git commit, Python version,
and dependency versions.

BlockWorld writes a config-hashed evaluator directory containing
`effective_config.json`, `run_metadata.json`, evaluator logs, and persisted
JSON results.

WebShop writes:

```text
<output-dir>/
  run_manifest.json
  effective_config.json
  run_metadata.json
  results.jsonl
  tasks/fixed_<index>.json
```

WebShop provenance includes the PlanU commit, pinned server commit, sanitized
server URL, config hash, model identifier, seed, task bounds, Python version,
and dependency versions. Reusing an output directory with incompatible
provenance is rejected.

## Reproduction Scope

Smoke runs establish that the software path works; they do not reproduce paper
metrics. Phase-one paper-scale comparison requires the intended 7B/8B
checkpoints, reference trajectory budgets, all five seeds, and adequate GPU
resources. The WebShop scripted `fixed_1` smoke does not validate the
model-backed reference profile or a benchmark-level WebShop metric.

The complete experiment configuration matrices, parity gates, and intentional
non-bit-identical differences are documented in
[Phase-One Experiment Validation](docs/experiment-validation.md).

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

Code adapted from
[LATS](https://github.com/lapisrocks/LanguageAgentTreeSearch) and
[Overcooked-AI](https://github.com/HumanCompatibleAI/overcooked_ai).
