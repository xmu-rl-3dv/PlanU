# Experiment Validation

This document separates paper/reference runs from executable smoke runs.
Smoke runs validate the complete software path but do not reproduce paper
metrics because they use a tiny public model and a reduced search budget.
For WebShop, the smoke uses a deterministic scripted provider against the
pinned real server instead of a paid model backend.

## Reference configurations

| Experiment | Effective reference configuration |
| --- | --- |
| Overcooked tomato salad | `Overcooked-LLMA-v4`, task `0`, rewards `0.2/1/-0.1/-0.001`, chop failure `0.5`, depth `15`, 1000 trajectories, seeds `1,10,20,30,40`, 51 quantiles, quantile learning rate `0.75`, RND weight `0.5`, `Neko-Institute-of-Science/LLaMA-7B-HF`, token normalization |
| VirtualHome food | `VirtualHome-v1`, open failure `0.2`, depth `15`, 1000 trajectories, seeds `1,10,20,30,40`, constant action prior, 51 quantiles, quantile learning rate `0.7`, RND weight `0.25` |
| VirtualHome entertainment | `VirtualHome-v2`, grab failure `0.2`, depth `15`, 1000 trajectories, seeds `1,10,20,30,40`, 51 quantiles, quantile learning rate `0.7`, RND weight `0.1`, `meta-llama/Meta-Llama-3.1-8B-Instruct`, token normalization, temperature `1.0` |
| BlockWorld | PlanBench commit `34e6841`, action success probability `0.8`, four-shot shuffled prompt, 51 quantiles, quantile learning rate `0.75`, range `[-10,100]`, risk distortion `0`, 10 search iterations by default, depth `10` for 2/4/6-step splits and `12` otherwise |

The Overcooked legacy shell passed Meta-Llama and word normalization, but the
active legacy implementation ignored both values and used Neko LLaMA-7B with
token normalization. The reference launcher preserves the values that actually
produced the original results.

BlockWorld accepts `--model` so the paper model matrix can be selected without
editing source. Its default remains
`deepseek-ai/DeepSeek-R1-Distill-Llama-8B`.

## Algorithm correspondence

All phase-one runners use the same `PlanUSearch`, alternating State/Action
tree, quantile distribution, UCC selection, and suffix-return pinball backup.
The benchmark profiles retain the effective action priors, stochastic action,
reward transformation, quantile learning rate, RND weight, and search budget
from the original runnable implementations.

The following differences mean seeded trajectories are not expected to be
bit-identical:

- random draws use explicit NumPy generators instead of process-global Python
  and Torch RNG state;
- state keys include hidden environment counters and reward flags so a tree
  node represents a complete Markov state;
- environment time limits are recorded as truncations rather than successful
  terminal states.

These changes do not alter the configured transition probabilities or reward
rules. Paper-scale metric comparison still requires the original checkpoints,
GPU resources, and all five seeds.

## Executable validation

Install `requirements-experiments.txt` with
`requirements-experiments-lock.txt` as its resolved Python 3.9 constraint set,
including `openai==1.109.1` and its exact transitive dependencies. The lock
contains 144 exact pins: the prior 143 resolved distributions plus the
explicit build-tool pin `setuptools==66.1.1`, which is also present in the
direct requirements. Then install the three editable packages as shown in the
root README and run `scripts/smoke_phase_one.sh`. Full lock equality excludes
only `pip`, `wheel`, and the three editable local distributions; no registry
dependency may be omitted. The
OpenAI-compatible client is constructed lazily; dependency validation does
not require a real key or API request.

The script executes:

1. Overcooked with the real V4 environment, Hugging Face action scorer, RND,
   shared search, and backup.
2. VirtualHome food with the real V1 environment, constant prior, RND, shared
   search, and backup.
3. VirtualHome entertainment with the real V2 environment, Hugging Face action
   scorer, RND, shared search, and backup.
4. BlockWorld with one real PlanBench case, Hugging Face world model and fast
   reward, shared search, evaluator, provenance, and pickle persistence.

The smoke model defaults to
`hf-internal-testing/tiny-random-LlamaForCausalLM`. Override `SMOKE_MODEL` only
to test another checkpoint; use the reference launchers for paper-scale runs.

## Automated parity gates

- Overcooked prompt/action parity: 3,618 generated cases.
- VirtualHome food prompt/action parity: 512 generated cases.
- VirtualHome entertainment prompt/action parity: 8,456 generated cases.
- Quantile initialization, UCC selection, suffix-return backup, RND settings,
  terminal handling, and reference launcher values are covered by unit tests.
- Effective configs and dependency versions are written to TensorBoard or
  JSON provenance before evaluation.

## WebShop server setup

WebShop uses two isolated runtimes. PlanU and the shared search run under
Python 3.9. The official server runs under Python 3.8.13 with Java 11 and
Pyserini. The server checkout is pinned exactly to
`princeton-nlp/WebShop@64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd`.

From the PlanU repository root, bootstrap the external server with:

```bash
export WEBSHOP_ROOT="$PWD/external/WebShop"
export WEBSHOP_ENV_PREFIX="$WEBSHOP_ROOT/.conda-planu"
bash scripts/bootstrap_webshop.sh
```

The bootstrap validates the official remote, exact detached commit, Python
3.8.13, Java 11, Flask 2.1.2, Werkzeug 2.1.2, the small dataset, the Lucene
index, and server startup. It installs the exact direct pins from
`requirements-webshop-server.txt` with
`requirements-webshop-server-lock.txt` as `PIP_CONSTRAINT`. The latter is the
complete 138-distribution freeze of the validated server environment, including
the exact pip, setuptools, and wheel build tools; Conda-origin `file://`
references are normalized to versions. The direct transformer pin is 4.30.2
because the upstream 4.19.2 tokenizers dependency has no compatible macOS arm64
wheel. The three small data files must match the documented SHA-256 values
before their derived index is accepted. The setup marker stores the lock
SHA-256, so a lock change invalidates it. The external checkout, environment,
downloaded data, and indexes are runtime dependencies and are not repository
inputs. On the validated macOS arm64 environment, server `pip check` reports
`torch 1.11.0 is not supported on this platform` for the Conda-installed
build; the exact distribution remains in the lock and is validated because it
imports and the authoritative server smoke passes.

## WebShop reference configuration

This is the exact effective reference profile recovered from the legacy
launcher and represented by `webshop/planu.sh` and
`python -m planu_core.webshop.runner`:

| Setting | Value |
| --- | --- |
| Official server | `princeton-nlp/WebShop@64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd` |
| Backend | `qwen-plus` |
| Temperature | `0.8` |
| Prompt mode | `cot` |
| Generated candidates per expansion | `5` |
| Evaluation samples per candidate | `1` |
| Search iterations | `10` |
| Expanded tree depth | `10` |
| Task range | `fixed_1` through `fixed_49` |
| Quantiles | `51` midpoint quantiles |
| Quantile range | `[0, 1]` |
| Quantile learning rate | `0.9` |
| Stochastic latency distribution | log-normal `mu=0`, `sigma=10` milliseconds |
| Latency threshold | `200` milliseconds |

The range is half-open:
`--task-start-index 1 --task-end-index 50` selects `fixed_1` through
`fixed_49`. The remaining shared-core values are discount `1.0`, curiosity
weight `0.0`, preview reward enabled, categorical initialization disabled, and
risk distortion `0.0`. Model credentials and the compatible API base URL are
provided through environment variables and are excluded from artifacts.
Historical credentials in Git history remain out of scope for history
rewriting and must be rotated with the external provider.

## Intentional corrections from the legacy WebShop implementation

The migration preserves the effective profile above, but it does not preserve
broken or disconnected control flow:

- A done page is terminal for every valid reward in `[0, 1]`, including a
  partial-credit or zero-reward purchase. The legacy path continued unless the
  reward was exactly `1`.
- One single depth limit of `10` governs shared search. The separate depth-20
  rollout and its duplicate private tree traversal were removed.
- The action scorer initializes candidate quantiles directly. This replaces
  fixed `0.5` quantiles plus a disconnected scorer-selected rollout.
- Every completed trajectory uses the shared suffix-return quantile backup
  with learning rate `0.9`; action nodes, rather than state nodes, own return
  distributions.
- Stochastic latency uses the explicit NumPy generator supplied by
  `PlanUSearch`, not process-global random state. Search and Buy Now actions
  remain exempt from latency failure.
- Malformed or unavailable invalid generated actions are benchmark outcomes
  with reward `-1`; their learning target is clipped to the configured
  quantile support `[0, 1]`. Infrastructure failures still abort the run.

These corrections mean unified WebShop trajectories are not expected to be
bit-identical to the legacy private implementation.

## WebShop parity and evidence

Automated contract tests cover official route construction and structured
parsing for init, search, item, subpage, and done pages. Adapter tests cover
reset, search, item selection, option selection, subpage navigation, back and
pagination behavior, purchase rewards, partial-credit termination, invalid
actions, seeded latency outcomes, state snapshots, and shared quantile backup.
Runner tests pin the half-open reference configuration, credential handling,
provenance, resume behavior, and smoke artifact validation.

Those are automated contract tests using controlled responses. The
authoritative real-server smoke separately proves the local official checkout,
Flask/Lucene HTTP path, parser, action provider, adapter, shared
`PlanUSearch`, one suffix-return backup, and persistence for `fixed_1`.
Real-server option, subpage, back, and pagination coverage, the model-backed
provider/scorer, tasks after `fixed_1`, and benchmark-level quality are not
established by the smoke.

## WebShop real smoke

Run from a clean PlanU source checkout after bootstrapping the pinned server:

```bash
export PLANU_PYTHON="$PWD/.venv/bin/python"
export WEBSHOP_ROOT="$PWD/external/WebShop"
export WEBSHOP_ENV_PREFIX="$WEBSHOP_ROOT/.conda-planu"
bash scripts/smoke_webshop.sh
```

`RUN_ROOT` is optional. When it is unset, the script creates a unique temporary
directory and prints it; when it is set, it selects the output directory. It
is not an authoritative input. The evidence schema is:

- `<run-root>/run_manifest.json`
- `<run-root>/effective_config.json`
- `<run-root>/run_metadata.json`
- `<run-root>/results.jsonl`
- `<run-root>/tasks/fixed_1.json`
- `<run-root>/webshop-server.log`
- `<run-root>/server_runtime.json`

Before starting the server, the smoke executes `WEBSHOP_PYTHON` and requires
exactly Python 3.8.13, Flask 2.1.2, and Werkzeug 2.1.2. The atomic
`server_runtime.json` records those versions, the Java 11 version string, the
pinned WebShop commit, the complete normalized package mapping, and its
deterministic environment SHA-256. Final artifact validation recomputes that
hash, verifies every lock distribution, and checks the recorded lock SHA-256.
Client run metadata separately retains the selected `packages` field while
adding the complete canonical installed-distribution mapping and its SHA-256;
both full-map fields are part of `RUN_IDENTITY` during resume.

The final authoritative smoke facts for this migration are limited to one
task: exit `0`, `task_id: fixed_1`, `model_id: scripted`,
`http_transition_count: 3`, `quantile_backup_count: 1`,
`best_terminal_reward: 0.0`, and `success: false`. The HTTP trajectory is
`search[product] -> click[ASIN] -> click[Buy Now]`, against the exact official
server commit above. The zero reward is expected from the generic scripted
purchase. This smoke validates execution and provenance; it does not reproduce
paper metrics and does not support a success claim for the model-backed
reference profile or any task beyond `fixed_1`.
