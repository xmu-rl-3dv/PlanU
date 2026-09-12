# Phase-One Experiment Validation

This document separates paper/reference runs from executable smoke runs.
Smoke runs validate the complete software path but do not reproduce paper
metrics because they use a tiny public model and a reduced search budget.

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
then install the three editable packages as shown in the root README. Run
`scripts/smoke_phase_one.sh`.

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
