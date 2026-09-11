# Unified PlanU Core Design

Date: 2026-09-11

## 1. Goal

Create one shared PlanU algorithm implementation for BlockWorld, Overcooked,
and VirtualHome.

The existing Overcooked and VirtualHome experiments are the behavioral
reference because they produced the published results. Benchmark differences
must be expressed through configuration and adapters rather than copied search
implementations.

WebShop and TravelPlanner are deferred to phase two, but the interfaces created
in phase one must support their free-form text actions, tool-backed transitions,
and structured constraint rewards without requiring another search-core
rewrite.

## 2. Non-goals

- Do not modify the environment implementations under
  `gym-macro-overcooked/` or `virtual-home/`.
- Do not maintain separate "published" and "canonical" algorithm modes.
- Do not silently change active Overcooked or VirtualHome experiment behavior.
- Do not migrate WebShop or TravelPlanner during phase one.
- Do not provide every benchmark command in the top-level README.

## 3. Compatibility Policy

There will be one PlanU implementation. Overcooked and VirtualHome preserve
their effective experiment behavior through explicit parameter values:

- quantile update rate;
- number and range of quantiles;
- LLM action-prior enablement;
- action-score normalization;
- RND weight and training schedule;
- action-selection schedule;
- stochastic action class and failure probability;
- horizon and iteration budget.

Behaviorally inactive, duplicated, or unreachable code will not be preserved as
configuration. Examples include unused RAVE state, empty `simulate()` methods,
duplicate method definitions, and transposition values that are never read.

Any behavior-changing correction must be documented in the README and covered
by a focused test. BlockWorld may receive such corrections because its current
entry point is not runnable and its tree does not model repeated stochastic
outcomes correctly.

## 4. Package Layout

```text
PlanU/
  planu_core/
    __init__.py
    config.py
    nodes.py
    distribution.py
    selection.py
    backup.py
    curiosity.py
    interfaces.py
    search.py
    runner.py
    adapters/
      __init__.py
      overcooked.py
      virtualhome.py
      blockworld.py
  tests/
    planu_core/
      test_distribution.py
      test_selection.py
      test_backup.py
      test_stochastic_tree.py
      test_overcooked_adapter.py
      test_virtualhome_adapter.py
      test_blockworld_adapter.py
```

The existing experiment entry points remain at their current paths. They
become thin wrappers that construct an adapter and a `PlanUConfig`, then invoke
the shared runner.

## 5. Tree Model

The common tree alternates state and action nodes:

```text
LanguageNode(state)
  -> ActionNode(action, return distribution)
    -> LanguageNode(observed outcome)
```

`LanguageNode` owns:

- the opaque benchmark state;
- a stable state key supplied by the adapter;
- action children keyed by stable action ID;
- state visit count;
- terminal and truncated flags.

`ActionNode` owns:

- the action ID and display text;
- the LLM action prior;
- the quantile return distribution;
- action visit count;
- outcome children keyed by adapter state key;
- optional RND novelty statistics.

An action may have any number of outcomes. Overcooked and VirtualHome are
expected to produce two common outcomes for injected action failure, but this
is not enforced by the core.

The reward observed during actual execution belongs to the transition record,
not to the state. To reproduce the reference experiments, an adapter may also
provide a separately named `preview_reward` that participates in action
distribution initialization. The core must not confuse that preview with an
executed transition reward.

## 6. Core Interfaces

### EnvironmentAdapter

```python
reset(seed) -> State
clone(state) -> State
step(state, action, rng) -> TransitionResult
state_key(state) -> Hashable
is_terminal(state) -> bool
```

`TransitionResult` contains `next_state`, `reward`, `terminated`, `truncated`,
and benchmark-specific `info`.

The core treats states as opaque values. Gym environment copying, textual world
models, and HTTP or tool sessions remain adapter concerns.

### ActionProvider

```python
get_actions(state) -> list[ActionCandidate]
```

Each candidate contains a stable ID, environment payload, display text, and
optional metadata.

### ActionScorer

```python
score(state, actions) -> ActionScores
```

The initial implementation preserves teacher-forced candidate likelihood
scoring from Overcooked and VirtualHome. A null scorer returns a uniform prior.
The interface also supports free-form action generation for WebShop and
TravelPlanner in phase two.

### CuriosityProvider

```python
score(state) -> float
observe(state) -> None
train() -> metrics
```

The first implementation wraps the existing RND model. A null implementation
returns zero.

## 7. Selection

The common score is:

```text
score(s, a) =
    distort(action.quantiles)
    + curiosity_weight * normalized_novelty(s, a)
```

`distort` defaults to the arithmetic mean and may be configured for upper- or
lower-tail CVaR.

The RND term uses observed successor-state novelty and the same decay behavior
as the reference experiment. Selection is deterministic argmax unless the
benchmark configuration specifies the Overcooked early-iteration sampling
schedule.

The currently ineffective classical UCT term is not retained: the reference
runners never back up parent `LanguageNode.cum_rewards`, so that term evaluates
to zero. Removing the dead expression preserves effective behavior while
making the actual algorithm explicit.

Tie-breaking and action iteration order must remain deterministic and match the
adapter's action order.

## 8. Distribution Initialization And Backup

Each action has `n_quantiles` midpoint fractions and quantile values.

Phase one preserves the effective experiment initialization:

- scalar action score: initialize all quantiles to that scalar;
- five-level LLM distribution: map its cumulative mass onto the configured
  quantile count without the existing hard-coded value `50`;
- optional `preview_reward` contribution: controlled by configuration so the
  Overcooked and VirtualHome initial values match their experiment setup.

Backup consumes the selected action path and the trajectory reward vector. For
action at index `t`, its target is the undiscounted suffix return:

```text
G_t = sum(rewards[t:])
```

Each quantile is updated with the existing pinball-loss gradient:

```text
delta = G_t - quantile
weight = tau if delta > 0 else tau - 1
quantile += learning_rate * weight * abs(delta)
```

The learning rate and clipping range are configuration values. Overcooked and
VirtualHome retain their experiment-specific values.

The core increments action, state, and outcome visit counters consistently, but
selection uses only counters that affected the reference implementation.

## 9. Search Data Flow

One trajectory follows this sequence:

1. Reset the environment and locate the persistent root state.
2. Expand an unseen state into configured legal actions.
3. Score candidate actions and initialize their distributions.
4. Select an action using the shared selection policy.
5. Execute it through the adapter, including benchmark stochasticity.
6. Merge the observed result into the action's outcome children by state key.
7. Continue until termination or the configured horizon.
8. Back up suffix returns through the selected action nodes.
9. Feed observed states to the curiosity provider and train on its configured
   schedule.
10. Emit trajectory, reward, length, token, and curiosity metrics.

The root and learned distributions persist across trajectories, matching the
Overcooked and VirtualHome experiment structure.

## 10. Benchmark Adapters

### Overcooked

The adapter owns observation decoding, action text, macro-action IDs,
deep-copied Gym transitions, and injected `chop` failure. It preserves the
task-specific early sampling schedule and reward handling.

### VirtualHome

One adapter class supports food and entertainment through task configuration.
Task modules own observation decoding, action templates, action IDs, and the
`open` or `grab` stochastic failure rule.

### BlockWorld

The adapter wraps the existing `WorldModel` and `SearchConfig` concepts. It
uses the common alternating tree and samples the transition each time an action
is executed, allowing both successful and failed outcomes to accumulate under
one action node.

BlockWorld corrections include:

- make the `planu` CLI branch reachable;
- replace undefined node annotations;
- compute risk-neutral quantile value as the arithmetic mean;
- avoid the zero-fast-reward initialization discontinuity;
- align quantile clipping with the configured reward range;
- remove accepted but unimplemented output strategies.

## 11. Error Handling And Reproducibility

- Keep `terminated` and `truncated` separate.
- Never expand a terminal state.
- Treat an empty action set as a truncated leaf with a recorded reason.
- Reject NaN or infinite action scores and quantiles.
- Detect state-key collisions in debug mode.
- Route all Python, NumPy, Torch, and environment randomness through recorded
  seeds.
- Record effective configuration, model identifier, dependency versions, and
  Git commit with every run.
- Keep result directories distinct by benchmark, task, model, seed, and
  configuration hash.

## 12. Verification

Before replacing either reference implementation:

1. Add deterministic unit tests for quantile initialization, distortion,
   selection, suffix-return backup, and stochastic outcome merging.
2. Use fake scorers and tiny fake environments so core tests need no GPU.
3. Capture fixed-input outputs from the current Overcooked and VirtualHome
   implementations.
4. Compare action scores, selected action, quantile updates, and tree shape
   against the unified implementation.
5. Run one seeded smoke experiment for each environment when its dependencies
   and model weights are available.
6. Only switch an existing entry point after its differential checks pass.

Exact full-run reward equality is not guaranteed across GPU or library versions,
so reproduction is defined by matching configuration, deterministic components,
and statistically consistent multi-seed metrics.

## 13. README

The top-level README will:

- explain PlanU's alternating state/action tree;
- explain quantile return modeling and curiosity-based selection;
- describe the adapter boundary;
- list BlockWorld, Overcooked, VirtualHome, WebShop, and TravelPlanner as
  supported or planned benchmarks;
- provide one complete Overcooked installation and execution example;
- point other environments to their adapter modules without duplicating all
  commands.

## 14. Delivery Phases

Phase one:

1. Characterize current Overcooked and VirtualHome behavior.
2. Implement and test the shared core.
3. Migrate Overcooked.
4. Migrate VirtualHome food and entertainment.
5. Migrate and correct BlockWorld.
6. Update README and run available verification.

Phase two:

1. Import WebShop through a session-state adapter.
2. Add TravelPlanner as a tool-action and structured-constraint adapter.
3. Add their benchmark-specific evaluation and documentation.
