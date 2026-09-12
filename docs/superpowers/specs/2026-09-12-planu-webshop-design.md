# PlanU WebShop Migration Design

Date: 2026-09-12

## 1. Goal

Migrate the repository's WebShop experiment to the shared `planu_core`
implementation and make the experiment reproducible against a pinned official
WebShop server.

The migration must preserve the effective benchmark settings and interaction
semantics that are recoverable from the existing launcher. The broken,
duplicated search implementation in `webshop/planu.py` may be corrected to use
the unified PlanU tree, quantile update, UCC selection, stochastic outcome
model, and suffix-return backup.

## 2. Baselines And Version Pins

The migration has two independently versioned sources:

- PlanU migration baseline: repository commit `5c0803d`.
- WebShop environment: `https://github.com/princeton-nlp/WebShop.git` commit
  `64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd`.

The official WebShop server runs outside the PlanU Python environment. Its
documented Python 3.8 environment and data setup are isolated from the PlanU
Python 3.9 experiment environment. PlanU communicates with the server only
through HTTP.

The effective legacy PlanU launch profile is:

| Setting | Value |
| --- | --- |
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

The task range is half-open because the legacy launcher passes
`--task_start_index 1 --task_end_index 50`. The new CLI keeps half-open range
semantics and records both bounds in provenance.

## 3. Non-goals

- Do not preserve the private `Node`, selection, rollout, or backup
  implementations in `webshop/planu.py`.
- Do not reproduce invalid control flow, undefined variables, global mutable
  search state, or inconsistent return signatures.
- Do not vendor the official WebShop server or its product data into this
  repository.
- Do not require a paid model for unit tests or the executable smoke test.
- Do not claim that a tiny-model smoke run reproduces paper metrics.
- Do not alter Overcooked, VirtualHome, or BlockWorld behavior while extending
  the core.

## 4. Architecture

The shared search remains the only PlanU implementation:

```text
PlanUSearch
  -> ActionProvider
  -> ActionScorer
  -> WebShopAdapter
       -> WebShopHttpClient
       -> official WebShop server
```

`WebShopAdapter` owns benchmark state, legal transition validation, HTTP
interaction, reward extraction, terminal classification, and latency failure
sampling. `WebShopActionProvider` owns free-form candidate generation and
parsing. `WebShopActionScorer` owns candidate value estimation.

The old `webshop/run.py` and `webshop/planu.py` paths remain as compatibility
entry points. They become thin wrappers or re-exports and do not own search
state.

## 5. Shared Core Extension

Phase one exposes action enumeration through
`EnvironmentAdapter.actions(state, state_visit_count)`. WebShop and
TravelPlanner need model-generated candidates that are not environment
responsibilities.

Add an optional protocol:

```python
class ActionProvider(Protocol):
    def actions(
        self,
        state: EnvironmentState,
        state_visit_count: int = 0,
    ) -> Sequence[ActionCandidate]:
        ...
```

`PlanUSearch` accepts `action_provider=None`. When omitted, it delegates to
`adapter.actions`, preserving all phase-one call sites and behavior. When
provided, expansion calls the provider instead.

Action generation and action scoring remain separate:

- the provider generates a finite, ordered, deduplicated candidate set;
- the scorer returns one finite scalar per candidate;
- scalar scores initialize each candidate's quantile distribution;
- the adapter alone determines transition reward and termination.

No benchmark-specific branch is added to `PlanUSearch`.

## 6. WebShop State And Actions

`WebShopState` is represented inside `EnvironmentState` and contains:

- session ID;
- page type;
- search query;
- page number;
- selected ASIN;
- selected product options;
- current subpage;
- visible observation text;
- available buttons, ASINs, and option labels;
- step count;
- termination and truncation markers.

The state key is a canonical tuple of all transition-relevant fields. The
session ID is included so independent tasks cannot share a root accidentally.
Transient HTTP objects, loggers, and model clients are excluded.

The provider emits `ActionCandidate` objects for:

- `search[query]`;
- `click[label]`;
- `think[text]`.

Candidate keys use normalized action type and payload. The adapter validates
click targets against the buttons and options in the current state. Duplicate
normalized actions are rejected before expansion.

Model-generated malformed or unavailable actions are benchmark outcomes, not
infrastructure failures. They produce an `Invalid action!` observation and
reward `-1`; quantile updates clip that target to the configured `[0, 1]`
support.

## 7. HTTP And Snapshot Semantics

`WebShopHttpClient` receives its endpoint from `WEBSHOP_URL` or an explicit CLI
option. It has bounded connect and read timeouts and raises contextual errors
for unreachable servers, non-2xx responses, and malformed pages.

Environment state is reconstructed from immutable local session data. Preview
and step requests operate on cloned state; neither may mutate a state already
stored in the tree. The official routes are called with the same session,
page, query, ASIN, and option values as the legacy client.

HTML is parsed with a structured parser. Visible text, button labels, product
links, option labels, and final reward are extracted without depending on
process-global state.

Reaching the `done` page terminates the trajectory for every reward in
`[0, 1]`. The legacy code terminated only when reward equaled `1`, which
incorrectly continued partial-credit purchases and is not preserved.

The PlanU depth limit is a truncation, not a successful terminal state.

## 8. Stochastic Failure Model

The adapter samples latency from:

```text
latency_ms ~ LogNormal(mu=0, sigma=10)
success = latency_ms <= 200
```

The random draw uses the explicit NumPy generator supplied by `PlanUSearch`.
The model applies to non-search, non-`Buy Now` actions, matching the effective
legacy rollout condition. A failed action:

- returns the pre-action state;
- emits a latency-failure marker in transition info;
- yields zero environment reward;
- is stored as an outcome under the selected action;
- increments an episode-local failure counter.

`search[...]` and `click[Buy Now]` are never latency-failed. Multiple observed
success and failure outcomes remain attached to one action node through their
state keys.

## 9. PlanU Configuration

The WebShop profile uses:

```text
n_quantiles = 51
value_min = 0.0
value_max = 1.0
quantile_learning_rate = 0.9
discount = 1.0
curiosity_weight = 0.0
include_preview_reward = true
categorical_initialization = false
max_depth = 10
max_iterations = 10
risk_distortion = 0.0
```

The single depth limit replaces the legacy split between depth-10 expansion
and a separate depth-20 rollout. The separate rollout is removed because it
duplicates tree traversal and does not use the common PlanU algorithm.

The legacy action nodes initialized every quantile to `0.5` while a separate
LLM value selected rollout candidates. The migration removes that disconnected
rollout policy: the scorer's normalized value in `[0, 1]` initializes the
quantiles directly, after which all selection and learning use the shared UCC
and backup path.

Because WebShop has zero intermediate task reward and `discount=1`, shared
suffix-return backup equals the legacy terminal-return target while correctly
updates action nodes only.

## 10. Model Boundary

The generation and scoring clients share one backend abstraction with:

- explicit model identifier, base URL, timeout, retry limit, and temperature;
- no module-level network client;
- no model calls during import;
- bounded retries only for declared transient transport failures;
- token accounting returned as structured metrics.

The reference profile preserves `qwen-plus`, temperature `0.8`, CoT prompts,
five generated candidates, and one value sample. The smoke profile uses a
deterministic scripted provider and scorer so it validates the environment and
search without credentials. A local Hugging Face provider may be added only if
it implements the same `ActionProvider` and `ActionScorer` contracts.

## 11. Runner And Outputs

The runner exposes:

- server URL;
- model backend and base URL;
- task start and exclusive end;
- seed;
- iteration and depth overrides;
- output directory;
- request timeout;
- smoke mode.

Each task writes a structured result containing:

- task/session ID;
- best terminal reward;
- success indicator;
- selected trajectory;
- latency failure count;
- search iterations and depth;
- token usage;
- error status;
- provenance.

Provenance includes the PlanU commit, WebShop commit, config hash, model
identifier, server URL without credentials, seed, Python version, dependency
versions, and task bounds.

## 12. Security And Failure Handling

All hard-coded credentials are removed from `webshop/planu.py` and
`webshop/models.py`, including commented credentials. Credentials are accepted
only through environment variables and are never written to provenance or
logs. The exposed credentials must be treated as compromised and rotated
outside this repository.

Infrastructure failures abort with a nonzero exit status:

- missing server;
- HTTP timeout or invalid response;
- unavailable configured model;
- missing credentials for a credentialed backend;
- malformed scorer output;
- persistence failure.

Expected benchmark failures remain normal transitions:

- invalid generated action;
- sampled latency failure;
- partial-credit purchase;
- depth truncation.

Search expansion remains atomic. If provider, scorer, preview, or state-key
construction fails, the tree returns to its pre-expansion state.

## 13. Verification

Unit and contract tests cover:

- optional `ActionProvider` fallback compatibility;
- provider ordering, deduplication, and action parsing;
- complete state-key coverage and snapshot isolation;
- HTML parsing for every page type;
- transition and terminal semantics;
- deterministic latency outcomes under fixed seeds;
- arbitrary stochastic outcomes under one action;
- quantile initialization and learning rate `0.9`;
- depth truncation;
- model-client retry and token accounting;
- absence of credential-shaped literals;
- runner configuration and provenance.

Reference parity tests compare the new adapter against captured official
server responses for reset, search, item, option, subpage, back, and purchase
transitions.

The real smoke test must:

1. start or connect to the pinned official small WebShop server;
2. wait for an explicit health check;
3. reset a real `fixed_*` session;
4. run one complete task through `PlanUSearch`;
5. execute at least one real HTTP action and one backup;
6. persist result and provenance;
7. stop only processes it started;
8. exit nonzero if any real dependency was replaced by a mock.

The phase-two regression suite also reruns the existing Overcooked,
VirtualHome food, VirtualHome entertainment, and BlockWorld smoke paths.

## 14. Acceptance Criteria

The WebShop migration is complete when:

- the runner uses `PlanUSearch` and contains no private tree or backup;
- the effective legacy profile is represented by explicit configuration;
- the documented algorithm corrections are covered by tests;
- no credential remains in tracked source, and the previously exposed
  credentials are documented as requiring external rotation;
- the pinned official server completes a real single-task smoke run;
- the complete unit suite and all phase-one smoke experiments still pass;
- smoke results are clearly separated from paper-scale metric claims.
