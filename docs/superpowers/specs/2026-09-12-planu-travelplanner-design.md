# PlanU TravelPlanner Migration Design

Date: 2026-09-12

## 1. Goal

Add TravelPlanner two-stage tool-use as a new benchmark adapter over the shared
PlanU search. The migration must preserve the official tool and output
semantics while keeping search, quantile learning, UCC selection, stochastic
outcomes, and suffix-return backup in `planu_core`.

TravelPlanner was not an original PlanU experiment. Its configuration is
therefore a documented phase-two migration profile, not a claim that a
TravelPlanner score appeared in the PlanU paper.

## 2. Baselines And Version Pins

The migration pins:

- PlanU migration baseline: repository commit `5c0803d`.
- TravelPlanner source:
  `https://github.com/OSU-NLP-Group/TravelPlanner.git` at
  `e52c87f4ac348a3410c46dc3553c519db5ec5e23`.
- Hugging Face dataset:
  `osunlp/TravelPlanner@8736504ecfc31b7f8b7e40122873c337e83fff7c`.

The official database archive linked by the pinned TravelPlanner README is an
external runtime dependency. It is not committed to this repository. The
bootstrap and validation commands record a content manifest and checksums for
the extracted database files in each run's provenance.

The official `sole-planning` implementation remains a baseline and evaluator
input path. It does not receive a second PlanU adapter. The migration targets
the two-stage process: gather information with tools, maintain a notebook, and
submit a complete plan.

## 3. Non-goals

- Do not vendor the full TravelPlanner repository, database, images, generated
  outputs, or model weights.
- Do not import or copy the upstream `ReactAgent` control loop.
- Do not implement a separate MCTS or quantile tree for TravelPlanner.
- Do not optimize directly against validation or test evaluator feedback.
- Do not use hidden evaluator rules or structured test constraints in model
  prompts.
- Do not claim that the migration profile reproduces a PlanU-paper
  TravelPlanner metric.
- Do not require a paid API for unit tests or smoke validation.

## 4. Architecture

The PlanU process and the upstream runtime are isolated:

```text
PlanUSearch
  -> TravelPlannerActionProvider
  -> TravelPlannerActionScorer
  -> TravelPlannerAdapter
       -> TravelPlannerToolBridge
            -> pinned official tools and database
       -> TerminalRewardProvider
```

The shared optional `ActionProvider` interface is defined in the WebShop
design. TravelPlanner uses the same interface. Existing phase-one adapters
continue to enumerate their own static actions.

The upstream repository is discovered through `TRAVELPLANNER_ROOT` or an
explicit CLI option. The PlanU process does not import
`agents/tool_agents.py`: that module changes the working directory, mutates
`sys.path`, initializes model clients, and combines generation with transition
logic.

Instead, a small JSON-lines bridge runs in a separate TravelPlanner
environment with its working directory fixed to the pinned checkout. The
bridge imports the official database tool classes and exposes only typed,
stateless tool calls. This preserves official query behavior while isolating
dependency and working-directory side effects.

`TerminalRewardProvider` is a separate protocol:

```python
class TerminalRewardProvider(Protocol):
    def score(
        self,
        query: Mapping[str, Any],
        notebook: Sequence[Mapping[str, Any]],
        plan: Sequence[Mapping[str, Any]],
    ) -> float:
        ...
```

The adapter invokes it only for a valid `SubmitPlan`. Training and model-based
providers implement the same contract, but dataset policy determines which
implementation can be constructed.

## 5. State Model

`TravelPlannerState` is represented inside `EnvironmentState` and contains:

- dataset revision, split, and query ID;
- the natural-language query and public query fields;
- ordered action and observation history;
- notebook entries;
- the latest tool result;
- per-tool retry counters;
- consecutive-action history;
- current step;
- final submitted plan, if any;
- termination and truncation markers.

Notebook entries and tool results are normalized to JSON-compatible values.
DataFrame row order and column order are preserved. Runtime clients, live
subprocess handles, model objects, and evaluator functions are excluded.

The state key is a canonical hash of all transition-relevant fields. Two
states with different notebook contents, latest tool results, retry counters,
or step numbers cannot merge.

The default maximum depth is `30`, matching the official two-stage agent's
effective step limit. Reaching it is a truncation.

## 6. Action Model

The provider emits three typed action families:

```text
ToolCall(tool_name, arguments)
NotebookWrite(short_description)
SubmitPlan(plan)
```

`ToolCall` supports the official information-gathering actions:

- `FlightSearch[departure, destination, date]`;
- `AttractionSearch[city]`;
- `AccommodationSearch[city]`;
- `RestaurantSearch[city]`;
- `CitySearch[state]`;
- `GoogleDistanceMatrix[origin, destination, mode]`.

`NotebookWrite` records the latest tool result with a short description.
`SubmitPlan` replaces the upstream `Planner[query]` control-flow action. Its
payload is the complete plan generated from the query and notebook.

Actions are parsed into typed payloads before they reach the adapter. Stable
keys use canonical action type and normalized arguments. Duplicate candidates
are removed while preserving provider order.

Malformed, unsupported, or invalid-argument model actions produce explicit
observations and update the corresponding retry counter. They do not crash the
search. Repeating one invalid action three consecutive times or exceeding
three retries for one tool truncates that trajectory, preserving the official
agent's effective limits.

## 7. Tool Bridge

The bridge accepts one JSON object per request and returns one JSON object:

```text
request:  request_id, tool, arguments
response: request_id, ok, result, error_type, error_message
```

Only the six read-only information tools are exposed. Notebook state stays in
the PlanU adapter. Planner model calls and evaluator calls are not bridge
operations.

The bridge validates:

- the upstream Git commit;
- database directory and required files;
- supported tool names;
- argument count and types;
- date format;
- city membership;
- JSON-serializable output.

Tool calls are deterministic for identical arguments and database contents.
They are cached by tool name, normalized arguments, source commit, and database
manifest hash. The cache contains public tool results only and never stores API
credentials.

Preview may query this read-only cache or bridge on a cloned state. It must not
write notebook entries or mutate a stored tree state.

## 8. Candidate Generation And Scoring

`TravelPlannerActionProvider` receives the public query, action history,
notebook summaries, latest tool observation, and current step. It returns a
finite ordered candidate set. The prompt contains the official public action
schema but no evaluator implementation, validation labels, test labels, or
hidden constraint-derived hints.

`TravelPlannerActionScorer` returns one finite value in `[0, 1]` for each
candidate. Scores initialize the candidate quantiles. The provider and scorer
may share a model backend, but they expose separate interfaces and token
accounting.

The reference model matrix remains configurable and includes the official
two-stage backends. The smoke profile uses a deterministic scripted provider
and scorer to avoid credentials while still executing the real database
tools, adapter, search, and backup.

Provider output must be either typed JSON or text accepted by the strict
official-action parser. A provider cannot invoke a tool itself.

## 9. Terminal Plan And Rewards

`SubmitPlan` terminates the trajectory after strict conversion to the official
daily-plan JSON schema. Invalid or incomplete syntax yields a normal failed
submission with reward `0`; it does not invoke an LLM postprocessor silently.

Terminal reward is split by dataset policy:

- **Training:** the official commonsense and hard-constraint evaluator may
  return a binary reward of `1` only when all applicable constraints pass;
  otherwise it returns `0`.
- **Validation:** search uses a configured model-based terminal scorer that
  sees only the public query, notebook, and submitted plan. The official
  evaluator runs only after the trajectory is complete and cannot update the
  tree.
- **Test:** search follows the validation rule. Test scoring is performed only
  by the official leaderboard; local hidden-label or reconstructed-evaluator
  feedback is prohibited.

Offline validation metrics remain the official:

- Delivery Rate;
- Commonsense Constraint Micro Pass Rate;
- Commonsense Constraint Macro Pass Rate;
- Hard Constraint Micro Pass Rate;
- Hard Constraint Macro Pass Rate;
- Final Pass Rate.

No weighted private surrogate is reported as an official metric.

## 10. PlanU Configuration

The initial TravelPlanner migration profile is:

```text
n_quantiles = 51
value_min = 0.0
value_max = 1.0
quantile_learning_rate = 0.75
discount = 1.0
curiosity_weight = 0.0
include_preview_reward = false
categorical_initialization = false
max_depth = 30
max_iterations = 10
risk_distortion = 0.0
```

These values make the algorithm explicit and comparable with the shared PlanU
core. They are not attributed to the TravelPlanner paper. The paper-scale
launcher may override model, candidate count, search iterations, and seed, but
every override is written to provenance.

Tool transitions have zero reward. Only `SubmitPlan` supplies terminal reward.
With `discount=1`, suffix-return backup propagates the terminal reward to each
selected action without introducing unsupported per-tool reward shaping.

RND is disabled in the initial profile because the official TravelPlanner
baseline has no RND representation or training schedule. It may be evaluated
later as an explicit ablation, not silently enabled in the reference profile.

## 11. Dataset And Evaluator Isolation

Dataset loading always specifies the pinned revision and requested split. The
runner records split and query ID in every result.

The adapter receives a capability object determined before search:

```text
train       -> training terminal evaluator allowed
validation  -> official evaluator unavailable during search
test        -> official evaluator unavailable locally
```

The capability is enforced structurally. Validation and test adapters are not
constructed with an evaluator callable, so model code cannot obtain feedback
through an accidental branch.

Official validation evaluation runs in a separate process over finalized JSONL
output. The evaluator output is never reloaded into an active search tree.

## 12. Runner And Outputs

The runner exposes:

- TravelPlanner checkout and database paths;
- pinned dataset split and query selection;
- provider/scorer model and endpoint;
- seed;
- candidate count;
- iterations and depth;
- output directory;
- terminal reward policy;
- smoke mode.

Each result contains:

- split and query ID;
- final structured plan;
- full action/observation trace;
- notebook summary;
- terminal or truncation reason;
- search iterations and depth;
- token and tool-call counts;
- training reward when allowed;
- offline metrics in a separate evaluation artifact;
- provenance.

Provenance includes the PlanU commit, TravelPlanner commit, dataset revision,
database manifest hash, config hash, model identifiers, seed, Python versions,
tool-environment dependency versions, and result schema version.

## 13. Security And Failure Handling

API keys are read only from environment variables. They are never accepted as
CLI values, serialized in state, sent to the bridge, or written to logs.

Infrastructure errors abort the run with a nonzero status:

- missing or wrong TravelPlanner checkout;
- missing database files;
- bridge startup or protocol failure;
- unavailable configured model;
- missing credentials for a credentialed backend;
- malformed scorer dimensions or non-finite values;
- dataset revision mismatch;
- output persistence failure.

Expected agent errors remain transitions:

- malformed generated action;
- invalid date, city, mode, or argument count;
- empty tool result;
- invalid notebook write;
- repeated action;
- invalid submitted plan;
- depth or retry truncation.

Provider, scorer, preview, tool execution, and backup failures use the existing
PlanU mutation journal so a failed iteration cannot leave a partial tree
update.

## 14. Verification

Unit and contract tests cover:

- shared `ActionProvider` compatibility and provider ordering;
- typed action parsing and canonical keys;
- complete state keys and deep snapshot isolation;
- notebook write/read semantics;
- every tool request and bridge error class;
- parity between bridge output and official tool output on representative
  database queries;
- retry and repeated-action truncation;
- plan schema validation;
- train/validation/test evaluator capability isolation;
- binary training reward;
- zero intermediate rewards and suffix-return backup;
- configuration and provenance;
- absence of credential leakage.

The real TravelPlanner smoke test must:

1. verify the pinned checkout, dataset revision, and database manifest;
2. load one real training query;
3. start the real tool bridge in its isolated environment;
4. execute at least one successful official database tool call;
5. write the real result to the notebook;
6. execute a complete `SubmitPlan`;
7. run one PlanU iteration and backup;
8. persist the plan, trace, reward, and provenance;
9. exit nonzero if the tool backend or dataset was replaced by a mock.

The smoke plan may be scripted and need not pass all constraints. Its purpose
is complete-path validation, not metric reproduction.

The complete phase-two gate also runs:

- the WebShop real-server smoke test;
- Overcooked real-environment smoke;
- VirtualHome food and entertainment real-environment smoke;
- BlockWorld real-evaluator smoke;
- all Python 3.9 unit tests, package builds, and shell syntax checks.

## 15. Delivery Order

TravelPlanner implementation starts only after the WebShop migration and
shared `ActionProvider` extension pass their tests. Work is then divided into:

1. source/data validation and tool bridge;
2. immutable state and typed actions;
3. provider, scorer, and terminal reward policies;
4. thin runner and official output conversion;
5. real training-sample smoke;
6. full cross-benchmark regression.

Each step is independently testable and may be committed separately.

## 16. Acceptance Criteria

The TravelPlanner migration is complete when:

- the runner uses the same `PlanUSearch` as every other benchmark;
- no upstream agent loop, tree, selection, or backup is copied;
- six official information tools execute through the pinned real database;
- a real training query completes the full tool-to-plan smoke path;
- validation and test search cannot access official evaluator feedback;
- finalized validation output is accepted by the official evaluator;
- all source, dataset, database, model, config, and dependency provenance is
  recorded;
- WebShop and all phase-one experiments continue to pass their real smoke
  tests;
- documentation distinguishes software-path validation from published or
  paper-scale metrics.
