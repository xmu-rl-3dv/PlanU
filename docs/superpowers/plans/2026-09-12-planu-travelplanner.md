# PlanU TravelPlanner Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible TravelPlanner two-stage tool-use experiment backed by the shared PlanU core, pinned official tools/data, and evaluator isolation.

**Architecture:** Run official TravelPlanner database tools behind a JSON-lines subprocess bridge, while immutable search state, typed actions, model providers, terminal reward policy, and experiment orchestration live in `planu_core`. Training may use official terminal feedback; validation and test search are structurally denied evaluator access.

**Tech Stack:** Python 3.9, NumPy, pandas, Hugging Face Datasets, JSON Lines, subprocess, pytest, Bash, official `OSU-NLP-Group/TravelPlanner@e52c87f`.

**Design:** `docs/superpowers/specs/2026-09-12-planu-travelplanner-design.md`

**Prerequisite:** Complete
`docs/superpowers/plans/2026-09-12-planu-webshop.md`, including the shared
`ActionProvider` extension and its phase-one regression gate.

---

## File Map

**Create**

- `planu_core/travelplanner/__init__.py`: public TravelPlanner API.
- `planu_core/travelplanner/actions.py`: typed actions and strict parsing.
- `planu_core/travelplanner/source.py`: source, database, and dataset pins.
- `planu_core/travelplanner/bridge.py`: JSON-lines bridge client and protocol.
- `planu_core/travelplanner/tool_bridge_server.py`: isolated official tool host.
- `planu_core/travelplanner/providers.py`: scripted/model action and value policies.
- `planu_core/travelplanner/rewards.py`: terminal reward providers and split policy.
- `planu_core/travelplanner/runner.py`: CLI, search, results, and provenance.
- `planu_core/adapters/travelplanner.py`: state, notebook, and tool transitions.
- `tests/planu_core/test_travelplanner_actions.py`
- `tests/planu_core/test_travelplanner_source.py`
- `tests/planu_core/test_travelplanner_bridge.py`
- `tests/planu_core/test_travelplanner_adapter.py`
- `tests/planu_core/test_travelplanner_providers.py`
- `tests/planu_core/test_travelplanner_rewards.py`
- `tests/planu_core/test_travelplanner_runner.py`
- `requirements-travelplanner-bridge.txt`
- `scripts/bootstrap_travelplanner.sh`
- `scripts/smoke_travelplanner.sh`
- `scripts/smoke_phase_two.sh`

**Modify**

- `planu_core/adapters/__init__.py`: export TravelPlanner adapter/config.
- `planu_core/provenance.py`: include TravelPlanner package versions.
- `setup.py`: package dependencies and optional TravelPlanner extras.
- `README.md`: setup, reference profile, run, smoke, metric caveat.
- `docs/experiment-validation.md`: add TravelPlanner provenance and test matrix.
- `tests/planu_core/test_packaging.py`: package membership and dependencies.
- `tests/planu_core/test_readme.py`: documentation contract.

---

### Task 1: Pin And Validate External Sources

**Files:**
- Create: `planu_core/travelplanner/__init__.py`
- Create: `planu_core/travelplanner/source.py`
- Create: `tests/planu_core/test_travelplanner_source.py`
- Create: `requirements-travelplanner-bridge.txt`
- Create: `scripts/bootstrap_travelplanner.sh`

- [ ] **Step 1: Write failing source-validation tests**

```python
# tests/planu_core/test_travelplanner_source.py
from pathlib import Path

import pytest

from planu_core.travelplanner.source import (
    DATASET_REVISION,
    TRAVELPLANNER_COMMIT,
    database_manifest,
    validate_checkout,
)


def test_pins_are_immutable_hashes():
    assert TRAVELPLANNER_COMMIT == (
        "e52c87f4ac348a3410c46dc3553c519db5ec5e23"
    )
    assert DATASET_REVISION == (
        "8736504ecfc31b7f8b7e40122873c337e83fff7c"
    )


def test_validate_checkout_rejects_missing_database(tmp_path):
    (tmp_path / ".git").mkdir()
    with pytest.raises(RuntimeError, match="database"):
        validate_checkout(tmp_path, git_commit_fn=lambda root: TRAVELPLANNER_COMMIT)


def test_database_manifest_is_path_sorted(tmp_path):
    database = tmp_path / "database"
    (database / "b").mkdir(parents=True)
    (database / "a.txt").write_text("a", encoding="utf-8")
    (database / "b" / "c.txt").write_text("c", encoding="utf-8")

    manifest = database_manifest(database)

    assert [entry["path"] for entry in manifest] == ["a.txt", "b/c.txt"]
    assert all(len(entry["sha256"]) == 64 for entry in manifest)
```

- [ ] **Step 2: Run tests and verify the module is missing**

Run:

```bash
python -m pytest tests/planu_core/test_travelplanner_source.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement pins, validation, and manifesting**

```python
# planu_core/travelplanner/source.py
import hashlib
from pathlib import Path
from typing import Callable, Dict, List


TRAVELPLANNER_COMMIT = "e52c87f4ac348a3410c46dc3553c519db5ec5e23"
DATASET_REVISION = "8736504ecfc31b7f8b7e40122873c337e83fff7c"
REQUIRED_DATABASE_FILES = (
    "background/citySet.txt",
    "background/citySet_with_states.txt",
    "flights/clean_Flights_2022.csv",
    "accommodations/clean_accommodations_2022.csv",
    "restaurants/clean_restaurant_2022.csv",
    "attractions/attractions.csv",
    "googleDistanceMatrix/distance.csv",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def database_manifest(database: Path) -> List[Dict[str, object]]:
    entries = []
    for path in sorted(item for item in database.rglob("*") if item.is_file()):
        entries.append(
            {
                "path": path.relative_to(database).as_posix(),
                "size": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return entries
```

Implement `validate_checkout(root, git_commit_fn=...)` to verify the exact Git
commit, required upstream modules, and every required database file. Export
the constants and validation helpers.

Create the isolated bridge requirement set:

```text
numpy==1.26.4
pandas==2.2.3
requests==2.32.5
datasets==4.5.0
tqdm==4.70.1
gdown==5.2.0
```

- [ ] **Step 4: Add the pinned bootstrap script**

```bash
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${TRAVELPLANNER_ROOT:-$PWD/external/TravelPlanner}"
COMMIT="e52c87f4ac348a3410c46dc3553c519db5ec5e23"
DATABASE_ARCHIVE="${TRAVELPLANNER_DATABASE_ARCHIVE:-$ROOT/.cache/database.zip}"
if [[ ! -d "$ROOT/.git" ]]; then
  git clone https://github.com/OSU-NLP-Group/TravelPlanner.git "$ROOT"
fi
git -C "$ROOT" fetch origin "$COMMIT"
git -C "$ROOT" checkout --detach "$COMMIT"
python3.9 -m venv "$ROOT/.planu-bridge-venv"
"$ROOT/.planu-bridge-venv/bin/python" -m pip install \
  -r "$REPO_ROOT/requirements-travelplanner-bridge.txt"
if [[ ! -f "$ROOT/database/flights/clean_Flights_2022.csv" ]]; then
  mkdir -p "$(dirname "$DATABASE_ARCHIVE")"
  if [[ ! -f "$DATABASE_ARCHIVE" ]]; then
    "$ROOT/.planu-bridge-venv/bin/python" -m gdown \
      "https://drive.google.com/uc?id=1pF1Sw6pBmq2sFkJvm-LzJOqrmfWoQgxE" \
      -O "$DATABASE_ARCHIVE"
  fi
  unzip -q -o "$DATABASE_ARCHIVE" -d "$ROOT"
fi
```

After extraction, invoke `validate_checkout` and fail if any required CSV is
absent. Do not silently use the checked-in `*_ref_info.jsonl` files as a tool
database.

- [ ] **Step 5: Verify tests and shell syntax**

Run:

```bash
python -m pytest tests/planu_core/test_travelplanner_source.py -v
bash -n scripts/bootstrap_travelplanner.sh
```

Expected: tests PASS and shell syntax exits `0`.

- [ ] **Step 6: Commit**

```bash
git add planu_core/travelplanner/__init__.py planu_core/travelplanner/source.py tests/planu_core/test_travelplanner_source.py requirements-travelplanner-bridge.txt scripts/bootstrap_travelplanner.sh
git commit -m "feat: pin TravelPlanner sources"
```

---

### Task 2: Define Typed Actions And Plan Schema

**Files:**
- Create: `planu_core/travelplanner/actions.py`
- Create: `tests/planu_core/test_travelplanner_actions.py`
- Modify: `planu_core/travelplanner/__init__.py`

- [ ] **Step 1: Write parser and schema tests**

```python
# tests/planu_core/test_travelplanner_actions.py
import pytest

from planu_core.travelplanner.actions import (
    NotebookWrite,
    PlanDay,
    SubmitPlan,
    ToolCall,
    parse_action,
    validate_plan,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "FlightSearch[Boston, Denver, 2026-10-01]",
            ToolCall("FlightSearch", ("Boston", "Denver", "2026-10-01")),
        ),
        (
            "RestaurantSearch[Denver]",
            ToolCall("RestaurantSearch", ("Denver",)),
        ),
        (
            "NotebookWrite[Denver restaurants]",
            NotebookWrite("Denver restaurants"),
        ),
    ],
)
def test_parse_official_actions(text, expected):
    assert parse_action(text) == expected


def test_submit_plan_requires_official_fields():
    plan = [
        {
            "day": 1,
            "current_city": "from Boston to Denver",
            "transportation": "Flight Number: F1",
            "breakfast": "-",
            "lunch": "Cafe, Denver",
            "dinner": "Bistro, Denver",
            "attraction": "Museum, Denver;",
            "accommodation": "Hotel, Denver",
        }
    ]
    assert validate_plan(plan) == (
        PlanDay(
            day=1,
            current_city="from Boston to Denver",
            transportation="Flight Number: F1",
            breakfast="-",
            lunch="Cafe, Denver",
            dinner="Bistro, Denver",
            attraction="Museum, Denver;",
            accommodation="Hotel, Denver",
        ),
    )


def test_submit_plan_rejects_missing_field():
    with pytest.raises(ValueError, match="accommodation"):
        validate_plan([{"day": 1}])
```

Include cases for all six tools, JSON `SubmitPlan`, invalid dates, invalid
argument counts, unsupported tools, blank notebook descriptions, duplicate
days, and non-JSON values.

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
python -m pytest tests/planu_core/test_travelplanner_actions.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement immutable action types**

```python
# planu_core/travelplanner/actions.py
from dataclasses import asdict, dataclass
import json
from typing import Sequence, Tuple, Union


@dataclass(frozen=True)
class ToolCall:
    tool_name: str
    arguments: Tuple[str, ...]

    @property
    def key(self):
        return "tool", self.tool_name, self.arguments


@dataclass(frozen=True)
class NotebookWrite:
    short_description: str

    @property
    def key(self):
        return "notebook", self.short_description


@dataclass(frozen=True)
class PlanDay:
    day: int
    current_city: str
    transportation: str
    breakfast: str
    lunch: str
    dinner: str
    attraction: str
    accommodation: str


@dataclass(frozen=True)
class SubmitPlan:
    plan: Tuple[PlanDay, ...]

    @property
    def key(self):
        serialized = json.dumps(
            [asdict(day) for day in self.plan],
            sort_keys=True,
            separators=(",", ":"),
        )
        return "submit", serialized


TravelPlannerAction = Union[ToolCall, NotebookWrite, SubmitPlan]
```

Implement strict anchored parsing. `SubmitPlan[...]` contains JSON only; do
not call `eval`. `validate_plan()` requires exactly the official eight fields
for each day and returns a defensive immutable representation.

- [ ] **Step 4: Run action tests**

Run:

```bash
python -m pytest tests/planu_core/test_travelplanner_actions.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add planu_core/travelplanner/actions.py planu_core/travelplanner/__init__.py tests/planu_core/test_travelplanner_actions.py
git commit -m "feat: define TravelPlanner actions"
```

---

### Task 3: Implement The JSON-Lines Tool Bridge

**Files:**
- Create: `planu_core/travelplanner/bridge.py`
- Create: `planu_core/travelplanner/tool_bridge_server.py`
- Create: `tests/planu_core/test_travelplanner_bridge.py`
- Modify: `planu_core/travelplanner/__init__.py`

- [ ] **Step 1: Write protocol and lifecycle tests**

```python
# tests/planu_core/test_travelplanner_bridge.py
import json

import pytest

from planu_core.travelplanner.bridge import (
    BridgeProtocolError,
    TravelPlannerToolBridge,
)


def test_bridge_round_trip(fake_bridge_process):
    fake_bridge_process.queue_response(
        {"request_id": 1, "ok": True, "result": [{"Name": "Museum"}]}
    )
    bridge = TravelPlannerToolBridge(process=fake_bridge_process)

    result = bridge.call("AttractionSearch", ("Denver",))

    assert result == [{"Name": "Museum"}]
    assert json.loads(fake_bridge_process.stdin_lines[0]) == {
        "request_id": 1,
        "tool": "AttractionSearch",
        "arguments": ["Denver"],
    }


def test_bridge_rejects_mismatched_response_id(fake_bridge_process):
    fake_bridge_process.queue_response(
        {"request_id": 99, "ok": True, "result": []}
    )
    bridge = TravelPlannerToolBridge(process=fake_bridge_process)
    with pytest.raises(BridgeProtocolError, match="request_id"):
        bridge.call("AttractionSearch", ("Denver",))


def test_bridge_closes_only_its_process(fake_bridge_process):
    bridge = TravelPlannerToolBridge(process=fake_bridge_process)
    bridge.close()
    assert fake_bridge_process.terminated is True
```

Also test malformed JSON, EOF, timeout, stderr capture, upstream error
propagation, cache key stability, and idempotent close.

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
python -m pytest tests/planu_core/test_travelplanner_bridge.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the client**

```python
# planu_core/travelplanner/bridge.py
class TravelPlannerToolBridge:
    def __init__(self, process, timeout_seconds=30.0):
        self._process = process
        self._timeout_seconds = timeout_seconds
        self._next_request_id = 1
        self._cache = {}

    def call(self, tool_name, arguments):
        key = (tool_name, tuple(arguments))
        if key in self._cache:
            return copy.deepcopy(self._cache[key])
        request_id = self._next_request_id
        self._next_request_id += 1
        self._write_request(request_id, tool_name, arguments)
        response = self._read_response(request_id)
        if not response["ok"]:
            raise ToolExecutionError(
                response["error_type"],
                response["error_message"],
            )
        self._cache[key] = response["result"]
        return copy.deepcopy(response["result"])
```

Add `start(root, python, database_manifest_hash)` to launch:

```bash
python -m planu_core.travelplanner.tool_bridge_server --root ROOT
```

with line-buffered pipes and a bounded ready handshake.

- [ ] **Step 4: Implement the official tool server**

The server validates the checkout before imports, changes its own subprocess
working directory to `<root>/agents`, prepends `<root>` to `sys.path`, and
constructs:

```python
TOOLS = {
    "FlightSearch": Flights(),
    "AttractionSearch": Attractions(),
    "AccommodationSearch": Accommodations(),
    "RestaurantSearch": Restaurants(),
    "CitySearch": Cities(),
    "GoogleDistanceMatrix": GoogleDistanceMatrix(),
}
```

Dispatch exact signatures:

```python
def dispatch(tool_name, arguments):
    if tool_name == "FlightSearch":
        return TOOLS[tool_name].run(*arguments)
    if tool_name in {
        "AttractionSearch",
        "AccommodationSearch",
        "RestaurantSearch",
        "CitySearch",
    }:
        return TOOLS[tool_name].run(arguments[0])
    if tool_name == "GoogleDistanceMatrix":
        return TOOLS[tool_name].run(*arguments)
    raise ValueError("unsupported TravelPlanner tool: {}".format(tool_name))
```

Convert DataFrames to `{"columns": [...], "records": [...]}` while preserving
column and row order. Emit only JSON on stdout; diagnostics go to stderr.

- [ ] **Step 5: Run bridge tests**

Run:

```bash
python -m pytest tests/planu_core/test_travelplanner_bridge.py -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add planu_core/travelplanner/bridge.py planu_core/travelplanner/tool_bridge_server.py planu_core/travelplanner/__init__.py tests/planu_core/test_travelplanner_bridge.py
git commit -m "feat: bridge TravelPlanner tools"
```

---

### Task 4: Implement TravelPlanner State And Adapter

**Files:**
- Create: `planu_core/adapters/travelplanner.py`
- Create: `tests/planu_core/test_travelplanner_adapter.py`
- Modify: `planu_core/adapters/__init__.py`
- Modify: `planu_core/travelplanner/__init__.py`

- [ ] **Step 1: Write adapter behavior tests**

```python
# tests/planu_core/test_travelplanner_adapter.py
import numpy as np

from planu_core.adapters.travelplanner import TravelPlannerAdapter
from planu_core.interfaces import ActionCandidate
from planu_core.travelplanner.actions import NotebookWrite, ToolCall


def test_tool_call_records_result_without_mutating_parent(fake_tool_backend):
    adapter = TravelPlannerAdapter(train_query(), fake_tool_backend)
    state = adapter.reset(seed=7)
    parent_key = adapter.state_key(state)
    action = ToolCall("RestaurantSearch", ("Denver",))

    result = adapter.step(
        state,
        ActionCandidate(action.key, action, "RestaurantSearch[Denver]"),
        np.random.default_rng(7),
    )

    assert adapter.state_key(state) == parent_key
    assert result.state.runtime.latest_tool_result["records"]
    assert result.reward == 0.0
    assert result.terminated is False


def test_notebook_write_copies_latest_result(fake_tool_backend):
    adapter = TravelPlannerAdapter(train_query(), fake_tool_backend)
    state = state_with_latest_result(adapter)
    action = NotebookWrite("Denver restaurants")

    result = adapter.step(
        state,
        ActionCandidate(action.key, action, "NotebookWrite[Denver restaurants]"),
        np.random.default_rng(8),
    )

    assert result.state.runtime.notebook[0]["short_description"] == (
        "Denver restaurants"
    )
    assert result.state.runtime.notebook[0]["content"] is not (
        state.runtime.latest_tool_result
    )
```

Add tests for reset, all six tools, preview isolation, complete state key,
invalid action observations, three repeated actions, per-tool retry limits,
empty notebook writes, submit termination, and depth truncation.

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
python -m pytest tests/planu_core/test_travelplanner_adapter.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the runtime and adapter**

```python
@dataclass
class TravelPlannerRuntime:
    split: str
    query_id: str
    query: Mapping[str, Any]
    history: List[Mapping[str, Any]] = field(default_factory=list)
    notebook: List[Mapping[str, Any]] = field(default_factory=list)
    latest_tool_result: Any = None
    retry_counts: Dict[str, int] = field(default_factory=dict)
    recent_action_keys: List[Hashable] = field(default_factory=list)
    step_count: int = 0
    final_plan: Optional[Tuple[PlanDay, ...]] = None
    terminated: bool = False
    truncated: bool = False
    truncation_reason: Optional[str] = None
```

Implement `TravelPlannerAdapter` with clone-before-mutation and one
`_transition()` path used by preview and step. `actions()` raises a clear
`RuntimeError("TravelPlanner requires an ActionProvider")`.

Tool and notebook transitions return reward `0.0`. `SubmitPlan` delegates to
the injected terminal reward provider and terminates. Model mistakes return
structured observations; bridge protocol and process failures propagate.

Build state keys from a recursively frozen, canonical representation of every
runtime field listed above.

- [ ] **Step 4: Run adapter and core search tests**

Run:

```bash
python -m pytest \
  tests/planu_core/test_travelplanner_adapter.py \
  tests/planu_core/test_search.py \
  tests/planu_core/test_backup.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add planu_core/adapters/travelplanner.py planu_core/adapters/__init__.py planu_core/travelplanner/__init__.py tests/planu_core/test_travelplanner_adapter.py
git commit -m "feat: add TravelPlanner adapter"
```

---

### Task 5: Add TravelPlanner Providers And Scoring

**Files:**
- Create: `planu_core/travelplanner/providers.py`
- Create: `tests/planu_core/test_travelplanner_providers.py`
- Modify: `planu_core/travelplanner/__init__.py`

- [ ] **Step 1: Write provider isolation tests**

```python
# tests/planu_core/test_travelplanner_providers.py
from planu_core.travelplanner.providers import (
    ModelTravelPlannerActionProvider,
    ModelTravelPlannerActionScorer,
    ScriptedTravelPlannerActionProvider,
)


def test_prompt_contains_public_state_but_no_evaluator(fake_backend):
    provider = ModelTravelPlannerActionProvider(fake_backend, candidate_count=2)
    fake_backend.texts = [
        "RestaurantSearch[Denver]",
        "NotebookWrite[restaurants]",
    ]

    provider.actions(public_state(), 0)

    prompt = fake_backend.prompts[0]
    assert "RestaurantSearch" in prompt
    assert "notebook" in prompt.lower()
    assert "commonsense_constraint.py" not in prompt
    assert "hard_constraint.py" not in prompt
    assert "validation label" not in prompt.lower()


def test_scorer_returns_one_bounded_value_per_candidate(fake_backend):
    fake_backend.texts = ['{"scores": [0.25, 0.75]}']
    scorer = ModelTravelPlannerActionScorer(fake_backend)
    assert scorer.score("state", two_candidates()) == [0.25, 0.75]


def test_scripted_provider_executes_tool_write_and_submit():
    provider = ScriptedTravelPlannerActionProvider(smoke_plan())
    assert provider.actions(initial_state(), 0)[0].payload.tool_name == (
        "RestaurantSearch"
    )
    assert provider.actions(state_with_result(), 0)[0].text.startswith(
        "NotebookWrite["
    )
    assert provider.actions(state_with_notebook(), 0)[0].text.startswith(
        "SubmitPlan["
    )
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
python -m pytest tests/planu_core/test_travelplanner_providers.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement providers**

Reuse `TextBackend` from the WebShop migration. Implement:

```python
class ModelTravelPlannerActionProvider:
    def actions(self, state, state_visit_count=0):
        prompt = build_action_prompt(state)
        generated = self.backend.generate(
            prompt,
            n=self.candidate_count,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stop=("\nObservation",),
        )
        return parse_and_deduplicate(generated.texts)
```

The prompt includes only the public query, prior actions/observations, notebook
summary, latest result, current step, and official action grammar.

Implement scorer parsing from exactly:

```json
{"scores": [0.25, 0.75]}
```

Reject wrong lengths, values outside `[0, 1]`, NaN, infinity, and additional
top-level fields. Implement the scripted smoke provider with the sequence
`RestaurantSearch`, `NotebookWrite`, `SubmitPlan`.

- [ ] **Step 4: Run provider tests**

Run:

```bash
python -m pytest tests/planu_core/test_travelplanner_providers.py tests/planu_core/test_webshop_providers.py -v
```

Expected: all tests PASS, including the shared text backend regression.

- [ ] **Step 5: Commit**

```bash
git add planu_core/travelplanner/providers.py planu_core/travelplanner/__init__.py tests/planu_core/test_travelplanner_providers.py
git commit -m "feat: add TravelPlanner action policies"
```

---

### Task 6: Enforce Terminal Reward And Dataset Isolation

**Files:**
- Create: `planu_core/travelplanner/rewards.py`
- Create: `tests/planu_core/test_travelplanner_rewards.py`
- Modify: `planu_core/adapters/travelplanner.py`
- Modify: `planu_core/travelplanner/__init__.py`

- [ ] **Step 1: Write reward-policy tests**

```python
# tests/planu_core/test_travelplanner_rewards.py
import pytest

from planu_core.travelplanner.rewards import (
    DatasetSplit,
    ModelTerminalReward,
    TrainingConstraintReward,
    terminal_reward_for_split,
)


def test_training_reward_is_one_only_when_every_applicable_rule_passes():
    reward = TrainingConstraintReward(
        commonsense=lambda query, plan: {
            "complete": (True, None),
            "sandbox": (True, None),
        },
        hard=lambda query, plan: {
            "budget": (True, None),
            "cuisine": (None, None),
        },
    )
    assert reward.score({}, [], valid_plan()) == 1.0


def test_training_reward_is_zero_for_one_failed_rule():
    reward = TrainingConstraintReward(
        commonsense=lambda query, plan: {"complete": (False, "missing")},
        hard=lambda query, plan: {},
    )
    assert reward.score({}, [], valid_plan()) == 0.0


def test_validation_cannot_receive_official_evaluator():
    with pytest.raises(ValueError, match="validation"):
        terminal_reward_for_split(
            DatasetSplit.VALIDATION,
            official_evaluator=object(),
            model_reward=ModelTerminalReward(fake_backend()),
        )
```

Add equivalent test-set denial, absent model reward, `None` constraints,
invalid plan reward `0`, and no evaluator object reachable from validation
adapter state.

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
python -m pytest tests/planu_core/test_travelplanner_rewards.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement split policy and reward providers**

```python
class DatasetSplit(Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class TrainingConstraintReward:
    def score(self, query, notebook, plan):
        del notebook
        groups = (
            self._commonsense(query, plan),
            self._hard(query, plan),
        )
        applicable = [
            result[0]
            for group in groups
            for result in group.values()
            if result[0] is not None
        ]
        return float(bool(applicable) and all(applicable))
```

`ModelTerminalReward` passes only public query, notebook, and plan to a
`TextBackend`, parses a strict scalar JSON response, and caches by canonical
input hash.

`terminal_reward_for_split()` returns the official reward only for train.
Validation and test require `official_evaluator is None` and a model reward.
Wire the selected provider into `TravelPlannerAdapter`.

- [ ] **Step 4: Run reward and adapter tests**

Run:

```bash
python -m pytest tests/planu_core/test_travelplanner_rewards.py tests/planu_core/test_travelplanner_adapter.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add planu_core/travelplanner/rewards.py planu_core/travelplanner/__init__.py planu_core/adapters/travelplanner.py tests/planu_core/test_travelplanner_rewards.py tests/planu_core/test_travelplanner_adapter.py
git commit -m "feat: isolate TravelPlanner rewards"
```

---

### Task 7: Build The Runner And Official Output

**Files:**
- Create: `planu_core/travelplanner/runner.py`
- Create: `tests/planu_core/test_travelplanner_runner.py`
- Modify: `planu_core/provenance.py`
- Modify: `setup.py`
- Modify: `tests/planu_core/test_packaging.py`

- [ ] **Step 1: Write runner and provenance tests**

```python
# tests/planu_core/test_travelplanner_runner.py
import json

from planu_core.travelplanner.runner import parse_args, run


def test_migration_profile_defaults():
    args = parse_args([])
    assert args.split == "train"
    assert args.iterations == 10
    assert args.depth == 30
    assert args.n_quantiles == 51
    assert args.quantile_learning_rate == 0.75
    assert args.curiosity_weight == 0.0


def test_runner_pins_dataset_revision(tmp_path, fake_dependencies):
    code = run(
        parse_args(
            [
                "--smoke",
                "--query-index", "0",
                "--output-dir", str(tmp_path),
            ]
        ),
        dependencies=fake_dependencies,
    )

    assert code == 0
    metadata = json.loads((tmp_path / "run_metadata.json").read_text())
    assert metadata["travelplanner_commit"].startswith("e52c87f")
    assert metadata["dataset_revision"].startswith("8736504")
    assert metadata["database_manifest_sha256"]
```

Add tests that validation/test reject `--terminal-reward official`, dataset
loading always receives `revision=DATASET_REVISION`, result JSONL uses the
official plan fields, and bridge cleanup runs after model or persistence
errors.

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
python -m pytest tests/planu_core/test_travelplanner_runner.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement CLI and dependency construction**

Construct the approved profile:

```python
config = PlanUConfig(
    n_quantiles=args.n_quantiles,
    value_min=0.0,
    value_max=1.0,
    quantile_learning_rate=args.quantile_learning_rate,
    discount=1.0,
    curiosity_weight=0.0,
    include_preview_reward=False,
    categorical_initialization=False,
    max_depth=args.depth,
    max_iterations=args.iterations,
    risk_distortion=0.0,
)
search = PlanUSearch(
    adapter,
    scorer,
    config,
    action_provider=provider,
)
```

Load data with:

```python
load_dataset(
    "osunlp/TravelPlanner",
    args.split,
    revision=DATASET_REVISION,
)[args.split]
```

Open the bridge in a context manager, run the selected query range, persist
`results.jsonl`, `effective_config.json`, `run_metadata.json`, and close the
bridge on every exit path.

For validation, add a separate `--evaluate-output` command that invokes the
pinned official evaluator only after result finalization. It writes
`offline_evaluation.json` and never passes scores back to `PlanUSearch`.

- [ ] **Step 4: Update packaging**

Add `datasets` and the new package members to packaging assertions. Add:

```python
"travelplanner": [
    "datasets==4.5.0",
    "pandas==2.2.3",
]
```

to optional dependencies without changing phase-one extras.

- [ ] **Step 5: Run runner and packaging tests**

Run:

```bash
python -m pytest tests/planu_core/test_travelplanner_runner.py tests/planu_core/test_packaging.py -v
python setup.py check --strict
```

Expected: all tests PASS and setup validation exits `0`.

- [ ] **Step 6: Commit**

```bash
git add planu_core/travelplanner/runner.py planu_core/provenance.py setup.py tests/planu_core/test_travelplanner_runner.py tests/planu_core/test_packaging.py
git commit -m "feat: run TravelPlanner through PlanU core"
```

---

### Task 8: Add Real TravelPlanner Smoke Validation

**Files:**
- Create: `scripts/smoke_travelplanner.sh`
- Create: `scripts/smoke_phase_two.sh`
- Modify: `tests/planu_core/test_travelplanner_runner.py`

- [ ] **Step 1: Write smoke-script contract tests**

```python
def test_smoke_scripts_require_real_sources():
    root = Path(__file__).resolve().parents[2]
    travel = (root / "scripts" / "smoke_travelplanner.sh").read_text()
    phase_two = (root / "scripts" / "smoke_phase_two.sh").read_text()
    assert "TRAVELPLANNER_ROOT" in travel
    assert "e52c87f4ac348a3410c46dc3553c519db5ec5e23" in travel
    assert "--split train" in travel
    assert "smoke_phase_one.sh" in phase_two
    assert "smoke_webshop.sh" in phase_two
    assert "smoke_travelplanner.sh" in phase_two
```

- [ ] **Step 2: Run the contract test and verify scripts are missing**

Run:

```bash
python -m pytest tests/planu_core/test_travelplanner_runner.py::test_smoke_scripts_require_real_sources -v
```

Expected: FAIL because the scripts do not exist.

- [ ] **Step 3: Implement TravelPlanner smoke**

```bash
#!/usr/bin/env bash
set -euo pipefail

: "${TRAVELPLANNER_ROOT:?set TRAVELPLANNER_ROOT to the pinned checkout}"
BRIDGE_PYTHON="${BRIDGE_PYTHON:-$TRAVELPLANNER_ROOT/.planu-bridge-venv/bin/python}"
RUN_ROOT="${RUN_ROOT:-$(mktemp -d "${TMPDIR:-/tmp}/planu-travelplanner.XXXXXX")}"

test "$(git -C "$TRAVELPLANNER_ROOT" rev-parse HEAD)" = \
  "e52c87f4ac348a3410c46dc3553c519db5ec5e23"

python -m planu_core.travelplanner.runner \
  --smoke \
  --split train \
  --query-index 0 \
  --iterations 1 \
  --depth 3 \
  --travelplanner-root "$TRAVELPLANNER_ROOT" \
  --bridge-python "$BRIDGE_PYTHON" \
  --output-dir "$RUN_ROOT"
```

The scripted provider must execute one real `RestaurantSearch`, one
`NotebookWrite`, and one `SubmitPlan`. The runner asserts that bridge
provenance reports the real source and database manifest.

`scripts/smoke_phase_two.sh` runs, in order:

```bash
scripts/smoke_phase_one.sh
scripts/smoke_webshop.sh
scripts/smoke_travelplanner.sh
```

- [ ] **Step 4: Validate shell syntax**

Run:

```bash
bash -n scripts/smoke_travelplanner.sh
bash -n scripts/smoke_phase_two.sh
```

Expected: both commands exit `0`.

- [ ] **Step 5: Commit**

```bash
git add scripts/smoke_travelplanner.sh scripts/smoke_phase_two.sh tests/planu_core/test_travelplanner_runner.py
git commit -m "test: add real TravelPlanner smoke workflow"
```

---

### Task 9: Run The Real TravelPlanner Experiment Path

**Files:**
- Modify only if a real-run defect is first captured by a focused failing test.

- [ ] **Step 1: Prepare source and database**

Run:

```bash
TRAVELPLANNER_ROOT="$PWD/external/TravelPlanner" scripts/bootstrap_travelplanner.sh
```

Expected: checkout HEAD is
`e52c87f4ac348a3410c46dc3553c519db5ec5e23`, all seven required database files
exist, and the bridge environment imports pandas and datasets.

- [ ] **Step 2: Execute the real smoke**

Run:

```bash
TRAVELPLANNER_ROOT="$PWD/external/TravelPlanner" \
RUN_ROOT=/tmp/planu-travelplanner-smoke \
scripts/smoke_travelplanner.sh
```

Expected: exit `0`; one real training query executes
`RestaurantSearch -> NotebookWrite -> SubmitPlan`, one PlanU backup completes,
and result/provenance files are written.

- [ ] **Step 3: Inspect persisted evidence**

Run:

```bash
python -m json.tool /tmp/planu-travelplanner-smoke/effective_config.json
python -m json.tool /tmp/planu-travelplanner-smoke/run_metadata.json
python -c "import json; row=json.loads(open('/tmp/planu-travelplanner-smoke/results.jsonl').readline()); print(row['split'], row['query_id'], row['tool_call_count'])"
```

Expected: `train`, a real query ID, and tool call count at least `1`.

- [ ] **Step 4: Run bridge parity probes**

Run representative flight, restaurant, attraction, accommodation, city, and
distance calls through both the bridge and the pinned upstream classes. Compare
the normalized JSON payloads byte-for-byte.

Expected: all six normalized outputs match.

- [ ] **Step 5: Convert real defects into regression tests**

For every discrepancy, add a failing test to the narrowest TravelPlanner test
module, run it to prove the defect, implement the smallest correction, then
rerun both the focused test and real smoke. Do not replace real database calls
with fixtures in the smoke path.

- [ ] **Step 6: Commit real-run corrections**

```bash
git add planu_core scripts tests/planu_core
git commit -m "fix: complete TravelPlanner experiment smoke"
```

Skip this commit only when no source change is required.

---

### Task 10: Document And Regress Phase Two

**Files:**
- Modify: `README.md`
- Modify: `docs/experiment-validation.md`
- Modify: `tests/planu_core/test_readme.py`

- [ ] **Step 1: Write failing documentation tests**

```python
def test_readme_documents_travelplanner_scope_and_isolation():
    readme = README.read_text(encoding="utf-8")
    assert "scripts/bootstrap_travelplanner.sh" in readme
    assert "scripts/smoke_travelplanner.sh" in readme
    assert "e52c87f4ac348a3410c46dc3553c519db5ec5e23" in readme
    assert "8736504ecfc31b7f8b7e40122873c337e83fff7c" in readme
    assert "validation and test" in readme
    assert "offline evaluator" in readme
    assert "not a PlanU-paper metric" in readme
```

- [ ] **Step 2: Run the documentation test**

Run:

```bash
python -m pytest tests/planu_core/test_readme.py::test_readme_documents_travelplanner_scope_and_isolation -v
```

Expected: FAIL because TravelPlanner is still marked planned.

- [ ] **Step 3: Update documentation**

Change the status table to:

```text
WebShop       | Phase two | Supported
TravelPlanner | Phase two | Supported
```

Document pinned source/data/database setup, bridge environment, migration
profile, reference model overrides, split-specific reward policy, official
offline metrics, and real smoke command. State explicitly that TravelPlanner is
a new PlanU application and not an original PlanU-paper result.

Extend `docs/experiment-validation.md` with the exact migration profile and
real smoke evidence schema.

- [ ] **Step 4: Run all automated verification**

Run:

```bash
python -m pytest -q
python3.9 -m compileall -q planu_core webshop tests
python setup.py check --strict
python setup.py sdist bdist_wheel
bash -n scripts/bootstrap_travelplanner.sh scripts/smoke_travelplanner.sh scripts/smoke_phase_two.sh
git diff --check
```

Expected: all tests PASS; compile, package, shell, and diff checks exit `0`.

- [ ] **Step 5: Run the complete real smoke matrix**

Run:

```bash
scripts/smoke_phase_two.sh
```

Expected: Overcooked, VirtualHome food, VirtualHome entertainment, BlockWorld,
WebShop, and TravelPlanner all exit `0` using their real environment/tool
backends.

- [ ] **Step 6: Audit secrets and generated files**

Run:

```bash
rg -n --hidden --glob '!/.git/**' 'sk-[A-Za-z0-9_-]{16,}|OPENAI_API_KEY\\s*=\\s*[\"'\"'][^\"'\"']+' .
git status --short
```

Expected: no credential literals. Only intentional source, test, script, and
documentation changes are committed; generated outputs and external datasets
remain untracked or ignored.

- [ ] **Step 7: Commit**

```bash
git add README.md docs/experiment-validation.md tests/planu_core/test_readme.py
git commit -m "docs: document phase-two experiments"
```

---

## Phase-Two Completion Gate

Record the following in the final implementation report:

```text
PlanU commit
WebShop source commit and server data profile
TravelPlanner source commit
TravelPlanner dataset revision
TravelPlanner database manifest hash
Python and dependency versions for each environment
unit-test count
all six smoke command exit codes
artifact directories
```

Formal paper-scale claims require the configured production checkpoints,
experiment budgets, complete task ranges, and repeated seeds. Passing this
plan's smoke matrix proves end-to-end executability and configuration
integrity; it does not by itself establish metric equivalence.
