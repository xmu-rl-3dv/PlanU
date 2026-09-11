# Unified PlanU Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the duplicated BlockWorld, Overcooked, and VirtualHome PlanU implementations with one configurable search implementation while preserving the effective Overcooked and VirtualHome experiment behavior.

**Architecture:** Add a Python 3.9-compatible `planu_core` package containing the alternating language/action tree, quantile distribution, selection, backup, curiosity interface, and trajectory runner. Keep benchmark state parsing, legal actions, environment copying, stochastic failures, and reward conventions in adapters. Existing experiment scripts remain the user-facing entry points and delegate to the shared implementation.

**Tech Stack:** Python 3.9, NumPy, PyTorch, Gym 0.21, Transformers, DI-engine RND, pytest.

**Reference Design:** `docs/superpowers/specs/2026-09-11-planu-unified-core-design.md`

---

## Scope And File Map

Create:

- `requirements-dev.txt`: test-only dependencies.
- `pytest.ini`: local test discovery and import configuration.
- `planu_core/__init__.py`: public core API.
- `planu_core/config.py`: immutable algorithm and selection parameters.
- `planu_core/distribution.py`: quantile initialization, distortion, and update.
- `planu_core/interfaces.py`: action, transition, adapter, scorer, and curiosity contracts.
- `planu_core/nodes.py`: shared `LanguageNode` and `ActionNode`.
- `planu_core/selection.py`: deterministic and scheduled sampling policies.
- `planu_core/backup.py`: suffix-return computation and quantile backup.
- `planu_core/search.py`: expansion, transition merging, and trajectory execution.
- `planu_core/curiosity.py`: null and RND curiosity providers.
- `planu_core/adapters/__init__.py`: adapter exports.
- `planu_core/adapters/overcooked.py`: Overcooked observation/action/transition behavior.
- `planu_core/adapters/virtualhome.py`: food and entertainment behavior.
- `planu_core/adapters/blockworld.py`: `WorldModel` and `SearchConfig` bridge.
- `tests/planu_core/`: focused unit, characterization, and adapter tests.

Modify:

- `mcts/overcooked/PlanU_mcts.py`: compatibility exports and LLM scorer.
- `mcts/overcooked/PlanU_inference.py`: build config/adapter and invoke shared runner.
- `mcts/overcooked/rnd.py`: compatibility re-export.
- `mcts/virtualhome/PlanU_v1.py`: compatibility exports and food prompt scorer.
- `mcts/virtualhome/PlanU_v2.py`: compatibility exports and entertainment prompt scorer.
- `mcts/virtualhome/PlanU_inference_food.py`: invoke shared runner.
- `mcts/virtualhome/PlanU_entertainment.py`: invoke shared runner.
- `mcts/virtualhome/rnd.py`: compatibility re-export.
- `mcts/virtualhome/base_reward_model.py`: compatibility re-export.
- `blockworld/reasoners/algorithm/planU.py`: compatibility wrapper around shared core.
- `blockworld/evaluate_stochastic.py`: fix launch defects and construct the adapter.
- `README.md`: algorithm overview, supported environments, and one Overcooked command.
- `scripts/PlanU_overcooked.sh`: retain the documented reproduction command.
- `scripts/PlanU_Virtualhome.sh`: fix the stale entry-point filename.

Do not modify:

- `gym-macro-overcooked/gym_macro_overcooked/**`
- `virtual-home/virtual_home/**`
- `webshop/**` during phase one

### Task 1: Establish The Test Harness And Configuration Contract

**Files:**
- Create: `requirements-dev.txt`
- Create: `pytest.ini`
- Create: `tests/planu_core/test_config.py`
- Create: `planu_core/__init__.py`
- Create: `planu_core/config.py`

- [ ] **Step 1: Add a failing configuration test**

```python
# tests/planu_core/test_config.py
from planu_core.config import PlanUConfig, SelectionSchedule


def test_reference_defaults_are_explicit():
    config = PlanUConfig()

    assert config.n_quantiles == 51
    assert config.value_min == -1.0
    assert config.value_max == 1.0
    assert config.discount == 1.0
    assert config.quantile_learning_rate == 0.75
    assert config.curiosity_weight == 0.0
    assert config.include_preview_reward is True


def test_overcooked_sampling_schedule():
    schedule = SelectionSchedule(
        always_sample_before=50,
        probabilistic_sample_before=100,
        sample_probability=0.2,
    )

    assert schedule.always_samples(49)
    assert not schedule.always_samples(50)
    assert schedule.sample_probability_at(75) == 0.2
    assert schedule.sample_probability_at(100) == 0.0
```

- [ ] **Step 2: Add pytest configuration and install the test dependency**

```text
# requirements-dev.txt
pytest>=7.4,<9
```

```ini
# pytest.ini
[pytest]
testpaths = tests
addopts = -ra
```

Run:

```bash
cd /Users/bytedance/personal_paper_code/PlanU
python3 -m pip install -r requirements-dev.txt
python3 -m pytest tests/planu_core/test_config.py -q
```

Expected: collection fails because `planu_core.config` does not exist.

- [ ] **Step 3: Implement the immutable configuration types**

```python
# planu_core/config.py
from dataclasses import dataclass, field


@dataclass(frozen=True)
class SelectionSchedule:
    always_sample_before: int = 0
    probabilistic_sample_before: int = 0
    sample_probability: float = 0.0

    def __post_init__(self) -> None:
        if self.always_sample_before < 0:
            raise ValueError("always_sample_before must be non-negative")
        if self.probabilistic_sample_before < self.always_sample_before:
            raise ValueError("probabilistic_sample_before must not precede always_sample_before")
        if not 0.0 <= self.sample_probability <= 1.0:
            raise ValueError("sample_probability must be in [0, 1]")

    def always_samples(self, iteration: int) -> bool:
        return iteration < self.always_sample_before

    def sample_probability_at(self, iteration: int) -> float:
        if self.always_samples(iteration):
            return 1.0
        if iteration < self.probabilistic_sample_before:
            return self.sample_probability
        return 0.0


@dataclass(frozen=True)
class PlanUConfig:
    n_quantiles: int = 51
    value_min: float = -1.0
    value_max: float = 1.0
    quantile_learning_rate: float = 0.75
    discount: float = 1.0
    curiosity_weight: float = 0.0
    include_preview_reward: bool = True
    max_depth: int = 15
    max_iterations: int = 1000
    selection_temperature: float = 1.0
    risk_distortion: float = 0.0
    selection_schedule: SelectionSchedule = field(default_factory=SelectionSchedule)

    def __post_init__(self) -> None:
        if self.n_quantiles <= 0:
            raise ValueError("n_quantiles must be positive")
        if self.value_min >= self.value_max:
            raise ValueError("value_min must be less than value_max")
        if not 0.0 < self.quantile_learning_rate <= 1.0:
            raise ValueError("quantile_learning_rate must be in (0, 1]")
        if not 0.0 <= self.discount <= 1.0:
            raise ValueError("discount must be in [0, 1]")
        if self.max_depth <= 0 or self.max_iterations <= 0:
            raise ValueError("search budgets must be positive")
        if self.selection_temperature <= 0:
            raise ValueError("selection_temperature must be positive")
```

Export `PlanUConfig` and `SelectionSchedule` from `planu_core/__init__.py`.

- [ ] **Step 4: Run the test**

Run: `python3 -m pytest tests/planu_core/test_config.py -q`

Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add requirements-dev.txt pytest.ini planu_core tests/planu_core/test_config.py
git commit -m "test: establish PlanU core configuration"
```

### Task 2: Implement Quantile Distribution Semantics

**Files:**
- Create: `planu_core/distribution.py`
- Create: `tests/planu_core/test_distribution.py`
- Modify: `planu_core/__init__.py`

- [ ] **Step 1: Write failing tests for scalar and categorical initialization**

```python
# tests/planu_core/test_distribution.py
import numpy as np

from planu_core.distribution import QuantileDistribution


def test_scalar_initialization_creates_point_mass():
    dist = QuantileDistribution.from_scalar(0.4, count=5, value_min=-1, value_max=1)
    np.testing.assert_allclose(dist.values, [0.4] * 5)
    np.testing.assert_allclose(dist.fractions, [0.1, 0.3, 0.5, 0.7, 0.9])


def test_categorical_initialization_fills_every_quantile():
    dist = QuantileDistribution.from_categorical(
        levels=[0.1, 0.3, 0.5, 0.7, 0.9],
        probabilities=[0.0, 0.0, 0.0, 0.0, 1.0],
        count=51,
        value_min=-1,
        value_max=1,
    )
    np.testing.assert_allclose(dist.values, [0.9] * 51)


def test_risk_neutral_value_is_arithmetic_mean():
    dist = QuantileDistribution.from_values([-1.0, 0.0, 1.0], -1.0, 1.0)
    assert dist.distorted_value(0.0) == 0.0


def test_update_rejects_non_finite_target():
    dist = QuantileDistribution.from_scalar(0.0, count=5, value_min=-1, value_max=1)
    with pytest.raises(ValueError, match="finite"):
        dist.update_scalar(float("nan"), learning_rate=0.75)
```

- [ ] **Step 2: Run the tests and verify the missing module failure**

Run: `python3 -m pytest tests/planu_core/test_distribution.py -q`

Expected: FAIL with `ModuleNotFoundError: planu_core.distribution`.

- [ ] **Step 3: Implement initialization and distortion**

```python
# planu_core/distribution.py
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class QuantileDistribution:
    fractions: np.ndarray
    values: np.ndarray
    value_min: float
    value_max: float

    @classmethod
    def from_scalar(cls, value: float, count: int, value_min: float, value_max: float):
        fractions = (np.arange(count, dtype=np.float64) + 0.5) / count
        values = np.full(count, float(value), dtype=np.float64)
        return cls(fractions, values, value_min, value_max)

    @classmethod
    def from_values(cls, values: Sequence[float], value_min: float, value_max: float):
        array = np.asarray(values, dtype=np.float64)
        fractions = (np.arange(len(array), dtype=np.float64) + 0.5) / len(array)
        return cls(fractions, array.copy(), value_min, value_max)

    @classmethod
    def from_categorical(
        cls,
        levels: Sequence[float],
        probabilities: Sequence[float],
        count: int,
        value_min: float,
        value_max: float,
    ):
        probs = np.asarray(probabilities, dtype=np.float64)
        if len(levels) != len(probs) or np.any(probs < 0) or probs.sum() <= 0:
            raise ValueError("invalid categorical distribution")
        probs = probs / probs.sum()
        fractions = (np.arange(count, dtype=np.float64) + 0.5) / count
        indices = np.searchsorted(np.cumsum(probs), fractions, side="left")
        values = np.asarray(levels, dtype=np.float64)[np.minimum(indices, len(levels) - 1)]
        return cls(fractions, values, value_min, value_max)

    def distorted_value(self, risk_distortion: float = 0.0) -> float:
        if risk_distortion == 0.0:
            return float(np.mean(self.values))
        if risk_distortion > 0:
            mask = self.fractions >= 1.0 - risk_distortion
        else:
            mask = self.fractions <= -risk_distortion
        selected = self.values[mask]
        return float(np.mean(selected if len(selected) else self.values))

    def update_scalar(self, target: float, learning_rate: float) -> None:
        if not np.isfinite(target):
            raise ValueError("quantile target must be finite")
        delta = float(target) - self.values
        weights = np.where(delta > 0, self.fractions, self.fractions - 1.0)
        self.values += learning_rate * weights * np.abs(delta)
        self.values = np.clip(self.values, self.value_min, self.value_max)
```

Add `import pytest` to the test module.

- [ ] **Step 4: Add a regression test for the reference pinball update**

```python
def test_scalar_update_matches_reference_formula():
    dist = QuantileDistribution.from_scalar(0.5, count=3, value_min=-1, value_max=1)
    dist.update_scalar(target=1.0, learning_rate=0.75)
    expected = 0.5 + 0.75 * np.array([1 / 6, 1 / 2, 5 / 6]) * 0.5
    np.testing.assert_allclose(dist.values, expected)
```

Run: `python3 -m pytest tests/planu_core/test_distribution.py -q`

Expected: `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add planu_core/distribution.py planu_core/__init__.py tests/planu_core/test_distribution.py
git commit -m "feat: add shared quantile distribution"
```

### Task 3: Define Shared Contracts And Alternating Tree Nodes

**Files:**
- Create: `planu_core/interfaces.py`
- Create: `planu_core/nodes.py`
- Create: `tests/planu_core/test_nodes.py`
- Modify: `planu_core/__init__.py`

- [ ] **Step 1: Write the failing outcome-merging test**

```python
# tests/planu_core/test_nodes.py
from planu_core.distribution import QuantileDistribution
from planu_core.interfaces import ActionCandidate
from planu_core.nodes import ActionNode, LanguageNode


def test_action_node_merges_equal_outcomes():
    parent = LanguageNode(state=[0], state_key=(0,))
    action = ActionNode(
        parent=parent,
        action=ActionCandidate("open", 4, "open microwave"),
        distribution=QuantileDistribution.from_scalar(0.2, 5, -1, 1),
    )

    first = action.get_or_create_outcome([1], (1,), reward=0.0, terminated=False, truncated=False)
    second = action.get_or_create_outcome([1], (1,), reward=0.0, terminated=False, truncated=False)

    assert first is second
    assert len(action.children) == 1
    assert first.outcome_visits == 2
```

- [ ] **Step 2: Run it and verify failure**

Run: `python3 -m pytest tests/planu_core/test_nodes.py -q`

Expected: FAIL because the contracts and nodes do not exist.

- [ ] **Step 3: Implement shared contracts**

```python
# planu_core/interfaces.py
from dataclasses import dataclass, field
from typing import Any, Hashable, Mapping, Protocol, Sequence


@dataclass(frozen=True)
class ActionCandidate:
    key: Hashable
    payload: Any
    text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class EnvironmentState:
    observation: Any
    runtime: Any


@dataclass
class TransitionResult:
    state: EnvironmentState
    reward: float
    terminated: bool
    truncated: bool
    info: Mapping[str, Any] = field(default_factory=dict)


class EnvironmentAdapter(Protocol):
    def reset(self) -> EnvironmentState: ...
    def actions(
        self,
        state: EnvironmentState,
        state_visit_count: int = 0,
    ) -> Sequence[ActionCandidate]: ...
    def preview(self, state: EnvironmentState, action: ActionCandidate) -> TransitionResult: ...
    def step(self, state: EnvironmentState, action: ActionCandidate) -> TransitionResult: ...
    def state_key(self, observation: Any) -> Hashable: ...


class ActionScorer(Protocol):
    def score(self, observation: Any, actions: Sequence[ActionCandidate]):
        ...


class CuriosityProvider(Protocol):
    def score(self, observation: Any) -> float: ...
    def observe(self, observation: Any) -> None: ...
    def train(self) -> None: ...
```

- [ ] **Step 4: Implement the nodes**

```python
# planu_core/nodes.py
from dataclasses import dataclass, field
from typing import Any, Dict, Hashable, List, Optional

from .distribution import QuantileDistribution
from .interfaces import ActionCandidate


@dataclass
class LanguageNode:
    state: Any
    state_key: Hashable
    parent: Optional["ActionNode"] = None
    children: Dict[Hashable, "ActionNode"] = field(default_factory=dict)
    visit_count: int = 0
    terminated: bool = False
    truncated: bool = False
    outcome_visits: int = 0


@dataclass
class ActionNode:
    parent: LanguageNode
    action: ActionCandidate
    distribution: QuantileDistribution
    children: Dict[Hashable, LanguageNode] = field(default_factory=dict)
    visit_count: int = 0
    cumulative_returns: List[float] = field(default_factory=list)
    preview_state: Any = None

    def get_or_create_outcome(
        self,
        state: Any,
        state_key: Hashable,
        reward: float,
        terminated: bool,
        truncated: bool,
        increment_visit: bool = True,
    ) -> LanguageNode:
        child = self.children.get(state_key)
        if child is None:
            child = LanguageNode(
                state=state,
                state_key=state_key,
                parent=self,
                terminated=terminated,
                truncated=truncated,
            )
            self.children[state_key] = child
        if increment_visit:
            child.outcome_visits += 1
        return child
```

- [ ] **Step 5: Run and commit**

Run: `python3 -m pytest tests/planu_core/test_nodes.py -q`

Expected: `1 passed`.

```bash
git add planu_core/interfaces.py planu_core/nodes.py planu_core/__init__.py tests/planu_core/test_nodes.py
git commit -m "feat: add shared PlanU tree contracts"
```

### Task 4: Implement Reference-Compatible Selection

**Files:**
- Create: `planu_core/selection.py`
- Create: `tests/planu_core/test_selection.py`

- [ ] **Step 1: Write failing selection tests**

```python
# tests/planu_core/test_selection.py
import numpy as np

from planu_core.config import PlanUConfig, SelectionSchedule
from planu_core.distribution import QuantileDistribution
from planu_core.interfaces import ActionCandidate
from planu_core.nodes import ActionNode, LanguageNode
from planu_core.selection import select_action


def make_action(parent, key, value):
    return ActionNode(
        parent=parent,
        action=ActionCandidate(key, key, str(key)),
        distribution=QuantileDistribution.from_scalar(value, 5, -1, 1),
    )


def test_greedy_selection_uses_quantile_value():
    root = LanguageNode([0], (0,))
    root.children = {"low": make_action(root, "low", 0.1), "high": make_action(root, "high", 0.8)}
    selected, scores = select_action(root, PlanUConfig(), iteration=200, rng=np.random.default_rng(7))
    assert selected.action.key == "high"
    assert scores["high"] > scores["low"]


def test_reference_sampling_schedule_is_seeded():
    root = LanguageNode([0], (0,))
    root.children = {"a": make_action(root, "a", 0.0), "b": make_action(root, "b", 0.0)}
    config = PlanUConfig(
        selection_schedule=SelectionSchedule(50, 100, 0.2),
    )
    selected, _ = select_action(root, config, iteration=0, rng=np.random.default_rng(4))
    assert selected.action.key in {"a", "b"}
```

- [ ] **Step 2: Run and verify failure**

Run: `python3 -m pytest tests/planu_core/test_selection.py -q`

Expected: FAIL because `planu_core.selection` does not exist.

- [ ] **Step 3: Implement stable scoring and scheduled sampling**

```python
# planu_core/selection.py
from typing import Dict, Tuple

import numpy as np

from .config import PlanUConfig
from .nodes import ActionNode, LanguageNode


def _stable_softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    shifted = values / temperature - np.max(values / temperature)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum()


def select_action(
    node: LanguageNode,
    config: PlanUConfig,
    iteration: int,
    rng: np.random.Generator,
    novelty: Dict[object, float] = None,
) -> Tuple[ActionNode, Dict[object, float]]:
    if not node.children:
        raise ValueError("cannot select from an unexpanded state")
    novelty = novelty or {}
    raw_novelty = np.array([novelty.get(key, 0.0) for key in node.children], dtype=np.float64)
    norm = np.abs(raw_novelty).sum()
    normalized = raw_novelty / norm if norm > 0 else raw_novelty
    actions = list(node.children.values())
    values = np.array(
        [action.distribution.distorted_value(config.risk_distortion) for action in actions],
        dtype=np.float64,
    )
    values += config.curiosity_weight * normalized
    scores = {action.action.key: float(value) for action, value in zip(actions, values)}
    probability = config.selection_schedule.sample_probability_at(iteration)
    if probability == 1.0 or (probability > 0 and rng.random() < probability):
        index = int(rng.choice(len(actions), p=_stable_softmax(values, config.selection_temperature)))
    else:
        index = int(np.argmax(values))
    return actions[index], scores
```

- [ ] **Step 4: Run and commit**

Run: `python3 -m pytest tests/planu_core/test_selection.py -q`

Expected: `2 passed`.

```bash
git add planu_core/selection.py tests/planu_core/test_selection.py
git commit -m "feat: add shared PlanU action selection"
```

### Task 5: Implement Suffix-Return Backup

**Files:**
- Create: `planu_core/backup.py`
- Create: `tests/planu_core/test_backup.py`

- [ ] **Step 1: Write failing backup tests**

```python
# tests/planu_core/test_backup.py
import numpy as np

from planu_core.backup import backup_trajectory, suffix_returns
from planu_core.config import PlanUConfig
from planu_core.distribution import QuantileDistribution
from planu_core.interfaces import ActionCandidate
from planu_core.nodes import ActionNode, LanguageNode


def test_suffix_returns_match_reference_undiscounted_sum():
    np.testing.assert_allclose(suffix_returns([0.1, -0.1, 1.0], 1.0), [1.0, 0.9, 1.0])


def test_backup_updates_each_selected_action_once():
    root = LanguageNode([0], (0,))
    first = ActionNode(root, ActionCandidate("a", 0, "a"), QuantileDistribution.from_scalar(0, 3, -2, 2))
    second_state = LanguageNode([1], (1,), parent=first)
    second = ActionNode(second_state, ActionCandidate("b", 1, "b"), QuantileDistribution.from_scalar(0, 3, -2, 2))
    config = PlanUConfig(n_quantiles=3, value_min=-2, value_max=2, quantile_learning_rate=0.5)

    returns = backup_trajectory([first, second], [0.25, 1.0], config)

    np.testing.assert_allclose(returns, [1.25, 1.0])
    assert first.cumulative_returns == [1.25]
    assert second.cumulative_returns == [1.0]
    assert first.visit_count == 1
    assert second.visit_count == 1
```

- [ ] **Step 2: Run and verify failure**

Run: `python3 -m pytest tests/planu_core/test_backup.py -q`

Expected: FAIL because `planu_core.backup` does not exist.

- [ ] **Step 3: Implement backup**

```python
# planu_core/backup.py
from typing import Iterable, List, Sequence

from .config import PlanUConfig
from .nodes import ActionNode


def suffix_returns(rewards: Sequence[float], discount: float) -> List[float]:
    returns = [0.0] * len(rewards)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running = float(rewards[index]) + discount * running
        returns[index] = running
    return returns


def backup_trajectory(
    actions: Iterable[ActionNode],
    rewards: Sequence[float],
    config: PlanUConfig,
) -> List[float]:
    action_path = list(actions)
    if len(action_path) != len(rewards):
        raise ValueError("action path and reward lengths differ")
    returns = suffix_returns(rewards, config.discount)
    for action, target in zip(action_path, returns):
        action.visit_count += 1
        action.cumulative_returns.append(target)
        action.distribution.update_scalar(target, config.quantile_learning_rate)
    return returns
```

- [ ] **Step 4: Run and commit**

Run: `python3 -m pytest tests/planu_core/test_backup.py -q`

Expected: `2 passed`.

```bash
git add planu_core/backup.py tests/planu_core/test_backup.py
git commit -m "feat: add shared PlanU trajectory backup"
```

### Task 6: Build The Shared Search Engine Against A Fake Environment

**Files:**
- Create: `planu_core/search.py`
- Create: `tests/planu_core/fakes.py`
- Create: `tests/planu_core/test_search.py`
- Modify: `planu_core/__init__.py`

- [ ] **Step 1: Add a deterministic fake adapter and failing integration test**

```python
# tests/planu_core/fakes.py
import copy

import numpy as np

from planu_core.interfaces import ActionCandidate, EnvironmentState, TransitionResult


class FakeAdapter:
    def __init__(self):
        self.actions_taken = []

    def reset(self):
        return EnvironmentState(observation=np.array([0]), runtime={"position": 0})

    def actions(self, state, state_visit_count=0):
        return [ActionCandidate("advance", 0, "advance")]

    def preview(self, state, action):
        runtime = copy.deepcopy(state.runtime)
        runtime["position"] += 1
        return TransitionResult(
            EnvironmentState(np.array([runtime["position"]]), runtime),
            reward=1.0,
            terminated=runtime["position"] == 2,
            truncated=False,
        )

    def step(self, state, action):
        self.actions_taken.append(action.key)
        return self.preview(state, action)

    def state_key(self, observation):
        return tuple(np.asarray(observation).tolist())


class UniformScorer:
    def score(self, observation, actions):
        return np.ones(len(actions), dtype=np.float64) / len(actions)
```

```python
# tests/planu_core/test_search.py
import numpy as np

from planu_core.config import PlanUConfig
from planu_core.search import PlanUSearch
from tests.planu_core.fakes import FakeAdapter, UniformScorer


def test_two_step_trajectory_builds_alternating_tree_and_backs_up():
    adapter = FakeAdapter()
    search = PlanUSearch(adapter, UniformScorer(), PlanUConfig(max_depth=3, max_iterations=1))

    result = search.run_iteration(iteration=0, rng=np.random.default_rng(1))

    assert result.rewards == [1.0, 1.0]
    assert adapter.actions_taken == ["advance", "advance"]
    root_action = search.root.children["advance"]
    assert root_action.cumulative_returns == [2.0]
    assert list(root_action.children) == [(1,)]
    assert root_action.children[(1,)].outcome_visits == 1
```

- [ ] **Step 2: Run and verify failure**

Run: `python3 -m pytest tests/planu_core/test_search.py -q`

Expected: FAIL because `PlanUSearch` does not exist.

- [ ] **Step 3: Implement the engine**

Implement `PlanUSearch` with these concrete public methods:

```python
# planu_core/search.py
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from .backup import backup_trajectory
from .config import PlanUConfig
from .distribution import QuantileDistribution
from .interfaces import ActionScorer, EnvironmentAdapter, EnvironmentState
from .nodes import ActionNode, LanguageNode
from .selection import select_action


@dataclass
class TrajectoryResult:
    actions: List[Any]
    rewards: List[float]
    terminated: bool
    truncated: bool
    final_observation: Any
    selection_scores: List[Dict[Any, float]]
    action_path: List[ActionNode]
    state_path: List[LanguageNode]


class PlanUSearch:
    def __init__(self, adapter: EnvironmentAdapter, scorer: ActionScorer, config: PlanUConfig):
        self.adapter = adapter
        self.scorer = scorer
        self.config = config
        self.root: Optional[LanguageNode] = None

    def _ensure_root(self, state: EnvironmentState) -> LanguageNode:
        key = self.adapter.state_key(state.observation)
        if self.root is None:
            self.root = LanguageNode(state.observation, key)
        elif self.root.state_key != key:
            raise ValueError("reset state does not match persistent PlanU root")
        return self.root

    def expand(self, node: LanguageNode, state: EnvironmentState) -> None:
        if node.terminated or node.truncated or node.children:
            return
        candidates = list(self.adapter.actions(state, node.visit_count))
        priors = np.asarray(self.scorer.score(state.observation, candidates), dtype=np.float64)
        if len(priors) != len(candidates):
            raise ValueError("scorer returned the wrong number of action scores")
        if not np.all(np.isfinite(priors)):
            raise ValueError("action scores must be finite")
        for candidate, prior in zip(candidates, priors):
            preview = self.adapter.preview(state, candidate)
            initial = float(prior)
            if self.config.include_preview_reward:
                initial += float(preview.reward)
            action = ActionNode(
                parent=node,
                action=candidate,
                distribution=QuantileDistribution.from_scalar(
                    initial,
                    self.config.n_quantiles,
                    self.config.value_min,
                    self.config.value_max,
                ),
                preview_state=preview.state.observation,
            )
            preview_key = self.adapter.state_key(preview.state.observation)
            if preview.info.get("record_outcome", True):
                action.get_or_create_outcome(
                    preview.state.observation,
                    preview_key,
                    preview.reward,
                    preview.terminated,
                    preview.truncated,
                    increment_visit=False,
                )
            node.children[candidate.key] = action

    def run_iteration(self, iteration: int, rng: np.random.Generator) -> TrajectoryResult:
        state = self.adapter.reset()
        node = self._ensure_root(state)
        action_path = []
        state_path = [node]
        actions = []
        rewards = []
        score_history = []
        terminated = truncated = False
        for _ in range(self.config.max_depth):
            node.visit_count += 1
            self.expand(node, state)
            if not node.children:
                truncated = True
                break
            action_node, scores = select_action(node, self.config, iteration, rng)
            result = self.adapter.step(state, action_node.action)
            next_key = self.adapter.state_key(result.state.observation)
            next_node = action_node.get_or_create_outcome(
                result.state.observation,
                next_key,
                result.reward,
                result.terminated,
                result.truncated,
            )
            action_path.append(action_node)
            actions.append(action_node.action.key)
            rewards.append(float(result.reward))
            score_history.append(scores)
            state = result.state
            node = next_node
            state_path.append(next_node)
            terminated = result.terminated
            truncated = result.truncated
            if terminated or truncated:
                break
        backup_trajectory(action_path, rewards, self.config)
        return TrajectoryResult(
            actions,
            rewards,
            terminated,
            truncated,
            state.observation,
            score_history,
            action_path,
            state_path,
        )
```

- [ ] **Step 4: Run the core suite**

Run: `python3 -m pytest tests/planu_core/test_config.py tests/planu_core/test_distribution.py tests/planu_core/test_nodes.py tests/planu_core/test_selection.py tests/planu_core/test_backup.py tests/planu_core/test_search.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add planu_core/search.py planu_core/__init__.py tests/planu_core/fakes.py tests/planu_core/test_search.py
git commit -m "feat: add shared PlanU search engine"
```

### Task 7: Consolidate Curiosity Without Changing RND Math

**Files:**
- Create: `planu_core/curiosity.py`
- Create: `planu_core/rnd.py`
- Create: `planu_core/base_reward_model.py`
- Create: `tests/planu_core/test_curiosity.py`
- Modify: `mcts/overcooked/rnd.py`
- Modify: `mcts/virtualhome/rnd.py`
- Modify: `mcts/virtualhome/base_reward_model.py`
- Modify: `planu_core/search.py`

- [ ] **Step 1: Write null-provider and training-schedule tests**

```python
# tests/planu_core/test_curiosity.py
import numpy as np

from planu_core.curiosity import NullCuriosity


def test_null_curiosity_has_no_side_effects():
    curiosity = NullCuriosity()
    assert curiosity.score(np.array([1.0])) == 0.0
    curiosity.observe(np.array([1.0]))
    assert curiosity.ready_to_train is False
    assert curiosity.train() is None
```

- [ ] **Step 2: Run and verify failure**

Run: `python3 -m pytest tests/planu_core/test_curiosity.py -q`

Expected: FAIL because `planu_core.curiosity` does not exist.

- [ ] **Step 3: Add the common providers**

```python
# planu_core/curiosity.py
class NullCuriosity:
    ready_to_train = False

    def score(self, observation):
        return 0.0

    def observe(self, observation):
        return None

    def train(self):
        return None


class RndCuriosity:
    def __init__(self, model, minimum_samples=15):
        self.model = model
        self.minimum_samples = minimum_samples
        self.sample_count = 0

    @property
    def ready_to_train(self):
        return self.sample_count > self.minimum_samples

    def score(self, observation):
        return float(self.model.estimate(observation).detach().cpu().item())

    def observe(self, observation):
        self.model.collect_data(observation)
        self.sample_count += 1

    def train(self):
        if self.ready_to_train:
            return self.model.train()
        return None
```

Copy the existing identical RND implementation from
`mcts/virtualhome/rnd.py` to `planu_core/rnd.py`, copy
`mcts/virtualhome/base_reward_model.py` to `planu_core/base_reward_model.py`,
and change the import in `planu_core/rnd.py` to:

```python
from .base_reward_model import BaseRewardModel
```

Replace both benchmark RND files with compatibility exports:

```python
from planu_core.rnd import RndNetwork, RndRewardModel, collect_states

__all__ = ["RndNetwork", "RndRewardModel", "collect_states"]
```

Replace `mcts/virtualhome/base_reward_model.py` with:

```python
from planu_core.base_reward_model import BaseRewardModel

__all__ = ["BaseRewardModel"]
```

- [ ] **Step 4: Wire curiosity into search**

Extend `PlanUSearch.__init__` with `curiosity=None`, default it to
`NullCuriosity`, compute one novelty value per action from
`action.preview_state`, pass the values to `select_action`, call `observe()` on
each executed next observation, and call `train()` once after each trajectory.

```python
self.curiosity = curiosity or NullCuriosity()
novelty = {
    key: self.curiosity.score(action.preview_state)
    for key, action in node.children.items()
}
action_node, scores = select_action(node, self.config, iteration, rng, novelty)
```

- [ ] **Step 5: Run and commit**

Run: `python3 -m pytest tests/planu_core/test_curiosity.py tests/planu_core/test_search.py -q`

Expected: all tests pass.

```bash
git add planu_core mcts/overcooked/rnd.py mcts/virtualhome/rnd.py mcts/virtualhome/base_reward_model.py tests/planu_core
git commit -m "refactor: share PlanU curiosity implementation"
```

### Task 8: Extract The Overcooked Adapter And Characterize Reference Behavior

**Files:**
- Create: `planu_core/adapters/overcooked.py`
- Create: `tests/planu_core/test_overcooked_adapter.py`
- Modify: `mcts/overcooked/PlanU_mcts.py`
- Modify: `mcts/overcooked/PlanU_inference.py`

- [ ] **Step 1: Add tests for action order, stochastic failure, and profile values**

Use the tomato-salad observation fixture below without constructing Gym:

```python
# tests/planu_core/test_overcooked_adapter.py
import numpy as np

from planu_core.adapters.overcooked import OvercookedAdapter, overcooked_config


def test_tomato_salad_action_order_matches_reference():
    observation = np.array(
        [[0, 5, 0, 1, 6, 0, 2, 6, 0, 6, 5, 0, 1, 0, 2, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0]],
        dtype=np.float32,
    )
    adapter = OvercookedAdapter.for_observation_tests(task=3)
    actions = adapter.actions_from_observation(observation)
    assert [action.payload for action in actions] == list(range(8))
    assert actions[0].text == "pick up the tomato"


def test_overcooked_reference_parameters():
    config = overcooked_config(task=0, rnd=True, max_iterations=1000, max_depth=15)
    assert config.quantile_learning_rate == 0.75
    assert config.curiosity_weight == 0.5
    assert config.selection_schedule.always_sample_before == 50
```

- [ ] **Step 2: Run and verify failure**

Run: `python3 -m pytest tests/planu_core/test_overcooked_adapter.py -q`

Expected: FAIL because the adapter does not exist.

- [ ] **Step 3: Implement the adapter**

Move the exact observation-to-prompt/action logic from
`mcts/overcooked/PlanU_mcts.py:1176-1509` into
`OvercookedAdapter.describe_observation()`. Keep list order unchanged.

The adapter constructor and transition boundary must be:

```python
class OvercookedAdapter:
    def __init__(self, envs, task, stochastic_probability, rng):
        self.envs = envs
        self.task = task
        self.stochastic_probability = stochastic_probability
        self.rng = rng

    def reset(self):
        observation = self.envs.reset()
        return EnvironmentState(observation=observation, runtime=self.envs)

    def preview(self, state, action):
        env_copy = copy.deepcopy(state.runtime)
        observation, reward, done, info = env_copy.step(np.array([action.payload]))
        return TransitionResult(
            EnvironmentState(observation, env_copy),
            float(np.asarray(reward).reshape(-1)[0]),
            bool(np.asarray(done).reshape(-1)[0]),
            False,
            {"raw_info": info},
        )

    def step(self, state, action):
        before = copy.deepcopy(state.runtime)
        observation, reward, done, info = state.runtime.step(np.array([action.payload]))
        if "chop" in action.text and self.rng.random() < self.stochastic_probability:
            observation = state.observation
            reward = np.array([-0.001])
            done = np.array([False])
            runtime = before
        else:
            runtime = state.runtime
        return TransitionResult(
            EnvironmentState(observation, runtime),
            float(np.asarray(reward).reshape(-1)[0]),
            bool(np.asarray(done).reshape(-1)[0]),
            False,
            {"raw_info": info},
        )

    def state_key(self, observation):
        return tuple(np.asarray(observation).reshape(-1).tolist())
```

`overcooked_config()` must preserve `0.75`, the task-0 sampling schedule, and
the existing RND weight.

- [ ] **Step 4: Convert the old algorithm module to compatibility exports**

Retain the LLM tokenizer/model scorer in `PlanU_mcts.py`, rename it
`OvercookedActionScorer`, and remove its tree ownership. Export:

```python
from planu_core.nodes import ActionNode, LanguageNode
from planu_core.search import PlanUSearch
```

Update `PlanU_inference.py` to construct `OvercookedAdapter`, the shared
`PlanUSearch`, and `overcooked_config()`. Keep its argument names, result paths,
seed initialization, and TensorBoard tags unchanged. Record the effective
algorithm configuration before the loop:

```python
from dataclasses import asdict

writer.add_text(
    "planu/effective_config",
    json.dumps(asdict(config), sort_keys=True),
    global_step=0,
)
```

- [ ] **Step 5: Verify and commit**

Run:

```bash
python3 -m pytest tests/planu_core -q
python3 -m compileall -q planu_core mcts/overcooked
```

Expected: tests pass and compilation emits no errors.

```bash
git add planu_core/adapters/overcooked.py mcts/overcooked/PlanU_mcts.py mcts/overcooked/PlanU_inference.py tests/planu_core/test_overcooked_adapter.py
git commit -m "refactor: migrate Overcooked to shared PlanU core"
```

### Task 9: Migrate Both VirtualHome Tasks

**Files:**
- Create: `planu_core/adapters/virtualhome.py`
- Create: `tests/planu_core/test_virtualhome_adapter.py`
- Modify: `mcts/virtualhome/PlanU_v1.py`
- Modify: `mcts/virtualhome/PlanU_v2.py`
- Modify: `mcts/virtualhome/PlanU_inference_food.py`
- Modify: `mcts/virtualhome/PlanU_entertainment.py`
- Modify: `scripts/PlanU_Virtualhome.sh`

- [ ] **Step 1: Write failing profile and stochastic-action tests**

```python
# tests/planu_core/test_virtualhome_adapter.py
from planu_core.adapters.virtualhome import (
    VirtualHomeTask,
    virtualhome_config,
)


def test_food_reference_parameters():
    config = virtualhome_config(VirtualHomeTask.FOOD, rnd=True)
    assert config.quantile_learning_rate == 0.7
    assert config.curiosity_weight == 0.25


def test_entertainment_reference_parameters():
    config = virtualhome_config(VirtualHomeTask.ENTERTAINMENT, rnd=True)
    assert config.quantile_learning_rate == 0.7
    assert config.curiosity_weight == 0.1


def test_stochastic_action_names_are_task_specific():
    assert VirtualHomeTask.FOOD.stochastic_verb == "open"
    assert VirtualHomeTask.ENTERTAINMENT.stochastic_verb == "grab"
```

- [ ] **Step 2: Run and verify failure**

Run: `python3 -m pytest tests/planu_core/test_virtualhome_adapter.py -q`

Expected: FAIL because the adapter does not exist.

- [ ] **Step 3: Implement the shared VirtualHome adapter**

Create a task enum and one adapter:

```python
class VirtualHomeTask(Enum):
    FOOD = ("VirtualHome-v1", "open", False, 0.25)
    ENTERTAINMENT = ("VirtualHome-v2", "grab", True, 0.1)

    @property
    def env_id(self):
        return self.value[0]

    @property
    def stochastic_verb(self):
        return self.value[1]

    @property
    def uses_llm_prior(self):
        return self.value[2]

    @property
    def curiosity_weight(self):
        return self.value[3]
```

Move food observation decoding and action mapping from
`mcts/virtualhome/PlanU_v1.py:242-373` and entertainment decoding/mapping from
`mcts/virtualhome/PlanU_v2.py:1421-1980` into task-specific private functions.
The common adapter uses the same reset, preview, step, and `state_key` methods
as Overcooked, changing only the stochastic verb and observation/action
functions.

Use this configuration factory:

```python
def virtualhome_config(task, rnd, max_iterations=1000, max_depth=15):
    return PlanUConfig(
        quantile_learning_rate=0.7,
        curiosity_weight=task.curiosity_weight if rnd else 0.0,
        max_iterations=max_iterations,
        max_depth=max_depth,
    )
```

- [ ] **Step 4: Replace both runners with shared-core construction**

Keep all existing CLI flags and result paths. Replace direct node manipulation
with:

```python
adapter = VirtualHomeAdapter(
    envs=envs,
    task=VirtualHomeTask.FOOD,
    stochastic_probability=args.stochastic,
    rng=random.Random(args.seed),
)
search = PlanUSearch(adapter, scorer, virtualhome_config(VirtualHomeTask.FOOD, args.rnd))
for iteration in range(args.maxiterations):
    result = search.run_iteration(iteration, np.random.default_rng(args.seed + iteration))
```

Use `VirtualHomeTask.ENTERTAINMENT` in the entertainment runner. Convert
`PlanU_v1.py` and `PlanU_v2.py` to compatibility exports plus their respective
LLM scorer implementations. Record `json.dumps(asdict(config), sort_keys=True)`
under the same `planu/effective_config` TensorBoard tag in both runners.

Change the shell script target from the nonexistent
`PlanU_inference_entertainment.py` to:

```bash
python mcts/virtualhome/PlanU_entertainment.py \
```

- [ ] **Step 5: Verify and commit**

Run:

```bash
python3 -m pytest tests/planu_core -q
python3 -m compileall -q planu_core mcts/virtualhome
```

Expected: tests pass and compilation emits no errors.

```bash
git add \
  planu_core/adapters/virtualhome.py \
  mcts/virtualhome/PlanU_v1.py \
  mcts/virtualhome/PlanU_v2.py \
  mcts/virtualhome/PlanU_inference_food.py \
  mcts/virtualhome/PlanU_entertainment.py \
  scripts/PlanU_Virtualhome.sh \
  tests/planu_core/test_virtualhome_adapter.py
git commit -m "refactor: migrate VirtualHome to shared PlanU core"
```

### Task 10: Migrate And Correct BlockWorld

**Files:**
- Create: `planu_core/adapters/blockworld.py`
- Create: `tests/planu_core/test_blockworld_adapter.py`
- Modify: `blockworld/reasoners/algorithm/planU.py`
- Modify: `blockworld/evaluate_stochastic.py`

- [ ] **Step 1: Write a stochastic-outcome adapter test**

```python
# tests/planu_core/test_blockworld_adapter.py
from dataclasses import dataclass

from planu_core.adapters.blockworld import BlockWorldAdapter


@dataclass(frozen=True)
class FakeState:
    name: str


class FakeWorldModel:
    def __init__(self):
        self.outcomes = iter([True, False] * 10)

    def init_state(self):
        return FakeState("root")

    def step(self, state, action):
        success = next(self.outcomes)
        return FakeState("success" if success else "failure"), {"success": success}

    def is_terminal(self, state):
        return False


class FakeSearchConfig:
    def get_actions(self, state):
        return ["move"]

    def fast_reward(self, node, action):
        return 0.2, {"intuition": 0.1, "self_eval": 0.1}

    def reward(self, node, action, intuition, self_eval, success):
        return (1.0 if success else 0.0), {"success": success}


def test_same_action_can_accumulate_success_and_failure_states():
    adapter = BlockWorldAdapter(FakeWorldModel(), FakeSearchConfig())
    state = adapter.reset()
    action = adapter.actions(state, state_visit_count=0)[0]
    outcomes = {
        adapter.state_key(adapter.step(state, action).state.observation)
        for _ in range(20)
    }
    assert outcomes == {FakeState("success"), FakeState("failure")}
```

- [ ] **Step 2: Run and verify failure**

Run: `python3 -m pytest tests/planu_core/test_blockworld_adapter.py -q`

Expected: FAIL because the adapter does not exist.

- [ ] **Step 3: Implement the bridge**

```python
from types import SimpleNamespace

import numpy as np

from planu_core.interfaces import ActionCandidate, EnvironmentState, TransitionResult


class BlockWorldScorer:
    def score(self, observation, actions):
        return np.asarray(
            [candidate.metadata["prior_score"] for candidate in actions],
            dtype=np.float64,
        )


class BlockWorldAdapter:
    def __init__(self, world_model, search_config):
        self.world_model = world_model
        self.search_config = search_config

    def reset(self):
        state = self.world_model.init_state()
        return EnvironmentState(state, None)

    def actions(self, state, state_visit_count=0):
        actions = self.search_config.get_actions(state.observation)
        result = []
        node_view = SimpleNamespace(
            state=state.observation,
            cum_rewards=[0.0] * state_visit_count,
        )
        for action in actions:
            score, details = self.search_config.fast_reward(node_view, action)
            metadata = {
                "prior_score": float(score),
                "fast_reward_details": details,
                "node_view": node_view,
            }
            result.append(ActionCandidate(action, action, str(action), metadata))
        return result

    def preview(self, state, action):
        return TransitionResult(
            state,
            0.0,
            False,
            False,
            {"preview_only": True, "record_outcome": False},
        )

    def step(self, state, action):
        next_state, aux = self.world_model.step(state.observation, action.payload)
        reward, details = self.search_config.reward(
            action.metadata["node_view"],
            action.payload,
            **action.metadata["fast_reward_details"],
            **aux,
        )
        return TransitionResult(
            EnvironmentState(next_state, None),
            float(reward),
            bool(self.world_model.is_terminal(next_state)),
            False,
            {"reward_details": details, **aux},
        )

    def state_key(self, observation):
        return observation
```

If a benchmark state is not hashable, `state_key()` must return a deterministic
tuple of its serialized fields rather than an object identity or Python hash.

- [ ] **Step 4: Replace the BlockWorld algorithm wrapper and fix launch errors**

Keep the existing `PlanUResult` named tuple and replace the old algorithm body
with this `SearchAlgorithm` wrapper:

```python
class PlanU(SearchAlgorithm):
    def __init__(
        self,
        n_iters=10,
        depth_limit=12,
        n_atoms=51,
        v_min=-10.0,
        v_max=10.0,
        risk_distortion=0.0,
        output_trace_in_each_iter=False,
        **unused,
    ):
        self.n_iters = n_iters
        self.config = PlanUConfig(
            n_quantiles=n_atoms,
            value_min=v_min,
            value_max=v_max,
            max_depth=depth_limit,
            max_iterations=n_iters,
            risk_distortion=risk_distortion,
            include_preview_reward=False,
        )
        self.output_trace_in_each_iter = output_trace_in_each_iter

    def __call__(self, world_model, search_config, **kwargs):
        adapter = BlockWorldAdapter(world_model, search_config)
        search = PlanUSearch(adapter, BlockWorldScorer(), self.config)
        iterations = []
        for index in range(self.n_iters):
            result = search.run_iteration(index, np.random.default_rng(index))
            iterations.append(result)
        completed = [result for result in iterations if result.terminated]
        best = max(completed or iterations, key=lambda item: sum(item.rewards))
        trace = (
            [node.state for node in best.state_path],
            [node.action.payload for node in best.action_path],
        )
        return PlanUResult(
            terminal_state=best.final_observation,
            cum_reward=sum(best.rewards),
            trace=trace,
            trace_of_nodes=best.state_path,
            tree_state=search.root,
            trace_in_each_iter=(
                [item.state_path for item in iterations]
                if self.output_trace_in_each_iter
                else None
            ),
            tree_state_after_each_iter=None,
            aggregated_result=None,
        )
```

Export `PlanUNode = LanguageNode` for import compatibility. Remove
`PlanUAggregation`, distribution visualization, RAVE state, pickle helpers, and
unsupported output strategies from this wrapper.

In `blockworld/evaluate_stochastic.py`:

```python
from reasoners.algorithm import MCTS, PlanU
from reasoners.algorithm.planU import PlanUNode
```

Replace both `NewMCTSNode` annotations with `PlanUNode`, replace
`elif args.algorithm == 'plau':` with `elif args.algorithm == 'planu':`, and
remove the unconditional `ValueError` from that branch.

- [ ] **Step 5: Verify and commit**

Run:

```bash
python3 -m pytest tests/planu_core/test_blockworld_adapter.py tests/planu_core -q
python3 -m compileall -q planu_core blockworld/reasoners blockworld/evaluate_stochastic.py
```

Expected: tests pass and compilation emits no errors.

```bash
git add planu_core/adapters/blockworld.py blockworld/reasoners/algorithm/planU.py blockworld/evaluate_stochastic.py tests/planu_core/test_blockworld_adapter.py
git commit -m "refactor: migrate BlockWorld to shared PlanU core"
```

### Task 11: Document The Algorithm And One Reproduction Command

**Files:**
- Modify: `README.md`
- Modify: `scripts/PlanU_overcooked.sh`
- Create: `tests/planu_core/test_readme.py`

- [ ] **Step 1: Write the README contract test**

```python
# tests/planu_core/test_readme.py
from pathlib import Path


def test_readme_documents_one_primary_command_and_supported_adapters():
    readme = Path("README.md").read_text()
    assert "State Node" in readme
    assert "Action Node" in readme
    assert "Quantile Distribution" in readme
    assert "Upper Confidence Bounds with Curiosity" in readme
    assert "scripts/PlanU_overcooked.sh" in readme
    assert "BlockWorld" in readme
    assert "VirtualHome" in readme
    assert "WebShop" in readme
    assert "TravelPlanner" in readme
    assert readme.count("```bash") == 2
```

The two shell blocks are environment setup and the Overcooked command. Do not
add full commands for the other benchmarks.

- [ ] **Step 2: Run and verify failure**

Run: `python3 -m pytest tests/planu_core/test_readme.py -q`

Expected: FAIL because the algorithm section is absent.

- [ ] **Step 3: Update README**

Add:

- an algorithm section describing alternating state/action/outcome nodes;
- quantile return initialization and suffix-return update;
- the shared selection score and optional RND curiosity;
- a table marking Overcooked, VirtualHome, and BlockWorld as supported in phase
  one and WebShop/TravelPlanner as planned adapters;
- environment installation commands;
- one runnable example:

```bash
cd PlanU
bash scripts/PlanU_overcooked.sh
```

Explain that benchmark behavior is selected through configuration rather than
separate search implementations.

- [ ] **Step 4: Validate README and shell syntax**

Run:

```bash
python3 -m pytest tests/planu_core/test_readme.py -q
bash -n scripts/PlanU_overcooked.sh scripts/PlanU_Virtualhome.sh
```

Expected: test passes and shell validation emits no errors.

- [ ] **Step 5: Commit**

```bash
git add README.md scripts/PlanU_overcooked.sh tests/planu_core/test_readme.py
git commit -m "docs: explain unified PlanU algorithm"
```

### Task 12: Run Final Static And Behavioral Verification

**Files:**
- Modify only files required to fix failures caused by Tasks 1-11.

- [ ] **Step 1: Run the complete unit suite**

Run:

```bash
cd /Users/bytedance/personal_paper_code/PlanU
python3 -m pytest tests/planu_core -q
```

Expected: all PlanU core and adapter tests pass.

- [ ] **Step 2: Run syntax validation**

Run:

```bash
python3 -m compileall -q \
  planu_core \
  mcts/overcooked \
  mcts/virtualhome \
  blockworld/reasoners \
  blockworld/evaluate_stochastic.py
```

Expected: exit code 0 with no syntax errors.

- [ ] **Step 3: Run import smoke tests**

Run:

```bash
python3 -c "from planu_core import PlanUConfig, PlanUSearch"
python3 -c "from planu_core.adapters.overcooked import OvercookedAdapter"
python3 -c "from planu_core.adapters.virtualhome import VirtualHomeAdapter"
PYTHONPATH=blockworld python3 -c "from reasoners.algorithm import PlanU"
```

Expected: all commands exit successfully without loading model weights.

- [ ] **Step 4: Run environment smoke tests when dependencies are available**

Run:

```bash
python3 mcts/overcooked/PlanU_inference.py \
  --num-envs 1 \
  --maxiterations 1 \
  --depth 2 \
  --task 0 \
  --env-id Overcooked-LLMA-v4
```

Expected: one two-step-or-shorter trajectory completes and writes TensorBoard
metrics. If model weights are unavailable, record this as an environment
blocker rather than weakening the unit tests.

- [ ] **Step 5: Inspect scope and commit final corrections**

Run:

```bash
git status --short
git diff --check
git diff --stat HEAD~10
```

Confirm that `gym-macro-overcooked/**`, `virtual-home/virtual_home/**`, and
`webshop/**` are unchanged.

If final verification required corrections, stage only the listed implementation
files and commit them:

```bash
git add \
  planu_core \
  tests/planu_core \
  README.md \
  scripts/PlanU_overcooked.sh \
  scripts/PlanU_Virtualhome.sh \
  mcts/overcooked/PlanU_mcts.py \
  mcts/overcooked/PlanU_inference.py \
  mcts/overcooked/rnd.py \
  mcts/virtualhome/PlanU_v1.py \
  mcts/virtualhome/PlanU_v2.py \
  mcts/virtualhome/PlanU_inference_food.py \
  mcts/virtualhome/PlanU_entertainment.py \
  mcts/virtualhome/rnd.py \
  mcts/virtualhome/base_reward_model.py \
  blockworld/reasoners/algorithm/planU.py \
  blockworld/evaluate_stochastic.py
git commit -m "test: verify unified PlanU integration"
```

If verification changed no files, do not create an empty commit.
