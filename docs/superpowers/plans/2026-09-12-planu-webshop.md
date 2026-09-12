# PlanU WebShop Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace WebShop's broken private search with the shared PlanU core and validate one real task against a pinned official WebShop server.

**Architecture:** Add an optional action-provider boundary to `PlanUSearch`, then implement WebShop HTTP parsing, immutable benchmark state, model-backed candidate generation/scoring, and a thin runner in focused modules. The official WebShop server remains an external pinned process; legacy files become compatibility wrappers.

**Tech Stack:** Python 3.9, NumPy, requests, Beautiful Soup 4, OpenAI-compatible chat API, pytest, Bash, official `princeton-nlp/WebShop@64fa2a5`.

**Design:** `docs/superpowers/specs/2026-09-12-planu-webshop-design.md`

---

## File Map

**Create**

- `planu_core/webshop/__init__.py`: public WebShop API.
- `planu_core/webshop/actions.py`: typed actions and strict parser.
- `planu_core/webshop/client.py`: HTTP routes and HTML parsing.
- `planu_core/webshop/providers.py`: scripted and model action providers/scorers.
- `planu_core/webshop/runner.py`: CLI, search orchestration, output, provenance.
- `planu_core/adapters/webshop.py`: WebShop state and transition adapter.
- `tests/planu_core/test_action_provider.py`: core provider compatibility.
- `tests/planu_core/test_webshop_actions.py`: action parser tests.
- `tests/planu_core/test_webshop_client.py`: HTTP/parser tests.
- `tests/planu_core/test_webshop_adapter.py`: state and stochastic transition tests.
- `tests/planu_core/test_webshop_providers.py`: generation/scoring tests.
- `tests/planu_core/test_webshop_runner.py`: CLI and persistence tests.
- `tests/planu_core/test_webshop_security.py`: credential regression test.
- `scripts/bootstrap_webshop.sh`: pinned official server setup.
- `scripts/smoke_webshop.sh`: real single-task smoke test.

**Modify**

- `planu_core/interfaces.py`: add `ActionProvider`.
- `planu_core/search.py`: use an optional provider during expansion.
- `planu_core/__init__.py`: export `ActionProvider`.
- `planu_core/adapters/__init__.py`: export WebShop adapter/config.
- `planu_core/provenance.py`: allow benchmark-specific package metadata.
- `webshop/models.py`: remove credentials and delegate model calls.
- `webshop/planu.py`: replace private search with compatibility exports.
- `webshop/run.py`: delegate to the new runner.
- `webshop/planu.sh`: invoke the new module with the legacy profile.
- `requirements-experiments.txt`: add the HTML parser used by the client.
- `requirements-experiments-lock.txt`: record the tested parser version.
- `setup.py`: add the WebShop optional dependency extra.
- `README.md`: document setup, reference run, and smoke run.
- `docs/experiment-validation.md`: add the WebShop configuration matrix.
- `scripts/smoke_phase_one.sh`: no behavior change; it remains the regression
  command invoked by the phase-two gate.

---

### Task 1: Remove Tracked Credentials

**Files:**
- Create: `tests/planu_core/test_webshop_security.py`
- Modify: `webshop/models.py`
- Modify: `webshop/planu.py`

- [ ] **Step 1: Write the failing credential scan**

```python
# tests/planu_core/test_webshop_security.py
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SECRET_PATTERN = re.compile(r"sk-[A-Za-z0-9_-]{16,}")


def test_webshop_sources_do_not_contain_api_key_literals():
    paths = [
        ROOT / "webshop" / "models.py",
        ROOT / "webshop" / "planu.py",
    ]
    matches = {
        path.name: SECRET_PATTERN.findall(path.read_text(encoding="utf-8"))
        for path in paths
    }
    assert matches == {"models.py": [], "planu.py": []}
```

- [ ] **Step 2: Run the test and verify exposure**

Run:

```bash
python -m pytest tests/planu_core/test_webshop_security.py -v
```

Expected: FAIL and report credential-shaped literals in both files.

- [ ] **Step 3: Remove literals and import-time clients**

Replace module-level key assignment in `webshop/models.py` with:

```python
def require_api_key() -> str:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is required for the configured WebShop backend"
        )
    return api_key
```

Delete all active and commented credential literals from both files. Do not
construct an OpenAI client at import time. The currently exposed credentials
must be rotated outside the repository; history rewriting is not part of this
change.

- [ ] **Step 4: Verify the focused test**

Run:

```bash
python -m pytest tests/planu_core/test_webshop_security.py -v
```

Expected: `1 passed`.

- [ ] **Step 5: Commit**

```bash
git add webshop/models.py webshop/planu.py tests/planu_core/test_webshop_security.py
git commit -m "fix: remove WebShop credentials"
```

---

### Task 2: Add The Shared Action Provider Boundary

**Files:**
- Create: `tests/planu_core/test_action_provider.py`
- Modify: `planu_core/interfaces.py:102`
- Modify: `planu_core/search.py:133-164`
- Modify: `planu_core/search.py:204`
- Modify: `planu_core/__init__.py`

- [ ] **Step 1: Write failing provider tests**

```python
# tests/planu_core/test_action_provider.py
import numpy as np

from planu_core import PlanUConfig
from planu_core.interfaces import ActionCandidate
from planu_core.search import PlanUSearch
from tests.planu_core.fakes import FakeAdapter, UniformScorer


class RecordingProvider:
    def __init__(self):
        self.calls = []

    def actions(self, state, state_visit_count=0):
        self.calls.append((state.runtime["position"], state_visit_count))
        return [ActionCandidate("provided", 1, "provided")]


def test_search_uses_explicit_action_provider():
    adapter = FakeAdapter()
    provider = RecordingProvider()
    search = PlanUSearch(
        adapter,
        UniformScorer(),
        PlanUConfig(max_depth=1, max_iterations=1),
        action_provider=provider,
    )

    search.run_iteration(0, np.random.default_rng(7))

    assert provider.calls == [(0, 0)]
    assert adapter.action_calls == 0


def test_search_defaults_to_adapter_actions():
    adapter = FakeAdapter()
    search = PlanUSearch(
        adapter,
        UniformScorer(),
        PlanUConfig(max_depth=1, max_iterations=1),
    )

    search.run_iteration(0, np.random.default_rng(7))

    assert adapter.action_calls == 1
```

- [ ] **Step 2: Verify that the new constructor argument fails**

Run:

```bash
python -m pytest tests/planu_core/test_action_provider.py -v
```

Expected: FAIL with `unexpected keyword argument 'action_provider'`.

- [ ] **Step 3: Add the protocol and fallback**

Add to `planu_core/interfaces.py`:

```python
class ActionProvider(Protocol):
    def actions(
        self,
        state: EnvironmentState,
        state_visit_count: int = 0,
    ) -> Sequence[ActionCandidate]:
        ...
```

Extend `PlanUSearch.__init__`:

```python
def __init__(
    self,
    adapter: EnvironmentAdapter,
    scorer: ActionScorer,
    config: PlanUConfig,
    curiosity: Optional[CuriosityProvider] = None,
    action_provider: Optional[ActionProvider] = None,
):
    self.adapter = adapter
    self.scorer = scorer
    self.config = config
    self.curiosity = NullCuriosity() if curiosity is None else curiosity
    self.action_provider = action_provider
    self.root = None
```

Replace expansion action lookup with:

```python
provider = self.adapter if self.action_provider is None else self.action_provider
candidates = list(provider.actions(state, node.visit_count))
```

Export `ActionProvider` from `planu_core/__init__.py`.

- [ ] **Step 4: Run provider and existing search tests**

Run:

```bash
python -m pytest \
  tests/planu_core/test_action_provider.py \
  tests/planu_core/test_search.py \
  tests/planu_core/test_nodes.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add planu_core/interfaces.py planu_core/search.py planu_core/__init__.py tests/planu_core/test_action_provider.py
git commit -m "feat: add external action providers"
```

---

### Task 3: Define WebShop Actions

**Files:**
- Create: `planu_core/webshop/__init__.py`
- Create: `planu_core/webshop/actions.py`
- Create: `tests/planu_core/test_webshop_actions.py`

- [ ] **Step 1: Write strict parser tests**

```python
# tests/planu_core/test_webshop_actions.py
import pytest

from planu_core.webshop.actions import WebShopAction, parse_action


@pytest.mark.parametrize(
    ("text", "kind", "argument"),
    [
        ("search[folding desk]", "search", "folding desk"),
        ("click[Buy Now]", "click", "Buy Now"),
        ("think[compare prices]", "think", "compare prices"),
    ],
)
def test_parse_supported_actions(text, kind, argument):
    assert parse_action(text) == WebShopAction(kind, argument)


@pytest.mark.parametrize(
    "text",
    ["", "search[]", "click[Buy Now", "Action: click[x]", "open[x]"],
)
def test_parse_rejects_malformed_actions(text):
    with pytest.raises(ValueError, match="WebShop action"):
        parse_action(text)


def test_action_key_is_normalized():
    assert WebShopAction("search", "  Folding   Desk ").key == (
        "search",
        "Folding Desk",
    )
```

- [ ] **Step 2: Verify the module is missing**

Run:

```bash
python -m pytest tests/planu_core/test_webshop_actions.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement immutable action parsing**

```python
# planu_core/webshop/actions.py
import re
from dataclasses import dataclass
from typing import Tuple


_ACTION = re.compile(r"^(search|click|think)\[([^\n\[\]]+)\]$", re.IGNORECASE)


@dataclass(frozen=True)
class WebShopAction:
    kind: str
    argument: str

    def __post_init__(self) -> None:
        kind = self.kind.strip().lower()
        argument = " ".join(self.argument.strip().split())
        if kind not in {"search", "click", "think"} or not argument:
            raise ValueError("invalid WebShop action")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "argument", argument)

    @property
    def key(self) -> Tuple[str, str]:
        return self.kind, self.argument

    def render(self) -> str:
        return "{}[{}]".format(self.kind, self.argument)


def parse_action(text: str) -> WebShopAction:
    match = _ACTION.fullmatch(text.strip())
    if match is None:
        raise ValueError("invalid WebShop action: {!r}".format(text))
    return WebShopAction(match.group(1), match.group(2))
```

Export `WebShopAction` and `parse_action` from
`planu_core/webshop/__init__.py`.

- [ ] **Step 4: Run parser tests**

Run:

```bash
python -m pytest tests/planu_core/test_webshop_actions.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add planu_core/webshop tests/planu_core/test_webshop_actions.py
git commit -m "feat: define WebShop actions"
```

---

### Task 4: Implement The HTTP Client And HTML Parser

**Files:**
- Create: `planu_core/webshop/client.py`
- Create: `tests/planu_core/test_webshop_client.py`
- Modify: `requirements-experiments.txt`
- Modify: `requirements-experiments-lock.txt`
- Modify: `setup.py`

- [ ] **Step 1: Write parser and request tests**

```python
# tests/planu_core/test_webshop_client.py
from planu_core.webshop.client import WebShopHttpClient, parse_page


HTML = """
<html><body>
<div>Instruction: buy a red mug</div>
<button>Back to Search</button>
<a class="product-link">ASIN-1</a>
<div>Color</div><label>red</label>
<div>Your score (min 0.0, max 1.0)</div><div>0.75</div>
</body></html>
"""


def test_parse_page_extracts_actions_and_reward():
    page = parse_page(HTML)
    assert page.buttons == ("Back to Search",)
    assert page.asins == ("ASIN-1",)
    assert page.option_types == (("red", "Color"),)
    assert page.reward == 0.75


def test_client_uses_encoded_official_route(fake_session):
    fake_session.response_text = HTML
    client = WebShopHttpClient(
        "http://127.0.0.1:5000",
        session=fake_session,
        timeout=(1.0, 2.0),
    )

    client.fetch(
        session_id="fixed_1",
        page_type="search",
        query_string="red mug",
        page_num=1,
    )

    assert fake_session.last_timeout == (1.0, 2.0)
    assert "/search_results/fixed_1/red%20mug/1" in fake_session.last_url
```

Implement `fake_session` in the same test file with a response whose
`raise_for_status()` is observable.

- [ ] **Step 2: Run tests and verify the client is missing**

Run:

```bash
python -m pytest tests/planu_core/test_webshop_client.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement structured parsing and bounded HTTP**

```python
# planu_core/webshop/client.py
from dataclasses import dataclass
from typing import Mapping, Optional, Tuple
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup


@dataclass(frozen=True)
class WebShopPage:
    observation: str
    buttons: Tuple[str, ...]
    asins: Tuple[str, ...]
    option_types: Tuple[Tuple[str, str], ...]
    reward: float


def parse_page(html: str) -> WebShopPage:
    soup = BeautifulSoup(html, "html.parser")
    buttons = tuple(node.get_text(strip=True) for node in soup.find_all("button"))
    asins = tuple(
        node.get_text(strip=True)
        for node in soup.select(".product-link")
    )
    option_types = []
    current_heading = ""
    for node in soup.find_all(["div", "label"]):
        text = node.get_text(" ", strip=True)
        if node.name == "label":
            option_types.append((text, current_heading))
        elif text:
            current_heading = text
    visible = tuple(soup.stripped_strings)
    reward = 0.0
    marker = "Your score (min 0.0, max 1.0)"
    if marker in visible:
        reward = float(visible[visible.index(marker) + 1])
    observation = "\n".join(visible)
    return WebShopPage(
        observation,
        buttons,
        asins,
        tuple(option_types),
        reward,
    )
```

Implement `WebShopHttpClient.fetch()` with the five official routes, URL
encoding for every path component, `requests.Session.get(..., timeout=...)`,
`raise_for_status()`, and a contextual `WebShopHttpError`.

Add `beautifulsoup4==4.13.5` to the direct and locked experiment
requirements. Add a `webshop` extra containing `beautifulsoup4` and `openai`
to `setup.py`.

- [ ] **Step 4: Run focused tests and dependency validation**

Run:

```bash
python -m pytest tests/planu_core/test_webshop_client.py tests/planu_core/test_packaging.py -v
python -m pip check
```

Expected: tests PASS and `pip check` prints `No broken requirements found.`

- [ ] **Step 5: Commit**

```bash
git add planu_core/webshop/client.py tests/planu_core/test_webshop_client.py requirements-experiments.txt requirements-experiments-lock.txt setup.py
git commit -m "feat: add WebShop HTTP client"
```

---

### Task 5: Implement The WebShop Adapter

**Files:**
- Create: `planu_core/adapters/webshop.py`
- Create: `tests/planu_core/test_webshop_adapter.py`
- Modify: `planu_core/adapters/__init__.py`
- Modify: `planu_core/webshop/__init__.py`

- [ ] **Step 1: Write state, transition, and stochastic tests**

```python
# tests/planu_core/test_webshop_adapter.py
import numpy as np

from planu_core.adapters.webshop import WebShopAdapter, WebShopRuntime
from planu_core.interfaces import ActionCandidate
from planu_core.webshop.actions import WebShopAction


def candidate(text):
    action = WebShopAction(*text)
    return ActionCandidate(action.key, action, action.render())


def test_partial_credit_purchase_is_terminal(fake_webshop_client):
    fake_webshop_client.reward = 0.6
    adapter = WebShopAdapter(fake_webshop_client, "fixed_1")
    state = adapter.state_for_test(page_type="item")

    result = adapter.step(
        state,
        candidate(("click", "Buy Now")),
        np.random.default_rng(2),
    )

    assert result.reward == 0.6
    assert result.terminated is True
    assert result.truncated is False


def test_latency_failure_preserves_state(fake_webshop_client, monkeypatch):
    adapter = WebShopAdapter(fake_webshop_client, "fixed_1")
    state = adapter.state_for_test(page_type="search", asins=("A1",))
    before = adapter.state_key(state)
    monkeypatch.setattr(adapter, "_latency_succeeds", lambda rng: False)

    result = adapter.step(
        state,
        candidate(("click", "A1")),
        np.random.default_rng(3),
    )

    assert adapter.state_key(result.state) == before
    assert result.info["latency_failure"] is True
    assert fake_webshop_client.calls == []


def test_preview_forces_success_without_consuming_step_rng(
    fake_webshop_client,
):
    adapter = WebShopAdapter(fake_webshop_client, "fixed_1")
    state = adapter.state_for_test(page_type="search", asins=("A1",))
    rng = np.random.default_rng(4)
    expected = np.random.default_rng(4).random()

    adapter.preview(state, candidate(("click", "A1")), rng)

    assert rng.random() == expected
```

Include tests for reset, search, option selection, back, next/previous,
`think[...]`, invalid click reward, complete state key, clone isolation, depth
truncation, and the configured log-normal draw.

- [ ] **Step 2: Run tests and verify the adapter is missing**

Run:

```bash
python -m pytest tests/planu_core/test_webshop_adapter.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement state and adapter**

Define the runtime:

```python
@dataclass
class WebShopRuntime:
    session_id: str
    page_type: str = "init"
    query_string: str = ""
    page_num: int = 1
    asin: str = ""
    options: Dict[str, str] = field(default_factory=dict)
    subpage: str = ""
    buttons: Tuple[str, ...] = ()
    asins: Tuple[str, ...] = ()
    option_types: Tuple[Tuple[str, str], ...] = ()
    step_count: int = 0
    failure_count: int = 0
    terminated: bool = False
    truncated: bool = False
```

Implement:

```python
class WebShopAdapter:
    def reset(self, seed=None) -> EnvironmentState: ...
    def clone(self, state) -> EnvironmentState: ...
    def actions(self, state, state_visit_count=0): ...
    def preview(self, state, action, rng) -> TransitionResult: ...
    def step(self, state, action, rng) -> TransitionResult: ...
    def state_key(self, state): ...
    def is_terminal(self, state) -> bool: ...
    def is_truncated(self, state) -> bool: ...
```

Use one `_apply(state, action, inject_latency, rng)` path for preview and step.
Preview calls it on a clone with `inject_latency=False`; step calls it with
`inject_latency=True`. Implement latency exactly as:

```python
def _latency_succeeds(self, rng):
    return bool(rng.lognormal(mean=0.0, sigma=10.0) <= 200.0)
```

Return `info={"record_outcome": False}` only when preview intentionally does
not represent a real successor. Export the adapter and a
`webshop_config()` factory with the approved values.

- [ ] **Step 4: Run adapter and core stochastic tests**

Run:

```bash
python -m pytest \
  tests/planu_core/test_webshop_adapter.py \
  tests/planu_core/test_search.py \
  tests/planu_core/test_backup.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add planu_core/adapters/webshop.py planu_core/adapters/__init__.py planu_core/webshop/__init__.py tests/planu_core/test_webshop_adapter.py
git commit -m "feat: add WebShop adapter"
```

---

### Task 6: Add Candidate Providers And Scorers

**Files:**
- Create: `planu_core/text_backend.py`
- Create: `planu_core/webshop/providers.py`
- Create: `tests/planu_core/test_webshop_providers.py`
- Modify: `planu_core/webshop/__init__.py`

- [ ] **Step 1: Write provider and scorer tests**

```python
# tests/planu_core/test_webshop_providers.py
from planu_core.webshop.providers import (
    ModelWebShopActionProvider,
    ModelWebShopActionScorer,
    ScriptedWebShopActionProvider,
)


def test_model_provider_parses_deduplicates_and_preserves_order(fake_backend):
    fake_backend.texts = [
        "Thought: inspect\nAction: click[A1]",
        "click[A1]",
        "Action: click[A2]",
    ]
    provider = ModelWebShopActionProvider(fake_backend, candidate_count=3)

    actions = provider.actions(webshop_state(), 0)

    assert [item.text for item in actions] == ["click[A1]", "click[A2]"]


def test_scorer_parses_exact_integer_score(fake_backend):
    fake_backend.texts = ["Thus the correctness score is 7"]
    scorer = ModelWebShopActionScorer(fake_backend)

    assert scorer.score("page", [candidate("click[A1]")]) == [0.7]


def test_scripted_provider_reaches_buy_without_model():
    provider = ScriptedWebShopActionProvider(search_query="mug")
    assert provider.actions(initial_state(), 0)[0].text == "search[mug]"
    assert provider.actions(results_state(asins=("A1",)), 0)[0].text == (
        "click[A1]"
    )
```

- [ ] **Step 2: Run tests and verify the provider module is missing**

Run:

```bash
python -m pytest tests/planu_core/test_webshop_providers.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the backend contract and providers**

```python
# planu_core/text_backend.py
from dataclasses import dataclass
from typing import Protocol, Sequence


@dataclass(frozen=True)
class GenerationResult:
    texts: Sequence[str]
    prompt_tokens: int = 0
    completion_tokens: int = 0


class TextBackend(Protocol):
    @property
    def model_identifier(self) -> str:
        ...

    def generate(
        self,
        prompt: str,
        n: int,
        temperature: float,
        max_tokens: int,
        stop: Sequence[str],
    ) -> GenerationResult:
        ...
```

Implement `OpenAICompatibleBackend` with lazy client construction,
environment-only credentials, bounded retries, and structured token usage.

Implement the model provider with the existing CoT prompt text, strict
`Action:` extraction, stable deduplication, and exactly
`candidate_count` requested completions. Implement the scorer with the existing
value prompt and exact regex:

```python
_SCORE = re.compile(r"\bcorrectness score is\s+(10|[1-9])\b", re.IGNORECASE)
```

Normalize scores by dividing by `10.0`; reject missing or non-finite scores.
Implement the scripted provider as page-state logic for real smoke use.

- [ ] **Step 4: Run provider and scorer tests**

Run:

```bash
python -m pytest tests/planu_core/test_webshop_providers.py tests/planu_core/test_scorers.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add planu_core/text_backend.py planu_core/webshop/providers.py planu_core/webshop/__init__.py tests/planu_core/test_webshop_providers.py
git commit -m "feat: add WebShop action policies"
```

---

### Task 7: Build The Thin Runner And Compatibility Entrypoints

**Files:**
- Create: `planu_core/webshop/runner.py`
- Create: `tests/planu_core/test_webshop_runner.py`
- Modify: `planu_core/provenance.py`
- Modify: `webshop/run.py`
- Modify: `webshop/planu.py`
- Modify: `webshop/models.py`
- Modify: `webshop/planu.sh`

- [ ] **Step 1: Write runner contract tests**

```python
# tests/planu_core/test_webshop_runner.py
import json

from planu_core.webshop.runner import parse_args, run


def test_reference_defaults():
    args = parse_args([])
    assert args.backend == "qwen-plus"
    assert args.temperature == 0.8
    assert args.task_start_index == 1
    assert args.task_end_index == 50
    assert args.n_generate_sample == 5
    assert args.n_evaluate_sample == 1
    assert args.iterations == 10
    assert args.depth == 10


def test_runner_writes_result_and_provenance(tmp_path, fake_dependencies):
    code = run(
        parse_args(
            [
                "--smoke",
                "--task-start-index", "1",
                "--task-end-index", "2",
                "--output-dir", str(tmp_path),
            ]
        ),
        dependencies=fake_dependencies,
    )

    assert code == 0
    result = json.loads((tmp_path / "results.jsonl").read_text().splitlines()[0])
    assert result["task_id"] == "fixed_1"
    assert result["provenance"]["webshop_commit"] == (
        "64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd"
    )
```

Also assert that `effective_config.json` and `run_metadata.json` are written
before the first search iteration and contain no credential values.

- [ ] **Step 2: Run tests and verify the runner is missing**

Run:

```bash
python -m pytest tests/planu_core/test_webshop_runner.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement CLI orchestration**

The runner must construct:

```python
config = PlanUConfig(
    n_quantiles=51,
    value_min=0.0,
    value_max=1.0,
    quantile_learning_rate=0.9,
    discount=1.0,
    curiosity_weight=0.0,
    include_preview_reward=True,
    max_depth=args.depth,
    max_iterations=args.iterations,
)
search = PlanUSearch(
    adapter,
    scorer,
    config,
    action_provider=provider,
)
```

For each half-open task index, use one persistent tree for that task, execute
up to `iterations`, choose the highest observed terminal reward, and append one
JSON line. Add WebShop source commit and requested package distributions to
the shared provenance helpers without changing phase-one output.

Replace `webshop/run.py` with:

```python
from planu_core.webshop.runner import main


if __name__ == "__main__":
    raise SystemExit(main())
```

Replace `webshop/planu.py` and `webshop/models.py` with documented compatibility
exports only. Update `webshop/planu.sh` to call:

```bash
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
  "$@"
```

- [ ] **Step 4: Run runner, security, and packaging tests**

Run:

```bash
python -m pytest \
  tests/planu_core/test_webshop_runner.py \
  tests/planu_core/test_webshop_security.py \
  tests/planu_core/test_packaging.py -v
bash -n webshop/planu.sh
```

Expected: all tests PASS and shell syntax exits `0`.

- [ ] **Step 5: Commit**

```bash
git add planu_core/webshop/runner.py planu_core/provenance.py webshop/run.py webshop/planu.py webshop/models.py webshop/planu.sh tests/planu_core/test_webshop_runner.py
git commit -m "feat: run WebShop through PlanU core"
```

---

### Task 8: Add Pinned Server Bootstrap And Real Smoke

**Files:**
- Create: `scripts/bootstrap_webshop.sh`
- Create: `scripts/smoke_webshop.sh`
- Modify: `tests/planu_core/test_webshop_runner.py`

- [ ] **Step 1: Write shell contract tests**

Add:

```python
def test_webshop_scripts_pin_source_and_reject_mock_smoke():
    root = Path(__file__).resolve().parents[2]
    bootstrap = (root / "scripts" / "bootstrap_webshop.sh").read_text()
    smoke = (root / "scripts" / "smoke_webshop.sh").read_text()
    assert "64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd" in bootstrap
    assert "princeton-nlp/WebShop.git" in bootstrap
    assert "--smoke" in smoke
    assert "WEBSHOP_URL" in smoke
    assert "mock" not in smoke.lower()
```

- [ ] **Step 2: Run the contract test and verify scripts are missing**

Run:

```bash
python -m pytest tests/planu_core/test_webshop_runner.py::test_webshop_scripts_pin_source_and_reject_mock_smoke -v
```

Expected: FAIL because the scripts do not exist.

- [ ] **Step 3: Implement bootstrap and smoke scripts**

`scripts/bootstrap_webshop.sh` must:

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT="${WEBSHOP_ROOT:-$PWD/external/WebShop}"
COMMIT="64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd"
ENV_PREFIX="${WEBSHOP_ENV_PREFIX:-$ROOT/.conda-planu}"
if [[ ! -d "$ROOT/.git" ]]; then
  git clone https://github.com/princeton-nlp/WebShop.git "$ROOT"
fi
git -C "$ROOT" fetch origin "$COMMIT"
git -C "$ROOT" checkout --detach "$COMMIT"
conda create -y -p "$ENV_PREFIX" python=3.8.13
(
  eval "$(conda shell.bash hook)"
  conda activate "$ENV_PREFIX"
  export CONDA_ALWAYS_YES=true
  cd "$ROOT"
  ./setup.sh -d small
)
```

`scripts/smoke_webshop.sh` must either use an already reachable
`WEBSHOP_URL` or start the pinned server with:

```bash
(
  cd "$WEBSHOP_ROOT"
  exec "${WEBSHOP_PYTHON:-$WEBSHOP_ROOT/.conda-planu/bin/python}" \
    -m web_agent_site.app --log --attrs
)
```

It polls `http://127.0.0.1:3000/fixed_1` with a bounded timeout, runs one
`fixed_*` task with the scripted provider, and traps only the PID it started.

The runner writes:

```text
effective_config.json
run_metadata.json
results.jsonl
```

- [ ] **Step 4: Validate shell syntax**

Run:

```bash
bash -n scripts/bootstrap_webshop.sh
bash -n scripts/smoke_webshop.sh
```

Expected: both commands exit `0`.

- [ ] **Step 5: Commit**

```bash
git add scripts/bootstrap_webshop.sh scripts/smoke_webshop.sh tests/planu_core/test_webshop_runner.py
git commit -m "test: add real WebShop smoke workflow"
```

---

### Task 9: Run The Real WebShop Experiment Path

**Files:**
- Modify only if a real-run defect is reproduced in a focused failing test.

- [ ] **Step 1: Prepare the pinned small server**

Run:

```bash
WEBSHOP_ROOT="$PWD/external/WebShop" scripts/bootstrap_webshop.sh
```

Expected: checkout HEAD is
`64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd` and small data setup completes.

- [ ] **Step 2: Execute the real smoke**

Run:

```bash
WEBSHOP_ROOT="$PWD/external/WebShop" \
RUN_ROOT=/tmp/planu-webshop-smoke \
scripts/smoke_webshop.sh
```

Expected: exit `0`; output reports a real `fixed_*` session, at least one HTTP
transition, at least one quantile backup, and no mock backend.

- [ ] **Step 3: Inspect persisted evidence**

Run:

```bash
python -m json.tool /tmp/planu-webshop-smoke/effective_config.json
python -m json.tool /tmp/planu-webshop-smoke/run_metadata.json
python -c "import json; print(json.loads(open('/tmp/planu-webshop-smoke/results.jsonl').readline())['task_id'])"
```

Expected: valid JSON and task ID beginning with `fixed_`.

- [ ] **Step 4: Convert any defect into a regression test**

For each defect, add one test to the narrowest existing WebShop test module,
run it to observe failure, implement the smallest correction, then rerun the
focused test and smoke command. Do not change reference parameters to make the
smoke pass.

- [ ] **Step 5: Commit real-run corrections**

```bash
git add planu_core webshop scripts tests/planu_core
git commit -m "fix: complete WebShop experiment smoke"
```

Skip this commit only when the real smoke requires no correction.

---

### Task 10: Document And Regress The WebShop Migration

**Files:**
- Modify: `README.md`
- Modify: `docs/experiment-validation.md`
- Modify: `tests/planu_core/test_readme.py`

- [ ] **Step 1: Write failing documentation tests**

```python
def test_readme_documents_webshop_reference_and_smoke_commands():
    readme = README.read_text(encoding="utf-8")
    assert "scripts/bootstrap_webshop.sh" in readme
    assert "scripts/smoke_webshop.sh" in readme
    assert "python -m planu_core.webshop.runner" in readme
    assert "64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd" in readme
    assert "does not reproduce paper metrics" in readme
```

- [ ] **Step 2: Run the documentation test**

Run:

```bash
python -m pytest tests/planu_core/test_readme.py::test_readme_documents_webshop_reference_and_smoke_commands -v
```

Expected: FAIL because WebShop is still documented as planned.

- [ ] **Step 3: Update documentation**

Change the status table to `WebShop | Phase two | Supported`. Add separate
sections for:

```text
WebShop server setup
WebShop reference configuration
WebShop real smoke
Intentional corrections from the legacy WebShop implementation
```

Add the exact configuration table from the approved design to
`docs/experiment-validation.md`. State that the smoke validates execution, not
metric reproduction.

- [ ] **Step 4: Run all automated verification**

Run:

```bash
python -m pytest -q
python3.9 -m compileall -q planu_core webshop tests
python setup.py check --strict
python setup.py sdist bdist_wheel
bash -n webshop/planu.sh scripts/bootstrap_webshop.sh scripts/smoke_webshop.sh scripts/smoke_phase_one.sh
git diff --check
```

Expected: all tests PASS; compile, packaging, shell, and diff checks exit `0`.

- [ ] **Step 5: Run cross-benchmark smoke**

Run:

```bash
scripts/smoke_phase_one.sh
scripts/smoke_webshop.sh
```

Expected: Overcooked, VirtualHome food, VirtualHome entertainment, BlockWorld,
and WebShop smoke paths all exit `0`.

- [ ] **Step 6: Commit**

```bash
git add README.md docs/experiment-validation.md tests/planu_core/test_readme.py
git commit -m "docs: document WebShop experiment"
```

---

## WebShop Completion Gate

Before starting the TravelPlanner plan, verify:

```bash
git status --short
git log --oneline -10
```

Only known generated metadata may remain uncommitted. Do not carry source or
test changes into the TravelPlanner tasks. Record the WebShop real-smoke output
directory and exact commit in the implementation report.
