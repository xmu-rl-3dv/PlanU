import copy
from dataclasses import fields, replace

import numpy as np
import pytest

from planu_core.adapters import WebShopAdapter as ExportedWebShopAdapter
from planu_core.adapters.webshop import (
    WebShopAdapter,
    WebShopRuntime,
    webshop_config,
)
from planu_core.interfaces import ActionCandidate, EnvironmentState
from planu_core.webshop import (
    WebShopAdapter as WebShopPackageAdapter,
    WebShopRuntime as WebShopPackageRuntime,
    WebShopAction,
    webshop_config as webshop_package_config,
)
from planu_core.webshop.client import WebShopPage


INIT_PAGE = WebShopPage(
    observation="WebShop\nInstruction: find a mug\n[Search]",
    buttons=("Search",),
)
SEARCH_PAGE = WebShopPage(
    observation="[Back to Search]\nPage 1\n[Next >]\n[A-1]\n[A-2]",
    buttons=("Back to Search", "Next >"),
    asins=("A-1", "A-2"),
)
ITEM_PAGE = WebShopPage(
    observation=(
        "[Back to Search]\n[< Prev]\nColor [Red][Blue]\n"
        "[Description]\n[Features]\n[Reviews]\n[Buy Now]"
    ),
    buttons=(
        "Back to Search",
        "< Prev",
        "Description",
        "Features",
        "Reviews",
        "Buy Now",
    ),
    option_types=(("Red", "Color"), ("Blue", "Color")),
)
SUBPAGE = WebShopPage(
    observation="[Back to Search]\n[< Prev]\nA useful description.",
    buttons=("Back to Search", "< Prev"),
)


class FakeClient:
    def __init__(self, reward=0.75):
        self.reward = reward
        self.calls = []

    def fetch(self, page_type, session_id, **kwargs):
        self.calls.append(
            (page_type, session_id, copy.deepcopy(kwargs))
        )
        if page_type == "init":
            return INIT_PAGE
        if page_type == "search":
            page_num = kwargs["page_num"]
            return replace(
                SEARCH_PAGE,
                observation=SEARCH_PAGE.observation.replace(
                    "Page 1",
                    "Page {}".format(page_num),
                ),
            )
        if page_type == "item":
            return ITEM_PAGE
        if page_type == "item_sub":
            return SUBPAGE
        if page_type == "end":
            return WebShopPage(
                observation="score: {}".format(self.reward),
                reward=self.reward,
            )
        raise AssertionError("unexpected page type: {}".format(page_type))


class RecordingRng:
    def __init__(self, latency):
        self.latency = latency
        self.calls = []

    def lognormal(self, *, mean, sigma):
        self.calls.append((mean, sigma))
        return self.latency


class ForbiddenRng:
    def lognormal(self, **kwargs):
        raise AssertionError("latency RNG must not be used")


def candidate(kind, argument):
    action = WebShopAction(kind, argument)
    return ActionCandidate(
        key=action.key,
        payload=action,
        text=action.render(),
    )


def make_state(
    page_type="item",
    observation=None,
    **runtime_overrides
):
    defaults = {
        "session_id": "session-7",
        "page_type": page_type,
        "query_string": "red mug",
        "page_num": 2,
        "asin": "A-1",
        "options": {"color": "Red"},
        "subpage": "",
        "buttons": ITEM_PAGE.buttons,
        "asins": (),
        "option_types": ITEM_PAGE.option_types,
        "step_count": 3,
        "failure_count": 1,
        "terminated": False,
        "truncated": False,
    }
    defaults.update(runtime_overrides)
    return EnvironmentState(
        observation=(
            ITEM_PAGE.observation if observation is None else observation
        ),
        runtime=WebShopRuntime(**defaults),
    )


def test_runtime_declares_the_complete_snapshot_schema():
    assert [field.name for field in fields(WebShopRuntime)] == [
        "session_id",
        "page_type",
        "query_string",
        "page_num",
        "asin",
        "options",
        "subpage",
        "buttons",
        "asins",
        "option_types",
        "step_count",
        "failure_count",
        "terminated",
        "truncated",
    ]


def test_adapter_symbols_are_exported_from_both_packages():
    assert ExportedWebShopAdapter is WebShopAdapter
    assert WebShopPackageAdapter is WebShopAdapter
    assert WebShopPackageRuntime is WebShopRuntime
    assert webshop_package_config is webshop_config


def test_reset_fetches_the_init_page_and_builds_a_fresh_episode_snapshot():
    client = FakeClient()
    adapter = WebShopAdapter(client, "session-7")

    first = adapter.reset(seed=11)
    first.runtime.options["color"] = "changed"
    second = adapter.reset(seed=11)

    assert client.calls == [
        ("init", "session-7", {}),
        ("init", "session-7", {}),
    ]
    assert second.observation == INIT_PAGE.observation
    assert second.runtime == WebShopRuntime(
        session_id="session-7",
        page_type="init",
        query_string="",
        page_num=1,
        asin="",
        options={},
        subpage="",
        buttons=INIT_PAGE.buttons,
        asins=(),
        option_types=(),
        step_count=0,
        failure_count=0,
        terminated=False,
        truncated=False,
    )


def test_clone_deeply_isolates_observation_and_runtime():
    adapter = WebShopAdapter(FakeClient(), "session-7")
    state = make_state(observation={"tokens": ["item"]})

    cloned = adapter.clone(state)
    cloned.observation["tokens"].append("changed")
    cloned.runtime.options["size"] = "large"

    assert cloned is not state
    assert cloned.runtime is not state.runtime
    assert state.observation == {"tokens": ["item"]}
    assert state.runtime.options == {"color": "Red"}


def test_state_key_covers_observation_and_every_runtime_field():
    adapter = WebShopAdapter(FakeClient(), "session-7")
    state = make_state()
    original_key = adapter.state_key(state)

    assert adapter.state_key(
        make_state(observation=state.observation + " changed")
    ) != original_key

    changed_values = {
        "session_id": "other-session",
        "page_type": "item_sub",
        "query_string": "blue cup",
        "page_num": 8,
        "asin": "A-2",
        "options": {"color": "Blue"},
        "subpage": "Description",
        "buttons": ("< Prev",),
        "asins": ("A-3",),
        "option_types": (("Large", "Size"),),
        "step_count": 4,
        "failure_count": 2,
        "terminated": True,
        "truncated": True,
    }
    for field_name, changed_value in changed_values.items():
        changed = adapter.clone(state)
        setattr(changed.runtime, field_name, changed_value)
        assert adapter.state_key(changed) != original_key, field_name


@pytest.mark.parametrize(
    ("terminated", "truncated"),
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_completion_queries_return_exact_runtime_flags(
    terminated,
    truncated,
):
    adapter = WebShopAdapter(FakeClient(), "session-7")
    state = make_state(terminated=terminated, truncated=truncated)

    assert adapter.is_terminal(state) is terminated
    assert adapter.is_truncated(state) is truncated


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (
            make_state(
                page_type="init",
                observation=INIT_PAGE.observation,
                query_string="",
                page_num=1,
                asin="",
                options={},
                buttons=INIT_PAGE.buttons,
                option_types=(),
            ),
            [],
        ),
        (
            make_state(
                page_type="search",
                observation=SEARCH_PAGE.observation,
                buttons=SEARCH_PAGE.buttons,
                asins=SEARCH_PAGE.asins,
                option_types=(),
            ),
            [
                "click[Back to Search]",
                "click[Next >]",
                "click[A-1]",
                "click[A-2]",
            ],
        ),
        (
            make_state(),
            [
                "click[Back to Search]",
                "click[< Prev]",
                "click[Description]",
                "click[Features]",
                "click[Reviews]",
                "click[Buy Now]",
                "click[Red]",
                "click[Blue]",
            ],
        ),
        (
            make_state(
                page_type="item_sub",
                observation=SUBPAGE.observation,
                subpage="Description",
                buttons=SUBPAGE.buttons,
                option_types=(),
            ),
            ["click[Back to Search]", "click[< Prev]"],
        ),
        (
            make_state(
                page_type="end",
                observation="score: 1",
                buttons=(),
                option_types=(),
                terminated=True,
            ),
            [],
        ),
    ],
)
def test_actions_return_deterministic_page_controls_only(state, expected):
    adapter = WebShopAdapter(FakeClient(), "session-7")

    actions = adapter.actions(state, state_visit_count=99)

    assert [action.text for action in actions] == expected
    assert [action.payload.render() for action in actions] == expected
    assert [action.key for action in actions] == [
        action.payload.key for action in actions
    ]


def test_preview_applies_successful_transition_without_consuming_rng():
    client = FakeClient()
    adapter = WebShopAdapter(client, "session-7")
    state = make_state(
        page_type="search",
        observation=SEARCH_PAGE.observation,
        buttons=SEARCH_PAGE.buttons,
        asins=SEARCH_PAGE.asins,
        option_types=(),
    )
    rng = np.random.default_rng(1234)
    rng_state = copy.deepcopy(rng.bit_generator.state)

    result = adapter.preview(state, candidate("click", "Next >"), rng)

    assert rng.bit_generator.state == rng_state
    assert result.state.runtime.page_num == 3
    assert result.state.runtime.step_count == 4
    assert result.reward == 0.0
    assert result.terminated is False
    assert result.truncated is False
    assert client.calls[-1] == (
        "search",
        "session-7",
        {"query_string": "red mug", "page_num": 3},
    )
    assert state.runtime.page_num == 2
    assert state.runtime.step_count == 3


def test_step_clones_parent_and_success_matches_preview_state_key():
    preview_client = FakeClient()
    step_client = FakeClient()
    preview_adapter = WebShopAdapter(preview_client, "session-7")
    step_adapter = WebShopAdapter(step_client, "session-7")
    state = make_state(
        page_type="search",
        observation=SEARCH_PAGE.observation,
        buttons=SEARCH_PAGE.buttons,
        asins=SEARCH_PAGE.asins,
        option_types=(),
    )
    parent_snapshot = copy.deepcopy(state)
    action = candidate("click", "A-2")

    preview = preview_adapter.preview(state, action, ForbiddenRng())
    result = step_adapter.step(state, action, RecordingRng(200.0))

    assert step_adapter.state_key(result.state) == preview_adapter.state_key(
        preview.state
    )
    assert state == parent_snapshot
    assert result.state.runtime.page_type == "item"
    assert result.state.runtime.asin == "A-2"


@pytest.mark.parametrize(
    ("page_type", "page_num", "action_text", "expected_type", "expected_page"),
    [
        ("search", 2, "< Prev", "search", 1),
        ("item", 2, "< Prev", "search", 2),
        ("item_sub", 2, "< Prev", "item", 2),
    ],
)
def test_prev_reverses_search_item_and_item_sub_navigation(
    page_type,
    page_num,
    action_text,
    expected_type,
    expected_page,
):
    client = FakeClient()
    adapter = WebShopAdapter(client, "session-7")
    if page_type == "search":
        state = make_state(
            page_type="search",
            observation=SEARCH_PAGE.observation,
            page_num=page_num,
            buttons=("Back to Search", "< Prev", "Next >"),
            asins=SEARCH_PAGE.asins,
            option_types=(),
        )
    elif page_type == "item_sub":
        state = make_state(
            page_type="item_sub",
            observation=SUBPAGE.observation,
            page_num=page_num,
            subpage="Description",
            buttons=SUBPAGE.buttons,
            option_types=(),
        )
    else:
        state = make_state(page_num=page_num)

    result = adapter.preview(
        state,
        candidate("click", action_text),
        ForbiddenRng(),
    )

    assert result.state.runtime.page_type == expected_type
    assert result.state.runtime.page_num == expected_page
    assert result.state.runtime.subpage == ""


@pytest.mark.parametrize("subpage", ["Description", "Features", "Reviews"])
def test_item_detail_buttons_open_the_matching_subpage(subpage):
    client = FakeClient()
    adapter = WebShopAdapter(client, "session-7")

    result = adapter.preview(
        make_state(),
        candidate("click", subpage),
        ForbiddenRng(),
    )

    assert result.state.runtime.page_type == "item_sub"
    assert result.state.runtime.subpage == subpage
    assert client.calls[-1][0] == "item_sub"
    assert client.calls[-1][2]["subpage"] == subpage


def test_item_option_click_updates_its_option_type_and_refetches_item():
    client = FakeClient()
    adapter = WebShopAdapter(client, "session-7")
    state = make_state(options={})

    result = adapter.preview(
        state,
        candidate("click", "Blue"),
        ForbiddenRng(),
    )

    assert result.state.runtime.options == {"color": "Blue"}
    assert state.runtime.options == {}
    assert client.calls[-1] == (
        "item",
        "session-7",
        {
            "query_string": "red mug",
            "page_num": 2,
            "asin": "A-1",
            "options": {"color": "Blue"},
        },
    )


def test_back_to_search_fetches_and_resets_all_page_navigation_fields():
    client = FakeClient()
    adapter = WebShopAdapter(client, "session-7")

    result = adapter.preview(
        make_state(
            page_type="item_sub",
            observation=SUBPAGE.observation,
            subpage="Description",
            buttons=SUBPAGE.buttons,
            option_types=(),
        ),
        candidate("click", "Back to Search"),
        ForbiddenRng(),
    )

    runtime = result.state.runtime
    assert client.calls[-1] == ("init", "session-7", {})
    assert runtime.page_type == "init"
    assert runtime.query_string == ""
    assert runtime.page_num == 1
    assert runtime.asin == ""
    assert runtime.options == {}
    assert runtime.subpage == ""
    assert runtime.buttons == INIT_PAGE.buttons
    assert runtime.asins == ()
    assert runtime.option_types == ()
    assert runtime.step_count == 4
    assert runtime.failure_count == 1


def test_search_is_legal_only_from_init_and_never_samples_latency():
    client = FakeClient()
    adapter = WebShopAdapter(client, "session-7")
    init = adapter.reset()

    result = adapter.step(
        init,
        candidate("search", "red ceramic mug"),
        ForbiddenRng(),
    )

    assert result.state.runtime.page_type == "search"
    assert result.state.runtime.query_string == "red ceramic mug"
    assert result.state.runtime.page_num == 1
    assert result.state.runtime.asin == ""
    assert result.state.runtime.options == {}
    assert result.state.runtime.step_count == 1
    assert client.calls[-1] == (
        "search",
        "session-7",
        {"query_string": "red ceramic mug", "page_num": 1},
    )

    invalid = adapter.step(
        result.state,
        candidate("search", "another query"),
        ForbiddenRng(),
    )
    assert invalid.state.observation == "Invalid action!"
    assert invalid.reward == -1.0
    assert invalid.terminated is False


@pytest.mark.parametrize("reward", [0.0, 0.25, 1.0])
def test_buy_now_always_terminates_for_the_full_reward_range(reward):
    client = FakeClient(reward=reward)
    adapter = WebShopAdapter(client, "session-7")

    result = adapter.step(
        make_state(),
        candidate("click", "Buy Now"),
        ForbiddenRng(),
    )

    assert result.state.runtime.page_type == "end"
    assert result.state.runtime.terminated is True
    assert result.reward == reward
    assert result.terminated is True
    assert result.truncated is False
    assert client.calls[-1] == (
        "end",
        "session-7",
        {"asin": "A-1", "options": {"color": "Red"}},
    )


def test_think_returns_ok_without_navigation_after_successful_latency():
    client = FakeClient()
    adapter = WebShopAdapter(client, "session-7")
    state = make_state()
    rng = RecordingRng(200.0)

    result = adapter.step(
        state,
        candidate("think", "compare these options"),
        rng,
    )

    assert result.state.observation == "OK."
    assert result.state.runtime.page_type == state.runtime.page_type
    assert result.state.runtime.query_string == state.runtime.query_string
    assert result.state.runtime.page_num == state.runtime.page_num
    assert result.state.runtime.asin == state.runtime.asin
    assert result.state.runtime.options == state.runtime.options
    assert result.state.runtime.step_count == state.runtime.step_count + 1
    assert result.reward == 0.0
    assert client.calls == []
    assert rng.calls == [(0, 10)]


@pytest.mark.parametrize(
    "action",
    [
        candidate("click", "not on this page"),
        candidate("search", "not legal from item"),
        ActionCandidate("bad", object(), "bad payload"),
    ],
)
def test_invalid_model_actions_are_structured_nonterminal_results(action):
    client = FakeClient()
    adapter = WebShopAdapter(client, "session-7")
    state = make_state()
    snapshot = copy.deepcopy(state)

    result = adapter.step(state, action, ForbiddenRng())

    assert result.state.observation == "Invalid action!"
    assert result.state.runtime == snapshot.runtime
    assert result.reward == -1.0
    assert result.terminated is False
    assert result.truncated is False
    assert result.info["invalid_action"] is True
    assert isinstance(result.info["reason"], str)
    assert client.calls == []
    assert state == snapshot


def test_latency_failure_preserves_page_snapshot_and_increments_only_failures():
    client = FakeClient()
    adapter = WebShopAdapter(client, "session-7")
    state = make_state(
        page_type="search",
        observation=SEARCH_PAGE.observation,
        buttons=SEARCH_PAGE.buttons,
        asins=SEARCH_PAGE.asins,
        option_types=(),
    )
    snapshot = copy.deepcopy(state)
    rng = RecordingRng(200.0001)

    result = adapter.step(
        state,
        candidate("click", "Next >"),
        rng,
    )

    assert result.state.observation == snapshot.observation
    before = snapshot.runtime
    after = result.state.runtime
    for field in fields(WebShopRuntime):
        if field.name != "failure_count":
            assert getattr(after, field.name) == getattr(before, field.name)
    assert after.failure_count == before.failure_count + 1
    assert result.reward == 0.0
    assert result.terminated is False
    assert result.truncated is False
    assert result.info["latency_failure"] is True
    assert result.info["latency"] == pytest.approx(200.0001)
    assert rng.calls == [(0, 10)]
    assert client.calls == []
    assert state == snapshot


def test_equal_latency_threshold_succeeds_and_uses_exact_distribution():
    client = FakeClient()
    adapter = WebShopAdapter(client, "session-7")
    state = make_state(
        page_type="search",
        observation=SEARCH_PAGE.observation,
        buttons=SEARCH_PAGE.buttons,
        asins=SEARCH_PAGE.asins,
        option_types=(),
    )
    rng = RecordingRng(200.0)

    result = adapter.step(
        state,
        candidate("click", "Next >"),
        rng,
    )

    assert result.state.runtime.page_num == 3
    assert result.info["latency_failure"] is False
    assert result.info["latency"] == 200.0
    assert rng.calls == [(0, 10)]


def test_seeded_latency_outcomes_are_reproducible():
    state = make_state(
        page_type="search",
        observation=SEARCH_PAGE.observation,
        buttons=SEARCH_PAGE.buttons,
        asins=SEARCH_PAGE.asins,
        option_types=(),
    )
    first = WebShopAdapter(FakeClient(), "session-7").step(
        state,
        candidate("click", "Next >"),
        np.random.default_rng(821),
    )
    second = WebShopAdapter(FakeClient(), "session-7").step(
        state,
        candidate("click", "Next >"),
        np.random.default_rng(821),
    )

    assert first == second


def test_webshop_config_matches_the_experiment_settings_exactly():
    config = webshop_config()

    assert config.n_quantiles == 51
    assert config.value_min == 0.0
    assert config.value_max == 1.0
    assert config.quantile_learning_rate == 0.9
    assert config.discount == 1.0
    assert config.curiosity_weight == 0.0
    assert config.train_curiosity is False
    assert config.include_preview_reward is True
    assert config.categorical_initialization is False
    assert config.max_depth == 10
    assert config.max_iterations == 10
    assert config.risk_distortion == 0.0
