import ast
import hashlib
import math
import sys
import types
from dataclasses import FrozenInstanceError
from inspect import signature
from pathlib import Path
from typing import get_type_hints

import pytest

from planu_core.adapters.webshop import WebShopAdapter
from planu_core.interfaces import ActionCandidate, EnvironmentState
from planu_core.text_backend import (
    GenerationResult,
    OpenAICompatibleBackend,
    TextBackend,
)
from planu_core.webshop import (
    ModelWebShopActionProvider as ExportedModelProvider,
    ModelWebShopActionScorer as ExportedModelScorer,
    ScriptedWebShopActionProvider as ExportedScriptedProvider,
)
from planu_core.webshop.actions import WebShopAction
from planu_core.webshop.providers import (
    LEGACY_COT_PROMPT,
    LEGACY_VALUE_PROMPT,
    ModelWebShopActionProvider,
    ModelWebShopActionScorer,
    ScriptedWebShopActionProvider,
)

ROOT = Path(__file__).resolve().parents[2]


class FakeBackend:
    model_identifier = "fake/model"

    def __init__(self, results=()):
        self.results = list(results)
        self.calls = []

    def generate(self, prompt, n, temperature, max_tokens, stop):
        self.calls.append(
            {
                "prompt": prompt,
                "n": n,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stop": stop,
            }
        )
        return self.results.pop(0)


def result(*texts, prompt_tokens=0, completion_tokens=0):
    return GenerationResult(
        texts,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


def state(
    page_type="item",
    observation="current public page",
    buttons=("Buy Now",),
    asins=(),
    option_types=(),
    options=None,
    terminated=False,
    truncated=False,
):
    runtime = types.SimpleNamespace(
        page_type=page_type,
        buttons=tuple(buttons),
        asins=tuple(asins),
        option_types=tuple(option_types),
        options={} if options is None else dict(options),
        terminated=terminated,
        truncated=truncated,
        private_secret="must-not-be-read",
    )
    return EnvironmentState(observation=observation, runtime=runtime)


def candidate(text, trajectory=None):
    action = WebShopAction(*text[:-1].split("[", 1))
    metadata = {}
    if trajectory is not None:
        metadata["trajectory"] = trajectory
    return ActionCandidate(action.key, action, action.render(), metadata)


def legacy_prompt_source(name):
    source = (ROOT / "webshop" / "prompt.py").read_text(encoding="utf-8")
    module = ast.parse(source)
    for node in module.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == name
        ):
            return ast.literal_eval(node.value)
    raise AssertionError("legacy prompt {!r} was not found".format(name))


def sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_cot_prompt_is_byte_exact_legacy_prompt2():
    legacy = legacy_prompt_source("prompt2")
    trajectory = "Webshop\nInstruction:\nfind {one} item\n[Search]"

    assert sha256(legacy) == (
        "5cc7e1f4b72973b27e1579b4706ce7292b01dc13cd4f12638715f35b6b8ee510"
    )
    assert LEGACY_COT_PROMPT == legacy
    assert LEGACY_COT_PROMPT.format(input=trajectory) == legacy.format(
        input=trajectory
    )


def test_value_prompt_is_byte_exact_legacy_score_prompt():
    legacy = legacy_prompt_source("score_prompt")
    evaluation = "Webshop\nInstruction:\nfind {one} item\n\nReflection: "

    assert sha256(legacy) == (
        "94b7184c580857bb3682c6f8d4f181601cff76d24a01d025c52a2f4183d93145"
    )
    assert LEGACY_VALUE_PROMPT == legacy
    assert LEGACY_VALUE_PROMPT.format(s="", input=evaluation) == legacy.format(
        s="",
        input=evaluation,
    )


def test_generation_result_and_backend_protocol_contract():
    generation = GenerationResult(["one", "two"], 3, 4)
    hints = get_type_hints(TextBackend.generate)
    parameters = signature(TextBackend.generate).parameters

    assert generation.texts == ("one", "two")
    assert generation.prompt_tokens == 3
    assert generation.completion_tokens == 4
    assert list(parameters) == [
        "self",
        "prompt",
        "n",
        "temperature",
        "max_tokens",
        "stop",
    ]
    assert hints["return"] is GenerationResult
    with pytest.raises(FrozenInstanceError):
        generation.prompt_tokens = 9


def test_webshop_provider_symbols_are_exported():
    assert ExportedModelProvider is ModelWebShopActionProvider
    assert ExportedModelScorer is ModelWebShopActionScorer
    assert ExportedScriptedProvider is ScriptedWebShopActionProvider


def test_model_provider_uses_effective_legacy_prompt_and_public_observation_only(
    monkeypatch,
):
    monkeypatch.setenv("OPENAI_API_KEY", "top-secret-api-key")
    backend = FakeBackend(
        [result("click[A1]", prompt_tokens=11, completion_tokens=2)]
    )
    provider = ModelWebShopActionProvider(backend, candidate_count=1)

    actions = provider.actions(state(), 7)

    prompt = backend.calls[0]["prompt"]
    assert "You are interacting with a Webshop environment." in prompt
    assert "Only output the action name, nothing else." in prompt
    assert "current public page" in prompt
    assert "must-not-be-read" not in prompt
    assert "top-secret-api-key" not in prompt
    assert actions[0].metadata["trajectory"] == "current public page"
    assert provider.prompt_tokens == 11
    assert provider.completion_tokens == 2


def test_model_provider_requests_exact_count_parses_and_stably_deduplicates():
    backend = FakeBackend(
        [
            result(
                "Thought: inspect\nAction: click[A1]",
                "click[A1]",
                "Action: click[A2]",
                "Action: click[A1]",
                prompt_tokens=17,
                completion_tokens=9,
            )
        ]
    )
    provider = ModelWebShopActionProvider(
        backend,
        candidate_count=4,
        temperature=0.8,
        max_tokens=100,
    )

    actions = provider.actions(state(page_type="search"), 0)

    assert [item.text for item in actions] == ["click[A1]", "click[A2]"]
    assert [item.key for item in actions] == [
        ("click", "A1"),
        ("click", "A2"),
    ]
    assert backend.calls[0]["n"] == 4
    assert backend.calls[0]["temperature"] == 0.8
    assert backend.calls[0]["max_tokens"] == 100
    assert backend.calls[0]["stop"] == ("Observation",)
    assert provider.token_usage == {
        "prompt_tokens": 17,
        "completion_tokens": 9,
        "total_tokens": 26,
    }


@pytest.mark.parametrize(
    "malformed",
    [
        "prefix click[A1]",
        "Action: click[A1] suffix",
        "Thought: Action: click[A1]",
        "Action: click[A1]\nAction: click[A2]",
        "Action:\nclick[A1]",
        "open[A1]",
        "click[A[1]]",
    ],
)
def test_model_provider_preserves_malformed_or_ambiguous_completions(malformed):
    backend = FakeBackend([result(malformed, "click[A2]")])
    provider = ModelWebShopActionProvider(backend, candidate_count=2)
    page_state = state()

    actions = provider.actions(page_state, 0)

    assert len(actions) == 2
    invalid, valid = actions
    assert invalid.key == ("invalid", " ".join(malformed.split()))
    assert invalid.payload == malformed
    assert invalid.text == malformed
    assert isinstance(invalid.metadata["parse_error"], str)
    assert invalid.metadata["parse_error"]
    for field in ("prompt", "trajectory", "model_identifier"):
        assert invalid.metadata[field] == valid.metadata[field]

    adapter = WebShopAdapter(
        types.SimpleNamespace(fetch=lambda *args, **kwargs: None),
        "session-invalid",
    )
    transition = adapter.step(page_state, invalid, object())

    assert transition.state.observation == "Invalid action!"
    assert transition.reward == -1.0
    assert transition.info["invalid_action"] is True


def test_model_provider_stably_deduplicates_invalid_and_valid_actions():
    backend = FakeBackend(
        [
            result(
                "bad output",
                "click[A1]",
                "bad output",
                "other bad output",
                "click[A1]",
            )
        ]
    )
    provider = ModelWebShopActionProvider(backend, candidate_count=5)

    actions = provider.actions(state(), 0)

    assert [item.key for item in actions] == [
        ("invalid", "bad output"),
        ("click", "A1"),
        ("invalid", "other bad output"),
    ]
    assert [item.payload for item in actions] == [
        "bad output",
        WebShopAction("click", "A1"),
        "other bad output",
    ]


def test_model_provider_preserves_empty_output_as_deterministic_invalid_action():
    backend = FakeBackend([result("", " \t\n")])
    provider = ModelWebShopActionProvider(backend, candidate_count=2)

    actions = provider.actions(state(), 0)

    assert len(actions) == 1
    assert actions[0].key == ("invalid", "")
    assert actions[0].payload == ""
    assert actions[0].text == ""
    assert actions[0].metadata["parse_error"]


def test_model_provider_rejects_wrong_backend_completion_count():
    backend = FakeBackend([result("click[A1]")])
    provider = ModelWebShopActionProvider(backend, candidate_count=2)

    with pytest.raises(ValueError, match="completion count"):
        provider.actions(state(), 0)


def test_scorer_uses_legacy_value_prompt_and_candidate_trajectory():
    backend = FakeBackend(
        [result("Thus the correctness score is 7", prompt_tokens=13)]
    )
    scorer = ModelWebShopActionScorer(backend)

    scores = scorer.score(
        "fallback observation",
        [candidate("click[A1]", trajectory="public trajectory")],
    )

    assert scores == [0.7]
    prompt = backend.calls[0]["prompt"]
    assert "ideal score of 1.0" in prompt
    assert "public trajectory" in prompt
    assert "Action: click[A1]" in prompt
    assert "fallback observation" not in prompt
    assert scorer.prompt_tokens == 13


def test_scorer_parses_exact_ten_without_the_legacy_one_substring_bug():
    backend = FakeBackend(
        [
            result(
                "Analysis mentions 1 early.\n"
                "Thus the correctness score is 10"
            )
        ]
    )
    scorer = ModelWebShopActionScorer(backend)

    assert scorer.score("page", [candidate("click[A1]")]) == [1.0]


@pytest.mark.parametrize(
    "text",
    [
        "Thus the correctness score is 0",
        "Thus the correctness score is 11",
        "Thus the correctness score is 100",
        "The correctness score might be high",
        "Thus the correctness score is nan",
    ],
)
def test_scorer_rejects_malformed_scores(text):
    backend = FakeBackend([result(text)])
    scorer = ModelWebShopActionScorer(backend)

    with pytest.raises(ValueError, match="correctness score"):
        scorer.score("page", [candidate("click[A1]")])


def test_scorer_aggregates_multiple_samples_deterministically():
    backend = FakeBackend(
        [
            result(
                "Thus the correctness score is 9",
                "Thus the correctness score is 3",
                "Thus the correctness score is 6",
                prompt_tokens=19,
                completion_tokens=8,
            )
        ]
    )
    scorer = ModelWebShopActionScorer(backend, n_evaluate_sample=3)

    scores = scorer.score("page", [candidate("click[A1]")])

    assert scores == [0.6]
    assert backend.calls[0]["n"] == 3
    assert scorer.token_usage == {
        "prompt_tokens": 19,
        "completion_tokens": 8,
        "total_tokens": 27,
    }


def test_scorer_returns_one_finite_normalized_value_per_candidate():
    backend = FakeBackend(
        [
            result("Thus the correctness score is 1"),
            result("Thus the correctness score is 10"),
        ]
    )
    scorer = ModelWebShopActionScorer(backend)
    candidates = [candidate("click[A1]"), candidate("click[A2]")]

    scores = scorer.score("page", candidates)

    assert len(scores) == len(candidates)
    assert scores == [0.1, 1.0]
    assert all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in scores)


def test_scorer_rejects_wrong_sample_dimensions():
    backend = FakeBackend(
        [
            result(
                "Thus the correctness score is 7",
                "Thus the correctness score is 8",
            )
        ]
    )
    scorer = ModelWebShopActionScorer(backend)

    with pytest.raises(ValueError, match="completion count"):
        scorer.score("page", [candidate("click[A1]")])


def test_scorer_rejects_nonfinite_aggregate(monkeypatch):
    backend = FakeBackend([result("Thus the correctness score is 7")])
    scorer = ModelWebShopActionScorer(backend)
    monkeypatch.setattr(scorer, "_parse_score", lambda text: math.nan)

    with pytest.raises(ValueError, match="finite"):
        scorer.score("page", [candidate("click[A1]")])


def test_scorer_defaults_to_one_evaluation_sample():
    backend = FakeBackend([result("Thus the correctness score is 8")])
    scorer = ModelWebShopActionScorer(backend)

    scorer.score("page", [candidate("click[A1]")])

    assert scorer.n_evaluate_sample == 1
    assert backend.calls[0]["n"] == 1


def test_scripted_provider_follows_complete_page_state_sequence():
    provider = ScriptedWebShopActionProvider(search_query="mug")

    scenarios = [
        (state(page_type="init"), "search[mug]"),
        (
            state(page_type="search", asins=("A1", "A2")),
            "click[A1]",
        ),
        (
            state(
                option_types=(
                    ("Red", "Color"),
                    ("Blue", "Color"),
                    ("Small", "Size"),
                ),
            ),
            "click[Red]",
        ),
        (
            state(
                option_types=(
                    ("Red", "Color"),
                    ("Blue", "Color"),
                    ("Small", "Size"),
                ),
                options={"color": "Red"},
            ),
            "click[Small]",
        ),
        (
            state(
                option_types=(
                    ("Red", "Color"),
                    ("Blue", "Color"),
                    ("Small", "Size"),
                ),
                options={"color": "Red", "size": "Small"},
            ),
            "click[Buy Now]",
        ),
        (
            state(page_type="item_sub", buttons=("Back to Search", "< Prev")),
            "click[< Prev]",
        ),
    ]

    for page_state, expected in scenarios:
        actions = provider.actions(page_state, 0)
        assert [action.text for action in actions] == [expected]


@pytest.mark.parametrize(
    "page_state",
    [
        state(page_type="search", asins=()),
        state(page_type="item", buttons=(), option_types=()),
        state(page_type="item_sub", buttons=("Back to Search",)),
        state(terminated=True),
        state(truncated=True),
    ],
)
def test_scripted_provider_returns_no_action_when_path_cannot_continue(
    page_state,
):
    provider = ScriptedWebShopActionProvider(search_query="mug")

    assert provider.actions(page_state, 0) == []


class FakeTransientError(Exception):
    pass


class FakeBadRequestError(Exception):
    pass


class FakeCompletions:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def response(texts, prompt_tokens=0, completion_tokens=0):
    return types.SimpleNamespace(
        choices=[
            types.SimpleNamespace(message=types.SimpleNamespace(content=text))
            for text in texts
        ],
        usage=types.SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        ),
    )


def fake_openai_module(completions, constructor_calls):
    class FakeOpenAI:
        def __init__(self, **kwargs):
            constructor_calls.append(kwargs)
            self.chat = types.SimpleNamespace(completions=completions)

    return types.SimpleNamespace(
        OpenAI=FakeOpenAI,
        APIConnectionError=FakeTransientError,
        APITimeoutError=type("FakeTimeoutError", (Exception,), {}),
        RateLimitError=type("FakeRateLimitError", (Exception,), {}),
        InternalServerError=type("FakeInternalServerError", (Exception,), {}),
        BadRequestError=FakeBadRequestError,
    )


def test_openai_backend_is_lazy_uses_env_key_and_accounts_tokens(
    monkeypatch,
):
    constructor_calls = []
    completions = FakeCompletions(
        [response(["first", "second"], prompt_tokens=23, completion_tokens=5)]
    )
    monkeypatch.setitem(
        sys.modules,
        "openai",
        fake_openai_module(completions, constructor_calls),
    )
    monkeypatch.setenv("OPENAI_API_KEY", "environment-only-key")
    backend = OpenAICompatibleBackend(
        model="qwen-plus",
        base_url="https://model.example.test/v1",
        timeout=12.5,
        retry_limit=2,
    )

    assert constructor_calls == []
    assert backend.model_identifier == "qwen-plus"

    generated = backend.generate(
        "public prompt",
        n=2,
        temperature=0.8,
        max_tokens=100,
        stop=("Observation",),
    )

    assert constructor_calls == [
        {
            "api_key": "environment-only-key",
            "base_url": "https://model.example.test/v1",
            "timeout": 12.5,
            "max_retries": 0,
        }
    ]
    assert completions.calls == [
        {
            "model": "qwen-plus",
            "messages": [{"role": "user", "content": "public prompt"}],
            "n": 2,
            "temperature": 0.8,
            "max_tokens": 100,
            "stop": ["Observation"],
        }
    ]
    assert generated == GenerationResult(("first", "second"), 23, 5)
    assert backend.token_usage == {
        "prompt_tokens": 23,
        "completion_tokens": 5,
        "total_tokens": 28,
    }


def test_openai_backend_import_and_missing_key_fail_only_on_generate(
    monkeypatch,
):
    monkeypatch.setitem(sys.modules, "openai", None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    backend = OpenAICompatibleBackend(model="qwen-plus")

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        backend.generate("prompt", 1, 0.8, 100, ())


def test_openai_backend_retries_only_transient_errors(monkeypatch):
    constructor_calls = []
    completions = FakeCompletions(
        [
            FakeTransientError("temporary"),
            FakeTransientError("temporary"),
            response(["ok"], prompt_tokens=3, completion_tokens=1),
        ]
    )
    monkeypatch.setitem(
        sys.modules,
        "openai",
        fake_openai_module(completions, constructor_calls),
    )
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    backend = OpenAICompatibleBackend(
        model="model",
        retry_limit=2,
        retry_delay=0,
    )

    assert backend.generate("prompt", 1, 0.2, 10, ()).texts == ("ok",)
    assert len(completions.calls) == 3


def test_openai_backend_bounds_retries(monkeypatch):
    constructor_calls = []
    completions = FakeCompletions(
        [FakeTransientError("temporary") for _ in range(4)]
    )
    monkeypatch.setitem(
        sys.modules,
        "openai",
        fake_openai_module(completions, constructor_calls),
    )
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    backend = OpenAICompatibleBackend(
        model="model",
        retry_limit=2,
        retry_delay=0,
    )

    with pytest.raises(FakeTransientError):
        backend.generate("prompt", 1, 0.2, 10, ())
    assert len(completions.calls) == 3


def test_openai_backend_does_not_retry_permanent_errors(monkeypatch):
    constructor_calls = []
    completions = FakeCompletions(
        [FakeBadRequestError("bad request"), response(["must not run"])]
    )
    monkeypatch.setitem(
        sys.modules,
        "openai",
        fake_openai_module(completions, constructor_calls),
    )
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    backend = OpenAICompatibleBackend(
        model="model",
        retry_limit=3,
        retry_delay=0,
    )

    with pytest.raises(FakeBadRequestError):
        backend.generate("prompt", 1, 0.2, 10, ())
    assert len(completions.calls) == 1
