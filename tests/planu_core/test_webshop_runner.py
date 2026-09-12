import ast
from dataclasses import replace
import json
from pathlib import Path
import traceback
from types import SimpleNamespace

import pytest

from planu_core.config import PlanUConfig
from planu_core.text_backend import OpenAICompatibleBackend
from planu_core.webshop import runner as webshop_runner
from planu_core.webshop.runner import (
    WEBSHOP_COMMIT,
    RunnerDependencies,
    WebShopRunnerError,
    parse_args,
    run,
)


ROOT = Path(__file__).resolve().parents[2]


class Closeable:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class FakeClient(Closeable):
    pass


class FakeBackend(Closeable):
    model_identifier = "fake-model"


class FakePolicy:
    def __init__(self, token_usage=None):
        self.token_usage = token_usage or {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }


def trajectory(
    reward,
    *,
    terminated,
    action="click[Buy Now]",
    same_state=False,
):
    parent_key = ("parent", action)
    child_key = parent_key if same_state else ("child", action)
    return SimpleNamespace(
        action_path=[
            SimpleNamespace(
                action=SimpleNamespace(text=action),
            )
        ],
        state_path=[
            SimpleNamespace(state_key=parent_key),
            SimpleNamespace(state_key=child_key),
        ],
        rewards=[reward],
        terminated=terminated,
        truncated=not terminated,
        truncation_reason=None if terminated else "max_depth",
        final_observation="final page",
    )


class FakeSearch:
    def __init__(self, outcomes, output_dir):
        self.outcomes = list(outcomes)
        self.output_dir = output_dir
        self.calls = []

    def run_iteration(self, iteration, rng, reset_seed=None):
        for filename in ("effective_config.json", "run_metadata.json"):
            payload = json.loads(
                (self.output_dir / filename).read_text(encoding="utf-8")
            )
            assert isinstance(payload, dict)
        self.calls.append((iteration, reset_seed, rng))
        return self.outcomes[iteration]


class DependencyHarness:
    def __init__(self, output_dir, outcomes_by_task=None, fail_iteration=False):
        self.output_dir = output_dir
        self.outcomes_by_task = outcomes_by_task or {}
        self.fail_iteration = fail_iteration
        self.client = FakeClient()
        self.backend = FakeBackend()
        self.clients = []
        self.backends = []
        self.adapters = []
        self.searches = []
        self.search_arguments = []
        self.model_provider_calls = []
        self.model_scorer_calls = []
        self.scripted_provider_calls = []
        self.scripted_scorer_calls = 0

    def client_factory(self, base_url, timeout):
        self.client.base_url = base_url
        self.client.timeout = timeout
        self.clients.append(self.client)
        return self.client

    def backend_factory(self, **kwargs):
        self.backend.kwargs = kwargs
        self.backends.append(self.backend)
        return self.backend

    def adapter_factory(self, client, session_id):
        adapter = SimpleNamespace(client=client, session_id=session_id)
        self.adapters.append(adapter)
        return adapter

    def model_provider_factory(self, backend, **kwargs):
        self.model_provider_calls.append((backend, kwargs))
        return FakePolicy(
            {
                "prompt_tokens": 11,
                "completion_tokens": 3,
                "total_tokens": 14,
            }
        )

    def model_scorer_factory(self, backend, **kwargs):
        self.model_scorer_calls.append((backend, kwargs))
        return FakePolicy(
            {
                "prompt_tokens": 7,
                "completion_tokens": 2,
                "total_tokens": 9,
            }
        )

    def scripted_provider_factory(self, search_query):
        self.scripted_provider_calls.append(search_query)
        return FakePolicy()

    def scripted_scorer_factory(self):
        self.scripted_scorer_calls += 1
        return FakePolicy()

    def search_factory(self, adapter, scorer, config, action_provider):
        self.search_arguments.append(
            {
                "adapter": adapter,
                "scorer": scorer,
                "config": config,
                "action_provider": action_provider,
            }
        )
        if self.fail_iteration:
            search = FakeSearch([], self.output_dir)

            def fail(*args, **kwargs):
                raise ConnectionError("server unavailable")

            search.run_iteration = fail
        else:
            outcomes = self.outcomes_by_task.get(
                adapter.session_id,
                [trajectory(0.0, terminated=False)],
            )
            search = FakeSearch(outcomes, self.output_dir)
        self.searches.append(search)
        return search

    @staticmethod
    def metadata_factory(package_distributions):
        assert package_distributions["beautifulsoup4"] == "beautifulsoup4"
        return {
            "git_commit": "planu-commit",
            "python_version": "3.9.6",
            "packages": {
                name: "test-version" for name in package_distributions
            },
        }

    def dependencies(self):
        return RunnerDependencies(
            client_factory=self.client_factory,
            backend_factory=self.backend_factory,
            adapter_factory=self.adapter_factory,
            search_factory=self.search_factory,
            model_provider_factory=self.model_provider_factory,
            model_scorer_factory=self.model_scorer_factory,
            scripted_provider_factory=self.scripted_provider_factory,
            scripted_scorer_factory=self.scripted_scorer_factory,
            metadata_factory=self.metadata_factory,
        )


def smoke_args(tmp_path, *extra):
    return parse_args(
        [
            "--smoke",
            "--task-start-index",
            "1",
            "--task-end-index",
            "2",
            "--iterations",
            "1",
            "--output-dir",
            str(tmp_path),
            *extra,
        ]
    )


def test_reference_defaults(monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.delenv("WEBSHOP_URL", raising=False)

    args = parse_args([])

    assert args.backend == "qwen-plus"
    assert args.temperature == 0.8
    assert args.prompt_mode == "cot"
    assert args.n_generate_sample == 5
    assert args.n_evaluate_sample == 1
    assert args.iterations == 10
    assert args.depth == 10
    assert args.task_start_index == 1
    assert args.task_end_index == 50
    assert args.seed == 0
    assert args.output_dir == "logs/webshop"
    assert args.webshop_url == "http://127.0.0.1:3000"
    assert args.request_timeout == 30.0
    assert args.base_url


def test_base_urls_use_environment_and_allow_cli_override(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://model-env.example/v1")
    monkeypatch.setenv("WEBSHOP_URL", "https://shop-env.example")

    environment = parse_args([])
    explicit = parse_args(
        [
            "--base-url",
            "https://model-cli.example/v1",
            "--webshop-url",
            "https://shop-cli.example",
        ]
    )

    assert environment.base_url == "https://model-env.example/v1"
    assert environment.webshop_url == "https://shop-env.example"
    assert explicit.base_url == "https://model-cli.example/v1"
    assert explicit.webshop_url == "https://shop-cli.example"


def test_runner_constructs_exact_approved_config_and_half_open_task_ids(
    tmp_path,
):
    harness = DependencyHarness(
        tmp_path,
        {
            "fixed_2": [trajectory(0.0, terminated=False)] * 2,
            "fixed_3": [trajectory(0.0, terminated=False)] * 2,
        },
    )
    args = parse_args(
        [
            "--smoke",
            "--task-start-index",
            "2",
            "--task-end-index",
            "4",
            "--iterations",
            "2",
            "--depth",
            "7",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert run(args, dependencies=harness.dependencies()) == 0

    assert [adapter.session_id for adapter in harness.adapters] == [
        "fixed_2",
        "fixed_3",
    ]
    assert len(harness.searches) == 2
    assert [len(search.calls) for search in harness.searches] == [2, 2]
    expected = PlanUConfig(
        n_quantiles=51,
        value_min=0.0,
        value_max=1.0,
        quantile_learning_rate=0.9,
        discount=1.0,
        curiosity_weight=0.0,
        include_preview_reward=True,
        max_depth=7,
        max_iterations=2,
    )
    assert [
        arguments["config"] for arguments in harness.search_arguments
    ] == [expected, expected]


def test_smoke_uses_scripted_policies_without_constructing_model_backend(
    tmp_path,
):
    harness = DependencyHarness(tmp_path)

    assert run(
        smoke_args(tmp_path),
        dependencies=harness.dependencies(),
    ) == 0

    assert harness.backends == []
    assert harness.model_provider_calls == []
    assert harness.model_scorer_calls == []
    assert harness.scripted_provider_calls == ["product"]
    assert harness.scripted_scorer_calls == 1
    assert harness.clients == [harness.client]


def test_model_mode_constructs_openai_backend_provider_and_scorer(tmp_path):
    harness = DependencyHarness(tmp_path)
    args = parse_args(
        [
            "--task-start-index",
            "1",
            "--task-end-index",
            "2",
            "--iterations",
            "1",
            "--output-dir",
            str(tmp_path),
            "--backend",
            "model-x",
            "--base-url",
            "https://models.example/v1",
            "--temperature",
            "0.4",
            "--n-generate-sample",
            "3",
            "--n-evaluate-sample",
            "2",
            "--request-timeout",
            "12",
        ]
    )

    assert run(args, dependencies=harness.dependencies()) == 0

    assert harness.backend.kwargs == {
        "model": "model-x",
        "base_url": "https://models.example/v1",
        "timeout": 12.0,
    }
    assert harness.model_provider_calls == [
        (
            harness.backend,
            {"candidate_count": 3, "temperature": 0.4},
        )
    ]
    assert harness.model_scorer_calls == [
        (
            harness.backend,
            {"n_evaluate_sample": 2, "temperature": 0.4},
        )
    ]


def test_result_uses_best_observed_terminal_trajectory_and_valid_schema(
    tmp_path,
):
    harness = DependencyHarness(
        tmp_path,
        {
            "fixed_1": [
                trajectory(
                    0.9,
                    terminated=False,
                    action="click[A1]",
                    same_state=True,
                ),
                trajectory(0.4, terminated=True, action="click[A2]"),
                trajectory(0.8, terminated=True, action="click[A3]"),
            ]
        },
    )

    assert run(
        smoke_args(tmp_path, "--iterations", "3", "--seed", "17"),
        dependencies=harness.dependencies(),
    ) == 0

    lines = (tmp_path / "results.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(lines) == 1
    result = json.loads(lines[0])
    assert result == {
        "task_id": "fixed_1",
        "task_index": 1,
        "best_terminal_reward": 0.8,
        "success": False,
        "trajectory": [
            {
                "action": "click[A3]",
                "observation": "final page",
                "reward": 0.8,
            }
        ],
        "latency_failure_count": 1,
        "search_iterations": 3,
        "max_depth": 10,
        "token_usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
        "error_status": None,
        "provenance": result["provenance"],
    }
    assert result["provenance"]["webshop_commit"] == WEBSHOP_COMMIT
    assert result["provenance"]["planu_git_commit"] == "planu-commit"
    assert result["provenance"]["seed"] == 17
    assert result["provenance"]["task_bounds"] == {
        "start": 1,
        "end_exclusive": 2,
    }


def test_provenance_is_written_before_search_and_redacts_urls(tmp_path):
    harness = DependencyHarness(tmp_path)
    args = smoke_args(
        tmp_path,
        "--webshop-url",
        "https://shop-user:shop-secret@shop.example:8443/path?key=hidden",
        "--base-url",
        "https://model-user:model-secret@model.example/v1?token=hidden",
    )

    assert run(args, dependencies=harness.dependencies()) == 0

    effective = json.loads(
        (tmp_path / "effective_config.json").read_text(encoding="utf-8")
    )
    metadata = json.loads(
        (tmp_path / "run_metadata.json").read_text(encoding="utf-8")
    )
    serialized = json.dumps(
        {"effective": effective, "metadata": metadata},
        sort_keys=True,
    )
    for secret in (
        "shop-user",
        "shop-secret",
        "model-user",
        "model-secret",
        "hidden",
    ):
        assert secret not in serialized
    assert effective["args"]["webshop_url"] == (
        "https://shop.example:8443/path"
    )
    assert effective["args"]["base_url"] == "https://model.example/v1"
    assert metadata["server_url"] == "https://shop.example:8443/path"
    assert metadata["config_hash"]
    assert metadata["model_id"] == "scripted"
    assert metadata["webshop_commit"] == WEBSHOP_COMMIT
    assert not list(tmp_path.glob(".*.tmp"))


def test_model_token_usage_is_combined_per_task(tmp_path):
    harness = DependencyHarness(tmp_path)
    args = parse_args(
        [
            "--task-start-index",
            "1",
            "--task-end-index",
            "2",
            "--iterations",
            "1",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert run(args, dependencies=harness.dependencies()) == 0

    result = json.loads(
        (tmp_path / "results.jsonl").read_text(encoding="utf-8")
    )
    assert result["token_usage"] == {
        "prompt_tokens": 18,
        "completion_tokens": 5,
        "total_tokens": 23,
    }


@pytest.mark.parametrize("smoke", [True, False])
def test_runner_closes_clients_on_success(tmp_path, smoke):
    harness = DependencyHarness(tmp_path)
    args = smoke_args(tmp_path)
    if not smoke:
        args.smoke = False

    assert run(args, dependencies=harness.dependencies()) == 0

    assert harness.client.closed is True
    assert harness.backend.closed is (not smoke)


def test_infrastructure_failure_is_safe_and_closes_clients(tmp_path):
    harness = DependencyHarness(tmp_path, fail_iteration=True)
    args = parse_args(
        [
            "--task-start-index",
            "4",
            "--task-end-index",
            "5",
            "--iterations",
            "1",
            "--output-dir",
            str(tmp_path),
        ]
    )

    with pytest.raises(WebShopRunnerError) as raised:
        run(args, dependencies=harness.dependencies())

    message = str(raised.value)
    assert "fixed_4" in message
    assert "ConnectionError" in message
    assert "server unavailable" not in message
    assert raised.value.__cause__ is None
    assert harness.client.closed is True
    assert harness.backend.closed is True
    assert not (tmp_path / "results.jsonl").exists()


@pytest.mark.parametrize(
    ("failure_point", "expected_context"),
    [
        ("client", "runner"),
        ("search", "fixed_7"),
    ],
)
def test_failure_summary_does_not_leak_third_party_exception_text(
    tmp_path,
    failure_point,
    expected_context,
):
    raw_message = (
        "request failed for "
        "https://url-user:url-password@shop.example/search"
        "?token=secret-value"
    )
    harness = DependencyHarness(
        tmp_path,
        fail_iteration=failure_point == "search",
    )
    dependencies = harness.dependencies()

    if failure_point == "client":

        def failing_client_factory(*args, **kwargs):
            del args, kwargs
            raise ConnectionError(raw_message)

        dependencies = replace(
            dependencies,
            client_factory=failing_client_factory,
        )
    else:
        search = FakeSearch([], tmp_path)

        def fail_search(*args, **kwargs):
            del args, kwargs
            raise ConnectionError(raw_message)

        search.run_iteration = fail_search
        dependencies = replace(
            dependencies,
            search_factory=lambda *args, **kwargs: search,
        )

    args = smoke_args(
        tmp_path,
        "--task-start-index",
        "7",
        "--task-end-index",
        "8",
    )

    with pytest.raises(WebShopRunnerError) as raised:
        run(args, dependencies=dependencies)

    message = str(raised.value)
    assert "WebShop infrastructure failure" in message
    assert expected_context in message
    assert "ConnectionError" in message
    for sensitive_text in (
        "url-user",
        "url-password",
        "?token=secret-value",
        "token",
        "secret-value",
        raw_message,
    ):
        assert sensitive_text not in message
    assert raised.value.__cause__ is None


@pytest.mark.parametrize("failure_point", ["client", "search"])
def test_failure_traceback_suppresses_third_party_exception_chain(
    tmp_path,
    failure_point,
):
    raw_message = (
        "request failed for "
        "https://url-user:url-password@shop.example/search"
        "?token=secret-value"
    )
    harness = DependencyHarness(
        tmp_path,
        fail_iteration=failure_point == "search",
    )
    dependencies = harness.dependencies()

    if failure_point == "client":

        def failing_client_factory(*args, **kwargs):
            del args, kwargs
            raise ConnectionError(raw_message)

        dependencies = replace(
            dependencies,
            client_factory=failing_client_factory,
        )
    else:
        search = FakeSearch([], tmp_path)

        def fail_search(*args, **kwargs):
            del args, kwargs
            raise ConnectionError(raw_message)

        search.run_iteration = fail_search
        dependencies = replace(
            dependencies,
            search_factory=lambda *args, **kwargs: search,
        )

    with pytest.raises(WebShopRunnerError) as raised:
        run(
            smoke_args(
                tmp_path,
                "--task-start-index",
                "7",
                "--task-end-index",
                "8",
            ),
            dependencies=dependencies,
        )

    formatted = "".join(
        traceback.format_exception(
            type(raised.value),
            raised.value,
            raised.value.__traceback__,
        )
    )
    assert "ConnectionError" in str(raised.value)
    for secret in (
        "url-user",
        "url-password",
        "token=secret-value",
        "secret-value",
        raw_message,
    ):
        assert secret not in formatted
    assert raised.value.__cause__ is None


@pytest.mark.parametrize(
    ("failure_point", "expected_context"),
    [
        ("metadata", "runner"),
        ("client", "runner"),
        ("task", "fixed_7"),
    ],
)
def test_dependency_supplied_runner_errors_are_not_trusted(
    tmp_path,
    monkeypatch,
    capsys,
    failure_point,
    expected_context,
):
    secret = "external-runner-error-secret"

    def dependency_factory():
        harness = DependencyHarness(tmp_path)
        dependencies = harness.dependencies()

        def fail(*args, **kwargs):
            del args, kwargs
            raise WebShopRunnerError(secret)

        if failure_point == "metadata":
            return replace(dependencies, metadata_factory=fail)
        if failure_point == "client":
            return replace(dependencies, client_factory=fail)

        search = FakeSearch([], tmp_path)
        search.run_iteration = fail
        return replace(
            dependencies,
            search_factory=lambda *args, **kwargs: search,
        )

    args = smoke_args(
        tmp_path,
        "--task-start-index",
        "7",
        "--task-end-index",
        "8",
    )
    with pytest.raises(WebShopRunnerError) as raised:
        run(args, dependencies=dependency_factory())

    message = str(raised.value)
    formatted = "".join(
        traceback.format_exception(
            type(raised.value),
            raised.value,
            raised.value.__traceback__,
        )
    )
    assert message == (
        "WebShop infrastructure failure for {} (WebShopRunnerError)".format(
            expected_context
        )
    )
    assert secret not in message
    assert secret not in formatted
    assert raised.value.__cause__ is None

    monkeypatch.setattr(
        webshop_runner,
        "RunnerDependencies",
        dependency_factory,
    )
    exit_code = webshop_runner.main(
        [
            "--smoke",
            "--task-start-index",
            "7",
            "--task-end-index",
            "8",
            "--iterations",
            "1",
            "--output-dir",
            str(tmp_path),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.out == ""
    assert captured.err.strip() == message
    assert secret not in captured.err


def test_openai_backend_close_releases_lazy_client_once():
    client = Closeable()
    backend = OpenAICompatibleBackend(model="model")
    backend._client = client

    backend.close()
    backend.close()

    assert client.closed is True
    assert backend._client is None


def test_invalid_task_bounds_are_rejected():
    with pytest.raises(SystemExit):
        parse_args(["--task-start-index", "5", "--task-end-index", "5"])


def test_legacy_entrypoints_are_thin_compatibility_modules():
    run_source = (ROOT / "webshop" / "run.py").read_text(encoding="utf-8")
    planu_source = (ROOT / "webshop" / "planu.py").read_text(encoding="utf-8")
    models_source = (ROOT / "webshop" / "models.py").read_text(
        encoding="utf-8"
    )
    run_tree = ast.parse(run_source)
    planu_tree = ast.parse(planu_source)
    models_tree = ast.parse(models_source)

    assert "planu_core.webshop.runner" in run_source
    assert not any(isinstance(node, ast.ClassDef) for node in run_tree.body)
    assert not any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        for node in planu_tree.body + models_tree.body
    )
    forbidden = (
        "class Node",
        "def planu_search",
        "requests.",
        "OpenAI(",
        "PlanUSearch(",
    )
    assert all(item not in planu_source for item in forbidden)
    assert all(item not in models_source for item in forbidden)
    assert "LEGACY_COT_PROMPT" in planu_source


def test_legacy_shell_invokes_exact_reference_profile():
    source = (ROOT / "webshop" / "planu.sh").read_text(encoding="utf-8")

    assert "python -m planu_core.webshop.runner" in source
    for option in (
        "--backend qwen-plus",
        "--temperature 0.8",
        "--prompt-mode cot",
        "--n-generate-sample 5",
        "--n-evaluate-sample 1",
        "--iterations 10",
        "--depth 10",
        "--task-start-index 1",
        "--task-end-index 50",
    ):
        assert option in source
    assert '"$@"' in source
