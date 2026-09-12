import ast
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
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
WEBSHOP_SCRIPT = ROOT / "scripts" / "smoke_webshop.sh"
WEBSHOP_BOOTSTRAP = ROOT / "scripts" / "bootstrap_webshop.sh"


def write_executable(path, source):
    path.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n" + source,
        encoding="utf-8",
    )
    path.chmod(0o755)


def script_environment(bin_dir, **values):
    environment = os.environ.copy()
    environment.update(values)
    environment["PATH"] = "{}:/usr/bin:/bin".format(bin_dir)
    return environment


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
        for filename in ("run_manifest.json",):
            payload = json.loads(
                (self.output_dir / filename).read_text(encoding="utf-8")
            )
            assert isinstance(payload, dict)
        for filename in ("effective_config.json", "run_metadata.json"):
            path = self.output_dir / filename
            if path.exists():
                assert isinstance(
                    json.loads(path.read_text(encoding="utf-8")),
                    dict,
                )
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


def test_terminal_maximum_stops_before_a_failing_second_iteration(tmp_path):
    harness = DependencyHarness(tmp_path)
    search = FakeSearch([], tmp_path)

    def stop_after_success(iteration, rng, reset_seed=None):
        del rng, reset_seed
        search.calls.append(iteration)
        if iteration == 0:
            return trajectory(1.0, terminated=True)
        raise AssertionError("second iteration must not execute")

    search.run_iteration = stop_after_success
    dependencies = replace(
        harness.dependencies(),
        search_factory=lambda *args, **kwargs: search,
    )

    assert run(
        smoke_args(tmp_path, "--iterations", "3"),
        dependencies=dependencies,
    ) == 0

    result = json.loads(
        (tmp_path / "tasks" / "fixed_1.json").read_text(encoding="utf-8")
    )
    assert search.calls == [0]
    assert result["best_terminal_reward"] == 1.0
    assert result["success"] is True
    assert result["search_iterations"] == 1


def test_provenance_manifest_is_published_last_before_search_and_redacts_urls(
    tmp_path,
    monkeypatch,
):
    harness = DependencyHarness(tmp_path)
    args = smoke_args(
        tmp_path,
        "--webshop-url",
        "https://shop-user:shop-secret@shop.example:8443/path?key=hidden",
        "--base-url",
        "https://model-user:model-secret@model.example/v1?token=hidden",
    )
    published = []
    real_atomic_write_json = webshop_runner.atomic_write_json

    def recording_atomic_write(path, payload):
        path = Path(path)
        if path.name == "run_manifest.json":
            assert (tmp_path / "effective_config.json").is_file()
            assert (tmp_path / "run_metadata.json").is_file()
        published.append(path.name)
        real_atomic_write_json(path, payload)

    monkeypatch.setattr(
        webshop_runner,
        "atomic_write_json",
        recording_atomic_write,
    )

    assert run(args, dependencies=harness.dependencies()) == 0

    effective = json.loads(
        (tmp_path / "effective_config.json").read_text(encoding="utf-8")
    )
    metadata = json.loads(
        (tmp_path / "run_metadata.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (tmp_path / "run_manifest.json").read_text(encoding="utf-8")
    )
    serialized = json.dumps(
        {
            "effective": effective,
            "metadata": metadata,
            "manifest": manifest,
        },
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
    assert manifest == {
        "effective_config": effective,
        "run_metadata": metadata,
    }
    assert published[:3] == [
        "effective_config.json",
        "run_metadata.json",
        "run_manifest.json",
    ]
    assert not list(tmp_path.glob(".*.tmp"))


@pytest.mark.parametrize(
    "mutation",
    [
        "seed",
        "task_bounds",
        "model_id",
        "server_url",
        "config_hash",
        "webshop_commit",
        "planu_git_commit",
        "git_commit",
    ],
)
def test_manifest_identity_mismatch_refuses_reuse_before_altering_artifacts(
    tmp_path,
    mutation,
):
    first_harness = DependencyHarness(tmp_path)
    args = smoke_args(tmp_path)
    assert run(args, dependencies=first_harness.dependencies()) == 0

    manifest_path = tmp_path / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata_path = tmp_path / "run_metadata.json"
    if mutation == "seed":
        replacement = 999
    elif mutation == "task_bounds":
        replacement = {"start": 9, "end_exclusive": 10}
    else:
        replacement = "wrong-identity"
    manifest["run_metadata"][mutation] = replacement
    metadata_path.write_text(
        json.dumps(manifest["run_metadata"], sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    before = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    second_harness = DependencyHarness(tmp_path)

    with pytest.raises(WebShopRunnerError):
        run(args, dependencies=second_harness.dependencies())

    after = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert after == before
    assert second_harness.clients == []


def test_manifest_effective_config_mismatch_refuses_reuse_before_artifact_changes(
    tmp_path,
):
    first_harness = DependencyHarness(tmp_path)
    args = smoke_args(tmp_path)
    assert run(args, dependencies=first_harness.dependencies()) == 0

    manifest_path = tmp_path / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["effective_config"]["args"]["seed"] = 999
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    before = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    second_harness = DependencyHarness(tmp_path)

    with pytest.raises(WebShopRunnerError):
        run(args, dependencies=second_harness.dependencies())

    after = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert after == before
    assert second_harness.clients == []


@pytest.mark.parametrize(
    ("sidecar_name", "manifest_key"),
    [
        ("effective_config.json", "effective_config"),
        ("run_metadata.json", "run_metadata"),
    ],
)
def test_manifest_reuse_rejects_mismatched_present_sidecar(
    tmp_path,
    sidecar_name,
    manifest_key,
):
    args = smoke_args(tmp_path)
    assert run(
        args,
        dependencies=DependencyHarness(tmp_path).dependencies(),
    ) == 0

    sidecar_path = tmp_path / sidecar_name
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["unexpected"] = "different-from-manifest"
    sidecar_path.write_text(
        json.dumps(sidecar, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = json.loads(
        (tmp_path / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert sidecar != manifest[manifest_key]
    before = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    second_harness = DependencyHarness(tmp_path)

    with pytest.raises(WebShopRunnerError):
        run(args, dependencies=second_harness.dependencies())

    after = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert after == before
    assert second_harness.clients == []


def test_matching_pre_manifest_sidecars_are_repaired_without_rewriting_them(
    tmp_path,
    monkeypatch,
):
    harness = DependencyHarness(tmp_path)
    args = smoke_args(tmp_path)
    assert run(args, dependencies=harness.dependencies()) == 0
    (tmp_path / "run_manifest.json").unlink()
    (tmp_path / "results.jsonl").unlink()
    for task_path in (tmp_path / "tasks").iterdir():
        task_path.unlink()
    (tmp_path / "tasks").rmdir()
    sidecars_before = {
        name: (tmp_path / name).read_bytes()
        for name in ("effective_config.json", "run_metadata.json")
    }
    published = []
    real_atomic_write_json = webshop_runner.atomic_write_json

    def recording_atomic_write(path, payload):
        published.append(Path(path).name)
        real_atomic_write_json(path, payload)

    monkeypatch.setattr(
        webshop_runner,
        "atomic_write_json",
        recording_atomic_write,
    )

    assert run(
        args,
        dependencies=DependencyHarness(tmp_path).dependencies(),
    ) == 0

    assert published[0] == "run_manifest.json"
    assert {
        name: (tmp_path / name).read_bytes()
        for name in sidecars_before
    } == sidecars_before


def test_incomplete_pre_manifest_sidecars_are_rejected_without_repair(
    tmp_path,
):
    harness = DependencyHarness(tmp_path)
    args = smoke_args(tmp_path)
    assert run(args, dependencies=harness.dependencies()) == 0
    (tmp_path / "run_manifest.json").unlink()
    (tmp_path / "run_metadata.json").unlink()
    (tmp_path / "results.jsonl").unlink()
    for task_path in (tmp_path / "tasks").iterdir():
        task_path.unlink()
    (tmp_path / "tasks").rmdir()
    before = (tmp_path / "effective_config.json").read_bytes()
    second_harness = DependencyHarness(tmp_path)

    with pytest.raises(WebShopRunnerError):
        run(args, dependencies=second_harness.dependencies())

    assert (tmp_path / "effective_config.json").read_bytes() == before
    assert not (tmp_path / "run_metadata.json").exists()
    assert not (tmp_path / "run_manifest.json").exists()
    assert second_harness.clients == []


def test_manifest_reuse_does_not_recreate_missing_convenience_sidecars(
    tmp_path,
):
    args = smoke_args(tmp_path)
    assert run(
        args,
        dependencies=DependencyHarness(tmp_path).dependencies(),
    ) == 0
    (tmp_path / "effective_config.json").unlink()
    (tmp_path / "run_metadata.json").unlink()

    assert run(
        args,
        dependencies=DependencyHarness(tmp_path).dependencies(),
    ) == 0

    assert not (tmp_path / "effective_config.json").exists()
    assert not (tmp_path / "run_metadata.json").exists()


def test_manifest_metadata_remains_authoritative_on_compatible_reuse(tmp_path):
    args = smoke_args(tmp_path)
    assert run(
        args,
        dependencies=DependencyHarness(tmp_path).dependencies(),
    ) == 0
    manifest_path = tmp_path / "run_manifest.json"
    manifest_before = manifest_path.read_bytes()
    second_harness = DependencyHarness(tmp_path)

    def changed_environment_metadata(package_distributions):
        metadata = second_harness.metadata_factory(package_distributions)
        metadata["python_version"] = "9.9.9"
        metadata["packages"]["numpy"] = "changed-version"
        return metadata

    dependencies = replace(
        second_harness.dependencies(),
        metadata_factory=changed_environment_metadata,
    )

    assert run(args, dependencies=dependencies) == 0

    result = json.loads(
        (tmp_path / "tasks" / "fixed_1.json").read_text(encoding="utf-8")
    )
    assert manifest_path.read_bytes() == manifest_before
    assert result["provenance"]["python_version"] == "3.9.6"
    assert result["provenance"]["packages"]["numpy"] == "test-version"


def test_output_directory_lock_refuses_concurrent_run_before_writes(
    tmp_path,
    monkeypatch,
):
    harness = DependencyHarness(tmp_path)

    def locked(*args, **kwargs):
        del args, kwargs
        raise BlockingIOError

    monkeypatch.setattr(webshop_runner.fcntl, "flock", locked)

    with pytest.raises(WebShopRunnerError):
        run(smoke_args(tmp_path), dependencies=harness.dependencies())

    assert list(tmp_path.iterdir()) == []
    assert harness.clients == []


def test_task_failure_preserves_completed_per_task_artifacts(tmp_path):
    harness = DependencyHarness(
        tmp_path,
        {
            "fixed_1": [trajectory(0.4, terminated=True)],
            "fixed_2": [],
        },
    )
    args = parse_args(
        [
            "--smoke",
            "--task-start-index",
            "1",
            "--task-end-index",
            "3",
            "--iterations",
            "1",
            "--output-dir",
            str(tmp_path),
        ]
    )

    with pytest.raises(WebShopRunnerError):
        run(args, dependencies=harness.dependencies())

    completed = json.loads(
        (tmp_path / "tasks" / "fixed_1.json").read_text(encoding="utf-8")
    )
    assert completed["task_id"] == "fixed_1"
    assert not (tmp_path / "tasks" / "fixed_2.json").exists()
    assert not (tmp_path / "results.jsonl").exists()


def test_results_are_consolidated_from_validated_task_files_in_task_order(
    tmp_path,
):
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir()
    second = {"task_id": "fixed_2", "task_index": 2}
    first = {"task_id": "fixed_1", "task_index": 1}
    for path, payload in (
        (tasks_dir / "fixed_2.json", second),
        (tasks_dir / "fixed_1.json", first),
    ):
        path.write_text(json.dumps(payload), encoding="utf-8")

    webshop_runner._publish_results(tmp_path, 1, 3)

    assert [
        json.loads(line)
        for line in (tmp_path / "results.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ] == [first, second]


def test_invalid_task_json_does_not_replace_existing_results(tmp_path):
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir()
    (tasks_dir / "fixed_1.json").write_text("{broken", encoding="utf-8")
    results_path = tmp_path / "results.jsonl"
    results_path.write_text('{"old":true}\n', encoding="utf-8")

    with pytest.raises((ValueError, json.JSONDecodeError)):
        webshop_runner._publish_results(tmp_path, 1, 2)

    assert results_path.read_text(encoding="utf-8") == '{"old":true}\n'


def test_runner_has_no_cumulative_jsonl_append_helper():
    source = Path(webshop_runner.__file__).read_text(encoding="utf-8")

    assert "_atomic_append_json_line" not in source


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


def test_sanitized_error_search_ignores_public_error_cycles():
    error = WebShopRunnerError("external-secret")
    error.__cause__ = error

    assert webshop_runner._find_sanitized_error(error) is None


def test_cleanup_failure_preserves_safe_task_context(
    tmp_path,
    monkeypatch,
    capsys,
):
    task_secret = "task-failure-secret"
    cleanup_secret = "cleanup-failure-secret"

    def dependency_factory():
        harness = DependencyHarness(tmp_path)
        dependencies = harness.dependencies()
        search = FakeSearch([], tmp_path)

        def fail_task(*args, **kwargs):
            del args, kwargs
            raise ConnectionError(task_secret)

        def fail_close():
            raise OSError(cleanup_secret)

        search.run_iteration = fail_task
        harness.client.close = fail_close
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
    assert "fixed_7" in message
    assert "ConnectionError" in message
    assert "OSError" in message
    assert raised.value.__cause__ is None
    for secret in (task_secret, cleanup_secret):
        assert secret not in message
        assert secret not in formatted

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
    for secret in (task_secret, cleanup_secret):
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


def test_webshop_scripts_pin_source_and_require_real_smoke_contract():
    bootstrap = WEBSHOP_BOOTSTRAP.read_text(encoding="utf-8")
    smoke = WEBSHOP_SCRIPT.read_text(encoding="utf-8")

    assert bootstrap.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
    assert "64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd" in bootstrap
    assert "https://github.com/princeton-nlp/WebShop.git" in bootstrap
    assert "BASH_SOURCE[0]" in bootstrap
    assert "python=3.8.13" in bootstrap
    assert "setup.sh -d small" in bootstrap
    assert "checkout --detach" in bootstrap
    assert "reset --hard" not in bootstrap
    assert "clean -f" not in bootstrap

    assert smoke.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
    assert "64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd" in smoke
    assert "https://github.com/princeton-nlp/WebShop.git" in smoke
    assert "BASH_SOURCE[0]" in smoke
    assert "WEBSHOP_ROOT" in smoke
    assert "WEBSHOP_URL" in smoke
    assert "http://127.0.0.1:3000/fixed_1" in smoke
    assert "-m web_agent_site.app --log --attrs" in smoke
    for argument in (
        "--smoke",
        "--task-start-index 1",
        "--task-end-index 2",
        "--iterations 1",
        "--depth 10",
        "--smoke-search-query",
    ):
        assert argument in smoke
    assert "mock" not in smoke.lower()
    assert "pkill" not in smoke
    assert "killall" not in smoke
    assert "trap cleanup EXIT" in smoke
    assert 'kill "${SERVER_PID}"' in smoke

    for script in (WEBSHOP_BOOTSTRAP, WEBSHOP_SCRIPT):
        syntax = subprocess.run(
            ["bash", "-n", str(script)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert syntax.returncode == 0, syntax.stderr


def test_webshop_bootstrap_checks_for_conda_before_clone(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    git_log = tmp_path / "git.log"
    write_executable(
        fake_bin / "git",
        'printf "%s\\n" "$*" >> "$FAKE_GIT_LOG"\n',
    )
    webshop_root = tmp_path / "WebShop"

    completed = subprocess.run(
        ["bash", str(WEBSHOP_BOOTSTRAP)],
        check=False,
        capture_output=True,
        text=True,
        env=script_environment(
            fake_bin,
            FAKE_GIT_LOG=str(git_log),
            WEBSHOP_ROOT=str(webshop_root),
        ),
    )

    assert completed.returncode != 0
    assert "conda" in completed.stderr.lower()
    assert not webshop_root.exists()
    assert not git_log.exists()


def test_webshop_bootstrap_reports_clone_network_failure(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    write_executable(
        fake_bin / "git",
        """
if [[ "${1:-}" == "clone" ]]; then
  printf 'transport failed\n' >&2
  exit 1
fi
""",
    )
    write_executable(fake_bin / "conda", "exit 0\n")

    completed = subprocess.run(
        ["bash", str(WEBSHOP_BOOTSTRAP)],
        check=False,
        capture_output=True,
        text=True,
        env=script_environment(
            fake_bin,
            WEBSHOP_ROOT=str(tmp_path / "WebShop"),
        ),
    )

    assert completed.returncode != 0
    assert "clone failed" in completed.stderr
    assert "network" in completed.stderr


def test_webshop_bootstrap_rejects_wrong_existing_remote_without_mutation(
    tmp_path,
):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    git_log = tmp_path / "git.log"
    webshop_root = tmp_path / "WebShop"
    (webshop_root / ".git").mkdir(parents=True)
    sentinel = webshop_root / "user-change.txt"
    sentinel.write_text("keep\n", encoding="utf-8")
    write_executable(
        fake_bin / "git",
        """
printf '%s\n' "$*" >> "$FAKE_GIT_LOG"
case "$*" in
  *"rev-parse --is-inside-work-tree"*) printf 'true\n' ;;
  *"remote get-url origin"*) printf 'https://example.com/not-webshop.git\n' ;;
esac
""",
    )
    write_executable(fake_bin / "conda", "exit 0\n")

    completed = subprocess.run(
        ["bash", str(WEBSHOP_BOOTSTRAP)],
        check=False,
        capture_output=True,
        text=True,
        env=script_environment(
            fake_bin,
            FAKE_GIT_LOG=str(git_log),
            WEBSHOP_ROOT=str(webshop_root),
        ),
    )

    commands = git_log.read_text(encoding="utf-8")
    assert completed.returncode != 0
    assert "remote" in completed.stderr.lower()
    assert "checkout" not in commands
    assert "fetch" not in commands
    assert sentinel.read_text(encoding="utf-8") == "keep\n"


def test_webshop_bootstrap_refuses_dirty_checkout_before_fetch(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    git_log = tmp_path / "git.log"
    webshop_root = tmp_path / "WebShop"
    (webshop_root / ".git").mkdir(parents=True)
    sentinel = webshop_root / "user-change.txt"
    sentinel.write_text("keep\n", encoding="utf-8")
    write_executable(
        fake_bin / "git",
        """
printf '%s\n' "$*" >> "$FAKE_GIT_LOG"
case "$*" in
  *"rev-parse --is-inside-work-tree"*) printf 'true\n' ;;
  *"remote get-url origin"*)
    printf 'https://github.com/princeton-nlp/WebShop.git\n'
    ;;
  *"status --porcelain"*) printf ' M user-change.txt\n' ;;
  *"rev-parse HEAD"*) printf '0000000000000000000000000000000000000000\n' ;;
esac
""",
    )
    write_executable(fake_bin / "conda", "exit 0\n")

    completed = subprocess.run(
        ["bash", str(WEBSHOP_BOOTSTRAP)],
        check=False,
        capture_output=True,
        text=True,
        env=script_environment(
            fake_bin,
            FAKE_GIT_LOG=str(git_log),
            WEBSHOP_ROOT=str(webshop_root),
        ),
    )

    commands = git_log.read_text(encoding="utf-8")
    assert completed.returncode != 0
    assert "uncommitted" in completed.stderr.lower()
    assert "checkout" not in commands
    assert "fetch" not in commands
    assert "reset" not in commands
    assert sentinel.read_text(encoding="utf-8") == "keep\n"


def test_webshop_bootstrap_is_idempotent_after_verified_small_setup(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    webshop_root = tmp_path / "WebShop"
    (webshop_root / ".git").mkdir(parents=True)
    setup_log = tmp_path / "setup.log"
    conda_log = tmp_path / "conda.log"
    setup_script = webshop_root / "setup.sh"
    write_executable(
        setup_script,
        """
printf 'setup %s\n' "$*" >> "$FAKE_SETUP_LOG"
mkdir -p data search_engine/indexes
touch data/items_shuffle_1000.json data/items_ins_v2_1000.json
""",
    )
    write_executable(
        fake_bin / "git",
        """
case "$*" in
  *"rev-parse --is-inside-work-tree"*) printf 'true\n' ;;
  *"remote get-url origin"*)
    printf 'https://github.com/princeton-nlp/WebShop.git\n'
    ;;
  *"status --porcelain"*) ;;
  *"cat-file -e"*) ;;
  *"rev-parse HEAD"*)
    printf '64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd\n'
    ;;
  *"symbolic-ref -q HEAD"*) exit 1 ;;
esac
""",
    )
    write_executable(
        fake_bin / "conda",
        """
printf 'conda %s\n' "$*" >> "$FAKE_CONDA_LOG"
if [[ "${1:-}" == "create" ]]; then
  while (($#)); do
    if [[ "$1" == "-p" ]]; then
      prefix="$2"
      break
    fi
    shift
  done
  mkdir -p "$prefix/bin"
  cat > "$prefix/bin/python" <<'EOF'
#!/usr/bin/env bash
if [[ "$*" == *"platform.python_version"* ]]; then
  printf '3.8.13\n'
fi
EOF
  chmod +x "$prefix/bin/python"
elif [[ "${1:-}" == "run" ]]; then
  bash ./setup.sh -d small
fi
""",
    )
    environment = script_environment(
        fake_bin,
        FAKE_CONDA_LOG=str(conda_log),
        FAKE_SETUP_LOG=str(setup_log),
        WEBSHOP_ROOT=str(webshop_root),
    )

    first = subprocess.run(
        ["bash", str(WEBSHOP_BOOTSTRAP)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    second = subprocess.run(
        ["bash", str(WEBSHOP_BOOTSTRAP)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    conda_commands = conda_log.read_text(encoding="utf-8").splitlines()
    assert sum("conda create " in line for line in conda_commands) == 1
    assert sum("conda run " in line for line in conda_commands) == 1
    assert setup_log.read_text(encoding="utf-8").splitlines() == [
        "setup -d small"
    ]


def test_webshop_smoke_uses_reachable_server_and_verifies_artifacts(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    webshop_root = tmp_path / "WebShop"
    (webshop_root / ".git").mkdir(parents=True)
    run_root = tmp_path / "run"
    command_log = tmp_path / "commands.log"
    write_executable(
        fake_bin / "git",
        """
case "$*" in
  *"rev-parse --is-inside-work-tree"*) printf 'true\n' ;;
  *"remote get-url origin"*)
    printf 'https://github.com/princeton-nlp/WebShop.git\n'
    ;;
  *"status --porcelain"*) ;;
  *"rev-parse HEAD"*)
    printf '64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd\n'
    ;;
esac
""",
    )
    write_executable(
        fake_bin / "curl",
        'printf "curl %s\\n" "$*" >> "$FAKE_COMMAND_LOG"\n',
    )
    write_executable(
        fake_bin / "planu-python",
        """
if [[ "${1:-}" == "-" ]]; then
  exec "$FAKE_REAL_PYTHON" "$@"
fi
printf 'python %s\n' "$*" >> "$FAKE_COMMAND_LOG"
output_dir=''
while (($#)); do
  if [[ "$1" == "--output-dir" ]]; then
    output_dir="$2"
    break
  fi
  shift
done
mkdir -p "$output_dir/tasks"
metadata='{"model_id":"scripted","server_url":"https://shop.example","webshop_commit":"64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd"}'
effective='{"args":{"depth":10,"iterations":1,"smoke":true,"task_end_index":2,"task_start_index":1}}'
task='{"task_id":"fixed_1","task_index":1,"trajectory":[{"action":"search[product]"},{"action":"click[A1]"},{"action":"click[Buy Now]"}],"search_iterations":1,"max_depth":10,"provenance":'"$metadata"'}'
printf '{"effective_config":%s,"run_metadata":%s}\n' \
  "$effective" "$metadata" > "$output_dir/run_manifest.json"
printf '%s\n' "$effective" > "$output_dir/effective_config.json"
printf '%s\n' "$metadata" > "$output_dir/run_metadata.json"
printf '%s\n' "$task" > "$output_dir/tasks/fixed_1.json"
printf '%s\n' "$task" > "$output_dir/results.jsonl"
""",
    )

    completed = subprocess.run(
        ["bash", str(WEBSHOP_SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
        env=script_environment(
            fake_bin,
            FAKE_COMMAND_LOG=str(command_log),
            FAKE_REAL_PYTHON=sys.executable,
            PLANU_PYTHON=str(fake_bin / "planu-python"),
            RUN_ROOT=str(run_root),
            WEBSHOP_ROOT=str(webshop_root),
            WEBSHOP_URL="https://shop.example",
        ),
    )

    commands = command_log.read_text(encoding="utf-8")
    assert completed.returncode == 0, completed.stderr
    assert "https://shop.example/fixed_1" in commands
    assert "--smoke" in commands
    assert "--iterations 1" in commands
    assert "--depth 10" in commands
    assert "--task-start-index 1 --task-end-index 2" in commands
    assert '"http_transition_count": 3' in completed.stdout
    assert '"quantile_backup_count": 1' in completed.stdout
