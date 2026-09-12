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
SCRIPT_TIMEOUT = 20


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


def write_smoke_runner_fake(path):
    write_executable(
        path,
        r"""
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
"$FAKE_REAL_PYTHON" - \
  "$output_dir" "$FAKE_SERVER_URL" "$FAKE_PLANU_HEAD" \
  "${FAKE_ARTIFACT_MODE:-valid}" <<'PY'
import copy
import hashlib
import json
from pathlib import Path
import sys

output_dir = Path(sys.argv[1])
server_url = sys.argv[2]
planu_head = sys.argv[3]
mode = sys.argv[4]
output_dir.mkdir(parents=True, exist_ok=True)
(output_dir / "tasks").mkdir()

if mode == "minimal":
    effective = {
        "args": {
            "depth": 10,
            "iterations": 1,
            "smoke": True,
            "task_end_index": 2,
            "task_start_index": 1,
        }
    }
    metadata = {
        "model_id": "scripted",
        "server_url": server_url,
        "webshop_commit": (
            "64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd"
        ),
    }
    task = {
        "task_id": "fixed_1",
        "task_index": 1,
        "trajectory": [
            {"action": "search[product]"},
            {"action": "click[A1]"},
            {"action": "click[Buy Now]"},
        ],
        "search_iterations": 1,
        "max_depth": 10,
        "provenance": metadata,
    }
else:
    args = {
        "backend": "qwen-plus",
        "base_url": (
            "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ),
        "depth": 10,
        "iterations": 1,
        "n_evaluate_sample": 1,
        "n_generate_sample": 5,
        "output_dir": str(output_dir),
        "prompt_mode": "cot",
        "request_timeout": 30.0,
        "seed": 0,
        "smoke": True,
        "smoke_search_query": "product",
        "task_end_index": 2,
        "task_start_index": 1,
        "temperature": 0.8,
        "webshop_url": server_url,
    }
    planu_config = {
        "categorical_initialization": False,
        "categorical_levels": [0.1, 0.3, 0.5, 0.7, 0.9],
        "curiosity_weight": 0.0,
        "debug_state_keys": False,
        "discount": 1.0,
        "include_preview_reward": True,
        "max_depth": 10,
        "max_iterations": 1,
        "n_quantiles": 51,
        "quantile_learning_rate": 0.9,
        "risk_distortion": 0.0,
        "selection_schedule": {
            "always_sample_before": 0,
            "probabilistic_sample_before": 0,
            "sample_probability": 0.0,
        },
        "selection_temperature": 1.0,
        "train_curiosity": True,
        "value_max": 1.0,
        "value_min": 0.0,
    }
    effective = {"args": args, "planu_config": planu_config}
    config_hash = hashlib.sha256(
        json.dumps(
            effective,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:12]
    metadata = {
        "config_hash": config_hash,
        "git_commit": planu_head,
        "model_id": "scripted",
        "packages": {
            name: "test-version"
            for name in (
                "beautifulsoup4",
                "ding",
                "gym",
                "numpy",
                "openai",
                "peft",
                "requests",
                "torch",
                "transformers",
            )
        },
        "planu_git_commit": planu_head,
        "python_version": "3.9.6",
        "seed": 0,
        "server_url": server_url,
        "task_bounds": {"end_exclusive": 2, "start": 1},
        "webshop_commit": (
            "64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd"
        ),
    }
    trajectory = [
        {
            "action": "search[product]",
            "observation": "Search results for a mockingbird product",
            "reward": 0.0,
        },
        {
            "action": "click[A1]",
            "observation": "Product detail with Buy Now",
            "reward": 0.0,
        },
        {
            "action": "click[Buy Now]",
            "observation": "Your score (min 0.0, max 1.0): 0.5",
            "reward": 0.5,
        },
    ]
    task = {
        "best_terminal_reward": 0.5,
        "error_status": None,
        "latency_failure_count": 0,
        "max_depth": 10,
        "provenance": metadata,
        "search_iterations": 1,
        "success": False,
        "task_id": "fixed_1",
        "task_index": 1,
        "token_usage": {
            "completion_tokens": 0,
            "prompt_tokens": 0,
            "total_tokens": 0,
        },
        "trajectory": trajectory,
    }
    if mode == "wrong_webshop_commit":
        metadata["webshop_commit"] = "wrong"
    elif mode == "empty_planu_commit":
        metadata["git_commit"] = ""
        metadata["planu_git_commit"] = ""
    elif mode == "wrong_model":
        metadata["model_id"] = "fake-scripted"
    elif mode == "wrong_server":
        metadata["server_url"] = "https://different.example"
    elif mode == "wrong_task":
        task["task_id"] = "fixed_2"
    elif mode == "zero_iterations":
        task["search_iterations"] = 0
    elif mode == "empty_trajectory":
        task["trajectory"] = []
    elif mode == "missing_observation":
        del task["trajectory"][0]["observation"]
    elif mode == "reward_mismatch":
        task["best_terminal_reward"] = 0.25

manifest = {
    "effective_config": effective,
    "run_metadata": metadata,
}
result_task = copy.deepcopy(task)
if mode == "results_mismatch":
    result_task["task_index"] = 2
for name, payload in (
    ("run_manifest.json", manifest),
    ("effective_config.json", effective),
    ("run_metadata.json", metadata),
    ("tasks/fixed_1.json", task),
):
    (output_dir / name).write_text(
        json.dumps(payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )
(output_dir / "results.jsonl").write_text(
    json.dumps(result_task, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
""",
    )


def local_smoke_environment(tmp_path, *, artifact_mode="valid", resistant=False):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    webshop_root = tmp_path / "WebShop"
    (webshop_root / ".git").mkdir(parents=True)
    env_prefix = webshop_root / "custom-env"
    (env_prefix / "bin").mkdir(parents=True)
    java_bin = env_prefix / "lib" / "jvm" / "bin"
    java_bin.mkdir(parents=True)
    write_executable(java_bin / "java", "exit 0\n")
    server_pid_log = tmp_path / "server.pid"
    child_pid_log = tmp_path / "child.pid"
    server_env_log = tmp_path / "server.env"
    server_source = r"""
printf 'JAVA_HOME=%s\nPATH=%s\n' \
  "${JAVA_HOME:-}" "$PATH" > "$FAKE_SERVER_ENV_LOG"
printf '%s\n' "$$" > "$FAKE_SERVER_PID_LOG"
if [[ "${FAKE_RESIST_TERM:-0}" == "1" ]]; then
  trap '' TERM INT
  bash -c '
    trap "" TERM INT
    printf "%s\n" "$$" > "$FAKE_CHILD_PID_LOG"
    while true; do sleep 1; done
  ' &
else
  trap 'exit 0' TERM INT
fi
while true; do sleep 1; done
"""
    write_executable(env_prefix / "bin" / "python", server_source)
    command_log = tmp_path / "commands.log"
    write_executable(
        fake_bin / "git",
        r"""
printf '%s\n' "$*" >> "$FAKE_COMMAND_LOG"
case "$*" in
  *"$FAKE_WEBSHOP_ROOT rev-parse --show-toplevel"*)
    printf '%s\n' "$FAKE_WEBSHOP_TOPLEVEL"
    ;;
  *"$FAKE_PROJECT_ROOT rev-parse HEAD"*)
    printf '%s\n' "$FAKE_PLANU_HEAD"
    ;;
  *"$FAKE_PROJECT_ROOT status --porcelain --untracked-files=no"*)
    printf '%s' "${FAKE_PLANU_TRACKED_STATUS:-}"
    ;;
  *"$FAKE_PROJECT_ROOT ls-files --others"*)
    printf '%s' "${FAKE_PLANU_UNTRACKED_FILES:-}"
    ;;
  *"rev-parse --is-inside-work-tree"*) printf 'true\n' ;;
  *"remote get-url origin"*)
    printf 'https://github.com/princeton-nlp/WebShop.git\n'
    ;;
  *"rev-parse HEAD"*)
    printf '64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd\n'
    ;;
  *"status --porcelain"*) ;;
esac
""",
    )
    write_executable(
        fake_bin / "curl",
        r"""
printf 'curl %s\n' "$*" >> "$FAKE_COMMAND_LOG"
[[ -s "$FAKE_SERVER_PID_LOG" ]] || exit 1
server_pid="$(cat "$FAKE_SERVER_PID_LOG")"
kill -0 "$server_pid" 2>/dev/null
""",
    )
    write_smoke_runner_fake(fake_bin / "planu-python")
    planu_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        timeout=SCRIPT_TIMEOUT,
    ).strip()
    environment = script_environment(
        fake_bin,
        FAKE_ARTIFACT_MODE=artifact_mode,
        FAKE_CHILD_PID_LOG=str(child_pid_log),
        FAKE_COMMAND_LOG=str(command_log),
        FAKE_PLANU_HEAD=planu_head,
        FAKE_PROJECT_ROOT=str(ROOT),
        FAKE_REAL_PYTHON=sys.executable,
        FAKE_RESIST_TERM="1" if resistant else "0",
        FAKE_SERVER_ENV_LOG=str(server_env_log),
        FAKE_SERVER_PID_LOG=str(server_pid_log),
        FAKE_SERVER_URL="http://127.0.0.1:3000",
        FAKE_WEBSHOP_ROOT=str(webshop_root),
        FAKE_WEBSHOP_TOPLEVEL=str(webshop_root),
        PLANU_PYTHON=str(fake_bin / "planu-python"),
        RUN_ROOT=str(tmp_path / "run"),
        WEBSHOP_ENV_PREFIX=str(env_prefix),
        WEBSHOP_ROOT=str(webshop_root),
        WEBSHOP_STOP_ATTEMPTS="2",
    )
    return environment, command_log, server_pid_log, child_pid_log


def bootstrap_server_environment(tmp_path, *, include_java=True):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    webshop_root = tmp_path / "WebShop"
    (webshop_root / ".git").mkdir(parents=True)
    for relative_path in (
        "data/items_shuffle_1000.json",
        "data/items_ins_v2_1000.json",
        "data/items_human_ins.json",
        "search_engine/resources/documents.jsonl",
        "search_engine/indexes/segments_1",
        "search_engine/indexes/_0.si",
    ):
        path = webshop_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("nonempty\n", encoding="utf-8")
    env_prefix = webshop_root / "custom-env"
    (env_prefix / "bin").mkdir(parents=True)
    if include_java:
        java_bin = env_prefix / "lib" / "jvm" / "bin"
        java_bin.mkdir(parents=True)
        write_executable(java_bin / "java", "exit 0\n")
    server_pid_log = tmp_path / "server.pid"
    server_env_log = tmp_path / "server.env"
    write_executable(
        env_prefix / "bin" / "python",
        """
if [[ "$*" == *"platform.python_version"* ]]; then
  printf '3.8.13\n'
elif [[ "$*" == *"import web_agent_site.app"* ]]; then
  java_home="$FAKE_ENV_PREFIX/lib/jvm"
  [[ "${JAVA_HOME:-}" == "$java_home" ]] || exit 21
  case ":$PATH:" in
    *":$java_home/bin:"*) ;;
    *) exit 22 ;;
  esac
elif [[ "$*" == *"-m web_agent_site.app"* ]]; then
  printf 'JAVA_HOME=%s\nPATH=%s\n' \
    "${JAVA_HOME:-}" "$PATH" > "$FAKE_SERVER_ENV_LOG"
  printf '%s\n' "$$" > "$FAKE_SERVER_PID_LOG"
  trap 'exit 0' TERM INT
  while true; do sleep 1; done
fi
""",
    )
    (env_prefix / (
        ".planu-webshop-small-"
        "64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd"
    )).touch()
    git_log = tmp_path / "git.log"
    write_executable(
        fake_bin / "git",
        """
printf '%s\n' "$*" >> "$FAKE_GIT_LOG"
case "$*" in
  *"rev-parse --is-inside-work-tree"*) printf 'true\n' ;;
  *"rev-parse --show-toplevel"*) printf '%s\n' "$FAKE_WEBSHOP_TOPLEVEL" ;;
  *"remote get-url origin"*)
    printf 'https://github.com/princeton-nlp/WebShop.git\n'
    ;;
  *"status --porcelain"*)
    if [[ "$*" != *":(exclude)custom-env"* ]]; then
      printf '?? custom-env/\n'
    fi
    ;;
  *"cat-file -e"*) ;;
  *"rev-parse HEAD"*)
    printf '64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd\n'
    ;;
  *"symbolic-ref -q HEAD"*) exit 1 ;;
esac
""",
    )
    write_executable(fake_bin / "conda", "exit 0\n")
    write_executable(
        fake_bin / "curl",
        """
[[ -s "$FAKE_SERVER_PID_LOG" ]] || exit 1
kill -0 "$(cat "$FAKE_SERVER_PID_LOG")" 2>/dev/null
""",
    )
    environment = script_environment(
        fake_bin,
        FAKE_GIT_LOG=str(git_log),
        FAKE_ENV_PREFIX=str(env_prefix),
        FAKE_SERVER_ENV_LOG=str(server_env_log),
        FAKE_SERVER_PID_LOG=str(server_pid_log),
        FAKE_WEBSHOP_TOPLEVEL=str(webshop_root),
        WEBSHOP_ENV_PREFIX=str(env_prefix),
        WEBSHOP_ROOT=str(webshop_root),
    )
    return environment, webshop_root, env_prefix, git_log, server_env_log


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
    assert "rev-parse --show-toplevel" in bootstrap
    assert "verify_server_startup" in bootstrap
    assert "reset --hard" not in bootstrap
    assert "clean -f" not in bootstrap

    assert smoke.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
    assert "64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd" in smoke
    assert "https://github.com/princeton-nlp/WebShop.git" in smoke
    assert "BASH_SOURCE[0]" in smoke
    assert "WEBSHOP_ROOT" in smoke
    assert "WEBSHOP_URL" in smoke
    assert "http://127.0.0.1:3000/fixed_1" in smoke
    assert '"-m", "web_agent_site.app", "--log", "--attrs"' in smoke
    assert "authoritative WebShop smoke is local-only" in smoke
    assert "rev-parse --show-toplevel" in smoke
    assert "status --porcelain --untracked-files=no" in smoke
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
    assert 'kill -TERM -- "-${SERVER_PGID}"' in smoke
    assert 'kill -KILL -- "-${SERVER_PGID}"' in smoke

    for script in (WEBSHOP_BOOTSTRAP, WEBSHOP_SCRIPT):
        syntax = subprocess.run(
            ["bash", "-n", str(script)],
            check=False,
            capture_output=True,
            text=True,
            timeout=SCRIPT_TIMEOUT,
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
        timeout=SCRIPT_TIMEOUT,
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
        timeout=SCRIPT_TIMEOUT,
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
  *"rev-parse --show-toplevel"*) printf '%s\n' "$FAKE_WEBSHOP_TOPLEVEL" ;;
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
            FAKE_WEBSHOP_TOPLEVEL=str(webshop_root),
            WEBSHOP_ROOT=str(webshop_root),
        ),
        timeout=SCRIPT_TIMEOUT,
    )

    commands = git_log.read_text(encoding="utf-8")
    assert completed.returncode != 0
    assert "remote" in completed.stderr.lower()
    assert "checkout" not in commands
    assert "fetch" not in commands
    assert sentinel.read_text(encoding="utf-8") == "keep\n"


def test_webshop_bootstrap_rejects_git_subdirectory_before_mutation(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    git_log = tmp_path / "git.log"
    webshop_root = tmp_path / "WebShop"
    nested_root = webshop_root / "web_agent_site"
    nested_root.mkdir(parents=True)
    write_executable(
        fake_bin / "git",
        """
printf '%s\n' "$*" >> "$FAKE_GIT_LOG"
case "$*" in
  *"rev-parse --is-inside-work-tree"*) printf 'true\n' ;;
  *"rev-parse --show-toplevel"*) printf '%s\n' "$FAKE_WEBSHOP_TOPLEVEL" ;;
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
            FAKE_WEBSHOP_TOPLEVEL=str(webshop_root),
            WEBSHOP_ROOT=str(nested_root),
        ),
        timeout=SCRIPT_TIMEOUT,
    )

    commands = git_log.read_text(encoding="utf-8")
    assert completed.returncode != 0
    assert "top-level" in completed.stderr
    assert "fetch" not in commands
    assert "checkout" not in commands


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
  *"rev-parse --show-toplevel"*) printf '%s\n' "$FAKE_WEBSHOP_TOPLEVEL" ;;
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
            FAKE_WEBSHOP_TOPLEVEL=str(webshop_root),
            WEBSHOP_ROOT=str(webshop_root),
        ),
        timeout=SCRIPT_TIMEOUT,
    )

    commands = git_log.read_text(encoding="utf-8")
    assert completed.returncode != 0
    assert "uncommitted" in completed.stderr.lower()
    assert "checkout" not in commands
    assert "fetch" not in commands
    assert "reset" not in commands
    assert sentinel.read_text(encoding="utf-8") == "keep\n"


def test_webshop_bootstrap_rejects_empty_lucene_index(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    webshop_root = tmp_path / "WebShop"
    (webshop_root / ".git").mkdir(parents=True)
    (webshop_root / "data").mkdir()
    (webshop_root / "data" / "items_shuffle_1000.json").write_text(
        "{}\n",
        encoding="utf-8",
    )
    (webshop_root / "data" / "items_ins_v2_1000.json").write_text(
        "{}\n",
        encoding="utf-8",
    )
    (webshop_root / "data" / "items_human_ins.json").write_text(
        "{}\n",
        encoding="utf-8",
    )
    resources = webshop_root / "search_engine" / "resources"
    resources.mkdir(parents=True)
    (resources / "documents.jsonl").write_text(
        "{}\n",
        encoding="utf-8",
    )
    (webshop_root / "search_engine" / "indexes").mkdir(parents=True)
    env_prefix = webshop_root / ".conda-planu"
    (env_prefix / "bin").mkdir(parents=True)
    java_bin = env_prefix / "lib" / "jvm" / "bin"
    java_bin.mkdir(parents=True)
    write_executable(java_bin / "java", "exit 0\n")
    write_executable(
        env_prefix / "bin" / "python",
        """
if [[ "$*" == *"platform.python_version"* ]]; then
  printf '3.8.13\n'
fi
""",
    )
    (env_prefix / (
        ".planu-webshop-small-"
        "64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd"
    )).touch()
    write_executable(
        fake_bin / "git",
        """
case "$*" in
  *"rev-parse --is-inside-work-tree"*) printf 'true\n' ;;
  *"rev-parse --show-toplevel"*) printf '%s\n' "$FAKE_WEBSHOP_TOPLEVEL" ;;
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
    write_executable(fake_bin / "conda", "exit 0\n")

    completed = subprocess.run(
        ["bash", str(WEBSHOP_BOOTSTRAP)],
        check=False,
        capture_output=True,
        text=True,
        env=script_environment(
            fake_bin,
            FAKE_WEBSHOP_TOPLEVEL=str(webshop_root),
            WEBSHOP_ROOT=str(webshop_root),
        ),
        timeout=SCRIPT_TIMEOUT,
    )

    assert completed.returncode != 0
    assert "index" in completed.stderr.lower()


def test_webshop_bootstrap_configures_resolved_conda_java_for_server(tmp_path):
    (
        environment,
        webshop_root,
        env_prefix,
        git_log,
        server_env_log,
    ) = bootstrap_server_environment(tmp_path)
    environment["WEBSHOP_ENV_PREFIX"] = "WebShop/custom-env"
    environment["WEBSHOP_ROOT"] = "WebShop"

    completed = subprocess.run(
        ["bash", str(WEBSHOP_BOOTSTRAP)],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=SCRIPT_TIMEOUT,
    )

    assert completed.returncode == 0, completed.stderr
    assert str(webshop_root.resolve()) in completed.stdout
    assert ":(exclude)custom-env" in git_log.read_text(encoding="utf-8")
    java_home = env_prefix.resolve() / "lib" / "jvm"
    assert server_env_log.read_text(encoding="utf-8").splitlines() == [
        "JAVA_HOME={}".format(java_home),
        "PATH={}:{}".format(java_home / "bin", environment["PATH"]),
    ]


def test_webshop_fresh_bootstrap_installs_and_exports_java_before_setup(
    tmp_path,
):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    webshop_root = tmp_path / "WebShop"
    (webshop_root / ".git").mkdir(parents=True)
    env_prefix = webshop_root / ".conda-planu"
    event_log = tmp_path / "events.log"
    server_pid_log = tmp_path / "server.pid"
    write_executable(
        webshop_root / "setup.sh",
        """
java_home="$FAKE_ENV_PREFIX/lib/jvm"
[[ "${JAVA_HOME:-}" == "$java_home" ]] || {
  printf 'setup missing JAVA_HOME\n' >&2
  exit 31
}
case ":$PATH:" in
  *":$java_home/bin:"*) ;;
  *)
    printf 'setup missing Java PATH\n' >&2
    exit 32
    ;;
esac
printf 'setup\n' >> "$FAKE_EVENT_LOG"
"$FAKE_ENV_PREFIX/bin/python" -c 'import pyserini'
mkdir -p data search_engine/resources search_engine/indexes
printf '{}\n' > data/items_shuffle_1000.json
printf '{}\n' > data/items_ins_v2_1000.json
printf '{}\n' > data/items_human_ins.json
printf '{}\n' > search_engine/resources/documents.jsonl
printf 'segments\n' > search_engine/indexes/segments_1
printf 'segment info\n' > search_engine/indexes/_0.si
""",
    )
    write_executable(
        fake_bin / "git",
        """
case "$*" in
  *"rev-parse --is-inside-work-tree"*) printf 'true\n' ;;
  *"rev-parse --show-toplevel"*) printf '%s\n' "$FAKE_WEBSHOP_TOPLEVEL" ;;
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
command="${1:-}"
printf 'conda %s\n' "$*" >> "$FAKE_EVENT_LOG"
if [[ "$command" == "create" ]]; then
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
elif [[ "$*" == *"import pyserini"* ]] ||
  [[ "$*" == *"import web_agent_site.app"* ]]; then
  java_home="$FAKE_ENV_PREFIX/lib/jvm"
  [[ -x "$java_home/bin/java" ]] || exit 41
  [[ "${JAVA_HOME:-}" == "$java_home" ]] || exit 42
  case ":$PATH:" in
    *":$java_home/bin:"*) ;;
    *) exit 43 ;;
  esac
  printf 'python import %s\n' "$*" >> "$FAKE_EVENT_LOG"
elif [[ "$*" == *"-m web_agent_site.app"* ]]; then
  printf '%s\n' "$$" > "$FAKE_SERVER_PID_LOG"
  trap 'exit 0' TERM INT
  while true; do sleep 1; done
fi
EOF
  chmod +x "$prefix/bin/python"
elif [[ "$command" == "install" ]]; then
  [[ "$*" == "install -y -p $FAKE_ENV_PREFIX -c conda-forge openjdk=11" ]] ||
    exit 51
  mkdir -p "$FAKE_ENV_PREFIX/lib/jvm/bin"
  printf '#!/usr/bin/env bash\nexit 0\n' \
    > "$FAKE_ENV_PREFIX/lib/jvm/bin/java"
  chmod +x "$FAKE_ENV_PREFIX/lib/jvm/bin/java"
elif [[ "$command" == "run" ]]; then
  java_home="$FAKE_ENV_PREFIX/lib/jvm"
  [[ "${JAVA_HOME:-}" == "$java_home" ]] || exit 52
  case ":$PATH:" in
    *":$java_home/bin:"*) ;;
    *) exit 53 ;;
  esac
  bash ./setup.sh -d small
fi
""",
    )
    write_executable(
        fake_bin / "curl",
        """
[[ -s "$FAKE_SERVER_PID_LOG" ]] || exit 1
kill -0 "$(cat "$FAKE_SERVER_PID_LOG")" 2>/dev/null
""",
    )
    environment = script_environment(
        fake_bin,
        FAKE_ENV_PREFIX=str(env_prefix),
        FAKE_EVENT_LOG=str(event_log),
        FAKE_SERVER_PID_LOG=str(server_pid_log),
        FAKE_WEBSHOP_TOPLEVEL=str(webshop_root),
        WEBSHOP_ROOT=str(webshop_root),
    )

    completed = subprocess.run(
        ["bash", str(WEBSHOP_BOOTSTRAP)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=SCRIPT_TIMEOUT,
    )

    assert completed.returncode == 0, completed.stderr
    events = event_log.read_text(encoding="utf-8").splitlines()
    assert events[:5] == [
        "conda create -y -p {} python=3.8.13".format(env_prefix),
        "conda install -y -p {} -c conda-forge openjdk=11".format(
            env_prefix
        ),
        "conda run --no-capture-output -p {} bash ./setup.sh -d small".format(
            env_prefix
        ),
        "setup",
        "python import -c import pyserini",
    ]
    assert "python import -c import web_agent_site.app" in events


def test_webshop_bootstrap_fails_clearly_when_conda_java_is_missing(tmp_path):
    environment, _, env_prefix, _, server_env_log = (
        bootstrap_server_environment(tmp_path, include_java=False)
    )

    completed = subprocess.run(
        ["bash", str(WEBSHOP_BOOTSTRAP)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=SCRIPT_TIMEOUT,
    )

    java = env_prefix.resolve() / "lib" / "jvm" / "bin" / "java"
    assert completed.returncode != 0
    assert str(java) in completed.stderr
    assert not server_env_log.exists()


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
mkdir -p data search_engine/resources search_engine/indexes
printf '{}\n' > data/items_shuffle_1000.json
printf '{}\n' > data/items_ins_v2_1000.json
printf '{}\n' > data/items_human_ins.json
printf '{}\n' > search_engine/resources/documents.jsonl
printf 'segments\n' > search_engine/indexes/segments_1
printf 'segment info\n' > search_engine/indexes/_0.si
""",
    )
    write_executable(
        fake_bin / "git",
        """
case "$*" in
  *"rev-parse --is-inside-work-tree"*) printf 'true\n' ;;
  *"rev-parse --show-toplevel"*) printf '%s\n' "$FAKE_WEBSHOP_TOPLEVEL" ;;
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
  mkdir -p "$prefix/bin" "$prefix/lib/jvm/bin"
  printf '#!/usr/bin/env bash\nexit 0\n' > "$prefix/lib/jvm/bin/java"
  chmod +x "$prefix/lib/jvm/bin/java"
  cat > "$prefix/bin/python" <<'EOF'
#!/usr/bin/env bash
if [[ "$*" == *"platform.python_version"* ]]; then
  printf '3.8.13\n'
elif [[ "$*" == *"import web_agent_site.app"* ]]; then
  exit 0
elif [[ "$*" == *"-m web_agent_site.app"* ]]; then
  printf '%s\n' "$$" >> "$FAKE_SERVER_PID_LOG"
  trap 'exit 0' TERM INT
  while true; do
    sleep 1
  done
fi
EOF
  chmod +x "$prefix/bin/python"
elif [[ "${1:-}" == "run" ]]; then
  bash ./setup.sh -d small
fi
""",
    )
    write_executable(
        fake_bin / "curl",
        """
[[ -s "$FAKE_SERVER_PID_LOG" ]] || exit 1
kill -0 "$(tail -n 1 "$FAKE_SERVER_PID_LOG")" 2>/dev/null
""",
    )
    server_pid_log = tmp_path / "server.pid"
    environment = script_environment(
        fake_bin,
        FAKE_CONDA_LOG=str(conda_log),
        FAKE_SERVER_PID_LOG=str(server_pid_log),
        FAKE_SETUP_LOG=str(setup_log),
        FAKE_WEBSHOP_TOPLEVEL=str(webshop_root),
        WEBSHOP_ROOT=str(webshop_root),
    )

    first = subprocess.run(
        ["bash", str(WEBSHOP_BOOTSTRAP)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=SCRIPT_TIMEOUT,
    )
    second = subprocess.run(
        ["bash", str(WEBSHOP_BOOTSTRAP)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=SCRIPT_TIMEOUT,
    )

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    conda_commands = conda_log.read_text(encoding="utf-8").splitlines()
    assert sum("conda create " in line for line in conda_commands) == 1
    assert sum("conda run " in line for line in conda_commands) == 1
    assert setup_log.read_text(encoding="utf-8").splitlines() == [
        "setup -d small"
    ]
    server_pids = server_pid_log.read_text(encoding="utf-8").splitlines()
    assert len(server_pids) == 2
    server_pid = int(server_pids[-1])
    with pytest.raises(ProcessLookupError):
        os.kill(server_pid, 0)


def test_webshop_smoke_rejects_external_server_even_with_asserted_commit(
    tmp_path,
):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    command_log = tmp_path / "commands.log"
    write_executable(
        fake_bin / "curl",
        'printf "curl %s\\n" "$*" >> "$FAKE_COMMAND_LOG"\n',
    )
    write_smoke_runner_fake(fake_bin / "planu-python")

    completed = subprocess.run(
        ["bash", str(WEBSHOP_SCRIPT)],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        env=script_environment(
            fake_bin,
            FAKE_COMMAND_LOG=str(command_log),
            FAKE_REAL_PYTHON=sys.executable,
            PLANU_PYTHON="bin/planu-python",
            RUN_ROOT="run",
            WEBSHOP_ROOT="missing-webshop-checkout",
            WEBSHOP_SERVER_COMMIT=WEBSHOP_COMMIT,
            WEBSHOP_URL="https://shop.example",
        ),
        timeout=SCRIPT_TIMEOUT,
    )

    assert completed.returncode != 0
    assert "authoritative" in completed.stderr.lower()
    assert "local" in completed.stderr.lower()
    assert not command_log.exists()


def test_webshop_smoke_starts_pinned_local_checkout_and_verifies_artifacts(
    tmp_path,
):
    environment, command_log, server_pid_log, _ = local_smoke_environment(
        tmp_path
    )
    completed = subprocess.run(
        ["bash", str(WEBSHOP_SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=SCRIPT_TIMEOUT,
    )

    assert completed.returncode == 0, completed.stderr
    commands = command_log.read_text(encoding="utf-8")
    assert "--smoke" in commands
    assert "--iterations 1" in commands
    assert "--depth 10" in commands
    assert "--task-start-index 1 --task-end-index 2" in commands
    assert '"http_transition_count": 3' in completed.stdout
    assert '"quantile_backup_count": 1' in completed.stdout
    java_home = (
        Path(environment["WEBSHOP_ENV_PREFIX"]).resolve() / "lib" / "jvm"
    )
    assert (tmp_path / "server.env").read_text(
        encoding="utf-8"
    ).splitlines() == [
        "JAVA_HOME={}".format(java_home),
        "PATH={}:{}".format(java_home / "bin", environment["PATH"]),
    ]
    server_pid = int(server_pid_log.read_text(encoding="utf-8"))
    with pytest.raises(ProcessLookupError):
        os.kill(server_pid, 0)


def test_webshop_smoke_fails_clearly_when_conda_java_is_missing(tmp_path):
    environment, _, server_pid_log, _ = local_smoke_environment(tmp_path)
    env_prefix = Path(environment["WEBSHOP_ENV_PREFIX"]).resolve()
    java = env_prefix / "lib" / "jvm" / "bin" / "java"
    java.unlink()

    completed = subprocess.run(
        ["bash", str(WEBSHOP_SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=SCRIPT_TIMEOUT,
    )

    assert completed.returncode != 0
    assert str(java) in completed.stderr
    assert not server_pid_log.exists()
    assert not (tmp_path / "server.env").exists()


def test_webshop_smoke_rejects_git_subdirectory_root(tmp_path):
    environment, command_log, server_pid_log, _ = local_smoke_environment(
        tmp_path
    )
    webshop_root = Path(environment["FAKE_WEBSHOP_ROOT"])
    nested_root = webshop_root / "web_agent_site"
    nested_root.mkdir()
    environment["FAKE_WEBSHOP_ROOT"] = str(nested_root)
    environment["WEBSHOP_ROOT"] = str(nested_root)

    completed = subprocess.run(
        ["bash", str(WEBSHOP_SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=SCRIPT_TIMEOUT,
    )

    assert completed.returncode != 0
    assert "top-level" in completed.stderr
    assert not server_pid_log.exists()
    assert "python -m planu_core.webshop.runner" not in (
        command_log.read_text(encoding="utf-8")
    )


@pytest.mark.parametrize(
    ("tracked_status", "untracked_files"),
    [
        (" M planu_core/search.py\n", ""),
        ("", "scripts/local_smoke_override.py\n"),
        (" M requirements.txt\n", ""),
    ],
)
def test_webshop_smoke_rejects_dirty_relevant_planu_source(
    tmp_path,
    tracked_status,
    untracked_files,
):
    environment, command_log, server_pid_log, _ = local_smoke_environment(
        tmp_path
    )
    environment["FAKE_PLANU_TRACKED_STATUS"] = tracked_status
    environment["FAKE_PLANU_UNTRACKED_FILES"] = untracked_files
    completed = subprocess.run(
        ["bash", str(WEBSHOP_SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=SCRIPT_TIMEOUT,
    )

    assert completed.returncode != 0
    assert "PlanU source checkout must be clean" in completed.stderr
    assert not server_pid_log.exists()
    assert "python -m planu_core.webshop.runner" not in (
        command_log.read_text(encoding="utf-8")
    )


def test_webshop_smoke_allows_external_caches_and_output_files(tmp_path):
    environment, _, _, _ = local_smoke_environment(tmp_path)
    environment["FAKE_PLANU_UNTRACKED_FILES"] = (
        "external/WebShop/local.py\n"
        "planu_core/__pycache__/generated.py\n"
        "planu_core/outputs/result.py\n"
    )

    completed = subprocess.run(
        ["bash", str(WEBSHOP_SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=SCRIPT_TIMEOUT,
    )

    assert completed.returncode == 0, completed.stderr


def test_webshop_smoke_kills_term_resistant_process_group(tmp_path):
    environment, _, server_pid_log, child_pid_log = local_smoke_environment(
        tmp_path,
        resistant=True,
    )
    completed = subprocess.run(
        ["bash", str(WEBSHOP_SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=SCRIPT_TIMEOUT,
    )

    assert completed.returncode == 0, completed.stderr
    for pid_log in (server_pid_log, child_pid_log):
        pid = int(pid_log.read_text(encoding="utf-8"))
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("minimal", "keys mismatch"),
        ("wrong_webshop_commit", "pinned WebShop commit"),
        ("empty_planu_commit", "checkout HEAD"),
        ("wrong_model", "scripted model policy"),
        ("wrong_server", "configured URL"),
        ("wrong_task", "fixed_1"),
        ("zero_iterations", "at least one backup"),
        ("empty_trajectory", "real HTTP trajectory"),
        ("missing_observation", "keys mismatch"),
        ("reward_mismatch", "terminal reward"),
        ("results_mismatch", "consolidated results"),
    ],
)
def test_webshop_smoke_rejects_tampered_authoritative_artifacts(
    tmp_path,
    mode,
    message,
):
    environment, _, _, _ = local_smoke_environment(
        tmp_path,
        artifact_mode=mode,
    )
    completed = subprocess.run(
        ["bash", str(WEBSHOP_SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=SCRIPT_TIMEOUT,
    )

    assert completed.returncode != 0
    assert message in completed.stderr
