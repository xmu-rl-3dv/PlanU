#!/usr/bin/env bash
set -euo pipefail

SOURCE_URL="https://github.com/princeton-nlp/WebShop.git"
COMMIT="64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd"
WEBSHOP_URL="${WEBSHOP_URL:-http://127.0.0.1:3000}"
WEBSHOP_URL="${WEBSHOP_URL%/}"
DEFAULT_HEALTH_URL="http://127.0.0.1:3000/fixed_1"
HEALTH_URL="${WEBSHOP_URL}/fixed_1"
SMOKE_QUERY="${WEBSHOP_SMOKE_QUERY:-product}"
START_ATTEMPTS="${WEBSHOP_START_ATTEMPTS:-120}"
STOP_ATTEMPTS="${WEBSHOP_STOP_ATTEMPTS:-20}"
SERVER_PID=""
SERVER_PGID=""
SERVER_COMMIT=""

fail() {
  printf 'WebShop smoke error: %s\n' "$*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 ||
    fail "required command '$1' is missing"
}

canonical_path() {
  python3 - "$1" <<'PY'
import os
import sys

print(os.path.realpath(os.path.abspath(sys.argv[1])))
PY
}

absolute_path() {
  python3 - "$1" <<'PY'
import os
import sys

print(os.path.abspath(sys.argv[1]))
PY
}

canonical_command() {
  local configured="$1"
  local resolved
  if [[ "${configured}" == */* ]]; then
    resolved="$(absolute_path "${configured}")"
    [[ -x "${resolved}" ]] ||
      fail "configured executable is missing: ${resolved}"
  else
    resolved="$(command -v "${configured}" 2>/dev/null || true)"
    [[ -n "${resolved}" ]] ||
      fail "required command '${configured}' is missing"
    resolved="$(absolute_path "${resolved}")"
  fi
  printf '%s\n' "${resolved}"
}

is_official_remote() {
  case "$1" in
    "https://github.com/princeton-nlp/WebShop.git" | \
      "https://github.com/princeton-nlp/WebShop" | \
      "git@github.com:princeton-nlp/WebShop.git" | \
      "ssh://git@github.com/princeton-nlp/WebShop.git")
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

server_group_alive() {
  [[ -n "${SERVER_PGID}" ]] &&
    kill -0 -- "-${SERVER_PGID}" 2>/dev/null
}

reap_server() {
  [[ -n "${SERVER_PID}" ]] || return
  wait "${SERVER_PID}" 2>/dev/null || true
  SERVER_PID=""
  SERVER_PGID=""
}

cleanup() {
  local attempt=0
  [[ -n "${SERVER_PID}" ]] || return 0

  if server_group_alive; then
    kill -TERM -- "-${SERVER_PGID}" 2>/dev/null || true
    while kill -0 "${SERVER_PID}" 2>/dev/null &&
      ((attempt < STOP_ATTEMPTS)); do
      attempt=$((attempt + 1))
      sleep 0.1
    done
    if server_group_alive; then
      kill -KILL -- "-${SERVER_PGID}" 2>/dev/null || true
    fi
  elif kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill -TERM "${SERVER_PID}" 2>/dev/null || true
    while kill -0 "${SERVER_PID}" 2>/dev/null &&
      ((attempt < STOP_ATTEMPTS)); do
      attempt=$((attempt + 1))
      sleep 0.1
    done
    if kill -0 "${SERVER_PID}" 2>/dev/null; then
      kill -KILL "${SERVER_PID}" 2>/dev/null || true
    fi
  fi
  reap_server
}

server_reachable() {
  curl --fail --silent \
    --connect-timeout 2 --max-time 5 "${HEALTH_URL}" >/dev/null
}

verify_planu_source_clean() {
  local tracked_status
  local untracked_status
  local untracked_path
  local -a pathspecs

  pathspecs=(
    planu_core
    webshop
    scripts
    setup.py
    pyproject.toml
    setup.cfg
    pytest.ini
    tox.ini
    requirements.txt
    requirements-dev.txt
    requirements-experiments.txt
    requirements-experiments-lock.txt
    ":(top,glob)*.toml"
    ":(top,glob)*.cfg"
    ":(top,glob)*.ini"
    ":(top,glob)*.yaml"
    ":(top,glob)*.yml"
    ":(top,glob)requirements*.txt"
    ":(exclude,glob)**/__pycache__/**"
    ":(exclude,glob)**/.pytest_cache/**"
    ":(exclude,glob)**/.mypy_cache/**"
    ":(exclude,glob)**/.ruff_cache/**"
    ":(exclude,glob)**/output/**"
    ":(exclude,glob)**/outputs/**"
    ":(exclude,glob)**/results/**"
    ":(exclude,glob)**/runs/**"
    ":(exclude,glob)**/logs/**"
  )
  tracked_status="$(
    git -C "${PROJECT_ROOT}" status --porcelain --untracked-files=no \
      -- "${pathspecs[@]}"
  )" || fail "could not inspect tracked PlanU source files"
  [[ -z "${tracked_status}" ]] ||
    fail "PlanU source checkout must be clean before an authoritative smoke run"

  untracked_status="$(
    git -C "${PROJECT_ROOT}" ls-files --others --exclude-standard \
      -- "${pathspecs[@]}"
  )" || fail "could not inspect untracked PlanU source files"
  while IFS= read -r untracked_path; do
    [[ -n "${untracked_path}" ]] || continue
    case "${untracked_path}" in
      */__pycache__/* | */.pytest_cache/* | */.mypy_cache/* | \
        */.ruff_cache/* | */output/* | */outputs/* | */results/* | \
        */runs/* | */logs/*)
        continue
        ;;
      planu_core/*.py | planu_core/*.pyi | \
        planu_core/*.sh | planu_core/*.toml | planu_core/*.cfg | \
        planu_core/*.ini | planu_core/*.yaml | planu_core/*.yml | \
        webshop/*.py | webshop/*.pyi | webshop/*.sh | webshop/*.toml | \
        webshop/*.cfg | webshop/*.ini | webshop/*.yaml | webshop/*.yml | \
        scripts/*.py | scripts/*.pyi | scripts/*.sh | scripts/*.toml | \
        scripts/*.cfg | scripts/*.ini | scripts/*.yaml | scripts/*.yml | \
        setup.py | *.toml | *.cfg | *.ini | *.yaml | *.yml | \
        requirements*.txt)
        fail "PlanU source checkout must be clean before an authoritative smoke run"
        ;;
    esac
  done <<<"${untracked_status}"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

require_command python3
require_command git
require_command curl
SCRIPT_PATH="$(canonical_path "${BASH_SOURCE[0]}")"
SCRIPT_DIR="${SCRIPT_PATH%/*}"
PROJECT_ROOT="${SCRIPT_DIR%/*}"
WEBSHOP_ROOT="$(canonical_path "${WEBSHOP_ROOT:-${PROJECT_ROOT}/external/WebShop}")"
if [[ -n "${WEBSHOP_ENV_PREFIX:-}" ]]; then
  WEBSHOP_ENV_PREFIX="$(canonical_path "${WEBSHOP_ENV_PREFIX}")"
elif [[ -n "${WEBSHOP_PYTHON:-}" ]]; then
  WEBSHOP_PYTHON="$(absolute_path "${WEBSHOP_PYTHON}")"
  WEBSHOP_ENV_PREFIX="${WEBSHOP_PYTHON%/bin/python}"
else
  WEBSHOP_ENV_PREFIX="${WEBSHOP_ROOT}/.conda-planu"
fi
WEBSHOP_ENV_PREFIX="$(canonical_path "${WEBSHOP_ENV_PREFIX}")"
WEBSHOP_PYTHON="$(absolute_path "${WEBSHOP_PYTHON:-${WEBSHOP_ENV_PREFIX}/bin/python}")"
PLANU_PYTHON="$(canonical_command "${PLANU_PYTHON:-python3}")"
RUN_ROOT="$(canonical_path "${RUN_ROOT:-$(mktemp -d "${TMPDIR:-/tmp}/planu-webshop-smoke.XXXXXX")}")"

case "${START_ATTEMPTS}" in
  "" | *[!0-9]*) fail "WEBSHOP_START_ATTEMPTS must be a positive integer" ;;
esac
((START_ATTEMPTS > 0)) ||
  fail "WEBSHOP_START_ATTEMPTS must be a positive integer"
case "${STOP_ATTEMPTS}" in
  "" | *[!0-9]*) fail "WEBSHOP_STOP_ATTEMPTS must be a positive integer" ;;
esac
((STOP_ATTEMPTS > 0)) ||
  fail "WEBSHOP_STOP_ATTEMPTS must be a positive integer"

[[ "${HEALTH_URL}" == "${DEFAULT_HEALTH_URL}" ]] ||
  fail "authoritative WebShop smoke is local-only; external URLs are non-authoritative"

verify_planu_source_clean
PLANU_COMMIT="$(git -C "${PROJECT_ROOT}" rev-parse HEAD 2>/dev/null || true)"
[[ -n "${PLANU_COMMIT}" ]] ||
  fail "could not resolve the current clean PlanU checkout commit"

mkdir -p "${RUN_ROOT}"

[[ "$(git -C "${WEBSHOP_ROOT}" rev-parse --is-inside-work-tree 2>/dev/null || true)" == "true" ]] ||
  fail "WEBSHOP_ROOT is not a bootstrapped Git working tree: ${WEBSHOP_ROOT}"
checkout_root="$(git -C "${WEBSHOP_ROOT}" rev-parse --show-toplevel 2>/dev/null || true)"
[[ -n "${checkout_root}" ]] ||
  fail "could not resolve the WebShop checkout top-level"
checkout_root="$(canonical_path "${checkout_root}")"
[[ "${WEBSHOP_ROOT}" == "${checkout_root}" ]] ||
  fail "WEBSHOP_ROOT must equal the Git checkout top-level: ${checkout_root}"

remote_url="$(git -C "${WEBSHOP_ROOT}" remote get-url origin 2>/dev/null || true)"
is_official_remote "${remote_url}" ||
  fail "origin remote must match ${SOURCE_URL}; found ${remote_url:-missing}"

actual_commit="$(git -C "${WEBSHOP_ROOT}" rev-parse HEAD 2>/dev/null || true)"
[[ "${actual_commit}" == "${COMMIT}" ]] ||
  fail "WEBSHOP_ROOT must be pinned at ${COMMIT}; found ${actual_commit:-unknown}"
SERVER_COMMIT="${actual_commit}"

pathspecs=(.)
if [[ "${WEBSHOP_ENV_PREFIX}" == "${WEBSHOP_ROOT}/"* ]]; then
  env_relative="${WEBSHOP_ENV_PREFIX#"${WEBSHOP_ROOT}/"}"
  pathspecs+=(":(exclude)${env_relative}")
fi
working_tree_status="$(
  git -C "${WEBSHOP_ROOT}" status --porcelain --untracked-files=normal \
    -- "${pathspecs[@]}"
)" || fail "could not inspect the WebShop working tree"
[[ -z "${working_tree_status}" ]] ||
  fail "WEBSHOP_ROOT must be clean before a real smoke run"

[[ -x "${WEBSHOP_PYTHON}" ]] ||
  fail "pinned WebShop Python is missing; run scripts/bootstrap_webshop.sh"
server_reachable &&
  fail "port 3000 is already serving a process not started by this smoke run"

python3 - "${WEBSHOP_ROOT}" "${WEBSHOP_PYTHON}" \
  >"${RUN_ROOT}/webshop-server.log" 2>&1 <<'PY' &
import os
import sys

os.chdir(sys.argv[1])
os.setsid()
os.execv(
    sys.argv[2],
    [sys.argv[2], "-m", "web_agent_site.app", "--log", "--attrs"],
)
PY
SERVER_PID=$!
SERVER_PGID="${SERVER_PID}"

attempt=0
until server_reachable; do
  attempt=$((attempt + 1))
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    reap_server
    fail "official WebShop server exited during startup; see ${RUN_ROOT}/webshop-server.log"
  fi
  if ((attempt >= START_ATTEMPTS)); then
    fail "official WebShop server did not become ready after ${START_ATTEMPTS} attempts"
  fi
  sleep 1
done
kill -0 "${SERVER_PID}" 2>/dev/null ||
  fail "official WebShop server exited after its health check"

cd "${PROJECT_ROOT}"
"${PLANU_PYTHON}" -m planu_core.webshop.runner \
  --smoke \
  --webshop-url "${WEBSHOP_URL}" \
  --task-start-index 1 \
  --task-end-index 2 \
  --iterations 1 \
  --depth 10 \
  --smoke-search-query "${SMOKE_QUERY}" \
  --output-dir "${RUN_ROOT}"

"${PLANU_PYTHON}" - \
  "${RUN_ROOT}" "${SERVER_COMMIT}" "${PLANU_COMMIT}" "${WEBSHOP_URL}" \
  "${SMOKE_QUERY}" <<'PY'
import hashlib
import json
import math
from pathlib import Path
import sys
from urllib.parse import urlsplit, urlunsplit

run_root = Path(sys.argv[1])
expected_commit = sys.argv[2]
expected_planu_commit = sys.argv[3]
configured_server_url = sys.argv[4]
expected_smoke_query = sys.argv[5]


def load_object(path):
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise SystemExit("{} must contain a JSON object".format(path))
    return value


def require_exact_keys(value, expected, name):
    actual = set(value)
    if actual != set(expected):
        missing = sorted(set(expected) - actual)
        extra = sorted(actual - set(expected))
        raise SystemExit(
            "{} keys mismatch (missing={}, extra={})".format(
                name,
                missing,
                extra,
            )
        )


def require_nonnegative_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SystemExit("{} must be a nonnegative integer".format(name))


def require_finite_reward(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise SystemExit("{} must be a finite reward in [0, 1]".format(name))


def redacted_url(url):
    parsed = urlsplit(url)
    hostname = parsed.hostname or ""
    if ":" in hostname and not hostname.startswith("["):
        hostname = "[{}]".format(hostname)
    if parsed.port is not None:
        hostname = "{}:{}".format(hostname, parsed.port)
    return urlunsplit((parsed.scheme, hostname, parsed.path, "", ""))


manifest = load_object(run_root / "run_manifest.json")
effective = load_object(run_root / "effective_config.json")
metadata = load_object(run_root / "run_metadata.json")
task = load_object(run_root / "tasks" / "fixed_1.json")
with (run_root / "results.jsonl").open("r", encoding="utf-8") as handle:
    raw_result_lines = [line for line in handle if line.strip()]
result_lines = [json.loads(line) for line in raw_result_lines]

require_exact_keys(
    manifest,
    {"effective_config", "run_metadata"},
    "run manifest",
)
require_exact_keys(
    effective,
    {"args", "planu_config"},
    "effective config",
)
if not isinstance(effective["args"], dict):
    raise SystemExit("effective config args must be an object")
if not isinstance(effective["planu_config"], dict):
    raise SystemExit("effective PlanU config must be an object")

if manifest.get("effective_config") != effective:
    raise SystemExit("effective config does not match the run manifest")
if manifest.get("run_metadata") != metadata:
    raise SystemExit("run metadata does not match the run manifest")
require_exact_keys(
    metadata,
    {
        "config_hash",
        "git_commit",
        "model_id",
        "packages",
        "planu_git_commit",
        "python_version",
        "seed",
        "server_url",
        "task_bounds",
        "webshop_commit",
    },
    "run metadata",
)
if metadata.get("webshop_commit") != expected_commit:
    raise SystemExit("run metadata does not contain the pinned WebShop commit")
if (
    not metadata.get("planu_git_commit")
    or metadata.get("planu_git_commit") != expected_planu_commit
    or metadata.get("git_commit") != expected_planu_commit
):
    raise SystemExit("run metadata PlanU commit does not match checkout HEAD")
if metadata.get("model_id") != "scripted":
    raise SystemExit("smoke run did not use the scripted model policy")
if metadata.get("server_url") != redacted_url(configured_server_url):
    raise SystemExit("run metadata server URL does not match configured URL")
if metadata.get("seed") != 0:
    raise SystemExit("run metadata seed does not match the smoke run")
if metadata.get("task_bounds") != {"start": 1, "end_exclusive": 2}:
    raise SystemExit("run metadata task bounds do not describe fixed_1")
if not isinstance(metadata.get("python_version"), str) or not metadata["python_version"]:
    raise SystemExit("run metadata Python version is missing")
packages = metadata.get("packages")
if not isinstance(packages, dict) or not packages:
    raise SystemExit("run metadata packages must be a nonempty object")
required_packages = {
    "beautifulsoup4",
    "ding",
    "gym",
    "numpy",
    "openai",
    "peft",
    "requests",
    "torch",
    "transformers",
}
if not required_packages.issubset(packages):
    raise SystemExit("run metadata packages are incomplete")
if any(not isinstance(value, str) or not value for value in packages.values()):
    raise SystemExit("run metadata package versions must be nonempty strings")
serialized_config = json.dumps(
    effective,
    sort_keys=True,
    separators=(",", ":"),
)
expected_hash = hashlib.sha256(
    serialized_config.encode("utf-8")
).hexdigest()[:12]
if metadata.get("config_hash") != expected_hash:
    raise SystemExit("run metadata config hash is invalid")

args = effective["args"]
require_exact_keys(
    args,
    {
        "backend",
        "base_url",
        "depth",
        "iterations",
        "n_evaluate_sample",
        "n_generate_sample",
        "output_dir",
        "prompt_mode",
        "request_timeout",
        "seed",
        "smoke",
        "smoke_search_query",
        "task_end_index",
        "task_start_index",
        "temperature",
        "webshop_url",
    },
    "effective config args",
)
expected_args = {
    "backend": "qwen-plus",
    "depth": 10,
    "iterations": 1,
    "n_evaluate_sample": 1,
    "n_generate_sample": 5,
    "output_dir": str(run_root),
    "prompt_mode": "cot",
    "request_timeout": 30.0,
    "seed": 0,
    "smoke": True,
    "smoke_search_query": expected_smoke_query,
    "task_start_index": 1,
    "task_end_index": 2,
    "temperature": 0.8,
    "webshop_url": redacted_url(configured_server_url),
}
if any(args.get(key) != value for key, value in expected_args.items()):
    raise SystemExit("effective config does not describe the real smoke task")
if (
    not isinstance(args.get("base_url"), str)
    or not args["base_url"]
    or not isinstance(args.get("request_timeout"), (int, float))
    or args["request_timeout"] <= 0
    or not isinstance(args.get("smoke_search_query"), str)
    or not args["smoke_search_query"].strip()
):
    raise SystemExit("effective config contains invalid smoke arguments")

expected_planu_config = {
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
if effective["planu_config"] != expected_planu_config:
    raise SystemExit("effective PlanU config does not match smoke profile")

require_exact_keys(
    task,
    {
        "best_terminal_reward",
        "error_status",
        "latency_failure_count",
        "max_depth",
        "provenance",
        "search_iterations",
        "success",
        "task_id",
        "task_index",
        "token_usage",
        "trajectory",
    },
    "fixed_1 task",
)
if task.get("task_id") != "fixed_1" or task.get("task_index") != 1:
    raise SystemExit("task artifact does not contain fixed_1")
if result_lines != [task]:
    raise SystemExit("consolidated results do not match the fixed_1 task artifact")
if len(raw_result_lines) != 1:
    raise SystemExit("results must contain exactly one fixed_1 JSON line")
if task.get("provenance") != metadata:
    raise SystemExit("task provenance does not match run metadata")
require_finite_reward(
    task.get("best_terminal_reward"),
    "best_terminal_reward",
)
if task.get("success") is not (task["best_terminal_reward"] == 1.0):
    raise SystemExit("task success does not match best terminal reward")
if task.get("error_status") is not None:
    raise SystemExit("smoke task contains an error status")
if task.get("max_depth") != 10:
    raise SystemExit("smoke task max depth is invalid")
require_nonnegative_int(
    task.get("latency_failure_count"),
    "latency_failure_count",
)
backup_count = task.get("search_iterations")
if (
    isinstance(backup_count, bool)
    or not isinstance(backup_count, int)
    or backup_count < 1
    or backup_count > args["iterations"]
):
    raise SystemExit("search_iterations must prove at least one backup")

token_usage = task.get("token_usage")
if not isinstance(token_usage, dict):
    raise SystemExit("token_usage must be an object")
require_exact_keys(
    token_usage,
    {"completion_tokens", "prompt_tokens", "total_tokens"},
    "token usage",
)
for key in token_usage:
    require_nonnegative_int(token_usage[key], "token_usage.{}".format(key))
if token_usage["total_tokens"] != (
    token_usage["prompt_tokens"] + token_usage["completion_tokens"]
):
    raise SystemExit("token usage total is inconsistent")

trajectory = task.get("trajectory")
if not isinstance(trajectory, list) or not trajectory:
    raise SystemExit("smoke task must contain a real HTTP trajectory")
for index, step in enumerate(trajectory):
    if not isinstance(step, dict):
        raise SystemExit("trajectory step {} must be an object".format(index))
    require_exact_keys(
        step,
        {"action", "observation", "reward"},
        "trajectory step {}".format(index),
    )
    if not isinstance(step["action"], str) or not step["action"].strip():
        raise SystemExit("trajectory action {} is empty".format(index))
    if (
        not isinstance(step["observation"], str)
        or not step["observation"].strip()
    ):
        raise SystemExit("trajectory observation {} is empty".format(index))
    require_finite_reward(
        step["reward"],
        "trajectory reward {}".format(index),
    )
actions = [step["action"] for step in trajectory]
if not actions[0].startswith("search[") or actions[-1] != "click[Buy Now]":
    raise SystemExit("scripted smoke trajectory did not reach purchase")
if trajectory[-1]["reward"] != task["best_terminal_reward"]:
    raise SystemExit("terminal reward does not match best_terminal_reward")

print(
    json.dumps(
        {
            "http_transition_count": len(trajectory),
            "model_id": metadata["model_id"],
            "quantile_backup_count": backup_count,
            "task_id": task["task_id"],
        },
        sort_keys=True,
    )
)
PY

printf 'WebShop smoke outputs: %s\n' "${RUN_ROOT}"
