#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SOURCE_URL="https://github.com/princeton-nlp/WebShop.git"
COMMIT="64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd"
WEBSHOP_ROOT="${WEBSHOP_ROOT:-${PROJECT_ROOT}/external/WebShop}"
WEBSHOP_URL="${WEBSHOP_URL:-http://127.0.0.1:3000}"
WEBSHOP_URL="${WEBSHOP_URL%/}"
DEFAULT_HEALTH_URL="http://127.0.0.1:3000/fixed_1"
HEALTH_URL="${WEBSHOP_URL}/fixed_1"
PLANU_PYTHON="${PLANU_PYTHON:-python3}"
WEBSHOP_PYTHON="${WEBSHOP_PYTHON:-${WEBSHOP_ROOT}/.conda-planu/bin/python}"
SMOKE_QUERY="${WEBSHOP_SMOKE_QUERY:-product}"
START_ATTEMPTS="${WEBSHOP_START_ATTEMPTS:-120}"
RUN_ROOT="${RUN_ROOT:-$(mktemp -d "${TMPDIR:-/tmp}/planu-webshop-smoke.XXXXXX")}"
SERVER_PID=""

fail() {
  printf 'WebShop smoke error: %s\n' "$*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 ||
    fail "required command '$1' is missing"
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

cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}

server_reachable() {
  curl --fail --silent \
    --connect-timeout 2 --max-time 5 "${HEALTH_URL}" >/dev/null
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

require_command git
require_command curl
require_command "${PLANU_PYTHON}"

[[ "$(git -C "${WEBSHOP_ROOT}" rev-parse --is-inside-work-tree 2>/dev/null || true)" == "true" ]] ||
  fail "WEBSHOP_ROOT is not a bootstrapped Git working tree: ${WEBSHOP_ROOT}"

remote_url="$(git -C "${WEBSHOP_ROOT}" remote get-url origin 2>/dev/null || true)"
is_official_remote "${remote_url}" ||
  fail "origin remote must match ${SOURCE_URL}; found ${remote_url:-missing}"

actual_commit="$(git -C "${WEBSHOP_ROOT}" rev-parse HEAD 2>/dev/null || true)"
[[ "${actual_commit}" == "${COMMIT}" ]] ||
  fail "WEBSHOP_ROOT must be pinned at ${COMMIT}; found ${actual_commit:-unknown}"

working_tree_status="$(
  git -C "${WEBSHOP_ROOT}" status --porcelain --untracked-files=normal \
    -- . ':(exclude).conda-planu'
)" || fail "could not inspect the WebShop working tree"
[[ -z "${working_tree_status}" ]] ||
  fail "WEBSHOP_ROOT must be clean before a real smoke run"

case "${START_ATTEMPTS}" in
  "" | *[!0-9]*) fail "WEBSHOP_START_ATTEMPTS must be a positive integer" ;;
esac
((START_ATTEMPTS > 0)) ||
  fail "WEBSHOP_START_ATTEMPTS must be a positive integer"

mkdir -p "${RUN_ROOT}"

if ! server_reachable; then
  [[ "${HEALTH_URL}" == "${DEFAULT_HEALTH_URL}" ]] ||
    fail "configured WEBSHOP_URL is not reachable: ${HEALTH_URL}"
  [[ -x "${WEBSHOP_PYTHON}" ]] ||
    fail "pinned WebShop Python is missing; run scripts/bootstrap_webshop.sh"

  (
    cd "${WEBSHOP_ROOT}"
    exec "${WEBSHOP_PYTHON}" -m web_agent_site.app --log --attrs
  ) >"${RUN_ROOT}/webshop-server.log" 2>&1 &
  SERVER_PID=$!

  attempt=0
  until server_reachable; do
    attempt=$((attempt + 1))
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      wait "${SERVER_PID}" 2>/dev/null || true
      fail "official WebShop server exited during startup; see ${RUN_ROOT}/webshop-server.log"
    fi
    if ((attempt >= START_ATTEMPTS)); then
      fail "official WebShop server did not become ready after ${START_ATTEMPTS} attempts"
    fi
    sleep 1
  done
fi

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

"${PLANU_PYTHON}" - "${RUN_ROOT}" "${COMMIT}" <<'PY'
import json
from pathlib import Path
import sys

run_root = Path(sys.argv[1])
expected_commit = sys.argv[2]


def load_object(path):
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise SystemExit("{} must contain a JSON object".format(path))
    return value


manifest = load_object(run_root / "run_manifest.json")
effective = load_object(run_root / "effective_config.json")
metadata = load_object(run_root / "run_metadata.json")
task = load_object(run_root / "tasks" / "fixed_1.json")
with (run_root / "results.jsonl").open("r", encoding="utf-8") as handle:
    result_lines = [json.loads(line) for line in handle if line.strip()]

if manifest.get("effective_config") != effective:
    raise SystemExit("effective config does not match the run manifest")
if manifest.get("run_metadata") != metadata:
    raise SystemExit("run metadata does not match the run manifest")
if metadata.get("webshop_commit") != expected_commit:
    raise SystemExit("run metadata does not contain the pinned WebShop commit")
if metadata.get("model_id") != "scripted":
    raise SystemExit("smoke run did not use the scripted model policy")

args = effective.get("args", {})
expected_args = {
    "smoke": True,
    "task_start_index": 1,
    "task_end_index": 2,
    "iterations": 1,
    "depth": 10,
}
if any(args.get(key) != value for key, value in expected_args.items()):
    raise SystemExit("effective config does not describe the real smoke task")
if task.get("task_id") != "fixed_1" or task.get("task_index") != 1:
    raise SystemExit("task artifact does not contain fixed_1")
if result_lines != [task]:
    raise SystemExit("consolidated results do not match the fixed_1 task artifact")
if task.get("provenance", {}).get("model_id") != "scripted":
    raise SystemExit("task provenance does not use the scripted model policy")

trajectory = task.get("trajectory")
if not isinstance(trajectory, list) or len(trajectory) < 3:
    raise SystemExit("smoke task did not persist real HTTP transitions")
actions = [step.get("action", "") for step in trajectory]
if not actions[0].startswith("search[") or "click[Buy Now]" not in actions:
    raise SystemExit("scripted smoke trajectory did not reach purchase")
backup_count = task.get("search_iterations")
if backup_count != 1:
    raise SystemExit("smoke task does not prove exactly one quantile backup")

serialized = json.dumps(
    {
        "manifest": manifest,
        "task": task,
        "results": result_lines,
    },
    sort_keys=True,
).casefold()
if ("mo" + "ck") in serialized:
    raise SystemExit("smoke artifacts contain a substituted test backend")

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
