#!/usr/bin/env bash
set -euo pipefail

SOURCE_URL="https://github.com/princeton-nlp/WebShop.git"
COMMIT="64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd"
START_ATTEMPTS="${WEBSHOP_BOOTSTRAP_START_ATTEMPTS:-120}"
STOP_ATTEMPTS="${WEBSHOP_STOP_ATTEMPTS:-20}"
SERVER_PID=""
SERVER_PGID=""

fail() {
  printf 'WebShop bootstrap error: %s\n' "$*" >&2
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

verify_clean_checkout() {
  local -a pathspecs
  local env_relative
  local status

  pathspecs=(.)
  if [[ "${ENV_PREFIX}" == "${ROOT}/"* ]]; then
    env_relative="${ENV_PREFIX#"${ROOT}/"}"
    pathspecs+=(":(exclude)${env_relative}")
  fi
  status="$(
    git -C "${ROOT}" status --porcelain --untracked-files=normal \
      -- "${pathspecs[@]}"
  )" ||
    fail "could not inspect the WebShop working tree"
  [[ -z "${status}" ]] ||
    fail "WebShop has uncommitted or untracked changes; refusing to modify it"
}

verify_environment() {
  local version
  [[ -x "${ENV_PREFIX}/bin/python" ]] || return 1
  version="$("${ENV_PREFIX}/bin/python" -c \
    'import platform; print(platform.python_version())')" || return 1
  [[ "${version}" == "3.8.13" ]]
}

verify_small_setup() {
  local index_dir="${ROOT}/search_engine/indexes"
  local -a segment_files
  local -a segment_info_files

  # These are the data, resource, and runtime index paths used by the pin.
  [[ -s "${ROOT}/data/items_shuffle_1000.json" ]] || return 1
  [[ -s "${ROOT}/data/items_ins_v2_1000.json" ]] || return 1
  [[ -s "${ROOT}/data/items_human_ins.json" ]] || return 1
  [[ -s "${ROOT}/search_engine/resources/documents.jsonl" ]] || return 1
  [[ -d "${index_dir}" ]] || return 1
  shopt -s nullglob
  segment_files=("${index_dir}"/segments_*)
  segment_info_files=("${index_dir}"/_*.si)
  shopt -u nullglob
  ((${#segment_files[@]} > 0)) || return 1
  ((${#segment_info_files[@]} > 0)) || return 1
  [[ -s "${segment_files[0]}" ]] || return 1
  [[ -s "${segment_info_files[0]}" ]] || return 1
  (
    cd "${ROOT}"
    "${ENV_PREFIX}/bin/python" -c "import web_agent_site.app"
  ) >/dev/null 2>&1
}

server_reachable() {
  curl --fail --silent \
    --connect-timeout 2 --max-time 5 \
    "http://127.0.0.1:3000/fixed_1" >/dev/null
}

configure_java_runtime() {
  local java_version_output
  local java_version_line

  JAVA_HOME="${ENV_PREFIX}/lib/jvm"
  [[ -x "${JAVA_HOME}/bin/java" ]] ||
    fail "Conda Java runtime is missing: ${JAVA_HOME}/bin/java"
  if ! java_version_output="$("${JAVA_HOME}/bin/java" -version 2>&1)"; then
    java_version_line="${java_version_output%%$'\n'*}"
    fail "could not run ${JAVA_HOME}/bin/java -version: ${java_version_line:-no output}"
  fi
  java_version_line="${java_version_output%%$'\n'*}"
  case "${java_version_line}" in
    'openjdk version "11.'* | 'java version "11.'*) ;;
    *)
      fail "Java 11 is required at ${JAVA_HOME}/bin/java; found: ${java_version_line:-no version output}"
      ;;
  esac
  export JAVA_HOME
  export PATH="${JAVA_HOME}/bin:${PATH}"
}

verify_server_startup() {
  local attempt=0
  local log_path="${ENV_PREFIX}/bootstrap-server.log"

  if server_reachable; then
    fail "port 3000 is already serving a process not started by this bootstrap"
  fi
  python3 - "${ROOT}" "${ENV_PREFIX}/bin/python" \
    >"${log_path}" 2>&1 <<'PY' &
import os
import sys

os.chdir(sys.argv[1])
os.setsid()
os.execv(sys.argv[2], [sys.argv[2], "-m", "web_agent_site.app"])
PY
  SERVER_PID=$!
  SERVER_PGID="${SERVER_PID}"

  until server_reachable; do
    attempt=$((attempt + 1))
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      reap_server
      fail "pinned WebShop server exited during validation; see ${log_path}"
    fi
    if ((attempt >= START_ATTEMPTS)); then
      fail "pinned WebShop server did not become ready after ${START_ATTEMPTS} attempts"
    fi
    sleep 1
  done
  kill -0 "${SERVER_PID}" 2>/dev/null ||
    fail "pinned WebShop server exited after its health check"
  cleanup
}

require_command python3
require_command git
require_command conda
require_command curl
CONDA_EXE="$(command -v conda)"
SCRIPT_PATH="$(canonical_path "${BASH_SOURCE[0]}")"
SCRIPT_DIR="${SCRIPT_PATH%/*}"
PROJECT_ROOT="${SCRIPT_DIR%/*}"
ROOT="$(canonical_path "${WEBSHOP_ROOT:-${PROJECT_ROOT}/external/WebShop}")"
ENV_PREFIX="$(canonical_path "${WEBSHOP_ENV_PREFIX:-${ROOT}/.conda-planu}")"
SETUP_MARKER="${ENV_PREFIX}/.planu-webshop-small-${COMMIT}"

case "${START_ATTEMPTS}" in
  "" | *[!0-9]*) fail "WEBSHOP_BOOTSTRAP_START_ATTEMPTS must be a positive integer" ;;
esac
((START_ATTEMPTS > 0)) ||
  fail "WEBSHOP_BOOTSTRAP_START_ATTEMPTS must be a positive integer"
case "${STOP_ATTEMPTS}" in
  "" | *[!0-9]*) fail "WEBSHOP_STOP_ATTEMPTS must be a positive integer" ;;
esac
((STOP_ATTEMPTS > 0)) ||
  fail "WEBSHOP_STOP_ATTEMPTS must be a positive integer"
[[ "${ENV_PREFIX}" != "${ROOT}" ]] ||
  fail "WEBSHOP_ENV_PREFIX must not equal WEBSHOP_ROOT"

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ ! -e "${ROOT}" ]]; then
  mkdir -p "$(dirname "${ROOT}")"
  if ! git clone "${SOURCE_URL}" "${ROOT}"; then
    fail "clone failed; check network access to ${SOURCE_URL}"
  fi
elif [[ "$(git -C "${ROOT}" rev-parse --is-inside-work-tree 2>/dev/null || true)" != "true" ]]; then
  fail "WEBSHOP_ROOT exists but is not a Git checkout: ${ROOT}"
fi

checkout_root="$(git -C "${ROOT}" rev-parse --show-toplevel 2>/dev/null || true)"
[[ -n "${checkout_root}" ]] ||
  fail "could not resolve the WebShop checkout top-level"
checkout_root="$(canonical_path "${checkout_root}")"
[[ "${ROOT}" == "${checkout_root}" ]] ||
  fail "WEBSHOP_ROOT must equal the Git checkout top-level: ${checkout_root}"

remote_url="$(git -C "${ROOT}" remote get-url origin 2>/dev/null || true)"
is_official_remote "${remote_url}" ||
  fail "existing origin remote is not the official WebShop repository: ${remote_url:-missing}"

verify_clean_checkout

if ! git -C "${ROOT}" cat-file -e "${COMMIT}^{commit}" 2>/dev/null; then
  if ! git -C "${ROOT}" fetch --no-tags origin "${COMMIT}"; then
    fail "fetch failed; check network access to ${SOURCE_URL}"
  fi
fi

current_commit="$(git -C "${ROOT}" rev-parse HEAD)" ||
  fail "could not resolve the current WebShop commit"
if [[ "${current_commit}" != "${COMMIT}" ]] ||
  git -C "${ROOT}" symbolic-ref -q HEAD >/dev/null 2>&1; then
  if ! git -C "${ROOT}" checkout --detach "${COMMIT}"; then
    fail "could not check out pinned WebShop commit ${COMMIT}"
  fi
fi

actual_commit="$(git -C "${ROOT}" rev-parse HEAD)" ||
  fail "could not verify the pinned WebShop commit"
[[ "${actual_commit}" == "${COMMIT}" ]] ||
  fail "expected WebShop commit ${COMMIT}, found ${actual_commit}"
verify_clean_checkout

if [[ -e "${ENV_PREFIX}" && ! -x "${ENV_PREFIX}/bin/python" ]]; then
  fail "Conda prefix exists but is incomplete: ${ENV_PREFIX}"
fi
if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
  if ! "${CONDA_EXE}" create -y -p "${ENV_PREFIX}" python=3.8.13; then
    fail "Conda environment creation failed; check package channels and network access"
  fi
fi
verify_environment ||
  fail "Conda prefix must contain exactly Python 3.8.13: ${ENV_PREFIX}"
if [[ ! -x "${ENV_PREFIX}/lib/jvm/bin/java" ]]; then
  if ! "${CONDA_EXE}" install -y -p "${ENV_PREFIX}" \
    -c conda-forge openjdk=11; then
    fail "OpenJDK 11 installation failed; check conda-forge access and network connectivity"
  fi
fi
configure_java_runtime

if [[ -f "${SETUP_MARKER}" ]]; then
  verify_small_setup ||
    fail "setup marker exists but required data or Lucene index artifacts are incomplete"
  verify_server_startup
elif verify_small_setup; then
  verify_server_startup
  touch "${SETUP_MARKER}"
else
  (
    cd "${ROOT}"
    export CONDA_ALWAYS_YES=true
    export PIP_DISABLE_PIP_VERSION_CHECK=1
    "${CONDA_EXE}" run --no-capture-output -p "${ENV_PREFIX}" \
      bash ./setup.sh -d small
  ) || fail "upstream small setup failed; check network access and its log output"
  verify_small_setup ||
    fail "upstream small setup finished without required data, Lucene index artifacts, or server module"
  verify_server_startup
  touch "${SETUP_MARKER}"
fi

verify_clean_checkout
printf 'WebShop ready at %s (commit %s, Python 3.8.13)\n' \
  "${ROOT}" "${COMMIT}"
