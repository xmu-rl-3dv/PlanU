#!/usr/bin/env bash
set -euo pipefail

SOURCE_URL="https://github.com/princeton-nlp/WebShop.git"
COMMIT="64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd"
WERKZEUG_REQUIREMENT="Werkzeug==2.1.2"
SPACY_MODEL_REQUIREMENT="en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.3.0/en_core_web_sm-3.3.0-py3-none-any.whl#sha256=84d7d8059bfbf53c09b39139782f76cd6ac7064851e7799dcc685c06ebf5fd4f"
ITEMS_SHUFFLE_SHA256="30a4765c3a327af72d9a9a95a6b2486d516f0fa1d3ecd83681901ce82a21b269"
ITEMS_INS_SHA256="f88a36314a397b53b3d9c3fa5878e5f7b26d35019a51ec83fbedeca61a948f6f"
ITEMS_HUMAN_SHA256="cf78667548a71786e1d9049c24b802e48e1084ad4bb021cae56ce1f6d96954a3"
START_ATTEMPTS="${WEBSHOP_BOOTSTRAP_START_ATTEMPTS:-120}"
STOP_ATTEMPTS="${WEBSHOP_STOP_ATTEMPTS:-20}"
SERVER_PID=""
SERVER_PGID=""
# importlib.metadata exposes no extra Conda-only distributions in the
# validated 138-distribution prefix.
SERVER_CONDA_METADATA_EXEMPTIONS=""

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

verify_python_environment() {
  local version
  [[ -x "${ENV_PREFIX}/bin/python" ]] || return 1
  version="$("${ENV_PREFIX}/bin/python" -c \
    'import platform; print(platform.python_version())')" || return 1
  [[ "${version}" == "3.8.13" ]]
}

verify_werkzeug() {
  local version
  version="$("${ENV_PREFIX}/bin/python" -c \
    'from importlib.metadata import version; print(version("Werkzeug"))'
  )" || return 1
  [[ "${version}" == "2.1.2" ]]
}

verify_flask() {
  local version
  version="$("${ENV_PREFIX}/bin/python" -c \
    'from importlib.metadata import version; print(version("Flask"))'
  )" || return 1
  [[ "${version}" == "2.1.2" ]]
}

verify_locked_environment() {
  [[ -z "${SERVER_CONDA_METADATA_EXEMPTIONS}" ]] ||
    fail "unexpected server distribution exemptions are configured"
  "${ENV_PREFIX}/bin/python" "${LOCK_CHECKER}" "${SERVER_LOCK}"
}

verify_environment() {
  verify_python_environment &&
    verify_flask &&
    verify_werkzeug &&
    verify_locked_environment
}

verify_data_file() {
  local path="$1"
  local expected="$2"
  local actual

  [[ -s "${path}" ]] || {
    printf 'WebShop bootstrap error: required data file is missing or empty: %s\n' \
      "${path}" >&2
    return 1
  }
  actual="$(
    python3 - "${path}" <<'PY'
import hashlib
import sys

digest = hashlib.sha256()
with open(sys.argv[1], "rb") as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(block)
print(digest.hexdigest())
PY
  )" || return 1
  [[ "${actual}" == "${expected}" ]] || {
    printf 'WebShop bootstrap error: SHA-256 mismatch for %s: expected %s, found %s\n' \
      "${path}" "${expected}" "${actual:-unavailable}" >&2
    return 1
  }
}

verify_small_setup() {
  local index_dir="${ROOT}/search_engine/indexes"
  local -a segment_files
  local -a segment_info_files

  verify_environment || return 1
  verify_data_file \
    "${ROOT}/data/items_shuffle_1000.json" "${ITEMS_SHUFFLE_SHA256}" ||
    return 1
  verify_data_file \
    "${ROOT}/data/items_ins_v2_1000.json" "${ITEMS_INS_SHA256}" ||
    return 1
  verify_data_file \
    "${ROOT}/data/items_human_ins.json" "${ITEMS_HUMAN_SHA256}" ||
    return 1
  # Verify source data before accepting its derived resource and index files.
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

marker_matches_lock() {
  [[ -f "${SETUP_MARKER}" ]] || return 1
  [[ "$(tr -d '\r\n' <"${SETUP_MARKER}")" == "${LOCK_SHA256}" ]]
}

write_setup_marker() {
  printf '%s\n' "${LOCK_SHA256}" >"${SETUP_MARKER}"
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
LOCK_CHECKER="${PROJECT_ROOT}/planu_core/distribution_lock.py"
DIRECT_REQUIREMENTS="${PROJECT_ROOT}/requirements-webshop-server.txt"
SERVER_LOCK="${PROJECT_ROOT}/requirements-webshop-server-lock.txt"
PIP_CONSTRAINT="${SERVER_LOCK}"
SETUP_MARKER="${ENV_PREFIX}/.planu-webshop-small-${COMMIT}"
[[ -f "${DIRECT_REQUIREMENTS}" ]] ||
  fail "server direct requirements file is missing: ${DIRECT_REQUIREMENTS}"
[[ -f "${SERVER_LOCK}" ]] ||
  fail "server lock file is missing: ${SERVER_LOCK}"
[[ -f "${LOCK_CHECKER}" ]] ||
  fail "distribution lock checker is missing: ${LOCK_CHECKER}"
LOCK_SHA256="$(
  python3 - "${SERVER_LOCK}" <<'PY'
import hashlib
import sys

with open(sys.argv[1], "rb") as handle:
    print(hashlib.sha256(handle.read()).hexdigest())
PY
)" || fail "could not hash server lock: ${SERVER_LOCK}"
export PIP_CONSTRAINT

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
verify_python_environment ||
  fail "Conda prefix must contain exactly Python 3.8.13: ${ENV_PREFIX}"
if [[ ! -x "${ENV_PREFIX}/lib/jvm/bin/java" ]]; then
  if ! "${CONDA_EXE}" install -y -p "${ENV_PREFIX}" \
    -c conda-forge openjdk=11; then
    fail "OpenJDK 11 installation failed; check conda-forge access and network connectivity"
  fi
fi
configure_java_runtime
if ! verify_werkzeug; then
  if ! "${ENV_PREFIX}/bin/python" -m pip install \
    "${WERKZEUG_REQUIREMENT}"; then
    fail "${WERKZEUG_REQUIREMENT} installation failed; check network access"
  fi
fi
verify_python_environment && verify_werkzeug ||
  fail "Conda prefix must contain Python 3.8.13 and ${WERKZEUG_REQUIREMENT}: ${ENV_PREFIX}"

if marker_matches_lock; then
  verify_small_setup ||
    fail "setup marker exists but required data or Lucene index artifacts are incomplete"
  verify_server_startup
elif verify_small_setup; then
  verify_server_startup
  write_setup_marker
else
  if ! "${ENV_PREFIX}/bin/python" -m pip install \
    "pip==24.3.1" "packaging==26.2" "setuptools==68.2.2" \
    "wheel==0.45.1" -c "${SERVER_LOCK}"; then
    fail "locked server build-tool installation failed"
  fi
  if ! "${ENV_PREFIX}/bin/python" -m pip install \
    -r "${DIRECT_REQUIREMENTS}" -c "${SERVER_LOCK}"; then
    fail "locked direct server dependency installation failed"
  fi
  if ! "${ENV_PREFIX}/bin/python" -m pip install \
    "${SPACY_MODEL_REQUIREMENT}" -c "${SERVER_LOCK}"; then
    fail "locked spaCy server model installation failed"
  fi
  (
    cd "${ROOT}"
    export CONDA_ALWAYS_YES=true
    export PIP_DISABLE_PIP_VERSION_CHECK=1
    "${CONDA_EXE}" run --no-capture-output -p "${ENV_PREFIX}" \
      bash ./setup.sh -d small
  ) || fail "upstream small setup failed; check network access and its log output"
  verify_small_setup ||
    fail "upstream small setup finished with a server lock mismatch or without required data, Lucene index artifacts, or server module"
  verify_server_startup
  write_setup_marker
fi

verify_clean_checkout
printf 'WebShop ready at %s (commit %s, Python 3.8.13, Flask 2.1.2, Werkzeug 2.1.2)\n' \
  "${ROOT}" "${COMMIT}"
