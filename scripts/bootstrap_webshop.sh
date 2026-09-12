#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SOURCE_URL="https://github.com/princeton-nlp/WebShop.git"
COMMIT="64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd"
ROOT="${WEBSHOP_ROOT:-${PROJECT_ROOT}/external/WebShop}"
ENV_PREFIX="${WEBSHOP_ENV_PREFIX:-${ROOT}/.conda-planu}"
SETUP_MARKER="${ENV_PREFIX}/.planu-webshop-small-${COMMIT}"

fail() {
  printf 'WebShop bootstrap error: %s\n' "$*" >&2
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

verify_clean_checkout() {
  local status
  status="$(
    git -C "${ROOT}" status --porcelain --untracked-files=normal \
      -- . ':(exclude).conda-planu'
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
  [[ -f "${ROOT}/data/items_shuffle_1000.json" ]] || return 1
  [[ -f "${ROOT}/data/items_ins_v2_1000.json" ]] || return 1
  [[ -d "${ROOT}/search_engine/indexes" ]] || return 1
  (
    cd "${ROOT}"
    "${ENV_PREFIX}/bin/python" -c "import web_agent_site.app"
  ) >/dev/null 2>&1
}

require_command git
require_command conda
CONDA_EXE="$(command -v conda)"

if [[ ! -e "${ROOT}" ]]; then
  mkdir -p "$(dirname "${ROOT}")"
  if ! git clone "${SOURCE_URL}" "${ROOT}"; then
    fail "clone failed; check network access to ${SOURCE_URL}"
  fi
elif [[ "$(git -C "${ROOT}" rev-parse --is-inside-work-tree 2>/dev/null || true)" != "true" ]]; then
  fail "WEBSHOP_ROOT exists but is not a Git checkout: ${ROOT}"
fi

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

if [[ -f "${SETUP_MARKER}" ]]; then
  verify_small_setup ||
    fail "setup marker exists but the pinned small WebShop setup is incomplete"
elif verify_small_setup; then
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
    fail "upstream small setup finished without the required data, index, or server module"
  touch "${SETUP_MARKER}"
fi

verify_clean_checkout
printf 'WebShop ready at %s (commit %s, Python 3.8.13)\n' \
  "${ROOT}" "${COMMIT}"
