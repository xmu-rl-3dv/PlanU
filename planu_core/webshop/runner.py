"""Command-line runner for WebShop experiments through the shared PlanU core."""

import argparse
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
import fcntl
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Callable, Dict, Mapping, Optional, Sequence
from urllib.parse import urlsplit, urlunsplit

import numpy as np

from planu_core.config import PlanUConfig
from planu_core.provenance import (
    PROVENANCE_DISTRIBUTIONS,
    atomic_write_json,
    atomic_write_text,
    build_effective_config,
    build_run_metadata,
    config_hash,
)
from planu_core.search import PlanUSearch
from planu_core.text_backend import (
    DEFAULT_OPENAI_BASE_URL,
    OpenAICompatibleBackend,
)
from planu_core.adapters.webshop import WebShopAdapter
from planu_core.webshop.client import WebShopHttpClient
from planu_core.webshop.providers import (
    ModelWebShopActionProvider,
    ModelWebShopActionScorer,
    ScriptedWebShopActionProvider,
)


WEBSHOP_COMMIT = "64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd"
DEFAULT_WEBSHOP_URL = "http://127.0.0.1:3000"
WEBSHOP_DISTRIBUTIONS = {
    **PROVENANCE_DISTRIBUTIONS,
    "requests": "requests",
    "beautifulsoup4": "beautifulsoup4",
    "openai": "openai",
}
RUN_MANIFEST = "run_manifest.json"
EFFECTIVE_CONFIG_SIDECAR = "effective_config.json"
RUN_METADATA_SIDECAR = "run_metadata.json"


class WebShopRunnerError(RuntimeError):
    """Raised when runner infrastructure cannot complete an experiment."""


class _SanitizedWebShopRunnerError(WebShopRunnerError):
    """Internally generated runner error whose message is safe to expose."""


def _safe_failure_summary(error: Exception, context: str) -> str:
    return "WebShop infrastructure failure for {} ({})".format(
        context,
        type(error).__name__,
    )


def _find_sanitized_error(
    error: BaseException,
) -> Optional[_SanitizedWebShopRunnerError]:
    pending = [error]
    seen = set()
    while pending:
        current = pending.pop()
        identity = id(current)
        if identity in seen:
            continue
        seen.add(identity)
        if type(current) is _SanitizedWebShopRunnerError:
            return current
        for linked in (current.__cause__, current.__context__):
            if linked is not None:
                pending.append(linked)
    return None


class ScriptedWebShopActionScorer:
    """Credential-free deterministic scorer used by real-HTTP smoke runs."""

    token_usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }

    def score(self, observation, candidates):
        del observation
        return [0.5 for _ in candidates]


@dataclass(frozen=True)
class RunnerDependencies:
    """Injectable runner construction points for tests and integrations."""

    client_factory: Callable[..., Any] = WebShopHttpClient
    backend_factory: Callable[..., Any] = OpenAICompatibleBackend
    adapter_factory: Callable[..., Any] = WebShopAdapter
    search_factory: Callable[..., Any] = PlanUSearch
    model_provider_factory: Callable[..., Any] = ModelWebShopActionProvider
    model_scorer_factory: Callable[..., Any] = ModelWebShopActionScorer
    scripted_provider_factory: Callable[..., Any] = (
        ScriptedWebShopActionProvider
    )
    scripted_scorer_factory: Callable[..., Any] = (
        ScriptedWebShopActionScorer
    )
    metadata_factory: Callable[..., Mapping[str, Any]] = build_run_metadata


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return parsed


def _model_base_url() -> str:
    return (
        os.environ.get("OPENAI_BASE_URL", "").strip()
        or os.environ.get("OPENAI_API_BASE", "").strip()
        or DEFAULT_OPENAI_BASE_URL
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run WebShop tasks through the shared PlanU core.",
    )
    parser.add_argument("--backend", default="qwen-plus")
    parser.add_argument(
        "--base-url",
        "--model-base-url",
        dest="base_url",
        default=_model_base_url(),
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument(
        "--prompt-mode",
        "--prompt_sample",
        dest="prompt_mode",
        choices=("cot",),
        default="cot",
    )
    parser.add_argument(
        "--n-generate-sample",
        "--n_generate_sample",
        dest="n_generate_sample",
        type=_positive_int,
        default=5,
    )
    parser.add_argument(
        "--n-evaluate-sample",
        "--n_evaluate_sample",
        dest="n_evaluate_sample",
        type=_positive_int,
        default=1,
    )
    parser.add_argument(
        "--iterations",
        type=_positive_int,
        default=10,
    )
    parser.add_argument("--depth", type=_positive_int, default=10)
    parser.add_argument(
        "--task-start-index",
        "--task_start_index",
        dest="task_start_index",
        type=_nonnegative_int,
        default=1,
    )
    parser.add_argument(
        "--task-end-index",
        "--task_end_index",
        dest="task_end_index",
        type=_nonnegative_int,
        default=50,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="logs/webshop")
    parser.add_argument(
        "--webshop-url",
        default=(
            os.environ.get("WEBSHOP_URL", "").strip()
            or DEFAULT_WEBSHOP_URL
        ),
    )
    parser.add_argument(
        "--request-timeout",
        type=_positive_float,
        default=30.0,
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--smoke-search-query",
        default="product",
        help="Credential-free query used by the scripted smoke policy.",
    )
    args = parser.parse_args(argv)
    if args.task_end_index <= args.task_start_index:
        parser.error("task end index must be greater than task start index")
    if not math.isfinite(args.temperature) or args.temperature < 0:
        parser.error("temperature must be a finite nonnegative number")
    if not args.backend.strip():
        parser.error("backend must be non-empty")
    if not args.base_url.strip():
        parser.error("base URL must be non-empty")
    if not args.webshop_url.strip():
        parser.error("WebShop URL must be non-empty")
    if not args.smoke_search_query.strip():
        parser.error("smoke search query must be non-empty")
    return args


def _redact_url(url: str) -> str:
    parsed = urlsplit(url)
    hostname = parsed.hostname or ""
    if ":" in hostname and not hostname.startswith("["):
        hostname = "[{}]".format(hostname)
    try:
        port = parsed.port
    except ValueError as error:
        raise ValueError("URL contains an invalid port") from error
    if port is not None:
        hostname = "{}:{}".format(hostname, port)
    return urlunsplit((parsed.scheme, hostname, parsed.path, "", ""))


def _approved_config(args: argparse.Namespace) -> PlanUConfig:
    return PlanUConfig(
        n_quantiles=51,
        value_min=0.0,
        value_max=1.0,
        quantile_learning_rate=0.9,
        discount=1.0,
        curiosity_weight=0.0,
        include_preview_reward=True,
        max_depth=args.depth,
        max_iterations=args.iterations,
    )


def _safe_effective_config(
    args: argparse.Namespace,
    config: PlanUConfig,
) -> Mapping[str, Any]:
    safe_args = dict(vars(args))
    safe_args["webshop_url"] = _redact_url(args.webshop_url)
    safe_args["base_url"] = _redact_url(args.base_url)
    return json.loads(
        json.dumps(
            build_effective_config(
                SimpleNamespace(**safe_args),
                config,
            ),
            allow_nan=False,
            sort_keys=True,
        )
    )


def _run_metadata(
    args: argparse.Namespace,
    effective_config: Mapping[str, Any],
    dependencies: RunnerDependencies,
) -> Dict[str, Any]:
    metadata = dict(
        dependencies.metadata_factory(
            package_distributions=WEBSHOP_DISTRIBUTIONS,
        )
    )
    metadata["planu_git_commit"] = metadata.get("git_commit", "unknown")
    metadata.update(
        {
            "webshop_commit": WEBSHOP_COMMIT,
            "config_hash": config_hash(effective_config),
            "model_id": "scripted" if args.smoke else args.backend,
            "server_url": _redact_url(args.webshop_url),
            "seed": args.seed,
            "task_bounds": {
                "start": args.task_start_index,
                "end_exclusive": args.task_end_index,
            },
        }
    )
    return metadata


def _register_close(
    stack: ExitStack,
    resource: Any,
    registered_ids: set,
) -> None:
    if id(resource) in registered_ids:
        return
    close = getattr(resource, "close", None)
    if callable(close):
        registered_ids.add(id(resource))
        stack.callback(close)


def _token_usage(*policies: Any) -> Dict[str, int]:
    usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    for policy in policies:
        current = getattr(policy, "token_usage", {})
        for name in ("prompt_tokens", "completion_tokens"):
            value = current.get(name, 0)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    "{} token usage must be a nonnegative integer".format(name)
                )
            usage[name] += value
    usage["total_tokens"] = (
        usage["prompt_tokens"] + usage["completion_tokens"]
    )
    return usage


def _latency_failures(result: Any) -> int:
    state_path = result.state_path
    return sum(
        parent.state_key == child.state_key
        for parent, child in zip(state_path, state_path[1:])
    )


def _trajectory_payload(result: Any) -> list:
    steps = []
    for index, (action_node, reward) in enumerate(
        zip(result.action_path, result.rewards)
    ):
        next_node = result.state_path[index + 1]
        observation = getattr(
            next_node,
            "state",
            result.final_observation,
        )
        steps.append(
            {
                "action": action_node.action.text,
                "observation": observation,
                "reward": float(reward),
            }
        )
    return steps


@contextmanager
def _lock_output_directory(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(str(output_dir), os.O_RDONLY)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("output directory is in use") from error
        yield
    finally:
        os.close(descriptor)


def _load_json_object(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("{} must contain a JSON object".format(path.name))
    return payload


def _validate_manifest(
    manifest: Mapping[str, Any],
    effective_config: Mapping[str, Any],
    run_metadata: Mapping[str, Any],
) -> None:
    stored_config = manifest.get("effective_config")
    stored_metadata = manifest.get("run_metadata")
    if stored_config != effective_config:
        raise ValueError("run manifest effective config mismatch")
    if not isinstance(stored_metadata, dict):
        raise ValueError("run manifest metadata is invalid")

    expected_hash = config_hash(effective_config)
    if (
        run_metadata.get("config_hash") != expected_hash
        or stored_metadata.get("config_hash") != expected_hash
    ):
        raise ValueError("run manifest config hash mismatch")
    for field in ("planu_git_commit", "webshop_commit"):
        if stored_metadata.get(field) != run_metadata.get(field):
            raise ValueError("run manifest pinned source mismatch")


def _prepare_run_artifacts(
    output_dir: Path,
    effective_config: Mapping[str, Any],
    run_metadata: Mapping[str, Any],
) -> Mapping[str, Any]:
    manifest_path = output_dir / RUN_MANIFEST
    effective_path = output_dir / EFFECTIVE_CONFIG_SIDECAR
    metadata_path = output_dir / RUN_METADATA_SIDECAR
    manifest_payload = {
        "effective_config": effective_config,
        "run_metadata": run_metadata,
    }

    if manifest_path.exists():
        manifest = _load_json_object(manifest_path)
        _validate_manifest(
            manifest,
            effective_config,
            run_metadata,
        )
        return manifest["run_metadata"]

    has_effective = effective_path.exists()
    has_metadata = metadata_path.exists()
    has_results = (output_dir / "results.jsonl").exists()
    has_tasks = (output_dir / "tasks").exists()
    if has_results or has_tasks:
        raise ValueError("run artifacts exist without an authoritative manifest")
    if has_effective != has_metadata:
        raise ValueError("incomplete pre-manifest provenance sidecars")
    if has_effective:
        stored_effective = _load_json_object(effective_path)
        stored_metadata = _load_json_object(metadata_path)
        if (
            stored_effective != effective_config
            or stored_metadata != run_metadata
        ):
            raise ValueError("pre-manifest provenance sidecars mismatch")
        atomic_write_json(manifest_path, manifest_payload)
        return stored_metadata

    atomic_write_json(effective_path, effective_config)
    atomic_write_json(metadata_path, run_metadata)
    atomic_write_json(manifest_path, manifest_payload)
    return run_metadata


def _task_path(output_dir: Path, task_index: int) -> Path:
    return output_dir / "tasks" / "fixed_{}.json".format(task_index)


def _load_task_result(output_dir: Path, task_index: int) -> Dict[str, Any]:
    result = _load_json_object(_task_path(output_dir, task_index))
    expected_id = "fixed_{}".format(task_index)
    if (
        result.get("task_id") != expected_id
        or result.get("task_index") != task_index
    ):
        raise ValueError("task result identity mismatch for {}".format(expected_id))
    return result


def _publish_results(
    output_dir: Path,
    start_index: int,
    end_index: int,
) -> None:
    lines = []
    for task_index in range(start_index, end_index):
        result = _load_task_result(output_dir, task_index)
        lines.append(
            json.dumps(
                result,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
    atomic_write_text(output_dir / "results.jsonl", "\n".join(lines) + "\n")


def _task_result(
    task_index: int,
    args: argparse.Namespace,
    search: Any,
    provider: Any,
    scorer: Any,
    metadata: Mapping[str, Any],
) -> Mapping[str, Any]:
    rng = np.random.default_rng(args.seed + task_index)
    best_terminal = None
    latency_failure_count = 0
    search_iterations = 0
    for iteration in range(args.iterations):
        result = search.run_iteration(iteration, rng)
        search_iterations += 1
        latency_failure_count += _latency_failures(result)
        if not result.terminated:
            continue
        reward = float(result.rewards[-1]) if result.rewards else 0.0
        if best_terminal is None or reward > best_terminal[0]:
            best_terminal = reward, result
        if reward == 1.0:
            break

    if best_terminal is None:
        best_reward = 0.0
        trajectory = []
    else:
        best_reward, best_result = best_terminal
        trajectory = _trajectory_payload(best_result)

    return {
        "task_id": "fixed_{}".format(task_index),
        "task_index": task_index,
        "best_terminal_reward": best_reward,
        "success": best_reward == 1.0,
        "trajectory": trajectory,
        "latency_failure_count": latency_failure_count,
        "search_iterations": search_iterations,
        "max_depth": args.depth,
        "token_usage": _token_usage(provider, scorer),
        "error_status": None,
        "provenance": dict(metadata),
    }


def _run(
    args: argparse.Namespace,
    dependencies: RunnerDependencies,
) -> int:
    config = _approved_config(args)
    effective_config = _safe_effective_config(args, config)
    metadata = _run_metadata(args, effective_config, dependencies)
    output_dir = Path(args.output_dir)

    with _lock_output_directory(output_dir):
        authoritative_metadata = _prepare_run_artifacts(
            output_dir,
            effective_config,
            metadata,
        )

        registered_ids = set()
        with ExitStack() as stack:
            client = dependencies.client_factory(
                args.webshop_url,
                timeout=(args.request_timeout, args.request_timeout),
            )
            _register_close(stack, client, registered_ids)

            backend = None
            if not args.smoke:
                backend = dependencies.backend_factory(
                    model=args.backend,
                    base_url=args.base_url,
                    timeout=args.request_timeout,
                )
                _register_close(stack, backend, registered_ids)

            for task_index in range(
                args.task_start_index,
                args.task_end_index,
            ):
                task_id = "fixed_{}".format(task_index)
                try:
                    adapter = dependencies.adapter_factory(client, task_id)
                    if args.smoke:
                        provider = dependencies.scripted_provider_factory(
                            search_query=args.smoke_search_query,
                        )
                        scorer = dependencies.scripted_scorer_factory()
                    else:
                        provider = dependencies.model_provider_factory(
                            backend,
                            candidate_count=args.n_generate_sample,
                            temperature=args.temperature,
                        )
                        scorer = dependencies.model_scorer_factory(
                            backend,
                            n_evaluate_sample=args.n_evaluate_sample,
                            temperature=args.temperature,
                        )
                    search = dependencies.search_factory(
                        adapter,
                        scorer,
                        config,
                        action_provider=provider,
                    )
                    result = _task_result(
                        task_index,
                        args,
                        search,
                        provider,
                        scorer,
                        authoritative_metadata,
                    )
                    atomic_write_json(
                        _task_path(output_dir, task_index),
                        result,
                    )
                except _SanitizedWebShopRunnerError:
                    raise
                except Exception as error:
                    raise _SanitizedWebShopRunnerError(
                        _safe_failure_summary(error, task_id)
                    ) from None

        try:
            _publish_results(
                output_dir,
                args.task_start_index,
                args.task_end_index,
            )
        except Exception as error:
            raise _SanitizedWebShopRunnerError(
                _safe_failure_summary(error, "results")
            ) from None
    return 0


def run(
    args: argparse.Namespace,
    dependencies: Optional[RunnerDependencies] = None,
) -> int:
    """Run the selected half-open WebShop task range."""
    resolved_dependencies = dependencies or RunnerDependencies()
    try:
        return _run(args, resolved_dependencies)
    except _SanitizedWebShopRunnerError:
        raise
    except Exception as error:
        prior_error = _find_sanitized_error(error)
        if prior_error is not None:
            raise _SanitizedWebShopRunnerError(
                "{}; cleanup failure ({})".format(
                    str(prior_error),
                    type(error).__name__,
                )
            ) from None
        raise _SanitizedWebShopRunnerError(
            _safe_failure_summary(error, "runner")
        ) from None


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        return run(parse_args(argv))
    except WebShopRunnerError as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_WEBSHOP_URL",
    "RunnerDependencies",
    "ScriptedWebShopActionScorer",
    "WEBSHOP_COMMIT",
    "WebShopRunnerError",
    "main",
    "parse_args",
    "run",
]
