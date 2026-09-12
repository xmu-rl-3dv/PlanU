from dataclasses import asdict, is_dataclass
import hashlib
from importlib import metadata as importlib_metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile
from typing import Any, Callable, Mapping, Optional


PROVENANCE_DISTRIBUTIONS = {
    "numpy": "numpy",
    "torch": "torch",
    "gym": "gym",
    "transformers": "transformers",
    "peft": "peft",
    # Keep the import label stable in output; the installed distribution differs.
    "ding": "DI-engine",
}


def build_effective_config(
    args: Any,
    config: Any,
    config_name: str = "planu_config",
):
    if is_dataclass(config):
        serialized_config = asdict(config)
    elif isinstance(config, Mapping):
        serialized_config = dict(config)
    else:
        serialized_config = dict(vars(config))
    return {
        "args": dict(vars(args)),
        config_name: serialized_config,
    }


def config_hash(effective_config: Mapping[str, Any]) -> str:
    serialized = json.dumps(
        effective_config,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]


def git_commit(repository_root: Optional[str] = None) -> str:
    root = (
        str(Path(__file__).resolve().parents[1])
        if repository_root is None
        else repository_root
    )
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return completed.stdout.strip() or "unknown"


def installed_versions(
    package_distributions=PROVENANCE_DISTRIBUTIONS,
    version_lookup=None,
):
    if version_lookup is None:
        version_lookup = importlib_metadata.version
    versions = {}
    for output_key, distribution_name in package_distributions.items():
        try:
            versions[output_key] = version_lookup(distribution_name)
        except importlib_metadata.PackageNotFoundError:
            versions[output_key] = "not-installed"
    return versions


def build_run_metadata(
    repository_root: Optional[str] = None,
    package_distributions=PROVENANCE_DISTRIBUTIONS,
    git_commit_fn: Optional[Callable[[], str]] = None,
    installed_versions_fn: Optional[Callable[[Mapping[str, str]], Any]] = None,
):
    if git_commit_fn is None:
        git_commit_fn = lambda: git_commit(repository_root)
    if installed_versions_fn is None:
        installed_versions_fn = installed_versions
    return {
        "git_commit": git_commit_fn(),
        "python_version": platform.python_version(),
        "packages": installed_versions_fn(package_distributions),
    }


def record_run_provenance(
    writer: Any,
    effective_config: Mapping[str, Any],
    run_metadata: Mapping[str, Any],
) -> None:
    writer.add_text(
        "planu/effective_config",
        json.dumps(effective_config, sort_keys=True),
        global_step=0,
    )
    writer.add_text(
        "planu/run_metadata",
        json.dumps(run_metadata, sort_keys=True),
        global_step=0,
    )


def atomic_write_text(path: Any, content: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".{}.".format(destination.name),
        suffix=".tmp",
        dir=str(destination.parent),
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: Any, payload: Mapping[str, Any]) -> None:
    serialized = json.dumps(
        payload,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    )
    atomic_write_text(path, serialized + "\n")


def write_json_provenance(
    log_dir: Any,
    effective_config: Mapping[str, Any],
    run_metadata: Mapping[str, Any],
) -> None:
    directory = Path(log_dir)
    for filename, payload in (
        ("effective_config.json", effective_config),
        ("run_metadata.json", run_metadata),
    ):
        atomic_write_json(directory / filename, payload)


__all__ = [
    "PROVENANCE_DISTRIBUTIONS",
    "atomic_write_json",
    "atomic_write_text",
    "build_effective_config",
    "build_run_metadata",
    "config_hash",
    "git_commit",
    "installed_versions",
    "record_run_provenance",
    "write_json_provenance",
]
