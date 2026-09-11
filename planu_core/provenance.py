from dataclasses import asdict
import hashlib
from importlib import metadata as importlib_metadata
import json
from pathlib import Path
import platform
import subprocess
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


def build_effective_config(args: Any, config: Any):
    return {
        "args": dict(vars(args)),
        "planu_config": asdict(config),
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


__all__ = [
    "PROVENANCE_DISTRIBUTIONS",
    "build_effective_config",
    "build_run_metadata",
    "config_hash",
    "git_commit",
    "installed_versions",
    "record_run_provenance",
]
