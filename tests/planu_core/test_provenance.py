import hashlib
import json
from types import SimpleNamespace

from planu_core.config import PlanUConfig


def test_effective_config_and_hash_are_canonical():
    from planu_core.provenance import build_effective_config, config_hash

    args = SimpleNamespace(seed=10, grid_dim=[7, 7])
    reordered = SimpleNamespace()
    reordered.grid_dim = [7, 7]
    reordered.seed = 10
    config = PlanUConfig(max_depth=3)

    payload = build_effective_config(args, config)
    expected_json = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    )

    assert config_hash(payload) == hashlib.sha256(
        expected_json.encode("utf-8")
    ).hexdigest()[:12]
    assert config_hash(build_effective_config(reordered, config)) == (
        config_hash(payload)
    )


def test_installed_versions_maps_ding_to_di_engine():
    from importlib import metadata as importlib_metadata

    from planu_core.provenance import installed_versions

    requested = []

    def fake_version(distribution):
        requested.append(distribution)
        if distribution == "DI-engine":
            return "0.5.3"
        raise importlib_metadata.PackageNotFoundError(distribution)

    versions = installed_versions(version_lookup=fake_version)

    assert requested == [
        "numpy",
        "torch",
        "gym",
        "transformers",
        "peft",
        "DI-engine",
    ]
    assert versions["ding"] == "0.5.3"
    assert versions["gym"] == "not-installed"


def test_record_run_provenance_uses_stable_tags():
    from planu_core.provenance import record_run_provenance

    calls = []

    class Writer:
        def add_text(self, tag, value, global_step):
            calls.append((tag, json.loads(value), global_step))

    effective_config = {"args": {"seed": 1}, "planu_config": {"max_depth": 3}}
    metadata = {"git_commit": "abc", "packages": {}, "python_version": "3.9"}

    record_run_provenance(Writer(), effective_config, metadata)

    assert calls == [
        ("planu/effective_config", effective_config, 0),
        ("planu/run_metadata", metadata, 0),
    ]
