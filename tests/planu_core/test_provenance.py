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


def test_effective_config_accepts_named_mapping_config():
    from planu_core.provenance import build_effective_config

    payload = build_effective_config(
        SimpleNamespace(algorithm="mcts", seed=7),
        {"n_iters": 10, "depth_limit": 12},
        config_name="mcts_config",
    )

    assert payload == {
        "args": {"algorithm": "mcts", "seed": 7},
        "mcts_config": {"n_iters": 10, "depth_limit": 12},
    }


def test_write_json_provenance_creates_stable_files(tmp_path):
    from planu_core.provenance import write_json_provenance

    log_dir = tmp_path / "seed=7" / "config=abc123"
    effective_config = {"args": {"seed": 7}}
    metadata = {"git_commit": "abc", "python_version": "3.9"}

    write_json_provenance(log_dir, effective_config, metadata)

    assert json.loads(
        (log_dir / "effective_config.json").read_text(encoding="utf-8")
    ) == effective_config
    assert json.loads(
        (log_dir / "run_metadata.json").read_text(encoding="utf-8")
    ) == metadata


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


def test_installed_distribution_identity_is_complete_normalized_and_hashed():
    from planu_core.provenance import installed_distribution_identity

    class Distribution:
        def __init__(self, name, version):
            self.metadata = {"Name": name}
            self.version = version

    identity = installed_distribution_identity(
        distributions_fn=lambda: [
            Distribution("Z_pkg", "3.0"),
            Distribution("alpha.pkg", "1.0"),
            Distribution("Alpha-Pkg", "1.0"),
            Distribution(None, "ignored"),
        ]
    )
    packages = {
        "alpha-pkg": "1.0",
        "z-pkg": "3.0",
    }

    assert identity == {
        "installed_distributions": packages,
        "installed_distributions_sha256": hashlib.sha256(
            json.dumps(
                packages,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
    }
    assert list(identity["installed_distributions"]) == sorted(packages)


def test_installed_distribution_identity_rejects_conflicting_canonical_names():
    import pytest

    from planu_core.provenance import installed_distribution_identity

    class Distribution:
        def __init__(self, name, version):
            self.metadata = {"Name": name}
            self.version = version

    with pytest.raises(ValueError, match="conflicting installed distributions"):
        installed_distribution_identity(
            distributions_fn=lambda: [
                Distribution("same_name", "1.0"),
                Distribution("same-name", "2.0"),
            ]
        )


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
