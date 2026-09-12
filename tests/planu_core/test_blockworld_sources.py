from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[2]
BLOCKWORLD_ROOT = REPO_ROOT / "blockworld"
STEPS = ("2", "4", "6", "8", "10", "12")
EXAMPLE_SUFFIXES = {".json", ".md", ".pddl", ".py", ".sh", ".yaml", ".yml"}


def _tracked_files():
    output = subprocess.check_output(
        ["git", "ls-files", "-z"],
        cwd=str(REPO_ROOT),
        timeout=30,
    )
    return {
        Path(path.decode("utf-8"))
        for path in output.split(b"\0")
        if path
    }


def _relative_files(root):
    return {
        path.relative_to(REPO_ROOT)
        for path in root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and ".pytest_cache" not in path.parts
        and path.suffix not in {".pyc", ".pyd", ".pyo"}
    }


def test_blockworld_clean_checkout_contains_runtime_sources_and_inputs():
    required = {
        Path("blockworld/.gitignore"),
        Path("blockworld/setup.py"),
        Path("blockworld/evaluate_stochastic.py"),
        Path("blockworld/reasoners/__init__.py"),
        Path("blockworld/reasoners/base.py"),
        Path("blockworld/reasoners/algorithm/mcts.py"),
        Path("blockworld/reasoners/algorithm/planU.py"),
        Path("blockworld/reasoners/benchmark/bw_utils.py"),
        Path("blockworld/reasoners/lm/hf_model.py"),
        Path("blockworld/examples/CoT/blocksworld/data/bw_config.yaml"),
        Path(
            "blockworld/examples/CoT/blocksworld/data/"
            "generated_domain.pddl"
        ),
        Path(
            "blockworld/examples/CoT/blocksworld/prompts/"
            "pool_prompt_v1.json"
        ),
    }
    required.update(
        Path(
            "blockworld/examples/CoT/blocksworld/data/"
            "split_{0}/split_{0}_step_{1}_data.json".format(version, step)
        )
        for version in ("v1", "v2")
        for step in STEPS
    )
    required.update(
        Path(
            "blockworld/examples/CoT/blocksworld/prompts/"
            "pool_prompt_v2_step_{}.json".format(step)
        )
        for step in STEPS
    )

    missing = sorted(
        str(path)
        for path in required
        if not (REPO_ROOT / path).is_file()
    )
    assert missing == []


def test_blockworld_deliverable_files_are_tracked_without_forbidden_paths():
    tracked = _tracked_files()
    expected = _relative_files(BLOCKWORLD_ROOT / "reasoners")
    expected.update(
        path
        for path in _relative_files(BLOCKWORLD_ROOT / "examples")
        if path.suffix in EXAMPLE_SUFFIXES
    )
    expected.update(
        {
            Path("blockworld/.gitignore"),
            Path("blockworld/setup.py"),
        }
    )

    assert sorted(str(path) for path in expected - tracked) == []

    forbidden = {
        path
        for path in tracked
        if path.parts[:1] == ("blockworld",)
        and (
            path.parts[:2] == ("blockworld", "exllama")
            or ".git" in path.parts
            or ".env" in path.parts
            or "__pycache__" in path.parts
            or ".pytest_cache" in path.parts
            or path.suffix in {".pyc", ".pyd", ".pyo"}
        )
    }
    assert sorted(str(path) for path in forbidden) == []
