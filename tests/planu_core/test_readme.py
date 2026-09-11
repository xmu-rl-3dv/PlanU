import re
import subprocess
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
README_PATH = REPOSITORY_ROOT / "README.md"
OVERCOOKED_SCRIPT = REPOSITORY_ROOT / "scripts" / "PlanU_overcooked.sh"


def _readme():
    return README_PATH.read_text(encoding="utf-8")


def _normalize_whitespace(value):
    return " ".join(value.split())


def test_readme_documents_python_39_and_unified_algorithm():
    readme = _readme()
    normalized_readme = _normalize_whitespace(readme)
    required_terms = (
        "one shared PlanU algorithm implementation",
        "no published/canonical split",
        "State Node",
        "Action Node",
        "outcome State Node",
        "Quantile Distribution",
        "action scorer",
        "optional preview reward",
        "distorted quantile value",
        "normalized optional RND curiosity",
        "persistent tree across trajectories",
        "Monte Carlo suffix-return quantile pinball backup",
        "stochastic outcomes under the same Action Node",
        "benchmark differences are limited to adapters and configuration",
    )

    assert "Python-3.9" in readme
    assert "Python-3.8" not in readme
    for term in required_terms:
        assert term in normalized_readme


def test_readme_documents_core_packages_and_boundaries():
    readme = _readme()

    for module in (
        "nodes",
        "distribution",
        "selection",
        "backup",
        "search",
        "curiosity",
        "scorers",
        "provenance",
        "adapters",
    ):
        assert f"`planu_core/{module}" in readme
    assert "boundary" in readme.lower()


def test_readme_has_supported_and_planned_benchmark_statuses():
    readme = _readme()
    expected_rows = (
        ("Overcooked", "Phase one", "Supported"),
        ("VirtualHome", "Phase one", "Supported"),
        ("BlockWorld", "Phase one", "Supported"),
        ("WebShop", "Phase two", "Planned"),
        ("TravelPlanner", "Phase two", "Planned"),
    )

    for benchmark, phase, status in expected_rows:
        row = rf"\|\s*{benchmark}\s*\|\s*{phase}\s*\|\s*{status}\s*\|"
        assert re.search(row, readme)
    assert "https://github.com/OSU-NLP-Group/TravelPlanner.git" in readme
    assert "Phase-two adapters are not implemented." in readme
    assert not re.search(
        r"(WebShop|TravelPlanner).{0,80}(adapter|implementation)\s+is implemented",
        readme,
        flags=re.IGNORECASE | re.DOTALL,
    )


def test_readme_installation_keeps_benchmark_environments_external():
    readme = _readme()

    assert "Python 3.9" in readme
    assert "python -m pip install -e ." in readme
    assert "`gym-macro-overcooked`" in readme
    assert "`virtual-home`" in readme
    assert "environments remain" in readme


def test_readme_has_exactly_one_overcooked_experiment_invocation():
    readme = _readme()
    experiment_commands = re.findall(
        r"^\s*(?:(?:bash|sh)\s+\S+\.sh|python\s+(?:mcts|webshop|blockworld)/\S+)",
        readme,
        flags=re.MULTILINE,
    )

    assert experiment_commands == ["bash scripts/PlanU_overcooked.sh"]
    assert "scripts/PlanU_Virtualhome.sh" not in readme
    assert not re.search(r"^\s*(?:bash|sh)\s+(?:webshop/)?planu\.sh", readme, re.MULTILINE)


def test_readme_removes_stale_model_edit_instructions_and_describes_outputs():
    readme = _readme()
    normalized_readme = _normalize_whitespace(readme)

    assert not re.search(
        r"set.{0,20}local LLM path",
        readme,
        flags=re.IGNORECASE,
    )
    assert not re.search(r"PlanU_(?:mcts|v1|v2)\.py#L\d+", readme)
    for term in (
        "config hash",
        "TensorBoard",
        "effective configuration",
        "git commit",
        "dependency versions",
    ):
        assert term in normalized_readme


def test_overcooked_script_is_valid_strict_bash_with_configurable_defaults():
    script = OVERCOOKED_SCRIPT.read_text(encoding="utf-8")
    syntax_check = subprocess.run(
        ["bash", "-n", str(OVERCOOKED_SCRIPT)],
        capture_output=True,
        text=True,
    )

    assert syntax_check.returncode == 0, syntax_check.stderr
    assert script.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
    assert "${CUDA_VISIBLE_DEVICES:-0,1,2,3}" in script
    assert "${BASE_MODEL:-meta-llama/Meta-Llama-3.1-8B-Instruct}" in script
    assert "${PYTHON:-python}" in script
    assert 'for seed in 1 10 20 30 40; do' in script
    assert '"${PYTHON}" mcts/overcooked/PlanU_inference.py' in script


def test_overcooked_script_preserves_experiment_flags():
    script = OVERCOOKED_SCRIPT.read_text(encoding="utf-8")
    required_arguments = (
        '--exp-name "tomato_salad_llm"',
        "--num-envs 1",
        "--depth 15",
        "--env-reward 0.2 1 0.1 0.001",
        "--task 0",
        '--env-id "Overcooked-LLMA-v4"',
        '--record-path "workdir"',
        '--normalization-mode "word"',
        "--maxiterations 1000",
        "--stochastic 0.5",
        '--base-model "${BASE_MODEL}"',
        '--seed "${seed}"',
        '--rnd "True"',
    )

    for argument in required_arguments:
        assert argument in script


def test_readme_referenced_local_paths_exist():
    for path in (
        OVERCOOKED_SCRIPT,
        REPOSITORY_ROOT / "gym-macro-overcooked",
        REPOSITORY_ROOT / "virtual-home",
    ):
        assert path.exists()
