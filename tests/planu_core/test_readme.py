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


def _section(readme, heading):
    match = re.search(
        rf"^## {re.escape(heading)}\s*$\n(.*?)(?=^## |\Z)",
        readme,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert match, f"README is missing the {heading!r} section"
    return match.group(1)


def test_readme_documents_python_39_and_unified_algorithm():
    readme = _readme()
    algorithm = _section(readme, "Unified PlanU Algorithm")
    required_terms = (
        "Upper Confidence Bounds with Curiosity (UCC)",
        "State Node",
        "Action Node",
        "outcome State Node",
        "Quantile Distribution",
        "action scorer",
        "distorted quantile value",
        "RND curiosity",
        "persistent tree",
        "suffix-return",
        "pinball",
    )

    assert "Python-3.9" in readme
    assert "Python-3.8" not in readme
    for term in required_terms:
        assert term in algorithm


def test_readme_scopes_unified_core_claims_to_migrated_phase_one_benchmarks():
    readme = _readme()
    algorithm = _normalize_whitespace(
        _section(readme, "Unified PlanU Algorithm")
    )

    assert not re.search(
        r"\b(?:all|every)\s+benchmarks?\b",
        algorithm,
        flags=re.IGNORECASE,
    )
    for term in (
        "migrated phase-one benchmarks",
        "Overcooked",
        "VirtualHome",
        "BlockWorld",
    ):
        assert term in algorithm

    for shared_component in (
        "tree",
        "selection",
        "quantile",
        "backup",
        "search core",
    ):
        assert shared_component in algorithm.lower()

    for benchmark_specific_boundary in (
        "adapters",
        "action scorers",
        "runner/evaluator orchestration",
        "numeric configuration",
        "outside",
    ):
        assert benchmark_specific_boundary in algorithm.lower()

    assert "differences are limited to adapters and configuration" not in algorithm

    status = _normalize_whitespace(
        _section(readme, "Unified-core migration status")
    )
    for term in (
        "WebShop",
        "TravelPlanner",
        "Phase two",
        "Planned",
        "legacy code",
        "not yet migrated",
    ):
        assert term in status


def test_readme_links_supported_benchmark_adapters():
    readme = _readme()

    assert "(planu_core/adapters/virtualhome.py)" in readme
    assert "(planu_core/adapters/blockworld.py)" in readme


def test_readme_documents_core_packages_and_boundaries():
    architecture = _section(_readme(), "Architecture")

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
        assert f"`planu_core/{module}" in architecture
    assert "boundary" in architecture.lower()


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
    assert not re.search(
        r"(WebShop|TravelPlanner).{0,80}(adapter|implementation)\s+is implemented",
        readme,
        flags=re.IGNORECASE | re.DOTALL,
    )


def test_readme_installation_keeps_benchmark_environments_external():
    installation = _section(_readme(), "Installation")

    for term in ("Python 3.9", "`gym-macro-overcooked`", "`virtual-home`", "external"):
        assert term in installation
    assert "python -m pip install -e ." in installation


def test_readme_documents_complete_overcooked_setup():
    readme = _readme()
    installation_section = readme.split("## Installation", 1)[1].split("\n## ", 1)[0]
    bash_blocks = re.findall(
        r"```bash\n(.*?)```",
        installation_section,
        flags=re.DOTALL,
    )

    assert len(bash_blocks) == 1
    installation_commands = bash_blocks[0].splitlines()
    required_commands = (
        "python -m pip install -r requirements.txt",
        "python -m pip install easydict DI-engine",
        "python -m pip install -e .",
        "python -m pip install -e gym-macro-overcooked",
        "python -m pip install -e virtual-home",
    )
    assert all(command in installation_commands for command in required_commands)
    assert [installation_commands.index(command) for command in required_commands] == sorted(
        installation_commands.index(command) for command in required_commands
    )

    for term in ("DI-engine", "easydict", "RND"):
        assert term in installation_section


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


def test_readme_scopes_model_and_device_configuration_by_benchmark():
    readme = _readme()
    configuration = _normalize_whitespace(_section(readme, "Configuration Scope"))

    for term in (
        "Overcooked",
        "VirtualHome",
        "runner arguments",
        "environment variables",
        "BlockWorld",
        "GPU",
        "seed",
        "iterations",
        "success probability",
        "`blockworld/evaluate_stochastic.py`",
        "HF model identifier",
        "source edit",
    ):
        assert term in configuration
    assert "without editing hardcoded source locations" not in readme


def test_readme_scopes_provenance_to_supported_runners():
    provenance = _normalize_whitespace(
        _section(_readme(), "Outputs And Provenance")
    )

    for term in (
        "Overcooked",
        "VirtualHome",
        "config-hashed",
        "TensorBoard",
        "effective configuration",
        "git commit",
        "dependency",
        "BlockWorld",
        "JSON",
        "config-hashed evaluator path",
        "effective_config.json",
        "run_metadata.json",
    ):
        assert term in provenance
    assert "TensorBoard text summaries" in provenance
    assert re.search(
        r"BlockWorld writes both `effective_config\.json` and "
        r"`run_metadata\.json`",
        provenance,
    )
    assert "Each run stores `effective_config.json`" not in provenance


def test_readme_documents_blockworld_behavior_corrections():
    migration = _normalize_whitespace(
        _section(_readme(), "BlockWorld Migration Notes")
    )

    for term in (
        "State Node",
        "Action Node",
        "outcome",
        "re-sampled",
        "arbitrary outcomes",
        "first result",
        "arithmetic mean",
        "-10",
        "100",
        "goal reward",
        "fast/action prior",
        "executed transition reward",
        "first-visit terminal goal reward",
        "suffix-return",
        "selection",
        "legacy output strategies",
        "non-default options",
        "rejected",
        "unified implementation results",
        "bit-identical",
        "previous broken entrypoint",
    ):
        assert term in migration


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
    assert (
        "${BASE_MODEL:-Neko-Institute-of-Science/LLaMA-7B-HF}"
        in script
    )
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
        '--normalization-mode "token"',
        "--maxiterations 1000",
        "--stochastic 0.5",
        '--base-model "${BASE_MODEL}"',
        '--seed "${seed}"',
        '--rnd "True"',
    )

    for argument in required_arguments:
        assert argument in script


def test_readme_discloses_effective_published_overcooked_settings():
    overcooked = _normalize_whitespace(
        _section(_readme(), "Run Overcooked")
    )

    for term in (
        "legacy source ignored",
        "`--base-model`",
        "`--normalization-mode`",
        "`Neko-Institute-of-Science/LLaMA-7B-HF`",
        "`token`",
        "unified runner now honors",
        "reference launcher pins",
        "overrides",
        "new configurations",
        "exact reproduction",
    ):
        assert term in overcooked


def test_readme_referenced_local_paths_exist():
    for path in (
        OVERCOOKED_SCRIPT,
        REPOSITORY_ROOT / "gym-macro-overcooked",
        REPOSITORY_ROOT / "virtual-home",
    ):
        assert path.exists()
