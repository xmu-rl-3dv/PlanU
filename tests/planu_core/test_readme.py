import re
import subprocess
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
README_PATH = REPOSITORY_ROOT / "README.md"
EXPERIMENT_VALIDATION_PATH = (
    REPOSITORY_ROOT / "docs" / "experiment-validation.md"
)
OVERCOOKED_SCRIPT = REPOSITORY_ROOT / "scripts" / "PlanU_overcooked.sh"
SMOKE_SCRIPT = REPOSITORY_ROOT / "scripts" / "smoke_phase_one.sh"
WEBSHOP_BOOTSTRAP = REPOSITORY_ROOT / "scripts" / "bootstrap_webshop.sh"
WEBSHOP_SMOKE = REPOSITORY_ROOT / "scripts" / "smoke_webshop.sh"
EXPERIMENT_REQUIREMENTS = (
    REPOSITORY_ROOT / "requirements-experiments.txt"
)
EXPERIMENT_LOCK = REPOSITORY_ROOT / "requirements-experiments-lock.txt"
WEBSHOP_SERVER_REQUIREMENTS = (
    REPOSITORY_ROOT / "requirements-webshop-server.txt"
)


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


def test_readme_documents_isolated_python_versions_and_unified_algorithm():
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
    assert "Python 3.8.13" in readme
    prerequisites = _normalize_whitespace(
        _section(readme, "Prerequisites")
    )
    assert "Python 3.9 for PlanU" in prerequisites
    assert "Python 3.8.13 for the official WebShop server" in prerequisites
    assert "separate" in prerequisites
    for term in required_terms:
        assert term in algorithm
    assert "distinct stochastic outcomes remain separate" in algorithm
    assert "repeated equal `state_key` outcomes merge" in algorithm
    assert "repeated stochastic outcomes remain separate" not in algorithm


def test_readme_scopes_unified_core_claims_to_supported_benchmarks():
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
        "WebShop",
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
        "action providers",
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
    for term in ("WebShop", "TravelPlanner", "Phase two", "Supported", "Planned"):
        assert term in status
    assert "WebShop search is implemented in `planu_core`" in status


def test_readme_links_supported_benchmark_adapters():
    readme = _readme()

    for adapter in ("virtualhome.py", "blockworld.py", "webshop.py"):
        assert f"`planu_core/adapters/{adapter}`" in readme


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
        ("Overcooked", "Phase one", "Supported", "`mcts/overcooked/PlanU_inference.py`"),
        ("VirtualHome", "Phase one", "Supported", "`mcts/virtualhome/PlanU_inference_food.py`, `PlanU_entertainment.py`"),
        ("BlockWorld", "Phase one", "Supported", "`blockworld/evaluate_stochastic.py`"),
        ("WebShop", "Phase two", "Supported", "`python -m planu_core.webshop.runner`"),
        ("TravelPlanner", "Phase two", "Planned", "Not implemented"),
    )

    for benchmark, phase, status, entry_point in expected_rows:
        row = (
            rf"\|\s*{benchmark}\s*\|\s*{phase}\s*\|\s*{status}\s*\|\s*"
            rf"{re.escape(entry_point)}\s*\|"
        )
        assert re.search(row, readme)
    assert "https://github.com/OSU-NLP-Group/TravelPlanner.git" in readme


def test_readme_installation_keeps_benchmark_environments_external():
    installation = _section(_readme(), "Installation")

    for term in ("Python 3.9", "`gym-macro-overcooked`", "`virtual-home`", "external"):
        assert term in installation
    assert "python -m pip install --no-deps -e ." in installation


def test_readme_documents_complete_overcooked_setup():
    readme = _readme()
    installation_section = readme.split("## Installation", 1)[1].split("\n## ", 1)[0]
    bash_blocks = re.findall(
        r"```bash\n(.*?)```",
        installation_section,
        flags=re.DOTALL,
    )

    assert len(bash_blocks) == 2
    installation_commands = bash_blocks[0].splitlines()
    required_commands = (
        (
            "python -m pip install -r requirements-experiments.txt "
            "-c requirements-experiments-lock.txt"
        ),
        "python -m pip install --no-deps -e .",
        "python -m pip install --no-deps -e gym-macro-overcooked",
        "python -m pip install --no-deps -e virtual-home",
    )
    assert all(command in installation_commands for command in required_commands)
    assert [installation_commands.index(command) for command in required_commands] == sorted(
        installation_commands.index(command) for command in required_commands
    )

    for term in ("DI-engine", "easydict", "RND"):
        assert term in installation_section


def test_readme_documents_webshop_setup_and_real_smoke():
    webshop = _normalize_whitespace(_section(_readme(), "Run WebShop"))

    for term in (
        "Python 3.8.13",
        "Flask 2.1.2",
        "Werkzeug 2.1.2",
        "Java 11",
        "Pyserini",
        "64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd",
        "`requirements-webshop-server.txt`",
        "`scripts/bootstrap_webshop.sh`",
        "`scripts/smoke_webshop.sh`",
        "`JAVA_HOME`",
        "`JVM_PATH`",
        "`PLANU_PYTHON`",
        "`WEBSHOP_PYTHON`",
        "`WEBSHOP_ROOT`",
        "`WEBSHOP_ENV_PREFIX`",
        "`WEBSHOP_URL`",
        "`RUN_ROOT`",
        "`OPENAI_API_KEY`",
        "`OPENAI_BASE_URL`",
        "`server_runtime.json`",
        "environment SHA-256",
        "http_transition_count",
        "quantile_backup_count",
        "search[product]",
        "click[Buy Now]",
    ):
        assert term in webshop

    for digest in (
        "30a4765c3a327af72d9a9a95a6b2486d516f0fa1d3ecd83681901ce82a21b269",
        "f88a36314a397b53b3d9c3fa5878e5f7b26d35019a51ec83fbedeca61a948f6f",
        "cf78667548a71786e1d9049c24b802e48e1084ad4bb021cae56ce1f6d96954a3",
    ):
        assert digest in webshop


def test_readme_limits_authoritative_webshop_evidence_to_fixed_one():
    webshop = _normalize_whitespace(_section(_readme(), "Run WebShop"))

    for term in (
        "`fixed_1`",
        "authoritative",
        "scripted",
        "`http_transition_count: 3`",
        "`quantile_backup_count: 1`",
        "does not reproduce paper metrics",
        "model-backed reference profile has not been validated",
    ):
        assert term in webshop


def test_experiment_validation_documents_exact_webshop_matrix():
    validation = EXPERIMENT_VALIDATION_PATH.read_text(encoding="utf-8")
    matrix = _normalize_whitespace(
        _section(validation, "WebShop reference configuration")
    )

    expected_rows = (
        ("Official server", "`princeton-nlp/WebShop@64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd`"),
        ("Backend", "`qwen-plus`"),
        ("Temperature", "`0.8`"),
        ("Prompt mode", "`cot`"),
        ("Generated candidates per expansion", "`5`"),
        ("Evaluation samples per candidate", "`1`"),
        ("Search iterations", "`10`"),
        ("Expanded tree depth", "`10`"),
        ("Task range", "`fixed_1` through `fixed_49`"),
        ("Quantiles", "`51` midpoint quantiles"),
        ("Quantile range", "`[0, 1]`"),
        ("Quantile learning rate", "`0.9`"),
        ("Stochastic latency distribution", "log-normal `mu=0`, `sigma=10` milliseconds"),
        ("Latency threshold", "`200` milliseconds"),
    )
    for setting, value in expected_rows:
        assert re.search(
            rf"\|\s*{re.escape(setting)}\s*\|\s*{re.escape(value)}\s*\|",
            matrix,
        )
    assert "half-open" in matrix
    assert "`--task-start-index 1 --task-end-index 50`" in matrix


def test_experiment_validation_documents_webshop_corrections_and_parity():
    validation = EXPERIMENT_VALIDATION_PATH.read_text(encoding="utf-8")
    corrections = _normalize_whitespace(
        _section(
            validation,
            "Intentional corrections from the legacy WebShop implementation",
        )
    )
    parity = _normalize_whitespace(
        _section(validation, "WebShop parity and evidence")
    )

    for term in (
        "partial-credit",
        "terminal",
        "single depth limit",
        "separate depth-20 rollout",
        "scorer",
        "quantiles",
        "suffix-return",
        "explicit NumPy generator",
        "invalid generated actions",
        "`-1`",
        "`[0, 1]`",
    ):
        assert term in corrections

    for term in (
        "init",
        "search",
        "item",
        "option",
        "subpage",
        "back",
        "purchase",
        "automated contract tests",
        "authoritative real-server smoke",
        "`fixed_1`",
        "not established by the smoke",
    ):
        assert term in parity


def test_experiment_validation_documents_stable_webshop_evidence_schema():
    validation = EXPERIMENT_VALIDATION_PATH.read_text(encoding="utf-8")
    smoke = _normalize_whitespace(
        _section(validation, "WebShop real smoke")
    )

    for term in (
        "`<run-root>/run_manifest.json`",
        "`<run-root>/effective_config.json`",
        "`<run-root>/run_metadata.json`",
        "`<run-root>/results.jsonl`",
        "`<run-root>/tasks/fixed_1.json`",
        "`<run-root>/webshop-server.log`",
        "`<run-root>/server_runtime.json`",
        "`RUN_ROOT` is optional",
        "exit `0`",
        "`model_id: scripted`",
        "`http_transition_count: 3`",
        "`quantile_backup_count: 1`",
        "`best_terminal_reward: 0.0`",
        "does not reproduce paper metrics",
    ):
        assert term in smoke
    assert "/tmp/planu-webshop-smoke-2141d95" not in validation


def test_experiment_lock_pins_resolved_transitive_dependencies():
    requirements = set(
        line.strip()
        for line in EXPERIMENT_LOCK.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    )

    for requirement in (
        "annotated-types==0.7.0",
        "distro==1.9.0",
        "gymnasium==1.1.1",
        "huggingface_hub==0.36.2",
        "jiter==0.16.0",
        "openai==1.109.1",
        "pydantic==2.13.5",
        "pydantic_core==2.46.5",
        "scipy==1.13.1",
        "setuptools==66.1.1",
        "typing-inspection==0.4.2",
        "wandb==0.12.16",
    ):
        assert requirement in requirements
    assert len(requirements) == 144


def test_docs_explain_experiment_lock_and_webshop_server_attestation():
    readme = _normalize_whitespace(_readme())
    validation = _normalize_whitespace(
        EXPERIMENT_VALIDATION_PATH.read_text(encoding="utf-8")
    )

    for document in (readme, validation):
        assert "144 exact pins" in document
        assert "143 resolved distributions" in document
        assert "`setuptools==66.1.1`" in document
        assert "`server_runtime.json`" in document
        assert "environment SHA-256" in document


def test_webshop_server_requirements_exactly_pin_upstream_direct_dependencies():
    requirements = {
        line.strip()
        for line in WEBSHOP_SERVER_REQUIREMENTS.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip() and not line.startswith("#")
    }

    assert requirements == {
        "beautifulsoup4==4.11.1",
        "cleantext==1.1.4",
        "env==0.1.0",
        "Flask==2.1.2",
        "gdown==5.2.0",
        "gradio==3.50.2",
        "gym==0.24.0",
        "numpy==1.22.4",
        "pandas==1.4.2",
        "pyserini==0.17.0",
        "pytest==7.4.4",
        "PyYAML==6.0",
        "rank_bm25==0.2.2",
        "requests==2.27.1",
        "requests-mock==1.12.1",
        "rich==12.4.4",
        "scikit_learn==1.1.1",
        "selenium==4.2.0",
        "spacy==3.3.0",
        "thefuzz==0.19.0",
        "torch==1.11.0",
        "tqdm==4.64.0",
        "train==0.0.5",
        "transformers==4.19.2",
        "Werkzeug==2.1.2",
    }


def test_experiment_requirements_pin_compatible_runtime_versions():
    requirements = set(
        line.strip()
        for line in EXPERIMENT_REQUIREMENTS.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    )

    for requirement in (
        "numpy==1.23.5",
        "gym==0.25.1",
        "opencv-python==4.8.1.78",
        "DI-engine==0.5.3",
        "openai==1.109.1",
        "Werkzeug==2.0.3",
        "torch==2.8.0",
        "transformers==4.57.6",
    ):
        assert requirement in requirements


def test_phase_one_smoke_script_covers_all_real_runners():
    script = SMOKE_SCRIPT.read_text(encoding="utf-8")

    assert script.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
    assert "34e6841f81ca7708f2f8b8241504bfe8a908e40b" in script
    assert "actual_planbench_commit" in script
    assert "mcts/overcooked/PlanU_inference.py" in script
    assert "mcts/virtualhome/PlanU_inference_food.py" in script
    assert "mcts/virtualhome/PlanU_entertainment.py" in script
    assert "blockworld/evaluate_stochastic.py" in script
    assert script.count("--maxiterations 1") == 3
    assert "--max-examples 1" in script
    assert "PLANBENCH_PATH" in script
    assert "SMOKE_MODEL" in script


def test_readme_links_phase_one_experiment_validation_matrix():
    assert (
        "[Phase-One Experiment Validation](docs/experiment-validation.md)"
        in _readme()
    )


def test_readme_has_run_examples_for_every_supported_benchmark():
    readme = _readme()

    for command in (
        "bash scripts/smoke_phase_one.sh",
        "python mcts/overcooked/PlanU_inference.py",
        "bash scripts/PlanU_overcooked.sh",
        "python mcts/virtualhome/PlanU_inference_food.py",
        "python mcts/virtualhome/PlanU_entertainment.py",
        "bash scripts/PlanU_Virtualhome.sh",
        "python blockworld/evaluate_stochastic.py",
        "bash scripts/bootstrap_webshop.sh",
        "bash scripts/smoke_webshop.sh",
        "python -m planu_core.webshop.runner",
    ):
        assert command in readme


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
        "`--model`",
        "`--device`",
        "`--max-examples`",
        "WebShop",
        "backend",
        "server URL",
        "`--smoke`",
    ):
        assert term in configuration
    assert "without editing hardcoded source locations" in configuration


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
        "config-hashed evaluator directory",
        "effective_config.json",
        "run_metadata.json",
        "WebShop",
        "run_manifest.json",
        "results.jsonl",
        "pinned server commit",
    ):
        assert term in provenance
    assert "TensorBoard result paths" in provenance
    assert re.search(
        r"BlockWorld writes a config-hashed evaluator directory containing "
        r"`effective_config\.json`, `run_metadata\.json`",
        provenance,
    )


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
        timeout=30,
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
        SMOKE_SCRIPT,
        WEBSHOP_BOOTSTRAP,
        WEBSHOP_SMOKE,
        REPOSITORY_ROOT / "gym-macro-overcooked",
        REPOSITORY_ROOT / "virtual-home",
        REPOSITORY_ROOT / "planu_core" / "webshop" / "runner.py",
        REPOSITORY_ROOT / "planu_core" / "adapters" / "webshop.py",
    ):
        assert path.exists()
