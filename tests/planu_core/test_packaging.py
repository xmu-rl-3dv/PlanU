import email
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
import zipfile

from packaging.requirements import Requirement


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_PACKAGE_MEMBERS = {
    "planu_core/search.py",
    "reasoners/algorithm/planU.py",
}
DECLARED_IMPORT_ROOTS = {
    "accelerate",
    "anthropic",
    "bitsandbytes",
    "datasets",
    "fairscale",
    "fire",
    "google",
    "huggingface_hub",
    "ninja",
    "numpy",
    "openai",
    "optimum",
    "peft",
    "scipy",
    "sentencepiece",
    "tarski",
    "torch",
    "tqdm",
    "transformers",
}
REQUIRED_RUNTIME_DEPENDENCIES = {
    "accelerate",
    "datasets",
    "numpy",
    "pddl==0.2.0",
    "peft",
    "pyyaml",
    "requests",
    "sentencepiece",
    "tarski",
    "torch",
    "tqdm",
    "transformers",
}
OPTIONAL_PROVIDER_DEPENDENCIES = {
    "anthropic",
    "bitsandbytes",
    "di-engine",
    "easydict",
    "fairscale",
    "google-generativeai",
    "huggingface-hub",
    "llama-cpp-python",
    "openai",
    "optimum",
    "scipy",
}


def _copy_packaging_source(destination):
    for relative_path in ("planu_core", "blockworld/reasoners"):
        shutil.copytree(
            REPOSITORY_ROOT / relative_path,
            destination / relative_path,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    for relative_path in ("README.md", "setup.py", "blockworld/setup.py"):
        source = REPOSITORY_ROOT / relative_path
        if source.exists():
            target = destination / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def _relative_archive_members(names):
    members = set()
    for member in names:
        relative_member = (
            member.split("/", 1)[-1] if "/" in member else member
        )
        members.add(relative_member)
        if relative_member.startswith("blockworld/"):
            members.add(relative_member.removeprefix("blockworld/"))
    return members


def _requirement_key(requirement):
    return "{}{}".format(requirement.name.lower(), requirement.specifier)


class PackagingTest(unittest.TestCase):
    def test_dev_requirements_cover_the_complete_test_toolchain(self):
        requirements = {
            line.strip()
            for line in (REPOSITORY_ROOT / "requirements-dev.txt")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }

        self.assertLessEqual(
            {
                "numpy>=1.24,<2",
                "pytest>=7.4,<9",
                "wheel>=0.37",
            },
            requirements,
        )

    def test_language_model_namespace_does_not_import_optional_providers(self):
        completed = subprocess.run(
            [
                sys.executable,
                "-I",
                "-c",
                """
import sys
import types

transformers = types.ModuleType("transformers")
transformers.StoppingCriteriaList = list
torch = types.ModuleType("torch")
tqdm = types.ModuleType("tqdm")
tqdm.tqdm = lambda items, **kwargs: items
sys.modules["transformers"] = transformers
sys.modules["torch"] = torch
sys.modules["tqdm"] = tqdm
sys.path.insert(0, {!r})

import reasoners.lm

assert "anthropic" not in sys.modules
assert "google.generativeai" not in sys.modules
assert "openai" not in sys.modules
assert "fairscale" not in sys.modules
""".format(os.fspath(REPOSITORY_ROOT / "blockworld")),
            ],
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_distribution_contains_and_imports_shared_planu_core(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            source_root = temporary_path / "source"
            source_root.mkdir()
            _copy_packaging_source(source_root)

            distribution_directory = temporary_path / "dist"
            for command in ("sdist", "bdist_wheel"):
                subprocess.run(
                    [
                        sys.executable,
                        "setup.py",
                        command,
                        "--dist-dir",
                        os.fspath(distribution_directory),
                    ],
                    cwd=source_root / "blockworld",
                    check=True,
                    capture_output=True,
                    text=True,
                )

            source_archive = next(distribution_directory.glob("*.tar.gz"))
            wheel_archive = next(distribution_directory.glob("*.whl"))
            with tarfile.open(source_archive) as archive:
                archive_names = archive.getnames()
                source_members = _relative_archive_members(archive_names)
                readme_names = [
                    name for name in archive_names if name.endswith("/README.md")
                ]
                self.assertEqual(len(readme_names), 1)
                readme_file = archive.extractfile(readme_names[0])
                self.assertIsNotNone(readme_file)
                readme = readme_file.read().decode("utf-8")
            with zipfile.ZipFile(wheel_archive) as archive:
                wheel_members = set(archive.namelist())
                metadata_name = next(
                    name for name in wheel_members if name.endswith("/METADATA")
                )
                metadata = email.message_from_bytes(archive.read(metadata_name))
            parsed_requirements = [
                Requirement(requirement)
                for requirement in metadata.get_all("Requires-Dist", [])
            ]

            self.assertLessEqual(EXPECTED_PACKAGE_MEMBERS, source_members)
            self.assertLessEqual(EXPECTED_PACKAGE_MEMBERS, wheel_members)
            self.assertIn("external benchmark data", readme.lower())
            self.assertIn("blockworld/examples", readme)
            self.assertEqual(metadata["Requires-Python"], ">=3.9")
            self.assertEqual(
                {
                    _requirement_key(requirement)
                    for requirement in parsed_requirements
                    if requirement.marker is None
                },
                REQUIRED_RUNTIME_DEPENDENCIES,
            )
            optional_requirements = {
                requirement.name.lower()
                for requirement in parsed_requirements
                if requirement.marker is not None
            }
            for dependency in OPTIONAL_PROVIDER_DEPENDENCIES:
                self.assertIn(dependency, optional_requirements)
            rnd_requirements = {
                _requirement_key(requirement)
                for requirement in parsed_requirements
                if requirement.marker is not None
                and requirement.marker.evaluate({"extra": "rnd"})
            }
            for requirement in (
                "di-engine==0.5.3",
                "easydict==1.13",
                "gym==0.25.1",
                "numpy==1.23.5",
                "werkzeug==2.0.3",
            ):
                self.assertIn(requirement, rnd_requirements)
            self.assertEqual(
                metadata["Home-page"],
                "https://github.com/xmu-rl-3dv/PlanU",
            )
            self.assertTrue(metadata["Author"])
            self.assertTrue(metadata["Author-email"])

            install_target = temporary_path / "installed"
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--no-deps",
                    "--target",
                    os.fspath(install_target),
                    os.fspath(wheel_archive),
                ],
                cwd=temporary_path,
                check=True,
                capture_output=True,
                text=True,
            )
            import_check = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-c",
                    """
import sys
import types

transformers = types.ModuleType("transformers")
transformers.StoppingCriteriaList = list
torch = types.ModuleType("torch")
tqdm = types.ModuleType("tqdm")
tqdm.tqdm = lambda items, **kwargs: items
tqdm.trange = lambda *args, **kwargs: range(*args)
requests = types.ModuleType("requests")
sys.modules["transformers"] = transformers
sys.modules["torch"] = torch
sys.modules["tqdm"] = tqdm
sys.modules["requests"] = requests
sys.path.insert(0, {!r})

import reasoners.algorithm.planU
import reasoners.visualization

assert callable(reasoners.visualization.main)
""".format(os.fspath(install_target)),
                ],
                cwd=temporary_path,
                capture_output=True,
                text=True,
            )
            if import_check.returncode:
                missing_module = None
                for line in reversed(import_check.stderr.splitlines()):
                    marker = "No module named "
                    if marker in line:
                        missing_module = line.split(marker, 1)[1].strip("'\"")
                        break
                self.assertIn(
                    missing_module,
                    DECLARED_IMPORT_ROOTS,
                    import_check.stderr,
                )


if __name__ == "__main__":
    unittest.main()
