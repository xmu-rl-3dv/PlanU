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


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_PACKAGE_MEMBERS = {
    "planu_core/search.py",
    "reasoners/algorithm/planU.py",
}
DECLARED_IMPORT_ROOTS = {
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


class PackagingTest(unittest.TestCase):
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

            self.assertLessEqual(EXPECTED_PACKAGE_MEMBERS, source_members)
            self.assertLessEqual(EXPECTED_PACKAGE_MEMBERS, wheel_members)
            self.assertIn("external benchmark data", readme.lower())
            self.assertIn("blockworld/examples", readme)
            self.assertEqual(metadata["Requires-Python"], ">=3.9")

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
sys.modules["transformers"] = transformers
sys.modules["torch"] = torch
sys.modules["tqdm"] = tqdm
sys.path.insert(0, {!r})

import reasoners.algorithm.planU
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
