import re
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
CREDENTIAL_PATTERN = re.compile(r"sk-[A-Za-z0-9_-]{16,}")
BYTE_CREDENTIAL_PATTERN = re.compile(rb"sk-[A-Za-z0-9_-]{16,}")


@pytest.mark.parametrize("relative_path", ["webshop/models.py", "webshop/planu.py"])
def test_webshop_source_contains_no_credential_literals(relative_path: str) -> None:
    source = (REPO_ROOT / relative_path).read_text(encoding="utf-8")

    assert CREDENTIAL_PATTERN.search(source) is None


def test_tracked_webshop_files_contain_no_credential_literals() -> None:
    result = subprocess.run(
        ["git", "ls-files", "-z", "--", "webshop"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        timeout=30,
    )
    tracked_paths = [
        path.decode("utf-8") for path in result.stdout.split(b"\0") if path
    ]

    offenders = [
        relative_path
        for relative_path in tracked_paths
        if BYTE_CREDENTIAL_PATTERN.search(
            (REPO_ROOT / relative_path).read_bytes()
        )
    ]

    assert offenders == []
