import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
CREDENTIAL_PATTERN = re.compile(r"sk-[A-Za-z0-9_-]{16,}")


@pytest.mark.parametrize("relative_path", ["webshop/models.py", "webshop/planu.py"])
def test_webshop_source_contains_no_credential_literals(relative_path: str) -> None:
    source = (REPO_ROOT / relative_path).read_text(encoding="utf-8")

    assert CREDENTIAL_PATTERN.search(source) is None
