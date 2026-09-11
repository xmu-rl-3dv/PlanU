import ast
from pathlib import Path
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
BLOCKWORLD_ROOT = REPO_ROOT / "blockworld"


def _has_future_annotations(tree):
    return any(
        isinstance(node, ast.ImportFrom)
        and node.module == "__future__"
        and any(alias.name == "annotations" for alias in node.names)
        for node in tree.body
    )


def _annotation_nodes(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.arg) and node.annotation is not None:
            yield node.annotation
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.returns is not None:
                yield node.returns
        elif isinstance(node, ast.AnnAssign):
            yield node.annotation


def test_tracked_blockworld_annotations_are_python39_runtime_safe():
    tracked = subprocess.check_output(
        ["git", "ls-files", "blockworld/**/*.py"],
        cwd=str(REPO_ROOT),
        text=True,
    ).splitlines()
    unsafe = []
    for relative_path in tracked:
        path = REPO_ROOT / relative_path
        tree = ast.parse(path.read_text(), filename=str(path))
        if _has_future_annotations(tree):
            continue
        if any(
            isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr)
            for annotation in _annotation_nodes(tree)
            for node in ast.walk(annotation)
        ):
            unsafe.append(relative_path)

    assert unsafe == []


def test_blockworld_package_declares_python39_support():
    tree = ast.parse(
        (BLOCKWORLD_ROOT / "setup.py").read_text(),
        filename="blockworld/setup.py",
    )
    setup_call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "setup"
    )
    python_requires = next(
        keyword.value
        for keyword in setup_call.keywords
        if keyword.arg == "python_requires"
    )

    assert ast.literal_eval(python_requires) == ">=3.9"


@pytest.mark.skipif(
    sys.version_info[:2] != (3, 9),
    reason="requires the Python 3.9 test environment",
)
def test_reasoners_algorithm_imports_under_python39():
    script = """
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

from reasoners.algorithm import MCTS, PlanU
assert MCTS.__name__ == "MCTS"
assert PlanU.__name__ == "PlanU"
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(REPO_ROOT),
        env={
            "PATH": str(Path(sys.executable).parent),
            "PYTHONPATH": "{}:{}".format(
                BLOCKWORLD_ROOT,
                REPO_ROOT,
            ),
        },
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
