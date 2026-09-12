from pathlib import Path
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
EVALUATE_PATH = REPO_ROOT / "blockworld" / "evaluate_stochastic.py"


@pytest.mark.parametrize(
    ("gpu_flag", "gpu_id"),
    [
        ("-g", "2"),
        ("--gpu", "5"),
    ],
)
def test_cuda_visibility_is_set_before_reasoners_and_torch_import(
    gpu_flag,
    gpu_id,
):
    script = """
import builtins
import importlib.util
import os
import sys
import types

expected_gpu = {gpu_id!r}
os.environ.pop("CUDA_VISIBLE_DEVICES", None)
sys.argv = [
    {evaluate_path!r},
    {gpu_flag!r},
    expected_gpu,
    "--algorithm",
    "planu",
]

reasoners = types.ModuleType("reasoners")
reasoners.__path__ = []
reasoners.LanguageModel = object
reasoners.Reasoner = object
reasoners.SearchConfig = object
reasoners.WorldModel = object
algorithm = types.ModuleType("reasoners.algorithm")
algorithm.MCTS = object
algorithm.PlanU = object
benchmark = types.ModuleType("reasoners.benchmark")
benchmark.__path__ = []
benchmark.BWEvaluator = object
bw_utils = types.ModuleType("reasoners.benchmark.bw_utils")
lm = types.ModuleType("reasoners.lm")
lm.ExLlamaModel = object
lm.HFModel = object
torch = types.ModuleType("torch")

sys.modules["reasoners"] = reasoners
sys.modules["reasoners.algorithm"] = algorithm
sys.modules["reasoners.benchmark"] = benchmark
sys.modules["reasoners.benchmark.bw_utils"] = bw_utils
sys.modules["reasoners.lm"] = lm
sys.modules["torch"] = torch

checked_imports = []
original_import = builtins.__import__
def checking_import(name, *args, **kwargs):
    if name == "torch" or name == "reasoners" or name.startswith("reasoners."):
        actual = os.environ.get("CUDA_VISIBLE_DEVICES")
        assert actual == expected_gpu, (name, actual, expected_gpu)
        checked_imports.append(name)
    return original_import(name, *args, **kwargs)

builtins.__import__ = checking_import
try:
    spec = importlib.util.spec_from_file_location(
        "_blockworld_cuda_order_test",
        {evaluate_path!r},
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
finally:
    builtins.__import__ = original_import

assert "reasoners" in checked_imports
assert "torch" in checked_imports
assert os.environ["CUDA_VISIBLE_DEVICES"] == expected_gpu
""".format(
        evaluate_path=str(EVALUATE_PATH),
        gpu_flag=gpu_flag,
        gpu_id=gpu_id,
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
