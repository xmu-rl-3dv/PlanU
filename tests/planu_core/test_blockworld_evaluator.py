import importlib.util
from pathlib import Path
import sys
import types


REPO_ROOT = Path(__file__).resolve().parents[2]
EVALUATOR_PATH = (
    REPO_ROOT / "blockworld" / "reasoners" / "benchmark" / "blocksworld.py"
)


def _load_evaluator(monkeypatch):
    datasets = types.ModuleType("datasets")
    tqdm_module = types.ModuleType("tqdm")
    tqdm_module.tqdm = lambda items, **kwargs: items
    torch = types.ModuleType("torch")
    torch.distributed = types.SimpleNamespace(
        is_initialized=lambda: False,
    )

    reasoners = types.ModuleType("reasoners")
    reasoners.__path__ = []
    reasoners.Evaluator = object
    benchmark = types.ModuleType("reasoners.benchmark")
    benchmark.__path__ = []
    bw_utils = types.ModuleType("reasoners.benchmark.bw_utils")
    bw_utils.read_config = lambda path: {
        "encoded_objects": {
            "a": "red block",
            "b": "blue block",
        }
    }

    monkeypatch.setitem(sys.modules, "datasets", datasets)
    monkeypatch.setitem(sys.modules, "tqdm", tqdm_module)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "reasoners", reasoners)
    monkeypatch.setitem(sys.modules, "reasoners.benchmark", benchmark)
    monkeypatch.setitem(
        sys.modules,
        "reasoners.benchmark.bw_utils",
        bw_utils,
    )

    module_name = "_blockworld_evaluator_test"
    spec = importlib.util.spec_from_file_location(
        module_name,
        EVALUATOR_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module.BWEvaluator


def test_eval_output_new_parses_realistic_on_top_of_goal(monkeypatch):
    evaluator_type = _load_evaluator(monkeypatch)
    evaluator = evaluator_type.__new__(evaluator_type)
    evaluator.config_file = "unused-test-config.yaml"

    final_state = (
        "the red block is on top of the blue block, "
        "the red block is clear and the blue block is on the table."
    )
    goal = "the red block is on top of the blue block."

    assert evaluator.eval_output_new(final_state, goal) is True
