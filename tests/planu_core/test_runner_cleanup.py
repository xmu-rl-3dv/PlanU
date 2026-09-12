import importlib
import sys
import types

import pytest


RUNNER_MODULES = (
    "mcts.overcooked.PlanU_inference",
    "mcts.virtualhome.PlanU_inference_food",
    "mcts.virtualhome.PlanU_entertainment",
)


class RecordingWriter:
    instances = []
    fail_on_call = None

    def __init__(self, path):
        self.path = path
        self.closed = False
        self.__class__.instances.append(self)
        if len(self.__class__.instances) == self.__class__.fail_on_call:
            raise RuntimeError("writer construction failed")

    def add_scalar(self, *args, **kwargs):
        pass

    def add_text(self, *args, **kwargs):
        pass

    def close(self):
        self.closed = True


class FailingCloseEnvironment:
    def __init__(self, discrete_type):
        self.single_action_space = discrete_type()
        self.closed = False

    def close(self):
        self.closed = True
        raise RuntimeError("environment close failed")


def _install_runtime_modules(monkeypatch, writer_type, environment=None):
    tensorboard = types.ModuleType("torch.utils.tensorboard")
    tensorboard.SummaryWriter = writer_type
    torch_utils = types.ModuleType("torch.utils")
    torch_utils.tensorboard = tensorboard

    cudnn = types.SimpleNamespace(deterministic=False, benchmark=True)
    torch = types.ModuleType("torch")
    torch.manual_seed = lambda seed: None
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: False,
        manual_seed_all=lambda seed: None,
    )
    torch.backends = types.SimpleNamespace(cudnn=cudnn)
    torch.device = lambda value: value
    torch.utils = torch_utils

    class Discrete:
        pass

    gym = types.ModuleType("gym")
    gym.spaces = types.SimpleNamespace(Discrete=Discrete)
    gym.vector = types.SimpleNamespace(
        SyncVectorEnv=lambda factories: environment
    )

    monkeypatch.setitem(sys.modules, "gym", gym)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.utils", torch_utils)
    monkeypatch.setitem(sys.modules, "torch.utils.tensorboard", tensorboard)
    monkeypatch.setitem(
        sys.modules,
        "virtual_home",
        types.ModuleType("virtual_home"),
    )
    return Discrete


@pytest.mark.parametrize("module_name", RUNNER_MODULES)
def test_runner_closes_first_writer_when_second_writer_construction_fails(
    module_name,
    monkeypatch,
):
    runner = importlib.import_module(module_name)
    RecordingWriter.instances = []
    RecordingWriter.fail_on_call = 2
    _install_runtime_modules(monkeypatch, RecordingWriter)

    with pytest.raises(RuntimeError, match="writer construction failed"):
        runner.run(runner.parse_args(["--maxiterations", "1"]))

    assert len(RecordingWriter.instances) == 2
    assert RecordingWriter.instances[0].closed is True


@pytest.mark.parametrize("module_name", RUNNER_MODULES)
def test_runner_closes_writers_even_when_environment_close_fails(
    module_name,
    monkeypatch,
    tmp_path,
):
    runner = importlib.import_module(module_name)
    RecordingWriter.instances = []
    RecordingWriter.fail_on_call = None

    class Discrete:
        pass

    environment = FailingCloseEnvironment(Discrete)
    installed_discrete = _install_runtime_modules(
        monkeypatch,
        RecordingWriter,
        environment,
    )
    environment.single_action_space = installed_discrete()
    scorer = types.SimpleNamespace(
        total_llm_tokenizer_token=0,
        total_llm_tokenizer_call=0,
    )
    search = types.SimpleNamespace(
        run_iteration=lambda iteration, rng: types.SimpleNamespace(
            rewards=[],
            action_path=[],
        )
    )
    monkeypatch.setattr(
        runner,
        "build_planu_components",
        lambda *args, **kwargs: (search, scorer, args[-1]),
    )
    monkeypatch.chdir(tmp_path)

    with pytest.raises(RuntimeError, match="environment close failed"):
        runner.run(runner.parse_args(["--maxiterations", "1"]))

    assert environment.closed is True
    assert len(RecordingWriter.instances) == 2
    assert all(writer.closed for writer in RecordingWriter.instances)
