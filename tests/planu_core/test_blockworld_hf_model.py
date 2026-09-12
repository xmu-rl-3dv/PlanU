import importlib.util
from pathlib import Path
import sys
import types


REPO_ROOT = Path(__file__).resolve().parents[2]
HF_MODEL_PATH = REPO_ROOT / "blockworld/reasoners/lm/hf_model.py"


def test_hf_model_honors_explicit_cpu_device(monkeypatch):
    load_calls = []

    class FakeTokenizer:
        pad_token_id = None
        bos_token_id = None
        eos_token_id = None

    class FakeModel:
        def __init__(self):
            self.config = types.SimpleNamespace()

        def eval(self):
            return self

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return FakeTokenizer()

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            load_calls.append(kwargs)
            return FakeModel()

    transformers = types.ModuleType("transformers")
    transformers.AutoModelForCausalLM = FakeAutoModel
    transformers.AutoTokenizer = FakeAutoTokenizer
    transformers.GenerationConfig = object
    transformers.BitsAndBytesConfig = object
    transformers.AutoConfig = object

    torch = types.ModuleType("torch")
    torch.float16 = object()
    torch.bfloat16 = object()
    torch.no_grad = lambda function=None: (
        (lambda decorated: decorated)
        if function is None
        else function
    )

    peft = types.ModuleType("peft")
    peft.PeftModel = object
    accelerate = types.ModuleType("accelerate")
    accelerate.infer_auto_device_map = lambda *args, **kwargs: {}
    accelerate.dispatch_model = lambda model, **kwargs: model

    reasoners = types.ModuleType("reasoners")
    reasoners.__path__ = []
    reasoners.LanguageModel = object
    reasoners.GenerateOutput = object
    reasoners_lm = types.ModuleType("reasoners.lm")
    reasoners_lm.__path__ = []

    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "peft", peft)
    monkeypatch.setitem(sys.modules, "accelerate", accelerate)
    monkeypatch.setitem(sys.modules, "reasoners", reasoners)
    monkeypatch.setitem(sys.modules, "reasoners.lm", reasoners_lm)

    module_name = "reasoners.lm._hf_model_device_test"
    spec = importlib.util.spec_from_file_location(module_name, HF_MODEL_PATH)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)

    module.HFModel("tiny-model", "tiny-model", device="cpu")

    assert load_calls == [
        {
            "device_map": {"": "cpu"},
            "trust_remote_code": True,
        }
    ]
