from importlib import import_module


_MODEL_MODULES = {
    "HFModel": ".hf_model",
    "LlamaModel": ".llama_model",
    "Llama2Model": ".llama_2_model",
    "Llama3Model": ".llama_3_model",
    "LlamaCppModel": ".llama_cpp_model",
    "OpenAIModel": ".openai_model",
    "ExLlamaModel": ".exllama_model",
    "BardCompletionModel": ".gemini_model",
    "ClaudeModel": ".anthropic_model",
}

__all__ = list(_MODEL_MODULES)


def __getattr__(name):
    module_name = _MODEL_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    model = getattr(import_module(module_name, __name__), name)
    globals()[name] = model
    return model
