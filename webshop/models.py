"""Compatibility exports for the shared, lazily constructed text backend."""

from planu_core.text_backend import (
    DEFAULT_OPENAI_BASE_URL,
    GenerationResult,
    OpenAICompatibleBackend,
    TextBackend,
)


__all__ = [
    "DEFAULT_OPENAI_BASE_URL",
    "GenerationResult",
    "OpenAICompatibleBackend",
    "TextBackend",
]
