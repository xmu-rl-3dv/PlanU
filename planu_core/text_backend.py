import importlib
import math
import os
import time
from dataclasses import dataclass
from numbers import Real
from typing import Any, Callable, Dict, Optional, Protocol, Sequence


DEFAULT_OPENAI_BASE_URL = (
    "https://dashscope.aliyuncs.com/compatible-mode/v1"
)


@dataclass(frozen=True)
class GenerationResult:
    texts: Sequence[str]
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.texts, (str, bytes)):
            raise ValueError("generation texts must be a sequence of strings")
        texts = tuple(self.texts)
        if not all(isinstance(text, str) for text in texts):
            raise ValueError("generation texts must contain only strings")
        for name in ("prompt_tokens", "completion_tokens"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(
                    "{} must be a nonnegative integer".format(name)
                )
        object.__setattr__(self, "texts", texts)


class TextBackend(Protocol):
    @property
    def model_identifier(self) -> str:
        ...

    def generate(
        self,
        prompt: str,
        n: int,
        temperature: float,
        max_tokens: int,
        stop: Sequence[str],
    ) -> GenerationResult:
        ...


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


class OpenAICompatibleBackend:
    def __init__(
        self,
        model: str,
        base_url: Optional[str] = DEFAULT_OPENAI_BASE_URL,
        timeout: float = 30.0,
        retry_limit: int = 2,
        retry_delay: float = 0.5,
        client_factory: Optional[Callable[..., Any]] = None,
        max_batch_size: int = 4,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")
        if base_url is not None and (
            not isinstance(base_url, str) or not base_url.strip()
        ):
            raise ValueError("base_url must be None or a non-empty string")
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, Real)
            or not math.isfinite(float(timeout))
            or timeout <= 0
        ):
            raise ValueError("timeout must be a finite positive number")
        if (
            isinstance(retry_limit, bool)
            or not isinstance(retry_limit, int)
            or retry_limit < 0
        ):
            raise ValueError("retry_limit must be a nonnegative integer")
        if (
            isinstance(retry_delay, bool)
            or not isinstance(retry_delay, Real)
            or not math.isfinite(float(retry_delay))
            or retry_delay < 0
        ):
            raise ValueError("retry_delay must be a finite nonnegative number")
        if client_factory is not None and not callable(client_factory):
            raise ValueError("client_factory must be callable")
        if (
            isinstance(max_batch_size, bool)
            or not isinstance(max_batch_size, int)
            or max_batch_size <= 0
        ):
            raise ValueError("max_batch_size must be a positive integer")

        self._model = model.strip()
        self.base_url = base_url.strip() if base_url is not None else None
        self.timeout = float(timeout)
        self.retry_limit = retry_limit
        self.retry_delay = float(retry_delay)
        self.max_batch_size = max_batch_size
        self._client_factory = client_factory
        self._client = None
        self._openai_module = None
        self.prompt_tokens = 0
        self.completion_tokens = 0

    @property
    def model_identifier(self) -> str:
        return self._model

    @property
    def token_usage(self) -> Dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
        }

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is required for the configured text backend"
            )

        openai_module = importlib.import_module("openai")
        factory = self._client_factory
        if factory is None:
            factory = openai_module.OpenAI
        kwargs = {
            "api_key": api_key,
            "timeout": self.timeout,
            # This class owns retry policy so the SDK cannot multiply retries.
            "max_retries": 0,
        }
        if self.base_url is not None:
            kwargs["base_url"] = self.base_url
        self._client = factory(**kwargs)
        self._openai_module = openai_module
        return self._client

    def _is_transient(self, error: BaseException) -> bool:
        module = self._openai_module
        transient_types = [TimeoutError, ConnectionError]
        if module is not None:
            for name in (
                "APIConnectionError",
                "APITimeoutError",
                "RateLimitError",
                "InternalServerError",
            ):
                error_type = getattr(module, name, None)
                if isinstance(error_type, type):
                    transient_types.append(error_type)
        if isinstance(error, tuple(transient_types)):
            return True

        status_code = getattr(error, "status_code", None)
        return (
            isinstance(status_code, int)
            and (
                status_code in (408, 409, 429)
                or 500 <= status_code <= 599
            )
        )

    @staticmethod
    def _usage_count(usage: Any, name: str) -> int:
        raw_value = _field(usage, name, 0)
        if raw_value is None:
            return 0
        if (
            isinstance(raw_value, bool)
            or not isinstance(raw_value, int)
            or raw_value < 0
        ):
            raise ValueError(
                "OpenAI response {} must be a nonnegative integer".format(
                    name
                )
            )
        return raw_value

    @staticmethod
    def _response_texts(response: Any) -> Sequence[str]:
        choices = _field(response, "choices")
        if choices is None:
            raise ValueError("OpenAI response is missing choices")

        texts = []
        for choice in choices:
            message = _field(choice, "message")
            content = _field(message, "content")
            if not isinstance(content, str):
                raise ValueError(
                    "OpenAI response choice is missing text content"
                )
            texts.append(content)
        return texts

    def _create_completion(
        self,
        client: Any,
        request: Dict[str, Any],
    ) -> Any:
        for attempt in range(self.retry_limit + 1):
            try:
                return client.chat.completions.create(**request)
            except Exception as error:
                if (
                    attempt >= self.retry_limit
                    or not self._is_transient(error)
                ):
                    raise
                if self.retry_delay:
                    time.sleep(self.retry_delay * (2 ** attempt))
        raise RuntimeError("OpenAI completion retries exhausted")

    def generate(
        self,
        prompt: str,
        n: int,
        temperature: float,
        max_tokens: int,
        stop: Sequence[str],
    ) -> GenerationResult:
        if not isinstance(prompt, str):
            raise ValueError("prompt must be a string")
        if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
            raise ValueError("n must be a positive integer")
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, Real)
            or not math.isfinite(float(temperature))
            or temperature < 0
        ):
            raise ValueError(
                "temperature must be a finite nonnegative number"
            )
        if (
            isinstance(max_tokens, bool)
            or not isinstance(max_tokens, int)
            or max_tokens <= 0
        ):
            raise ValueError("max_tokens must be a positive integer")
        if isinstance(stop, (str, bytes)):
            raise ValueError("stop must be a sequence of non-empty strings")
        stop_items = tuple(stop)
        if not all(
            isinstance(item, str) and item for item in stop_items
        ):
            raise ValueError("stop must be a sequence of non-empty strings")

        client = self._get_client()
        request = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": float(temperature),
            "max_tokens": max_tokens,
            "stop": list(stop_items) or None,
        }
        texts = []
        prompt_tokens = 0
        completion_tokens = 0
        for start in range(0, n, self.max_batch_size):
            batch_size = min(self.max_batch_size, n - start)
            response = self._create_completion(
                client,
                {**request, "n": batch_size},
            )
            texts.extend(self._response_texts(response))
            usage = _field(response, "usage")
            prompt_tokens += self._usage_count(usage, "prompt_tokens")
            completion_tokens += self._usage_count(
                usage,
                "completion_tokens",
            )
        generated = GenerationResult(
            texts,
            prompt_tokens,
            completion_tokens,
        )
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        return generated


__all__ = [
    "DEFAULT_OPENAI_BASE_URL",
    "GenerationResult",
    "OpenAICompatibleBackend",
    "TextBackend",
]
