import math
from typing import Any, Optional, Sequence

import numpy as np

from .interfaces import ActionCandidate


LEGACY_BASE_MODEL = "Neko-Institute-of-Science/LLaMA-7B-HF"
LEGACY_DISTRIBUTION_CRITERIA = (
    "very low",
    "somewhat low",
    "medium level",
    "somewhat high",
    "very high",
)
_NORMALIZATION_MODES = {"token", "word", "sum"}


def _device_map_for(device: str):
    resolved_device = str(device)
    if resolved_device.startswith("cuda"):
        return "auto"
    return {"": resolved_device}


def normalize_action_scores(
    log_likelihoods: Sequence[float],
    token_lengths: Sequence[int],
    actions: Sequence[str],
    normalization_mode: str,
    temperature: float,
) -> np.ndarray:
    if normalization_mode not in _NORMALIZATION_MODES:
        raise ValueError("normalization_mode must be token, word, or sum")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be a finite positive number")

    scores = np.asarray(log_likelihoods, dtype=np.float64)
    token_counts = np.asarray(token_lengths, dtype=np.float64)
    if scores.ndim != 1:
        raise ValueError("log_likelihoods must be one-dimensional")
    if token_counts.shape != scores.shape or len(actions) != len(scores):
        raise ValueError("scores, token lengths, and actions must align")
    if not np.all(np.isfinite(scores)):
        raise ValueError("log likelihoods must be finite")

    if normalization_mode == "token":
        denominators = token_counts
    elif normalization_mode == "word":
        denominators = np.asarray(
            [len(action.split()) for action in actions],
            dtype=np.float64,
        )
    else:
        denominators = np.ones_like(scores)
    if np.any(denominators <= 0):
        raise ValueError("normalization lengths must be positive")

    normalized = scores / denominators / temperature
    normalized -= np.max(normalized)
    probabilities = np.exp(normalized)
    total = probabilities.sum()
    if not math.isfinite(float(total)) or total <= 0.0:
        raise ValueError("action probabilities could not be normalized")
    return probabilities / total


class HuggingFaceActionScorer:
    def __init__(
        self,
        base_model: Optional[str],
        normalization_mode: str = "token",
        temperature: float = 1.0,
        tokenizer: Any = None,
        model: Any = None,
        device: Optional[str] = None,
        torch_dtype: Any = None,
        distribution_prompt: Optional[str] = None,
        log_likelihood_helper: Any = None,
    ) -> None:
        if normalization_mode not in _NORMALIZATION_MODES:
            raise ValueError("normalization_mode must be token, word, or sum")
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("temperature must be a finite positive number")

        self.base_model = (
            LEGACY_BASE_MODEL if base_model is None else base_model
        )
        self.normalization_mode = normalization_mode
        self.temperature = float(temperature)
        self.device = device
        self.torch_dtype = torch_dtype
        if distribution_prompt is not None and not isinstance(
            distribution_prompt,
            str,
        ):
            raise TypeError("distribution_prompt must be a string")
        if log_likelihood_helper is not None and not callable(
            log_likelihood_helper
        ):
            raise TypeError("log_likelihood_helper must be callable")
        self.distribution_prompt = distribution_prompt or ""
        self.log_likelihood_helper = log_likelihood_helper
        self.total_llm_tokenizer_token = 0
        self.total_llm_tokenizer_call = 0

        if tokenizer is None or model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(self.base_model)
                tokenizer.pad_token_id = 0
            if model is None:
                model_kwargs = {
                    "device_map": _device_map_for(self.device),
                }
                if self.torch_dtype is not None:
                    model_kwargs["torch_dtype"] = self.torch_dtype
                model = AutoModelForCausalLM.from_pretrained(
                    self.base_model,
                    **model_kwargs,
                )

        self.tokenizer = tokenizer
        self.model = model
        if self.device is None:
            try:
                self.device = str(next(model.parameters()).device)
            except (AttributeError, StopIteration, TypeError):
                self.device = "cpu"

    @staticmethod
    def _shared_prompt(candidates: Sequence[ActionCandidate]) -> str:
        prompts = [candidate.metadata.get("prompt") for candidate in candidates]
        if any(prompt is None for prompt in prompts):
            raise ValueError("every candidate must include prompt metadata")
        if len(set(prompts)) != 1:
            raise ValueError("all candidates must share the same prompt")
        return prompts[0]

    def _teacher_forced_log_likelihoods(
        self,
        prefix: str,
        completions: Sequence[str],
    ):
        sequences = [prefix + completion for completion in completions]

        import torch

        inputs = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
        )
        completion_inputs = self.tokenizer(
            completions,
            return_tensors="pt",
            padding=True,
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        completion_lengths = (
            torch.sum(completion_inputs["attention_mask"], dim=-1) - 1
        )
        completion_lengths_on_device = completion_lengths.to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
        token_count = int(outputs.logits.shape[0]) * int(outputs.logits.shape[1])

        sequence_lengths = torch.sum(attention_mask, dim=-1)
        completion_starts = (
            sequence_lengths - completion_lengths_on_device
        ).detach().cpu().tolist()
        completion_ends = sequence_lengths.detach().cpu().tolist()
        token_log_probs = torch.log_softmax(outputs.logits, dim=-1)
        token_log_probs = token_log_probs[:, :-1, :]
        shifted_ids = input_ids[:, 1:]
        generated_log_probs = torch.gather(
            token_log_probs,
            2,
            shifted_ids[:, :, None],
        ).squeeze(-1)
        completion_slices = [
            generated_log_probs[index, start - 1 : end - 1]
            for index, (start, end) in enumerate(
                zip(completion_starts, completion_ends)
            )
        ]
        log_likelihoods = [
            float(completion_slice.sum().detach().cpu().item())
            for completion_slice in completion_slices
        ]
        lengths = [
            int(length.detach().cpu().item())
            for length in completion_lengths
        ]
        return log_likelihoods, lengths, token_count

    def _normalized_completion_scores(
        self,
        prefix: str,
        completions: Sequence[str],
    ) -> np.ndarray:
        helper = (
            self._teacher_forced_log_likelihoods
            if self.log_likelihood_helper is None
            else self.log_likelihood_helper
        )
        result = helper(prefix, completions)
        if not isinstance(result, tuple) or len(result) not in (2, 3):
            raise ValueError(
                "log likelihood helper must return scores, lengths, "
                "and optionally token count"
            )
        log_likelihoods, lengths = result[:2]
        token_count = sum(lengths) if len(result) == 2 else result[2]
        self.total_llm_tokenizer_token += int(token_count)
        self.total_llm_tokenizer_call += 1
        return normalize_action_scores(
            log_likelihoods,
            lengths,
            completions,
            self.normalization_mode,
            self.temperature,
        )

    def score(
        self,
        observation: Any,
        candidates: Sequence[ActionCandidate],
    ) -> Sequence[float]:
        del observation
        if not candidates:
            return []

        prompt = self._shared_prompt(candidates)
        actions = [candidate.text for candidate in candidates]
        return self._normalized_completion_scores(prompt + " ", actions)

    def score_distributions(
        self,
        observation: Any,
        candidates: Sequence[ActionCandidate],
        levels: Sequence[float],
    ) -> Sequence[Sequence[float]]:
        del observation
        if len(levels) != len(LEGACY_DISTRIBUTION_CRITERIA):
            raise ValueError(
                "HuggingFaceActionScorer requires five categorical levels"
            )
        if not candidates:
            return []

        prompt = self._shared_prompt(candidates)
        rows = []
        for candidate in candidates:
            prefix = (
                self.distribution_prompt
                + prompt
                + candidate.text
                + "Evaluation: "
            )
            rows.append(
                self._normalized_completion_scores(
                    prefix,
                    LEGACY_DISTRIBUTION_CRITERIA,
                )
            )
        return rows


class ConstantActionScorer:
    def __init__(self, value: float = 1.0) -> None:
        try:
            resolved_value = float(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("value must be finite") from error
        if not math.isfinite(resolved_value):
            raise ValueError("value must be finite")
        self.value = resolved_value

    def score(
        self,
        observation: Any,
        candidates: Sequence[ActionCandidate],
    ) -> Sequence[float]:
        del observation
        return [self.value] * len(candidates)


OvercookedActionScorer = HuggingFaceActionScorer


__all__ = [
    "ConstantActionScorer",
    "HuggingFaceActionScorer",
    "LEGACY_BASE_MODEL",
    "LEGACY_DISTRIBUTION_CRITERIA",
    "OvercookedActionScorer",
    "_device_map_for",
    "normalize_action_scores",
]
