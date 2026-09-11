import math
from typing import Any, Optional, Sequence

import numpy as np

from planu_core.interfaces import ActionCandidate
from planu_core.nodes import ActionNode, LanguageNode
from planu_core.search import PlanUSearch


LEGACY_BASE_MODEL = "Neko-Institute-of-Science/LLaMA-7B-HF"
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


class OvercookedActionScorer:
    def __init__(
        self,
        base_model: Optional[str],
        normalization_mode: str = "token",
        temperature: float = 1.0,
        tokenizer: Any = None,
        model: Any = None,
        device: Optional[str] = None,
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
                model = AutoModelForCausalLM.from_pretrained(
                    self.base_model,
                    device_map=_device_map_for(self.device),
                )

        self.tokenizer = tokenizer
        self.model = model
        if self.device is None:
            try:
                self.device = str(next(model.parameters()).device)
            except (AttributeError, StopIteration, TypeError):
                self.device = "cpu"

    def score(
        self,
        observation: Any,
        candidates: Sequence[ActionCandidate],
    ) -> Sequence[float]:
        del observation
        if not candidates:
            return []

        prompts = [candidate.metadata.get("prompt") for candidate in candidates]
        if any(prompt is None for prompt in prompts):
            raise ValueError("every candidate must include prompt metadata")
        if len(set(prompts)) != 1:
            raise ValueError("all candidates must share the same prompt")

        actions = [candidate.text for candidate in candidates]
        prompt = prompts[0]
        sequences = [prompt + " " + action for action in actions]

        import torch

        inputs = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
        )
        action_inputs = self.tokenizer(
            actions,
            return_tensors="pt",
            padding=True,
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        action_lengths = (
            torch.sum(action_inputs["attention_mask"], dim=-1) - 1
        )
        action_lengths_on_device = action_lengths.to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
        self.total_llm_tokenizer_token += (
            int(outputs.logits.shape[0]) * int(outputs.logits.shape[1])
        )
        self.total_llm_tokenizer_call += 1

        sequence_lengths = torch.sum(attention_mask, dim=-1)
        action_starts = (
            sequence_lengths - action_lengths_on_device
        ).detach().cpu().tolist()
        action_ends = sequence_lengths.detach().cpu().tolist()
        token_log_probs = torch.log_softmax(outputs.logits, dim=-1)
        token_log_probs = token_log_probs[:, :-1, :]
        shifted_ids = input_ids[:, 1:]
        generated_log_probs = torch.gather(
            token_log_probs,
            2,
            shifted_ids[:, :, None],
        ).squeeze(-1)
        action_slices = [
            generated_log_probs[index, start - 1 : end - 1]
            for index, (start, end) in enumerate(zip(action_starts, action_ends))
        ]
        log_likelihoods = [
            float(action_slice.sum().detach().cpu().item())
            for action_slice in action_slices
        ]
        lengths = [
            int(length.detach().cpu().item())
            for length in action_lengths
        ]
        return normalize_action_scores(
            log_likelihoods,
            lengths,
            actions,
            self.normalization_mode,
            self.temperature,
        )


__all__ = [
    "ActionNode",
    "LanguageNode",
    "OvercookedActionScorer",
    "PlanUSearch",
    "normalize_action_scores",
]
