from typing import Any, Optional

import numpy as np

from planu_core.adapters.virtualhome import (
    VirtualHomeAdapter,
    VirtualHomeTask,
    describe_virtualhome_observation,
    virtualhome_config,
)
from planu_core.nodes import ActionNode, LanguageNode
from planu_core.scorers import HuggingFaceActionScorer
from planu_core.search import PlanUSearch


LLM = True
prompt_disturb = True
shuffle_prompt = False
LLMAgent = HuggingFaceActionScorer
VirtualHomeActionScorer = HuggingFaceActionScorer


def obs2text(
    observation: Any,
    rng: Optional[np.random.Generator] = None,
):
    prompt, actions = describe_virtualhome_observation(
        observation,
        VirtualHomeTask.ENTERTAINMENT,
        prompt_disturb=prompt_disturb,
        prompt_shuffle=shuffle_prompt,
        rng=rng,
    )
    return {
        "prompt": prompt,
        "action": [text for _, text in actions],
    }


__all__ = [
    "ActionNode",
    "HuggingFaceActionScorer",
    "LLM",
    "LLMAgent",
    "LanguageNode",
    "PlanUSearch",
    "VirtualHomeActionScorer",
    "VirtualHomeAdapter",
    "VirtualHomeTask",
    "describe_virtualhome_observation",
    "obs2text",
    "prompt_disturb",
    "shuffle_prompt",
    "virtualhome_config",
]
