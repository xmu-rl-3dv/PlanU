from typing import Any

from planu_core.adapters.virtualhome import (
    VirtualHomeAdapter,
    VirtualHomeTask,
    describe_virtualhome_observation,
    virtualhome_config,
)
from planu_core.nodes import ActionNode, LanguageNode
from planu_core.scorers import ConstantActionScorer
from planu_core.search import PlanUSearch


LLM = False
LLMAgent = ConstantActionScorer
VirtualHomeActionScorer = ConstantActionScorer


def obs2text(observation: Any):
    prompt, actions = describe_virtualhome_observation(
        observation,
        VirtualHomeTask.FOOD,
    )
    return {
        "prompt": prompt,
        "action": [text for _, text in actions],
    }


__all__ = [
    "ActionNode",
    "ConstantActionScorer",
    "LLM",
    "LLMAgent",
    "LanguageNode",
    "PlanUSearch",
    "VirtualHomeActionScorer",
    "VirtualHomeAdapter",
    "VirtualHomeTask",
    "describe_virtualhome_observation",
    "obs2text",
    "virtualhome_config",
]
