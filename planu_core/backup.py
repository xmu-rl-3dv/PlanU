import math
from typing import Iterable, List, Sequence

from .config import PlanUConfig
from .nodes import ActionNode


def suffix_returns(rewards: Sequence[float], discount: float) -> List[float]:
    discount = float(discount)
    if not math.isfinite(discount) or not 0.0 <= discount <= 1.0:
        raise ValueError("discount must be finite and between 0 and 1")

    try:
        reward_values = [float(reward) for reward in rewards]
    except (TypeError, ValueError) as error:
        raise ValueError("rewards must be finite") from error
    if not all(math.isfinite(reward) for reward in reward_values):
        raise ValueError("rewards must be finite")

    returns = [0.0] * len(reward_values)
    running = 0.0
    for index in range(len(reward_values) - 1, -1, -1):
        running = reward_values[index] + discount * running
        if not math.isfinite(running):
            raise ValueError("suffix returns must be finite")
        returns[index] = running
    return returns


def backup_trajectory(
    actions: Iterable[ActionNode],
    rewards: Sequence[float],
    config: PlanUConfig,
) -> List[float]:
    action_path = list(actions)
    if len(action_path) != len(rewards):
        raise ValueError("action path and reward lengths differ")

    returns = suffix_returns(rewards, config.discount)
    for action, target in zip(action_path, returns):
        action.visit_count += 1
        action.cumulative_returns.append(target)
        action.distribution.update_scalar(
            target,
            config.quantile_learning_rate,
        )
    return returns
