import sys
from pathlib import Path

_REPOSITORY_ROOT = str(Path(__file__).resolve().parents[2])
if _REPOSITORY_ROOT not in sys.path:
    sys.path.insert(0, _REPOSITORY_ROOT)

from planu_core.base_reward_model import BaseRewardModel

__all__ = ["BaseRewardModel"]
