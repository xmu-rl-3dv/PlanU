"""Compatibility exports for the retired private WebShop PlanU module.

New experiments should use :mod:`planu_core.webshop.runner`. The legacy
prompt constants remain available for notebooks that imported this module.
"""

from planu_core.adapters.webshop import WebShopAdapter, webshop_config
from planu_core.search import PlanUSearch
from planu_core.webshop.providers import (
    LEGACY_COT_PROMPT,
    LEGACY_VALUE_PROMPT,
    ModelWebShopActionProvider,
    ModelWebShopActionScorer,
    ScriptedWebShopActionProvider,
)


prompt2 = LEGACY_COT_PROMPT
score_prompt = LEGACY_VALUE_PROMPT


__all__ = [
    "LEGACY_COT_PROMPT",
    "LEGACY_VALUE_PROMPT",
    "ModelWebShopActionProvider",
    "ModelWebShopActionScorer",
    "PlanUSearch",
    "ScriptedWebShopActionProvider",
    "WebShopAdapter",
    "prompt2",
    "score_prompt",
    "webshop_config",
]
