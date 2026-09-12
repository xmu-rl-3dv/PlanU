from .blockworld import (
    BlockWorldAdapter,
    BlockWorldScorer,
    blockworld_config,
)
from .overcooked import (
    OvercookedAdapter,
    describe_overcooked_observation,
    overcooked_config,
)
from .virtualhome import (
    VirtualHomeAdapter,
    VirtualHomeTask,
    describe_virtualhome_observation,
    virtualhome_config,
)


def __getattr__(name):
    if name in {"WebShopAdapter", "WebShopRuntime", "webshop_config"}:
        from .webshop import WebShopAdapter, WebShopRuntime, webshop_config

        exports = {
            "WebShopAdapter": WebShopAdapter,
            "WebShopRuntime": WebShopRuntime,
            "webshop_config": webshop_config,
        }
        return exports[name]
    raise AttributeError("module {!r} has no attribute {!r}".format(
        __name__,
        name,
    ))

__all__ = [
    "BlockWorldAdapter",
    "BlockWorldScorer",
    "OvercookedAdapter",
    "VirtualHomeAdapter",
    "VirtualHomeTask",
    "WebShopAdapter",
    "WebShopRuntime",
    "blockworld_config",
    "describe_overcooked_observation",
    "describe_virtualhome_observation",
    "overcooked_config",
    "virtualhome_config",
    "webshop_config",
]
