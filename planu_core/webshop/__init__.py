from .actions import WebShopAction, parse_action


def __getattr__(name):
    if name in {"WebShopAdapter", "WebShopRuntime", "webshop_config"}:
        from planu_core.adapters.webshop import (
            WebShopAdapter,
            WebShopRuntime,
            webshop_config,
        )

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
    "WebShopAction",
    "WebShopAdapter",
    "WebShopRuntime",
    "parse_action",
    "webshop_config",
]
