import re
from dataclasses import dataclass
from typing import Tuple


_SUPPORTED_KINDS = frozenset(("search", "click", "think"))
_ACTION_PATTERN = re.compile(
    r"(?P<kind>search|click|think)\[(?P<argument>[^\[\]]+)\]",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class WebShopAction:
    kind: str
    argument: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str):
            raise ValueError("WebShop action kind must be a string")
        kind = self.kind.lower()
        if kind not in _SUPPORTED_KINDS:
            raise ValueError("unsupported WebShop action kind: {!r}".format(self.kind))

        if not isinstance(self.argument, str):
            raise ValueError("WebShop action argument must be a string")
        argument = " ".join(self.argument.split())
        if not argument:
            raise ValueError("WebShop action argument cannot be blank")

        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "argument", argument)

    @property
    def key(self) -> Tuple[str, str]:
        return self.kind, self.argument

    def render(self) -> str:
        return "{}[{}]".format(self.kind, self.argument)


def parse_action(value: str) -> WebShopAction:
    if not isinstance(value, str):
        raise ValueError("WebShop action must be a string")

    match = _ACTION_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError("invalid WebShop action: {!r}".format(value))

    return WebShopAction(match.group("kind"), match.group("argument"))


__all__ = ["WebShopAction", "parse_action"]
