import re
from dataclasses import dataclass
from typing import Tuple


_SUPPORTED_KINDS = frozenset(("search", "click", "think"))
_ACTION_PATTERN = re.compile(
    r"(?P<kind>search|click|think)\[(?P<argument>[^\[\]\r\n]+)\]",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class WebShopAction:
    kind: str
    argument: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str):
            raise ValueError("WebShop action kind must be a string")
        kind = self.kind.strip().lower()
        if kind not in _SUPPORTED_KINDS:
            raise ValueError("unsupported WebShop action kind: {!r}".format(self.kind))

        if not isinstance(self.argument, str):
            raise ValueError("WebShop action argument must be a string")
        if any(character in self.argument for character in "[]\r\n"):
            raise ValueError(
                "WebShop action argument cannot contain brackets or line breaks"
            )
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


def parse_action(text: str) -> WebShopAction:
    if not isinstance(text, str):
        raise ValueError("WebShop action must be a string")

    text = text.strip()
    match = _ACTION_PATTERN.fullmatch(text)
    if match is None:
        raise ValueError("invalid WebShop action: {!r}".format(text))

    return WebShopAction(match.group("kind"), match.group("argument"))


__all__ = ["WebShopAction", "parse_action"]
