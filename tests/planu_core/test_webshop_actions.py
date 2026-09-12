from dataclasses import FrozenInstanceError

import pytest

from planu_core.webshop import WebShopAction, parse_action
from planu_core.webshop.actions import parse_action as parse_action_from_actions


@pytest.mark.parametrize("kind", ["search", "click", "think"])
def test_supported_actions_render_and_expose_stable_keys(kind):
    action = WebShopAction(kind, "product details")

    assert action.key == (kind, "product details")
    assert action.render() == "{}[product details]".format(kind)


def test_action_normalizes_kind_and_argument_whitespace():
    action = WebShopAction(" ClIcK ", "  Buy \n\t Now  ")

    assert action == WebShopAction("click", "Buy Now")
    assert action.key == ("click", "Buy Now")
    assert action.render() == "click[Buy Now]"


@pytest.mark.parametrize(
    ("kind", "argument"),
    [
        ("open", "item"),
        ("click", ""),
        ("think", " \n\t "),
    ],
)
def test_action_rejects_unsupported_kinds_and_blank_arguments(kind, argument):
    with pytest.raises(ValueError):
        WebShopAction(kind, argument)


def test_action_is_hashable_and_immutable():
    action = WebShopAction("SEARCH", "blue shoes")

    assert {action} == {WebShopAction("search", "blue  shoes")}
    with pytest.raises(FrozenInstanceError):
        action.argument = "red shoes"


def test_parser_is_exported_from_actions_and_package():
    expected = WebShopAction("search", "blue shoes")

    assert parse_action(text="search[blue shoes]") == expected
    assert parse_action_from_actions("search[blue shoes]") == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("search[blue shoes]", WebShopAction("search", "blue shoes")),
        ("  search[blue shoes]  ", WebShopAction("search", "blue shoes")),
        ("CLICK[Buy Now]", WebShopAction("click", "Buy Now")),
        ("ThInK[  compare prices  ]", WebShopAction("think", "compare prices")),
    ],
)
def test_parser_accepts_only_supported_actions_case_insensitively(
    text,
    expected,
):
    assert parse_action(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "",
        "search[]",
        "search[   ]",
        "search",
        "search[query",
        "searchquery]",
        "search[query]]",
        "[search[query]]",
        "search[[query]]",
        "search[query[detail]]",
        "Action: search[query]",
        "prefix search[query]",
        "search[query] suffix",
        "think[compare\nprices]",
        "think[compare\rprices]",
        "open[item]",
    ],
)
def test_parser_rejects_empty_malformed_nested_and_prefixed_actions(text):
    with pytest.raises(ValueError, match="WebShop action"):
        parse_action(text)


@pytest.mark.parametrize("value", [None, 7, object(), ["search[item]"]])
def test_parser_defensively_rejects_non_string_values(value):
    with pytest.raises(ValueError, match="WebShop action"):
        parse_action(value)
