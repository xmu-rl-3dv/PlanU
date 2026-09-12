import json
import math
import re
from dataclasses import dataclass
from numbers import Real
from typing import Mapping, Optional, Tuple
from urllib.parse import quote, urlsplit, urlunsplit

import requests
from bs4 import BeautifulSoup
from bs4.element import (
    Comment,
    Declaration,
    Doctype,
    NavigableString,
    ProcessingInstruction,
    Tag,
)


_REWARD_MARKER = "Your score (min 0.0, max 1.0)"
_IGNORED_TAGS = frozenset(("style", "script", "head", "title", "meta"))
_DEFAULT_TIMEOUT = (3.05, 30.0)
_PATH_SEPARATOR_RUN = re.compile(r"[/\\]+")
_NON_CONTENT_STRINGS = (Comment, Declaration, Doctype, ProcessingInstruction)


class WebShopPageError(ValueError):
    """Raised when WebShop HTML cannot be converted into a page."""


class WebShopHttpError(RuntimeError):
    """Raised when a WebShop page cannot be fetched or parsed."""


@dataclass(frozen=True)
class WebShopPage:
    observation: str
    buttons: Tuple[str, ...] = ()
    asins: Tuple[str, ...] = ()
    option_types: Tuple[Tuple[str, str], ...] = ()
    reward: float = 0.0


def _is_visible_string(node: NavigableString) -> bool:
    if isinstance(node, _NON_CONTENT_STRINGS):
        return False

    parent = node.parent
    while isinstance(parent, Tag):
        if parent.name in _IGNORED_TAGS or parent.has_attr("hidden"):
            return False
        style = parent.get("style", "").replace(" ", "").lower()
        if "display:none" in style or "visibility:hidden" in style:
            return False
        parent = parent.parent
    return True


def _clean_text(value: str) -> str:
    return " ".join(value.split())


def _interactive_ancestor(node: NavigableString) -> Optional[Tag]:
    parent = node.parent
    while isinstance(parent, Tag):
        classes = parent.get("class", ())
        if (
            parent.name in ("button", "label")
            or "product-link" in classes
        ):
            return parent
        parent = parent.parent
    return None


def _option_type(label: Tag, fallback: str, soup: BeautifulSoup) -> str:
    input_id = label.get("for")
    if input_id:
        option_input = soup.find("input", id=input_id)
        if isinstance(option_input, Tag) and option_input.get("name"):
            return _clean_text(str(option_input["name"]))
    if label.get("name"):
        return _clean_text(str(label["name"]))
    return fallback


def _page_kind(soup: BeautifulSoup, visible_texts: Tuple[str, ...]) -> str:
    if _REWARD_MARKER in visible_texts:
        return "done"
    if soup.select_one("#search_input") is not None:
        return "init"
    if soup.select_one(".product-link") is not None:
        return "search"
    if soup.find("label") is not None:
        return "item"
    button_texts = {
        _clean_text(button.get_text(" ", strip=True))
        for button in soup.find_all("button")
    }
    if "Buy Now" in button_texts:
        return "item"
    return "subpage"


def _parse_reward(visible_texts: Tuple[str, ...]) -> float:
    if _REWARD_MARKER not in visible_texts:
        return 0.0

    marker_index = visible_texts.index(_REWARD_MARKER)
    if marker_index + 1 >= len(visible_texts):
        raise WebShopPageError(
            "WebShop reward marker is missing its following value"
        )

    reward_text = visible_texts[marker_index + 1]
    try:
        return float(reward_text)
    except (TypeError, ValueError):
        raise WebShopPageError(
            "WebShop reward value is malformed: {!r}".format(reward_text)
        ) from None


def parse_page(html: str) -> WebShopPage:
    """Parse WebShop HTML into its ordered text and interaction metadata."""
    if not isinstance(html, str):
        raise WebShopPageError("WebShop page HTML must be a string")

    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception as error:
        raise WebShopPageError("could not parse WebShop page HTML") from error

    visible_nodes = tuple(
        node
        for node in soup.find_all(string=True)
        if _is_visible_string(node) and _clean_text(str(node))
    )
    visible_texts = tuple(_clean_text(str(node)) for node in visible_nodes)
    reward = _parse_reward(visible_texts)
    page_kind = _page_kind(soup, visible_texts)

    observation_parts = []
    buttons = []
    asins = []
    option_types = []
    processed_interactive_tags = set()
    option_type = ""
    regular_text_count = 0
    product_count = 0
    distance_from_product = 0

    for node in visible_nodes:
        interactive = _interactive_ancestor(node)
        if interactive is not None:
            tag_identity = id(interactive)
            if tag_identity in processed_interactive_tags:
                continue
            processed_interactive_tags.add(tag_identity)
            text = _clean_text(interactive.get_text(" ", strip=True))

            if interactive.name == "button":
                buttons.append(text)
                observation_parts.append("\n[{}] ".format(text))
            elif interactive.name == "label":
                option_types.append(
                    (text, _option_type(interactive, option_type, soup))
                )
                observation_parts.append("[{}]".format(text))
            else:
                asins.append(text)
                if product_count < 10:
                    observation_parts.append("\n[{}] ".format(text))
                product_count += 1
                distance_from_product = 0
        else:
            text = _clean_text(str(node))
            processed_text = "\n{} ".format(text)
            if regular_text_count < 2 and page_kind != "init":
                processed_text = ""
            if distance_from_product <= 2 and product_count >= 4:
                processed_text = ""
            option_type = text
            regular_text_count += 1
            observation_parts.append(processed_text)

        distance_from_product += 1

    observation = "".join(observation_parts)
    if page_kind == "done":
        reward_text = visible_texts[
            visible_texts.index(_REWARD_MARKER) + 1
        ]
        observation = "{}: {}".format(_REWARD_MARKER, reward_text)

    return WebShopPage(
        observation=observation,
        buttons=tuple(buttons),
        asins=tuple(asins),
        option_types=tuple(option_types),
        reward=reward,
    )


def _encode_component(value: object) -> str:
    return quote(str(value), safe="")


def _encode_path_component(name: str, value: object) -> str:
    text = str(value)
    if _PATH_SEPARATOR_RUN.search(text):
        raise ValueError(
            "WebShop {} cannot contain '/' or '\\\\'".format(name)
        )
    return _encode_component(text)


def _encode_query_string(value: object) -> str:
    normalized = _PATH_SEPARATOR_RUN.sub(" ", str(value))
    return _encode_component(normalized)


def _canonical_options(options: Optional[Mapping[str, str]]) -> str:
    if options is None:
        options = {}
    if not isinstance(options, Mapping):
        raise ValueError("WebShop options must be a mapping")
    try:
        serialized = json.dumps(
            dict(options),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        return serialized.replace("/", "\\u002f")
    except (TypeError, ValueError) as error:
        raise ValueError("WebShop options must be JSON-serializable") from error


def _normalize_timeout(timeout: Tuple[float, float]) -> Tuple[float, float]:
    if not isinstance(timeout, tuple) or len(timeout) != 2:
        raise ValueError("WebShop timeout must be a (connect, read) tuple")

    normalized = []
    for value in timeout:
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(
                "WebShop timeout entries must be finite positive real numbers"
            )
        normalized.append(float(value))
    return normalized[0], normalized[1]


def _safe_url(url: str) -> str:
    parsed = urlsplit(url)
    hostname = parsed.hostname or ""
    if ":" in hostname and not hostname.startswith("["):
        hostname = "[{}]".format(hostname)
    if parsed.port is not None:
        hostname = "{}:{}".format(hostname, parsed.port)
    return urlunsplit(
        (parsed.scheme, hostname, parsed.path, parsed.query, parsed.fragment)
    )


class WebShopHttpClient:
    def __init__(
        self,
        base_url: str,
        session: Optional[requests.Session] = None,
        timeout: Tuple[float, float] = _DEFAULT_TIMEOUT,
    ) -> None:
        if not isinstance(base_url, str) or not base_url.rstrip("/"):
            raise ValueError("WebShop base URL must be a non-empty string")

        normalized_timeout = _normalize_timeout(timeout)
        self.base_url = base_url.rstrip("/")
        self._owns_session = session is None
        self.session = session if session is not None else requests.Session()
        self.timeout = normalized_timeout
        self._closed = False

    def __enter__(self) -> "WebShopHttpClient":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._owns_session:
            self.session.close()

    def _build_url(
        self,
        page_type: str,
        session_id: str,
        query_string: str,
        page_num: int,
        asin: str,
        options: Optional[Mapping[str, str]],
        subpage: str,
    ) -> str:
        encoded_session = _encode_path_component("session_id", session_id)
        encoded_asin = _encode_path_component("asin", asin)
        encoded_subpage = _encode_path_component("subpage", subpage)
        encoded_query = _encode_query_string(query_string)
        encoded_options = _encode_component(_canonical_options(options))
        done_route = (
            "done",
            encoded_session,
            encoded_asin,
            encoded_options,
        )
        routes = {
            "init": (encoded_session,),
            "search": (
                "search_results",
                encoded_session,
                encoded_query,
                _encode_component(page_num),
            ),
            "item": (
                "item_page",
                encoded_session,
                encoded_asin,
                encoded_query,
                _encode_component(page_num),
                encoded_options,
            ),
            "item_sub": (
                "item_sub_page",
                encoded_session,
                encoded_asin,
                encoded_query,
                _encode_component(page_num),
                encoded_subpage,
                encoded_options,
            ),
            "end": done_route,
            "done": done_route,
        }
        try:
            route = routes[page_type]
        except KeyError:
            raise ValueError(
                "unsupported WebShop page type: {!r}".format(page_type)
            ) from None
        return "{}/{}".format(self.base_url, "/".join(route))

    def fetch(
        self,
        page_type: str,
        session_id: str,
        query_string: str = "",
        page_num: int = 1,
        asin: str = "",
        options: Optional[Mapping[str, str]] = None,
        subpage: str = "",
    ) -> WebShopPage:
        url = self._build_url(
            page_type,
            session_id,
            query_string,
            page_num,
            asin,
            options,
            subpage,
        )
        safe_url = _safe_url(url)
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException:
            raise WebShopHttpError(
                "WebShop {} request failed at {}".format(
                    page_type,
                    safe_url,
                )
            ) from None

        try:
            return parse_page(response.text)
        except WebShopPageError as error:
            raise WebShopHttpError(
                "WebShop {} response could not be parsed at {}: {}".format(
                    page_type,
                    safe_url,
                    error,
                )
            ) from error


__all__ = [
    "WebShopHttpClient",
    "WebShopHttpError",
    "WebShopPage",
    "WebShopPageError",
    "parse_page",
]
