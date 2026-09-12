from dataclasses import FrozenInstanceError

import pytest
import requests

from planu_core.webshop.client import (
    WebShopHttpClient,
    WebShopHttpError,
    WebShopPage,
    WebShopPageError,
    parse_page,
)


SIMPLE_HTML = """
<html>
  <head><title>ignored</title><style>.hidden { display: none; }</style></head>
  <body><button>Continue</button></body>
</html>
"""


class FakeResponse:
    def __init__(self, text=SIMPLE_HTML, status_error=None):
        self.text = text
        self._status_error = status_error

    def raise_for_status(self):
        if self._status_error is not None:
            raise self._status_error


class FakeSession:
    def __init__(self, responses=None, error=None):
        self.responses = list(responses or [FakeResponse()])
        self.error = error
        self.calls = []

    def get(self, url, timeout):
        self.calls.append((url, timeout))
        if self.error is not None:
            raise self.error
        return self.responses.pop(0)


def test_page_is_frozen_and_defaults_to_empty_structured_values():
    page = WebShopPage("observation")

    assert page == WebShopPage(
        observation="observation",
        buttons=(),
        asins=(),
        option_types=(),
        reward=0.0,
    )
    with pytest.raises(FrozenInstanceError):
        page.reward = 1.0


def test_parse_init_page_ignores_nonvisible_text_and_extracts_button():
    page = parse_page(
        """
        <html>
          <head>
            <title>Not visible</title>
            <script>notVisible()</script>
          </head>
          <body>
            <!-- also ignored -->
            <h2>WebShop</h2>
            <div id="instruction-text"><h4>Instruction:<br>Find red shoes</h4></div>
            <input id="search_input">
            <button><i aria-hidden="true"></i>Search</button>
          </body>
        </html>
        """
    )

    assert page.observation == (
        "\nWebShop \nInstruction: \nFind red shoes \n[Search] "
    )
    assert page.buttons == ("Search",)
    assert page.asins == ()
    assert page.option_types == ()
    assert page.reward == 0.0


def test_parse_search_page_preserves_button_asin_and_visible_text_order():
    page = parse_page(
        """
        <body>
          <div>Instruction:</div><div>Find shoes</div>
          <button>Back to Search</button>
          <h3>Page 1</h3>
          <a class="product-link" href="/one">A-1</a>
          <h4>First shoe</h4><h5>$10</h5>
          <a class="product-link featured" href="/two">A-2</a>
          <h4>Second shoe</h4><h5>$20</h5>
        </body>
        """
    )

    assert page.buttons == ("Back to Search",)
    assert page.asins == ("A-1", "A-2")
    assert page.option_types == ()
    assert page.observation.index("[Back to Search]") < page.observation.index(
        "Page 1"
    )
    assert page.observation.index("[A-1]") < page.observation.index("First shoe")
    assert page.observation.index("First shoe") < page.observation.index("[A-2]")
    assert "Instruction:" not in page.observation
    assert "Find shoes" not in page.observation


def test_parse_item_page_extracts_buttons_and_ordered_option_types():
    page = parse_page(
        """
        <body>
          <div>Instruction:</div><div>Find shoes</div>
          <button>Back to Search</button><button>&lt; Prev</button>
          <h4>Color</h4>
          <input id="red" name="Color"><label for="red">Red</label>
          <input id="blue" name="Color"><label for="blue">Blue</label>
          <h4>Size</h4>
          <input id="large" name="Size"><label for="large">Large</label>
          <h2>Running Shoe</h2>
          <button>Description</button><button>Buy Now</button>
        </body>
        """
    )

    assert page.buttons == (
        "Back to Search",
        "< Prev",
        "Description",
        "Buy Now",
    )
    assert page.option_types == (
        ("Red", "Color"),
        ("Blue", "Color"),
        ("Large", "Size"),
    )
    assert page.asins == ()
    assert page.observation.index("[Red]") < page.observation.index("[Blue]")
    assert page.observation.index("[Blue]") < page.observation.index("[Large]")


def test_parse_subpage_keeps_navigation_and_product_information():
    page = parse_page(
        """
        <body>
          <div>Instruction:</div><div>Find shoes</div>
          <button>Back to Search</button><button>&lt; Prev</button>
          <p class="product-info">A durable running shoe.</p>
        </body>
        """
    )

    assert page.buttons == ("Back to Search", "< Prev")
    assert page.observation == (
        "\n[Back to Search] \n[< Prev] \nA durable running shoe. "
    )


def test_parse_done_page_reads_value_after_reward_marker():
    page = parse_page(
        """
        <body>
          <h1>Thank you</h1>
          <h3 id="reward">
            Your score (min 0.0, max 1.0)
            <pre>0.750</pre>
          </h3>
        </body>
        """
    )

    assert page.reward == 0.75
    assert page.observation == "Your score (min 0.0, max 1.0): 0.750"


def test_parse_page_defaults_reward_to_zero_without_marker():
    assert parse_page("<body><p>ordinary page</p></body>").reward == 0.0


@pytest.mark.parametrize(
    "html",
    [
        "<p>Your score (min 0.0, max 1.0)</p>",
        (
            "<p>Your score (min 0.0, max 1.0)</p>"
            "<pre>not-a-number</pre>"
        ),
    ],
)
def test_parse_page_rejects_missing_or_malformed_reward_contextually(html):
    with pytest.raises(WebShopPageError, match="reward"):
        parse_page(html)


@pytest.mark.parametrize(
    ("page_type", "kwargs", "expected_path"),
    [
        ("init", {}, "/session%20%2F%3F"),
        (
            "search",
            {"query": "red shoes/50%", "page": 2},
            "/search_results/session%20%2F%3F/red%20shoes%2F50%25/2",
        ),
        (
            "item",
            {
                "asin": "A/B ?",
                "query": "red shoes",
                "page": 3,
                "options": {"size": "M/L", "color": "red blue"},
            },
            (
                "/item_page/session%20%2F%3F/A%2FB%20%3F/red%20shoes/3/"
                "%7B%22color%22%3A%22red%20blue%22%2C"
                "%22size%22%3A%22M%2FL%22%7D"
            ),
        ),
        (
            "item_sub",
            {
                "asin": "A/B ?",
                "query": "red shoes",
                "page": 3,
                "subpage": "Reviews / Q&A",
                "options": {"color": "red blue"},
            },
            (
                "/item_sub_page/session%20%2F%3F/A%2FB%20%3F/"
                "red%20shoes/3/Reviews%20%2F%20Q%26A/"
                "%7B%22color%22%3A%22red%20blue%22%7D"
            ),
        ),
        (
            "done",
            {"asin": "A/B ?", "options": {"color": "red blue"}},
            (
                "/done/session%20%2F%3F/A%2FB%20%3F/"
                "%7B%22color%22%3A%22red%20blue%22%7D"
            ),
        ),
    ],
)
def test_fetch_builds_official_encoded_routes(
    page_type,
    kwargs,
    expected_path,
):
    session = FakeSession()
    client = WebShopHttpClient(
        "https://shop.example.test///",
        session=session,
        timeout=(1.5, 9.0),
    )

    page = client.fetch(page_type, "session /?", **kwargs)

    assert page.buttons == ("Continue",)
    assert session.calls == [
        ("https://shop.example.test" + expected_path, (1.5, 9.0))
    ]


def test_client_creation_does_not_make_a_request():
    session = FakeSession()

    WebShopHttpClient("https://shop.example.test", session=session)

    assert session.calls == []


def test_fetch_wraps_non_success_status_without_exposing_credentials():
    response = FakeResponse(
        status_error=requests.HTTPError(
            "401 for https://alice:secret@shop.example.test/private"
        )
    )
    client = WebShopHttpClient(
        "https://alice:secret@shop.example.test",
        session=FakeSession([response]),
    )

    with pytest.raises(WebShopHttpError) as error:
        client.fetch("init", "session")

    assert "init" in str(error.value)
    assert "shop.example.test" in str(error.value)
    assert "alice" not in str(error.value)
    assert "secret" not in str(error.value)


def test_fetch_wraps_transport_failure_with_route_context():
    client = WebShopHttpClient(
        "https://shop.example.test",
        session=FakeSession(error=requests.ConnectionError("offline")),
    )

    with pytest.raises(WebShopHttpError, match="search"):
        client.fetch("search", "session", query="shoes", page=1)


def test_fetch_wraps_page_parse_failure_with_route_context():
    client = WebShopHttpClient(
        "https://shop.example.test",
        session=FakeSession(
            [
                FakeResponse(
                    "<p>Your score (min 0.0, max 1.0)</p><pre>bad</pre>"
                )
            ]
        ),
    )

    with pytest.raises(WebShopHttpError, match="done"):
        client.fetch("done", "session", asin="A1")
