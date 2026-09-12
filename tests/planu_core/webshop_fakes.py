import copy

from planu_core.interfaces import ActionCandidate
from planu_core.webshop import WebShopAction
from planu_core.webshop.client import WebShopPage


class FakeWebShopClient:
    def __init__(self):
        self.calls = []

    def fetch(self, page_type, session_id, **kwargs):
        self.calls.append((page_type, session_id, copy.deepcopy(kwargs)))
        if page_type == "init":
            return WebShopPage(
                observation="WebShop\nInstruction: find a mug\n[Search]",
                buttons=("Search",),
            )
        if page_type == "search":
            return WebShopPage(
                observation="[A-1]",
                asins=("A-1",),
            )
        if page_type == "item":
            return WebShopPage(
                observation="[Attributes]\n[Buy Now]",
                buttons=("Attributes", "Buy Now"),
            )
        raise AssertionError("unexpected page type: {}".format(page_type))


def webshop_candidate(kind, argument):
    action = WebShopAction(kind, argument)
    return ActionCandidate(action.key, action, action.render())


class DeterministicWebShopActionProvider:
    def actions(self, state, state_visit_count=0):
        del state_visit_count
        if state.runtime.page_type == "init":
            return [webshop_candidate("search", "red mug")]
        if state.runtime.page_type == "search":
            return [webshop_candidate("click", "A-1")]
        return []
