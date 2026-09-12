import copy
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Hashable, Optional, Tuple

import numpy as np

from planu_core.config import PlanUConfig, SelectionSchedule
from planu_core.interfaces import (
    ActionCandidate,
    EnvironmentState,
    TransitionResult,
)
from planu_core.webshop.actions import WebShopAction, parse_action
from planu_core.webshop.client import WebShopPage


_BACK_TO_SEARCH = "back to search"
_NEXT_PAGE = "next >"
_PREVIOUS_PAGE = "< prev"
_END_BUTTON = "buy now"
_SUBPAGES = frozenset(("description", "features", "reviews", "attributes"))
_INVALID_OBSERVATION = "Invalid action!"
_THINK_OBSERVATION = "OK."


@dataclass
class WebShopRuntime:
    session_id: str
    page_type: str = "init"
    query_string: str = ""
    page_num: int = 1
    asin: str = ""
    options: dict = field(default_factory=dict)
    subpage: str = ""
    buttons: Tuple[str, ...] = ()
    asins: Tuple[str, ...] = ()
    option_types: Tuple[Tuple[str, str], ...] = ()
    history: Tuple[str, ...] = ()
    step_count: int = 0
    failure_count: int = 0
    terminated: bool = False
    truncated: bool = False


@dataclass(frozen=True)
class _ResolvedAction:
    action: WebShopAction
    argument: str
    target: str
    option_type: str = ""


def _freeze(value: Any) -> Hashable:
    if isinstance(value, Mapping):
        items = [(_freeze(key), _freeze(item)) for key, item in value.items()]
        items.sort(key=lambda pair: repr(pair[0]))
        return "mapping", tuple(items)
    if isinstance(value, list):
        return "list", tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return "tuple", tuple(_freeze(item) for item in value)
    if isinstance(value, np.ndarray):
        return (
            "ndarray",
            value.dtype.str,
            tuple(value.shape),
            _freeze(value.tolist()),
        )
    if isinstance(value, np.generic):
        return _freeze(value.item())
    try:
        hash(value)
    except TypeError as error:
        raise TypeError(
            "unsupported WebShop state value: {}".format(
                type(value).__qualname__
            )
        ) from error
    return type(value).__qualname__, value


def _matching_value(argument: str, values: Tuple[str, ...]) -> Optional[str]:
    normalized = argument.casefold()
    for value in values:
        if value.casefold() == normalized:
            return value
    return None


def _click_candidate(label: str) -> ActionCandidate:
    action = WebShopAction("click", label)
    return ActionCandidate(
        key=action.key,
        payload=action,
        text=action.render(),
    )


class WebShopAdapter:
    def __init__(self, client: Any, session_id: str) -> None:
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("WebShop session_id must be a non-empty string")
        if not callable(getattr(client, "fetch", None)):
            raise ValueError("WebShop client must provide fetch")
        self.client = client
        self.session_id = session_id

    def reset(self, seed: Optional[int] = None) -> EnvironmentState:
        del seed
        page = self.client.fetch("init", self.session_id)
        return EnvironmentState(
            observation=copy.deepcopy(page.observation),
            runtime=WebShopRuntime(
                session_id=self.session_id,
                buttons=tuple(page.buttons),
                asins=tuple(page.asins),
                option_types=tuple(page.option_types),
                history=(page.observation,),
            ),
        )

    def clone(self, state: EnvironmentState) -> EnvironmentState:
        return copy.deepcopy(state)

    def actions(
        self,
        state: EnvironmentState,
        state_visit_count: int = 0,
    ):
        del state_visit_count
        runtime = state.runtime
        if runtime.terminated or runtime.truncated:
            return []

        if runtime.page_type == "search":
            allowed_buttons = {
                _BACK_TO_SEARCH,
                _NEXT_PAGE,
                _PREVIOUS_PAGE,
            }
            labels = [
                button
                for button in runtime.buttons
                if button.casefold() in allowed_buttons
            ]
            labels.extend(runtime.asins)
        elif runtime.page_type == "item":
            allowed_buttons = {
                _BACK_TO_SEARCH,
                _PREVIOUS_PAGE,
                _END_BUTTON,
            } | _SUBPAGES
            labels = [
                button
                for button in runtime.buttons
                if button.casefold() in allowed_buttons
            ]
            labels.extend(option for option, _ in runtime.option_types)
        elif runtime.page_type == "item_sub":
            allowed_buttons = {_BACK_TO_SEARCH, _PREVIOUS_PAGE}
            labels = [
                button
                for button in runtime.buttons
                if button.casefold() in allowed_buttons
            ]
        else:
            # Initial search and free-form thoughts require an explicit
            # provider because their arguments cannot be inferred.
            labels = []

        candidates = []
        seen = set()
        for label in labels:
            candidate = _click_candidate(label)
            if candidate.key not in seen:
                candidates.append(candidate)
                seen.add(candidate.key)
        return candidates

    def preview(
        self,
        state: EnvironmentState,
        action: ActionCandidate,
        rng: np.random.Generator,
    ) -> TransitionResult:
        del rng
        next_state = self.clone(state)
        resolved, reason = self._resolve(next_state.runtime, action)
        if resolved is None:
            return self._invalid_result(next_state, action, reason)
        return self._transition(next_state, resolved)

    def step(
        self,
        state: EnvironmentState,
        action: ActionCandidate,
        rng: np.random.Generator,
    ) -> TransitionResult:
        next_state = self.clone(state)
        resolved, reason = self._resolve(next_state.runtime, action)
        if resolved is None:
            return self._invalid_result(next_state, action, reason)

        if resolved.target not in ("search", "end"):
            latency = float(rng.lognormal(mean=0, sigma=10))
            if latency > 200:
                next_state.runtime.failure_count += 1
                return TransitionResult(
                    state=next_state,
                    reward=0.0,
                    terminated=False,
                    truncated=False,
                    info={
                        "latency": latency,
                        "latency_failure": True,
                    },
                )
            result = self._transition(next_state, resolved)
            result.info = {
                **result.info,
                "latency": latency,
                "latency_failure": False,
            }
            return result

        return self._transition(next_state, resolved)

    def state_key(self, state: EnvironmentState) -> Hashable:
        runtime = state.runtime
        return (
            _freeze(state.observation),
            runtime.session_id,
            runtime.page_type,
            runtime.query_string,
            runtime.page_num,
            runtime.asin,
            _freeze(runtime.options),
            runtime.subpage,
            _freeze(runtime.buttons),
            _freeze(runtime.asins),
            _freeze(runtime.option_types),
            _freeze(runtime.history),
            runtime.terminated,
            runtime.truncated,
        )

    def is_terminal(self, state: EnvironmentState) -> bool:
        return bool(state.runtime.terminated)

    def is_truncated(self, state: EnvironmentState) -> bool:
        return bool(state.runtime.truncated)

    @staticmethod
    def _coerce_action(candidate: ActionCandidate) -> WebShopAction:
        payload = candidate.payload
        if isinstance(payload, WebShopAction):
            return payload
        if isinstance(payload, str):
            return parse_action(payload)
        raise ValueError("action payload must be WebShopAction or string")

    def _resolve(
        self,
        runtime: WebShopRuntime,
        candidate: ActionCandidate,
    ):
        if runtime.terminated:
            return None, "episode is terminated"
        if runtime.truncated:
            return None, "episode is truncated"
        try:
            action = self._coerce_action(candidate)
        except ValueError as error:
            return None, str(error)

        if action.kind == "think":
            return _ResolvedAction(action, action.argument, "think"), ""

        if action.kind == "search":
            if runtime.page_type != "init":
                return None, "search is only legal from the init page"
            return _ResolvedAction(action, action.argument, "search"), ""

        button = _matching_value(action.argument, runtime.buttons)
        asin = _matching_value(action.argument, runtime.asins)
        option = _matching_value(
            action.argument,
            tuple(value for value, _ in runtime.option_types),
        )
        normalized = action.argument.casefold()

        if (
            normalized == _BACK_TO_SEARCH
            and button is not None
            and runtime.page_type in ("search", "item", "item_sub")
        ):
            return _ResolvedAction(action, button, "init"), ""
        if (
            normalized == _NEXT_PAGE
            and button is not None
            and runtime.page_type == "search"
        ):
            return _ResolvedAction(action, button, "next"), ""
        if (
            normalized == _PREVIOUS_PAGE
            and button is not None
            and runtime.page_type in ("search", "item", "item_sub")
        ):
            return _ResolvedAction(action, button, "previous"), ""
        if (
            normalized == _END_BUTTON
            and button is not None
            and runtime.page_type == "item"
        ):
            return _ResolvedAction(action, button, "end"), ""
        if (
            normalized in _SUBPAGES
            and button is not None
            and runtime.page_type == "item"
        ):
            return _ResolvedAction(action, button, "item_sub"), ""
        if asin is not None and runtime.page_type == "search":
            return _ResolvedAction(action, asin, "item"), ""
        if option is not None and runtime.page_type == "item":
            for value, option_type in runtime.option_types:
                if value == option:
                    return (
                        _ResolvedAction(
                            action,
                            option,
                            "option",
                            option_type,
                        ),
                        "",
                    )
        return None, "action is not legal on the current page"

    @staticmethod
    def _invalid_result(
        state: EnvironmentState,
        action: ActionCandidate,
        reason: str,
    ) -> TransitionResult:
        state.observation = _INVALID_OBSERVATION
        try:
            rendered_action = WebShopAdapter._coerce_action(action).render()
        except ValueError:
            rendered_action = action.text
        WebShopAdapter._append_history(state, rendered_action)
        return TransitionResult(
            state=state,
            reward=-1.0,
            terminated=False,
            truncated=False,
            info={
                "invalid_action": True,
                "reason": reason,
            },
        )

    def _transition(
        self,
        state: EnvironmentState,
        resolved: _ResolvedAction,
    ) -> TransitionResult:
        runtime = state.runtime
        runtime.step_count += 1
        target = resolved.target

        if target == "think":
            state.observation = _THINK_OBSERVATION
            return self._result(state, 0.0, resolved.action)

        if target == "init":
            page = self.client.fetch("init", runtime.session_id)
            runtime.page_type = "init"
            runtime.query_string = ""
            runtime.page_num = 1
            runtime.asin = ""
            runtime.options = {}
            runtime.subpage = ""
        elif target == "search":
            runtime.page_type = "search"
            runtime.query_string = resolved.argument
            runtime.page_num = 1
            runtime.asin = ""
            runtime.options = {}
            runtime.subpage = ""
            page = self._fetch_search(runtime)
        elif target == "next":
            runtime.page_num += 1
            runtime.asin = ""
            runtime.options = {}
            runtime.subpage = ""
            page = self._fetch_search(runtime)
        elif target == "previous" and runtime.page_type == "search":
            runtime.page_num -= 1
            runtime.asin = ""
            runtime.options = {}
            runtime.subpage = ""
            page = self._fetch_search(runtime)
        elif target == "previous" and runtime.page_type == "item":
            runtime.page_type = "search"
            runtime.asin = ""
            runtime.options = {}
            runtime.subpage = ""
            page = self._fetch_search(runtime)
        elif target == "previous":
            runtime.page_type = "item"
            runtime.subpage = ""
            page = self._fetch_item(runtime)
        elif target == "item":
            runtime.page_type = "item"
            runtime.asin = resolved.argument.upper()
            runtime.options = {}
            runtime.subpage = ""
            page = self._fetch_item(runtime)
        elif target == "option":
            runtime.options[resolved.option_type.lower()] = resolved.argument
            page = self._fetch_item(runtime)
        elif target == "item_sub":
            runtime.page_type = "item_sub"
            runtime.subpage = resolved.argument
            page = self._fetch_item_sub(runtime)
        elif target == "end":
            runtime.page_type = "end"
            runtime.subpage = ""
            page = self.client.fetch(
                "end",
                runtime.session_id,
                asin=runtime.asin,
                options=runtime.options,
            )
            reward = float(page.reward)
            if not 0.0 <= reward <= 1.0:
                raise ValueError("WebShop reward must be between 0 and 1")
            runtime.terminated = True
            self._apply_page(state, page)
            return self._result(state, reward, resolved.action)
        else:
            raise RuntimeError(
                "unsupported resolved WebShop target: {}".format(target)
            )

        self._apply_page(state, page)
        if target == "option":
            state.observation = "You have clicked {}.".format(
                resolved.argument
            )
        return self._result(state, float(page.reward), resolved.action)

    def _fetch_search(self, runtime: WebShopRuntime) -> WebShopPage:
        return self.client.fetch(
            "search",
            runtime.session_id,
            query_string=runtime.query_string,
            page_num=runtime.page_num,
        )

    def _fetch_item(self, runtime: WebShopRuntime) -> WebShopPage:
        return self.client.fetch(
            "item",
            runtime.session_id,
            query_string=runtime.query_string,
            page_num=runtime.page_num,
            asin=runtime.asin,
            options=runtime.options,
        )

    def _fetch_item_sub(self, runtime: WebShopRuntime) -> WebShopPage:
        return self.client.fetch(
            "item_sub",
            runtime.session_id,
            query_string=runtime.query_string,
            page_num=runtime.page_num,
            asin=runtime.asin,
            options=runtime.options,
            subpage=runtime.subpage,
        )

    @staticmethod
    def _apply_page(state: EnvironmentState, page: WebShopPage) -> None:
        state.observation = copy.deepcopy(page.observation)
        state.runtime.buttons = tuple(page.buttons)
        state.runtime.asins = tuple(page.asins)
        state.runtime.option_types = tuple(page.option_types)

    @staticmethod
    def _result(
        state: EnvironmentState,
        reward: float,
        action: WebShopAction,
    ) -> TransitionResult:
        WebShopAdapter._append_history(state, action.render())
        return TransitionResult(
            state=state,
            reward=reward,
            terminated=bool(state.runtime.terminated),
            truncated=bool(state.runtime.truncated),
            info={},
        )

    @staticmethod
    def _append_history(
        state: EnvironmentState,
        rendered_action: str,
    ) -> None:
        history = tuple(getattr(state.runtime, "history", ()))
        state.runtime.history = history + (
            "Action: {}".format(rendered_action),
            "Observation: {}".format(state.observation),
        )


def webshop_config() -> PlanUConfig:
    return PlanUConfig(
        n_quantiles=51,
        value_min=0.0,
        value_max=1.0,
        quantile_learning_rate=0.9,
        discount=1.0,
        curiosity_weight=0.0,
        include_preview_reward=True,
        categorical_initialization=False,
        train_curiosity=False,
        max_depth=10,
        max_iterations=10,
        risk_distortion=0.0,
        selection_schedule=SelectionSchedule(),
    )


__all__ = ["WebShopAdapter", "WebShopRuntime", "webshop_config"]
