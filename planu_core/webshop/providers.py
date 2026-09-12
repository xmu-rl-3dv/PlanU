import math
import re
from typing import Any, Dict, List, Sequence

from planu_core.interfaces import ActionCandidate, EnvironmentState
from planu_core.text_backend import GenerationResult, TextBackend
from planu_core.webshop.actions import WebShopAction, parse_action


LEGACY_COT_PROMPT = """You are interacting with a Webshop environment.  
Your goal: Follow the given **user instruction** to find and purchase the correct item.  

Rules of interaction:  
1. You will see a simulated webpage with buttons in [square brackets].  
2. To interact, always output exactly one action in the format:  
   - `click[button name]` → to select an option or buy the product.  
   - `search[keywords]` → to perform a new search.  
3. Only output the action name, nothing else.  
4. You must only click [Buy Now] **after confirming that the selected item fully satisfies all requirements in the user instruction** (e.g., size, scent, skin type, price limit).  

Context:  
The current web page description is: {input}  
[] means a button you can click. You can select size, scent, pack, etc., by clicking the button.  

Your next action is:
"""


LEGACY_VALUE_PROMPT = """Given an item to purchase and a trajectory that aims to buy an item that exactly matches the specification, which corresponds to the ideal score of 1.0, analyze the following trajectory, then at the last line conclude "Thus the correctness score is {s}", where s is an integer from 1 to 10.

Here are some examples

Webshop
Instruction:
i am looking for dairy free and apple variety pack of chips, and price lower than 30.00 dollars
[Search]

Action: search[dairy free and apple variety pack of chips]
Observation:
[Back to Search]
Page 1 (Total results: 50)
[Next >]
[B07HRFJWP8]
Enjoy Life Foods Soft Baked Ovals, Breakfast Bars, Nut Free Bars, Soy Free, Dairy Free, Non GMO, Gluten Free, Vegan, Variety Pack, 4 Boxes (20 Bars Total)
$100.0
[B01KMHY5PG]
Enjoy Life Soft Baked Chewy Bars, Variety Pack, Nut Free Bars, Soy Free, Dairy Free, Gluten Free, 6 Boxes (30 Total Bars)
$21.49
[B008D2X8C4]
Enjoy Life Lentil Chips Variety Pack, Dairy Free Chips, Soy Free, Nut Free, Non GMO, Vegan, Gluten Free, 24 Bags (0.8 oz)
$100.0

Action: think[The first two items are not what I want. The third item is the dairy free and apple variety pack of chips I want.]
Observation: OK.

Action: click[B008D2X8C4]
Observation:
[Back to Search]
[< Prev]
flavor name [dill and sour cream][garlic & parmesan][light sea salt][margherita pizza][thai chili lime][variety pack]
size [0.8 ounce (pack of 24)][4 ounce (pack of 12)]
Price: $100.0
Rating: N.A.
[Description]
[Features]
[Reviews]
[Buy Now]

Action: think[The item has options 'variety pack' and '0.8 ounce (pack of 24)' and seems good to buy.]
Observation: OK.

Action: click[variety pack]
Observation: You have clicked variety pack.

Action: click[0.8 ounce (pack of 24)]
Observation: You have clicked 0.8 ounce (pack of 24).

Action: click[Buy Now]

Reflection: In this attempt, I was unsuccessful. I accidentally bought a product that was $100, which is more than my budget of $30. Either way, the initial search results were not good. Next time, I will do search["variety pack of chips"] and then check if the results meet the dairy free and the $30 budget constraints. I will continue to refine my searches so that I can find more products.
Thus the correctness score is 5

Webshop 
Instruction:  
i would like a 3 ounce bottle of bright citrus deodorant for sensitive skin, and price lower than 50.00 dollars 
[Search]  

Action: search[3 ounce bright citrus deodorant sensitive skin]
Observation: 
[Back to Search] 
Page 1 (Total results: 50) 
[Next >] 
[B078GWRC1J] 
Bright Citrus Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
$10.99 
[B078GTKVXY] 
Ginger Fresh Deodorant by Earth Mama | Natural and Safe for Sensitive Skin, Pregnancy and Breastfeeding, Contains Organic Calendula 3-Ounce 
$10.99 
[B08KBVJ4XN] 
Barrel and Oak - Aluminum-Free Deodorant, Deodorant for Men, Essential Oil-Based Scent, 24-Hour Odor Protection, Cedar & Patchouli Blend, Gentle on Sensitive Skin (Mountain Sage, 2.7 oz, 2-Pack) 
$15.95  

Action: think[B078GWRC1J and B078GTKVXY are bright citrus deodorant less then 50 dollars. I can check B078GWRC1J first.]
Observation: OK.
Reflection: Although the task is not yet complete, the first search is correct and returns relevant results. The thought is also logical and has potential to lead to the correct selection.
Thus the correctness score is 9
{input}
"""


_ACTION_LINE = re.compile(r"^Action:\s*(\S.*)\s*$")
_SCORE = re.compile(
    r"\bcorrectness score is\s+(10|[1-9])\b",
    re.IGNORECASE,
)


def _public_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return str(value)


def _candidate(action: WebShopAction, **metadata: Any) -> ActionCandidate:
    return ActionCandidate(
        key=action.key,
        payload=action,
        text=action.render(),
        metadata=metadata,
    )


def _invalid_candidate(
    raw_output: str,
    parse_error: ValueError,
    **metadata: Any
) -> ActionCandidate:
    return ActionCandidate(
        key=("invalid", " ".join(raw_output.split())),
        payload=raw_output,
        text=raw_output,
        metadata={
            **metadata,
            "parse_error": str(parse_error),
        },
    )


class _TokenAccounting:
    def __init__(self) -> None:
        self.prompt_tokens = 0
        self.completion_tokens = 0

    @property
    def token_usage(self) -> Dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
        }

    def _record_usage(self, generated: GenerationResult) -> None:
        self.prompt_tokens += generated.prompt_tokens
        self.completion_tokens += generated.completion_tokens


class ModelWebShopActionProvider(_TokenAccounting):
    def __init__(
        self,
        backend: TextBackend,
        candidate_count: int = 5,
        temperature: float = 0.8,
        max_tokens: int = 100,
    ) -> None:
        super().__init__()
        if (
            isinstance(candidate_count, bool)
            or not isinstance(candidate_count, int)
            or candidate_count <= 0
        ):
            raise ValueError("candidate_count must be a positive integer")
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(float(temperature))
            or temperature < 0
        ):
            raise ValueError(
                "temperature must be a finite nonnegative number"
            )
        if (
            isinstance(max_tokens, bool)
            or not isinstance(max_tokens, int)
            or max_tokens <= 0
        ):
            raise ValueError("max_tokens must be a positive integer")
        if not callable(getattr(backend, "generate", None)):
            raise ValueError("backend must provide generate")

        self.backend = backend
        self.candidate_count = candidate_count
        self.temperature = float(temperature)
        self.max_tokens = max_tokens

    @staticmethod
    def _extract_action(text: str) -> WebShopAction:
        stripped = text.strip()
        try:
            return parse_action(stripped)
        except ValueError:
            pass

        matches = [
            match.group(1).strip()
            for line in stripped.splitlines()
            for match in [_ACTION_LINE.fullmatch(line)]
            if match is not None
        ]
        if len(matches) != 1:
            raise ValueError("model output must contain exactly one action")
        return parse_action(matches[0])

    def actions(
        self,
        state: EnvironmentState,
        state_visit_count: int = 0,
    ) -> Sequence[ActionCandidate]:
        del state_visit_count
        history = getattr(state.runtime, "history", None)
        trajectory = (
            "\n".join(history)
            if history is not None
            else _public_text(state.observation)
        )
        prompt = LEGACY_COT_PROMPT.format(input=trajectory)
        generated = self.backend.generate(
            prompt,
            n=self.candidate_count,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stop=("Observation",),
        )
        self._record_usage(generated)
        if len(generated.texts) != self.candidate_count:
            raise ValueError(
                "text backend returned the wrong completion count"
            )

        candidates: List[ActionCandidate] = []
        seen = set()
        for text in generated.texts:
            try:
                action = self._extract_action(text)
                candidate = _candidate(
                    action,
                    prompt=prompt,
                    trajectory=trajectory,
                    model_identifier=self.backend.model_identifier,
                )
            except ValueError as error:
                candidate = _invalid_candidate(
                    text,
                    error,
                    prompt=prompt,
                    trajectory=trajectory,
                    model_identifier=self.backend.model_identifier,
                )
            if candidate.key in seen:
                continue
            seen.add(candidate.key)
            candidates.append(candidate)
        return candidates


class ModelWebShopActionScorer(_TokenAccounting):
    def __init__(
        self,
        backend: TextBackend,
        n_evaluate_sample: int = 1,
        temperature: float = 0.8,
        max_tokens: int = 100,
    ) -> None:
        super().__init__()
        if (
            isinstance(n_evaluate_sample, bool)
            or not isinstance(n_evaluate_sample, int)
            or n_evaluate_sample <= 0
        ):
            raise ValueError(
                "n_evaluate_sample must be a positive integer"
            )
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(float(temperature))
            or temperature < 0
        ):
            raise ValueError(
                "temperature must be a finite nonnegative number"
            )
        if (
            isinstance(max_tokens, bool)
            or not isinstance(max_tokens, int)
            or max_tokens <= 0
        ):
            raise ValueError("max_tokens must be a positive integer")
        if not callable(getattr(backend, "generate", None)):
            raise ValueError("backend must provide generate")

        self.backend = backend
        self.n_evaluate_sample = n_evaluate_sample
        self.temperature = float(temperature)
        self.max_tokens = max_tokens

    @staticmethod
    def _parse_score(text: str) -> float:
        matches = _SCORE.findall(text)
        if len(matches) != 1:
            raise ValueError(
                "model output must contain exactly one correctness score"
            )
        return int(matches[0]) / 10.0

    def score(
        self,
        observation: Any,
        candidates: Sequence[ActionCandidate],
    ) -> Sequence[float]:
        scores = []
        for candidate in candidates:
            trajectory = candidate.metadata.get(
                "trajectory",
                _public_text(observation),
            )
            if not isinstance(trajectory, str):
                raise ValueError("candidate trajectory must be a string")
            evaluation = "{}\n\nAction: {}\n\nReflection: ".format(
                trajectory.rstrip(),
                candidate.text,
            )
            prompt = LEGACY_VALUE_PROMPT.format(
                s="",
                input=evaluation,
            )
            generated = self.backend.generate(
                prompt,
                n=self.n_evaluate_sample,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stop=(),
            )
            self._record_usage(generated)
            if len(generated.texts) != self.n_evaluate_sample:
                raise ValueError(
                    "text backend returned the wrong completion count"
                )

            sample_scores = [
                self._parse_score(text)
                for text in generated.texts
            ]
            value = sum(sample_scores) / len(sample_scores)
            if not math.isfinite(value):
                raise ValueError("action score must be finite")
            if not 0.0 <= value <= 1.0:
                raise ValueError("action score must be between 0 and 1")
            scores.append(value)
        return scores


class ScriptedWebShopActionProvider:
    def __init__(self, search_query: str) -> None:
        if not isinstance(search_query, str) or not search_query.strip():
            raise ValueError("search_query must be a non-empty string")
        self.search_query = " ".join(search_query.split())

    @staticmethod
    def _matching_button(runtime: Any, expected: str) -> Any:
        for button in runtime.buttons:
            if button.casefold() == expected.casefold():
                return button
        return None

    def actions(
        self,
        state: EnvironmentState,
        state_visit_count: int = 0,
    ) -> Sequence[ActionCandidate]:
        del state_visit_count
        runtime = state.runtime
        if runtime.terminated or runtime.truncated:
            return []

        if runtime.page_type == "init":
            return [_candidate(WebShopAction("search", self.search_query))]
        if runtime.page_type == "search":
            if not runtime.asins:
                return []
            return [_candidate(WebShopAction("click", runtime.asins[0]))]
        if runtime.page_type == "item":
            selected_types = {
                str(option_type).casefold()
                for option_type in runtime.options
            }
            for option, option_type in runtime.option_types:
                if option_type.casefold() not in selected_types:
                    return [_candidate(WebShopAction("click", option))]
            buy_now = self._matching_button(runtime, "Buy Now")
            if buy_now is not None:
                return [_candidate(WebShopAction("click", buy_now))]
            return []
        if runtime.page_type in ("item_sub", "subpage"):
            previous = self._matching_button(runtime, "< Prev")
            if previous is not None:
                return [_candidate(WebShopAction("click", previous))]
        return []


__all__ = [
    "LEGACY_COT_PROMPT",
    "LEGACY_VALUE_PROMPT",
    "ModelWebShopActionProvider",
    "ModelWebShopActionScorer",
    "ScriptedWebShopActionProvider",
]
