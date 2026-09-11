from __future__ import annotations

from typing import Any, Dict, List, NamedTuple, Optional

import numpy as np

from planu_core import LanguageNode, PlanUSearch
from planu_core.adapters.blockworld import (
    BlockWorldAdapter,
    BlockWorldScorer,
    blockworld_config,
)

from .. import SearchAlgorithm


PlanUNode = LanguageNode


class PlanUResult(NamedTuple):
    terminal_state: Any
    cum_reward: float
    trace: Any
    trace_of_nodes: List[LanguageNode]
    tree_state: LanguageNode
    trace_in_each_iter: Optional[List[List[LanguageNode]]] = None
    tree_state_after_each_iter: Optional[List[LanguageNode]] = None
    aggregated_result: Optional[Any] = None


_UNSUPPORTED_DEFAULTS = {
    "w_exp": 1.0,
    "cum_reward": sum,
    "calc_q": np.mean,
    "simulate_strategy": "max",
    "output_strategy": "max_reward",
    "uct_with_fast_reward": True,
    "aggregator": None,
    "node_visualizer": None,
    "distortion_fn": None,
    "log_distributions": False,
    "distribution_log_path": None,
    "visualize_key_nodes": False,
    "chain_propagate": True,
}


def _reject_unsupported(options: Dict[str, Any]) -> None:
    for name, value in options.items():
        if name not in _UNSUPPORTED_DEFAULTS:
            raise ValueError("unsupported PlanU option: {}".format(name))
        default = _UNSUPPORTED_DEFAULTS[name]
        if callable(default):
            is_default = value is default
        else:
            is_default = value == default
        if not is_default:
            raise ValueError(
                "unsupported non-default PlanU option: {}".format(name)
            )


class PlanU(SearchAlgorithm):
    def __init__(
        self,
        n_iters: int = 10,
        depth_limit: int = 12,
        n_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 100.0,
        risk_distortion: float = 0.0,
        quantile_learning_rate: float = 0.75,
        output_trace_in_each_iter: bool = False,
        seed: int = 100,
        disable_tqdm: bool = True,
        **unsupported: Any
    ) -> None:
        super().__init__()
        _reject_unsupported(unsupported)
        self.n_iters = n_iters
        self.output_trace_in_each_iter = output_trace_in_each_iter
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.disable_tqdm = disable_tqdm
        self.config = blockworld_config(
            n_iters=n_iters,
            depth=depth_limit,
            n_atoms=n_atoms,
            risk=risk_distortion,
            lr=quantile_learning_rate,
            vmin=v_min,
            vmax=v_max,
        )
        self.search: Optional[PlanUSearch] = None

    def __call__(
        self,
        world_model: Any,
        search_config: Any,
        **kwargs: Any
    ) -> PlanUResult:
        unsupported_call_options = {
            name: value
            for name, value in kwargs.items()
            if name != "log_file" or value is not None
        }
        if unsupported_call_options:
            names = ", ".join(sorted(unsupported_call_options))
            raise ValueError(
                "unsupported PlanU call option(s): {}".format(names)
            )

        adapter = BlockWorldAdapter(world_model, search_config)
        search = PlanUSearch(
            adapter,
            BlockWorldScorer(),
            self.config,
        )
        self.search = search
        iterations = [
            search.run_iteration(index, self.rng)
            for index in range(self.n_iters)
        ]
        assert search.root is not None
        terminated = [result for result in iterations if result.terminated]
        best = max(
            terminated or iterations,
            key=lambda result: sum(result.rewards),
        )
        state_trace = [node.state for node in best.state_path]
        action_trace = [
            node.action.payload for node in best.action_path
        ]
        trace_in_each_iter = None
        if self.output_trace_in_each_iter:
            trace_in_each_iter = [
                result.state_path for result in iterations
            ]
        return PlanUResult(
            terminal_state=best.final_observation,
            cum_reward=sum(best.rewards),
            trace=(state_trace, action_trace),
            trace_of_nodes=best.state_path,
            tree_state=search.root,
            trace_in_each_iter=trace_in_each_iter,
            tree_state_after_each_iter=None,
            aggregated_result=None,
        )


__all__ = ["PlanU", "PlanUNode", "PlanUResult"]
