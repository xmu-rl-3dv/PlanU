from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import random
import re
import sys
from typing import Any, Mapping, Optional, Sequence, Tuple


def _preconfigure_cuda_visibility(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-g", "--gpu")
    args, _ = parser.parse_known_args(argv)
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)


_preconfigure_cuda_visibility(sys.argv[1:])

import numpy as np

PLANU_ROOT = str(Path(__file__).resolve().parents[1])
BLOCKWORLD_ROOT = Path(__file__).resolve().parent
if PLANU_ROOT not in sys.path:
    sys.path.insert(0, PLANU_ROOT)

from reasoners import LanguageModel, Reasoner, SearchConfig, WorldModel
from reasoners.algorithm import MCTS, PlanU
import reasoners.benchmark.bw_utils as utils
from reasoners.benchmark import BWEvaluator
from reasoners.lm import HFModel
from planu_core.adapters.blockworld import BWStateRAP
from planu_core.provenance import (
    build_effective_config,
    build_run_metadata,
    config_hash,
    write_json_provenance,
)
import torch

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def _positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run BlocksWorld evaluation.")
    parser.add_argument('-a', '--algorithm', choices=['mcts', 'planu'], required=True,
                        help="Specify the algorithm to use (MCTS or PlanU).")
    parser.add_argument('-g', '--gpu', type=int,
                    help="Specify the GPU ID to use, e.g., -g 0 for GPU 0.")
    parser.add_argument('-v', '--version', choices=['1', '2'], required=True,
                        help="Specify the version of the prompt to use (v1 or v2).")
    parser.add_argument('-s', '--steps', choices=['2','4','6','8','10','12'], required=True,
                        help="Specify the number of steps to use in the prompt.")
    parser.add_argument('-n', '--n_quantile', type=int, default=51,
                        help="Specify the number of quantiles to use for DMCTS algorithm.")
    parser.add_argument('--seed', type=int, default=100,
                        help="Seed for Python, NumPy, Torch, and PlanU.")
    parser.add_argument('--n-iters', type=int, default=10,
                        help="Number of search iterations.")
    parser.add_argument('--success-probability', type=float, default=0.8,
                        help="Probability that a BlockWorld action succeeds.")
    parser.add_argument(
        "--model",
        default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        help="Hugging Face model and tokenizer identifier.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Execution device; auto selects CUDA when available.",
    )
    parser.add_argument(
        "--max-examples",
        type=_positive_int,
        default=None,
        help="Optional dataset limit for smoke runs; omitted for paper runs.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional root directory for evaluator logs.",
    )

    return parser.parse_args(argv)


def resolve_device(requested: str, cuda_available: bool) -> str:
    if requested == "auto":
        return "cuda" if cuda_available else "cpu"
    if requested == "cuda" and not cuda_available:
        raise ValueError("CUDA was requested but is not available")
    return requested


def _model_placement_metadata(
    model: Any,
    requested_device: str,
    resolved_device: str,
    cuda_visible_devices: Optional[str],
    cuda_device_names: Sequence[str],
):
    raw_device_map = getattr(
        getattr(model, "model", None),
        "hf_device_map",
        None,
    )
    if isinstance(raw_device_map, Mapping):
        device_map = {
            str(name): str(device)
            for name, device in sorted(
                raw_device_map.items(),
                key=lambda item: str(item[0]),
            )
        }
    elif raw_device_map is None:
        device_map = None
    else:
        device_map = str(raw_device_map)
    return {
        "requested_device": requested_device,
        "resolved_device": resolved_device,
        "hf_device_map": device_map,
        "cuda_visible_devices": cuda_visible_devices,
        "cuda_device_names": list(cuda_device_names),
    }


def benchmark_paths(version: str, steps: str) -> tuple[str, str]:
    if version == "v1":
        prompt_path = (
            "examples/CoT/blocksworld/prompts/pool_prompt_v1.json"
        )
    else:
        prompt_path = (
            "examples/CoT/blocksworld/prompts/"
            f"pool_prompt_{version}_step_{steps}.json"
        )
    data_path = (
        "examples/CoT/blocksworld/data/"
        f"split_{version}/split_{version}_step_{steps}_data.json"
    )
    return prompt_path, data_path


def _build_effective_run_config(
    args,
    algorithm_config,
    model_identifier,
    depth,
    resolved_device,
):
    effective_config = build_effective_config(
        args,
        algorithm_config,
        config_name="{}_config".format(args.algorithm),
    )
    effective_config["args"].pop("output_dir", None)
    effective_config.update(
        {
            "model_identifier": model_identifier,
            "depth": depth,
            "success_probability": args.success_probability,
            "model": {
                "identifier": model_identifier,
                "requested_device": args.device,
                "device": resolved_device,
                "max_batch_size": 1,
                "max_new_tokens": 200,
                "max_length": 2048,
            },
            "benchmark": {
                "prompt_path": benchmark_paths(
                    "v{}".format(args.version),
                    args.steps,
                )[0],
                "data_path": benchmark_paths(
                    "v{}".format(args.version),
                    args.steps,
                )[1],
                "config_file": (
                    "examples/CoT/blocksworld/data/bw_config.yaml"
                ),
                "domain_file": (
                    "examples/CoT/blocksworld/data/generated_domain.pddl"
                ),
                "max_steps": 12,
                "num_shot": 4,
                "shuffle_prompt": True,
                "max_examples": args.max_examples,
            },
        }
    )
    return effective_config


def _build_log_dir(
    args,
    model_identifier,
    effective_config_hash,
    base_dir=BLOCKWORLD_ROOT / "logs",
):
    model_path = model_identifier.replace("/", "--")
    return (
        Path(base_dir)
        / "v{}_{}".format(args.version, args.steps)
        / "{}_{}".format(args.algorithm, model_path)
        / "seed={}".format(args.seed)
        / "config={}".format(effective_config_hash)
    )


BWAction = str


class BlocksWorldModelRAP(WorldModel):
    def __init__(self,
                 base_model: LanguageModel,
                 prompt: dict,
                 max_steps: int = 4,
                 batch_size: int = 1,
                 success_probability: float = 0.8,
                 rng: Optional[np.random.Generator] = None) -> None:
        super().__init__()
        self.max_steps = max_steps
        self.base_model = base_model
        self.prompt = prompt
        self.batch_size = batch_size
        self.success_probability = success_probability
        self.rng = np.random.default_rng() if rng is None else rng

    def init_state(self) -> BWStateRAP:
        return BWStateRAP(
            step_idx=0,
            last_blocks_state="",
            blocks_state=utils.extract_init_state(self.example),
            buffered_action=""
        )



    def step(self, state: BWStateRAP, action: BWAction) -> Tuple[BWStateRAP, dict]:
        state = copy.deepcopy(state)
        blocks_state = state.blocks_state
        step_idx = state.step_idx

        new_blocks_state = self.update_blocks(blocks_state, action)
        success = new_blocks_state != blocks_state
        new_buffered_action = action if state.buffered_action == "" else ""
        state = BWStateRAP(
            step_idx=step_idx + 1,
            last_blocks_state=state.blocks_state,
            blocks_state=new_blocks_state,
            buffered_action=new_buffered_action
        )
        return state, {
            "goal_reached": utils.goal_check(utils.extract_goals(self.example), new_blocks_state),
            "success": success
        }

    def update_blocks(self, block_states: str, action: BWAction) -> str:
        success = self.determine_success(block_states, action)

        if "pick" in action:
            key = "world_update_pickup"
        elif "unstack" in action:
            key = "world_update_unstack"
        elif "put" in action:
            key = "world_update_putdown"
        elif "stack" in action:
            key = "world_update_stack"
        else:
            raise ValueError("Invalid action")
        # if success == False and key == "world_update_stack":
        if success == False:
            print("Failed to execute action")
            new_state = block_states
        else:
            world_update_prompt = self.prompt[key].format(block_states, action.capitalize() + ".")
            # print("world_update_prompt: ")
            # print(world_update_prompt)
            world_output = self.base_model.generate(
                [world_update_prompt],
                eos_token_id="\n",
                hide_input=True,
                temperature=0
            ).text[0].strip()

            world_output = re.split(r'\n\[', world_output)[0]
            # print(f"world_output: {world_output}")

            new_state = utils.apply_change(world_output, block_states) if success else block_states
        return new_state

    def determine_success(self, block_states: str, action: BWAction) -> bool:
        return self.rng.random() < self.success_probability

    def is_terminal(self, state: BWStateRAP) -> bool:
        return bool(
            utils.goal_check(
                utils.extract_goals(self.example),
                state.blocks_state,
            )[0]
        )

class BWConfigRAP(SearchConfig):
    def __init__(self,
                 base_model: LanguageModel,
                 prompt: dict,
                 batch_size: int = 1,
                 reward_alpha: float = 0.5,
                 goal_reward_default: float = 0.,
                 goal_reached_reward: float = 100.) -> None:
        super().__init__()
        self.base_model = base_model
        self.example = None
        self.prompt = prompt
        self.batch_size = batch_size
        self.reward_alpha = reward_alpha
        self.goal_reward_default = goal_reward_default
        self.goal_reached_reward = goal_reached_reward

    def get_actions(self, state: BWStateRAP) -> list[BWAction]:
        blocks_state = state.blocks_state
        return utils.generate_all_actions(blocks_state)

    def fast_reward(self, node: Any, action: BWAction) -> tuple[float, dict]:
        if node.state.buffered_action == "":
            current_blocks_state = node.state.blocks_state
        else:
            current_blocks_state = node.state.last_blocks_state
        previous_action = node.state.buffered_action + "\n" if node.state.buffered_action != "" else ""

        # every two steps, we will also reduce the icl examples by 2 steps
        # so that the distribution of step length in examples is more reasonable
        N = len(node.cum_rewards)

        # icl_template = self.prompt["icl_list"][node.state.step_idx // 2]
        index = node.state.step_idx // 2
        if index < len(self.prompt["icl_list"]):
            icl_template = self.prompt["icl_list"][index]
        else:
            self.prompt["icl_list"].append(self.prompt["icl_list"][-1])
            icl_template = self.prompt["icl_list"][index]


        inputs = (icl_template.replace("<init_state>", current_blocks_state)
                              .replace("<goals>", utils.extract_goals(self.example, return_raw=True))
                              .replace("<action>", previous_action))
        intuition = self.base_model.get_loglikelihood(inputs, [inputs + action])[0]

        self_eval_prompt = (self.prompt["self-eval"]
                                .replace("<init_state>", current_blocks_state)
                                .replace("<goals>", utils.extract_goals(self.example, return_raw=True))
                                .replace("<action>", action))
        self_eval = self.base_model.get_loglikelihood(self_eval_prompt, [self_eval_prompt + "good"])[0]

        return (self.new_calculate_reward(intuition, self_eval, N),
                {'intuition': intuition, "self_eval": self_eval})

    def calculate_reward(self, intuition, self_eval, goal_reached=None) -> float:
        # to provide a unified interface for reward and fast_reward
        if goal_reached is None:
            goal_reward = self.goal_reward_default
        elif goal_reached[0]:
            goal_reward = self.goal_reached_reward
        else:
            goal_reward = goal_reached[1]
        return (intuition + self_eval) * self.reward_alpha + goal_reward * (1 - self.reward_alpha)

    def new_calculate_reward(
        self,
        intuition,
        self_eval,
        N,
        goal_reached=None,
    ) -> float:
        # to provide a unified interface for reward and fast_reward
        if goal_reached is None:
            return intuition + self_eval
        if goal_reached[0]:
            goal_reward = self.goal_reached_reward
        else:
            goal_reward = goal_reached[1]
        denominator = max(1, N)
        return (
            (intuition + self_eval)
            / (denominator ** self.reward_alpha + 1)
            + goal_reward / denominator
        )


    def reward(self, node: Any, action: BWAction,
               intuition: float = None,
               self_eval: float = None,
               goal_reached: tuple[bool, float] = None,
               **aux) -> tuple[float, dict]:
        success = aux.get('success', None)

        print(f"Auxiliary data received in reward: {aux}")

        N = len(node.cum_rewards)

        return (self.new_calculate_reward(intuition, self_eval, N,goal_reached),
                {'intuition': intuition, 'goal_reached': goal_reached})
        # return (self.calculate_reward(intuition, self_eval, goal_reached),
        #         {'intuition': intuition, 'goal_reached': goal_reached})


# a helper function to extract the action history from the output of the algorithm

def bfs_bw_extractor(algo_output):

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    try:
        return "\n".join(algo_output.trace[1])
    except Exception as e:
        print("Error in output extraction,", e)
        return ""

def validate_gpu_ids(gpu_ids):
    available_gpus = list(range(torch.cuda.device_count()))
    invalid_ids = [gpu for gpu in gpu_ids if gpu not in available_gpus]
    print(invalid_ids)
    if invalid_ids:
        raise ValueError(f"Invalid GPU IDs: {invalid_ids}. Available GPUs: {available_gpus}")


if __name__ == "__main__":
    args = parse_args()
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    version = f"v{args.version}"
    steps = args.steps
    prompt_path, data_path = benchmark_paths(version, steps)
    prompt_path = BLOCKWORLD_ROOT / prompt_path
    data_path = BLOCKWORLD_ROOT / data_path

    with open(prompt_path) as f:
        prompt = json.load(f)

    print(f"Loaded prompt from {prompt_path}")
    print(f"Data path set to {data_path}")
    llama_path = args.model
    resolved_device = resolve_device(
        args.device,
        torch.cuda.is_available(),
    )
    model = HFModel(
        model_pth=llama_path,
        tokenizer_pth=llama_path,
        device=torch.device(resolved_device),
        max_batch_size=1,
        max_new_tokens=200,
        max_length=2048,
    )

    print("Model and tokenizer successfully loaded!")

    world_model = BlocksWorldModelRAP(
        base_model=model,
        prompt=prompt,
        max_steps=12,
        success_probability=args.success_probability,
        rng=np.random.default_rng(args.seed),
    )
    config = BWConfigRAP(base_model=model, prompt=prompt)
    print("reward_alpha :", config.reward_alpha)

    if int(steps) < 8:
        depth_l = 10
    else:
        depth_l = 12


    if args.algorithm == 'mcts':
        algorithm = MCTS(
            depth_limit=depth_l,
            disable_tqdm=False,
            output_trace_in_each_iter=True,
            n_iters=args.n_iters,
        )
    elif args.algorithm == 'planu':
        algorithm = PlanU(
            depth_limit=depth_l,
            disable_tqdm=False,
            output_trace_in_each_iter=True,
            n_iters=args.n_iters,
            n_atoms=args.n_quantile,
            v_min=-10.0,
            v_max=100.0,
            risk_distortion=0.0,
            seed=args.seed,
        )
    else:
        raise ValueError(f"Invalid algorithm: {args.algorithm}")

    if args.algorithm == "planu":
        algorithm_config = algorithm.config
    else:
        algorithm_config = {
            "depth_limit": depth_l,
            "disable_tqdm": False,
            "output_trace_in_each_iter": True,
            "n_iters": args.n_iters,
        }
    effective_config = _build_effective_run_config(
        args,
        algorithm_config,
        llama_path,
        depth_l,
        resolved_device,
    )
    effective_config_hash = config_hash(effective_config)
    run_metadata = build_run_metadata(repository_root=PLANU_ROOT)
    run_metadata["model_placement"] = _model_placement_metadata(
        model,
        requested_device=args.device,
        resolved_device=resolved_device,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        cuda_device_names=[
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
    )
    log_dir = _build_log_dir(
        args,
        llama_path,
        effective_config_hash,
        base_dir=(
            args.output_dir
            if args.output_dir is not None
            else BLOCKWORLD_ROOT / "logs"
        ),
    )
    run_metadata["output_dir"] = os.fspath(log_dir)
    write_json_provenance(log_dir, effective_config, run_metadata)

    reasoner_rap = Reasoner(world_model=world_model, search_config=config, search_algo=algorithm)

    evaluator = BWEvaluator(config_file=os.fspath(
                                BLOCKWORLD_ROOT / "examples/CoT/blocksworld/data/bw_config.yaml"),
                            domain_file=os.fspath(
                                BLOCKWORLD_ROOT / "examples/CoT/blocksworld/data/generated_domain.pddl"),
                            data_path=data_path,
                            init_prompt=prompt,
                            output_extractor=bfs_bw_extractor)
    if args.max_examples is not None:
        evaluator.full_dataset = evaluator.full_dataset[:args.max_examples]

    os.environ["VERSION"] = f"v{args.version}"
    os.environ["STEPS"] = args.steps
    os.environ['MODEL_DIR'] = llama_path
    os.environ['QUANTILE_N'] = str(args.n_quantile)
    os.environ['SUCCESS_PROBABILITY'] = str(args.success_probability)

    evaluator.evaluate(
        reasoner_rap,
        shuffle_prompt=True,
        num_shot=4,
        resume=0,
        log_dir=os.fspath(log_dir),
    )

# nohup python evaluate_stochastic.py -a planu -g 1 -v 2 -s 2&
# nohup python evaluate_stochastic.py -a mcts -g 0 -v 2 -s 2&
