import argparse
import logging
import os
from pathlib import Path
import random
import sys
import time
from typing import Optional, Sequence

import numpy as np


_REPOSITORY_ROOT = str(Path(__file__).resolve().parents[2])
if _REPOSITORY_ROOT not in sys.path:
    sys.path.insert(0, _REPOSITORY_ROOT)

from planu_core.adapters.virtualhome import (
    VirtualHomeAdapter,
    VirtualHomeTask,
    virtualhome_config,
)
from planu_core.curiosity import RndCuriosity
from planu_core.provenance import (
    build_effective_config,
    build_run_metadata,
    config_hash,
    record_run_provenance,
)
from planu_core.scorers import ConstantActionScorer
from planu_core.search import PlanUSearch


_build_effective_config = build_effective_config
_config_hash = config_hash
_record_run_provenance = record_run_provenance


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    normalized = value.lower()
    if normalized in {"y", "yes", "t", "true", "on", "1"}:
        return True
    if normalized in {"n", "no", "f", "false", "off", "0"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid truth value: {value!r}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stochastic", type=float, default=0.2)
    parser.add_argument("--valueweight", type=float, default=0.5)
    parser.add_argument("--maxiterations", type=int, default=1000)
    parser.add_argument("--depth", type=int, default=15)
    parser.add_argument(
        "--transpositions",
        type=_parse_bool,
        default=False,
        nargs="?",
        const=True,
    )
    parser.add_argument(
        "--rnd",
        type=_parse_bool,
        default=False,
        nargs="?",
        const=True,
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
    )
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--num-envs", type=int, default=1)
    return parser


def parse_args(argv: Optional[Sequence[str]] = None):
    return build_parser().parse_args(argv)


def validate_args(args) -> None:
    assert args.num_envs == 1, "num_envs must be exactly 1"


def make_env(
    env_id,
    seed,
    idx,
    capture_video,
    run_name,
    env_params,
):
    def thunk():
        import gym

        env = gym.make(env_id, **env_params)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        return env

    return thunk


def rnd_settings(task: VirtualHomeTask):
    return {
        "type": "rnd",
        "intrinsic_reward_type": "assign",
        "learning_rate": 1e-5,
        "batch_size": 15,
        "obs_shape": task.obs_shape,
        "hidden_size_list": [64, 64, 128],
        "update_per_collect": 20,
        "obs_norm": True,
        "obs_norm_clamp_min": -1,
        "obs_norm_clamp_max": 1,
        "intrinsic_reward_weight": 0.01,
        "extrinsic_reward_norm": True,
        "extrinsic_reward_norm_max": 1,
    }


def _build_curiosity(task, device, writer):
    from easydict import EasyDict

    from mcts.virtualhome.rnd import RndRewardModel

    model = RndRewardModel(
        EasyDict(rnd_settings(task)),
        device=str(device),
        tb_logger=writer,
    )
    return RndCuriosity(model, minimum_samples=15)


def build_planu_components(args, envs, device, rnd_writer, config=None):
    task = VirtualHomeTask.FOOD
    scorer = ConstantActionScorer(1.0)
    adapter = VirtualHomeAdapter(
        envs,
        task=task,
        stochastic_probability=args.stochastic,
        rng=np.random.default_rng(args.seed),
    )
    if config is None:
        config = virtualhome_config(
            task,
            rnd=args.rnd,
            max_iterations=args.maxiterations,
            max_depth=args.depth,
        )
    curiosity = (
        _build_curiosity(task, device, rnd_writer) if args.rnd else None
    )
    return PlanUSearch(adapter, scorer, config, curiosity), scorer, config


def discounted_return(rewards, discount: float = 0.99) -> float:
    return float(
        sum(float(reward) * discount ** index for index, reward in enumerate(rewards))
    )


def is_success(episodic_return: float) -> bool:
    return episodic_return > 0.0


def _build_run_paths(args, config_digest):
    result_path = (
        f"./results/Model={args.base_model}/food/PlanU_nollm/"
        f"seed={args.seed}/stochastic={args.stochastic}/rnd={args.rnd}/"
        f"config={config_digest}"
    )
    rnd_path = (
        f"./rnd_results/Model={args.base_model}/food/PlanU_nollm/"
        f"seed={args.seed}/stochastic={args.stochastic}/rnd={args.rnd}/"
        f"config={config_digest}"
    )
    return result_path, rnd_path


def _token_log_path(args, config_digest):
    return (
        f"./results_new/Model={args.base_model}/fp/PlanU/"
        f"seed={args.seed}/stochastic={args.stochastic}/rnd={args.rnd}/"
        f"config={config_digest}"
    )


def run(args) -> None:
    validate_args(args)
    task = VirtualHomeTask.FOOD
    config = virtualhome_config(
        task,
        rnd=args.rnd,
        max_iterations=args.maxiterations,
        max_depth=args.depth,
    )
    effective_config = _build_effective_config(args, config)
    config_digest = _config_hash(effective_config)
    run_metadata = build_run_metadata(_REPOSITORY_ROOT)

    import gym
    import torch
    from torch.utils.tensorboard import SummaryWriter
    import virtual_home  # noqa: F401

    result_path, rnd_path = _build_run_paths(args, config_digest)
    writer = SummaryWriter(result_path)
    rnd_writer = SummaryWriter(rnd_path)
    envs = None
    scorer = None
    try:
        _record_run_provenance(writer, effective_config, run_metadata)
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s"
            % "\n".join(
                f"|{key}|{value}|" for key, value in vars(args).items()
            ),
        )

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        env_params = {"seed": args.seed, "debug": False}
        envs = gym.vector.SyncVectorEnv(
            [
                make_env(
                    task.env_id,
                    args.seed,
                    index,
                    False,
                    "tmp",
                    env_params,
                )
                for index in range(args.num_envs)
            ]
        )

        search, scorer, config = build_planu_components(
            args,
            envs,
            device,
            rnd_writer,
            config,
        )
        log_dir = "log/food_preparation/mcts+"
        os.makedirs(log_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        logging.basicConfig(
            filename=os.path.join(
                log_dir,
                f"stochastc={args.stochastic}_vw={args.valueweight}_"
                f"{timestamp}.txt",
            ),
            level=logging.INFO,
            format="%(asctime)s - %(message)s",
        )

        print("play virtual home v1")
        num_success = 0
        for iteration in range(args.maxiterations):
            result = search.run_iteration(
                iteration,
                np.random.default_rng(args.seed + iteration),
            )
            episodic_return = discounted_return(result.rewards)
            episodic_length = len(result.rewards)
            if is_success(episodic_return):
                num_success += 1
            print(iteration, episodic_length, episodic_return)
            logging.info(
                "steps : %s, rewards : %s",
                episodic_length,
                episodic_return,
            )
            writer.add_scalar(
                "charts/episodic_return",
                episodic_return,
                iteration,
            )
            writer.add_scalar(
                "charts/episodic_length",
                episodic_length,
                iteration,
            )

        consumed_tokens = getattr(scorer, "total_llm_tokenizer_token", 0)
        query_times = getattr(scorer, "total_llm_tokenizer_call", 0)
        writer.add_text(
            "scalrs/num_success",
            str(num_success),
            global_step=0,
        )
        writer.add_text(
            "scalrs/consumed_tokens",
            str(consumed_tokens),
            global_step=0,
        )
        writer.add_text(
            "scalrs/query_times",
            str(query_times),
            global_step=0,
        )
        with open("token_consumed.txt", "a") as token_log:
            token_log.write("--------------------------------------\n")
            token_log.write(f"{_token_log_path(args, config_digest)}\n")
            token_log.write(f"consumed_tokens={consumed_tokens}\n")
            token_log.write(f"query_times={query_times}\n")
            token_log.write("--------------------------------------\n")
    finally:
        if envs is not None:
            envs.close()
        writer.close()
        rnd_writer.close()


def main(argv: Optional[Sequence[str]] = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
