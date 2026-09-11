import argparse
from dataclasses import asdict
import hashlib
from importlib import metadata as importlib_metadata
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
from typing import Optional, Sequence

import numpy as np

_REPOSITORY_ROOT = str(Path(__file__).resolve().parents[2])
if _REPOSITORY_ROOT not in sys.path:
    sys.path.insert(0, _REPOSITORY_ROOT)

from mcts.overcooked.PlanU_mcts import OvercookedActionScorer
from planu_core.adapters.overcooked import OvercookedAdapter, overcooked_config
from planu_core.curiosity import RndCuriosity
from planu_core.search import PlanUSearch


_PROVENANCE_PACKAGES = (
    "numpy",
    "torch",
    "gym",
    "transformers",
    "peft",
    "ding",
)


def _build_effective_config(args, config):
    return {
        "args": dict(vars(args)),
        "planu_config": asdict(config),
    }


def _config_hash(effective_config) -> str:
    serialized = json.dumps(
        effective_config,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]


def _build_run_paths(args, run_name, config_hash):
    result_path = (
        f"./results/Model={args.base_model}/{run_name}/PlanU/"
        f"seed={args.seed}/stochastic={args.stochastic}/rnd={args.rnd}/"
        f"config={config_hash}"
    )
    rnd_path = (
        f"./rnd_reward/Model={args.base_model}/{run_name}/PlanU/"
        f"seed={args.seed}/{args.rnd}/transpositions={args.transpositions}/"
        f"config={config_hash}"
    )
    return result_path, rnd_path


def _git_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return completed.stdout.strip() or "unknown"


def _installed_versions(packages=_PROVENANCE_PACKAGES):
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _build_run_metadata():
    return {
        "git_commit": _git_commit(),
        "python_version": platform.python_version(),
        "packages": _installed_versions(_PROVENANCE_PACKAGES),
    }


def _record_run_provenance(writer, effective_config, run_metadata) -> None:
    writer.add_text(
        "planu/effective_config",
        json.dumps(effective_config, sort_keys=True),
        global_step=0,
    )
    writer.add_text(
        "planu/run_metadata",
        json.dumps(run_metadata, sort_keys=True),
        global_step=0,
    )


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
    parser.add_argument(
        "--exp-name",
        type=str,
        default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment",
    )
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument(
        "--cuda",
        type=_parse_bool,
        default=True,
        nargs="?",
        const=True,
    )
    parser.add_argument(
        "--capture-video",
        type=_parse_bool,
        default=False,
        nargs="?",
        const=True,
    )
    parser.add_argument("--total-timesteps", type=int, default=1000000)
    parser.add_argument("--num-steps", type=int, default=32)
    parser.add_argument("--depth", type=int, default=15)
    parser.add_argument(
        "--env-id",
        action="store",
        type=str,
        default="Overcooked-LLMA-v3",
    )
    parser.add_argument("--n-agent", action="store", type=int, default=1)
    parser.add_argument(
        "--grid-dim",
        action="store",
        type=int,
        nargs=2,
        default=[7, 7],
    )
    parser.add_argument("--task", action="store", type=int, default=3)
    parser.add_argument("--map-type", action="store", type=str, default="A")
    parser.add_argument("--obs-radius", action="store", type=int, default=2)
    parser.add_argument(
        "--env-reward",
        action="store",
        type=float,
        nargs=4,
        default=[0.1, 1, 0, 0.001],
    )
    parser.add_argument("--mode", action="store", type=str, default="vector")
    parser.add_argument(
        "--debug",
        action="store",
        type=_parse_bool,
        default=False,
    )
    parser.add_argument(
        "--save-path",
        action="store",
        type=str,
        default="saved_models",
    )
    parser.add_argument("--save-interval", action="store", type=int, default=10)
    parser.add_argument(
        "--record-path",
        action="store",
        type=str,
        default="llm5_runs",
    )
    parser.add_argument(
        "--normalization-mode",
        action="store",
        type=str,
        default="token",
    )
    parser.add_argument("--value-weight", action="store", type=float, default=0.5)
    parser.add_argument("--stochastic", action="store", type=float, default=0.2)
    parser.add_argument(
        "--transpositions",
        action="store",
        type=_parse_bool,
        default=False,
    )
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument(
        "--maxiterations",
        action="store",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--rnd",
        action="store",
        type=_parse_bool,
        default=False,
    )
    parser.add_argument(
        "--init_dist",
        action="store",
        type=_parse_bool,
        default=False,
    )
    parser.add_argument(
        "--base-model",
        action="store",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
    )
    parser.add_argument(
        "--temperature",
        action="store",
        type=float,
        default=1.0,
    )
    return parser


def parse_args(argv: Optional[Sequence[str]] = None):
    return build_parser().parse_args(argv)


def validate_args(args) -> None:
    assert args.num_envs == 1, "num_envs must be exactly 1"
    if args.init_dist:
        raise NotImplementedError(
            "init_dist=True is not supported by the unified scalar PlanU core"
        )


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
        from gym_macro_overcooked.macActEnvWrapper import MacEnvWrapper

        env = MacEnvWrapper(gym.make(env_id, **env_params))
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def _build_curiosity(args, device, writer):
    from easydict import EasyDict

    from mcts.overcooked.rnd import RndRewardModel

    config = EasyDict(
        type="rnd",
        intrinsic_reward_type="assign",
        learning_rate=1e-5,
        batch_size=15,
        obs_shape=18 if args.task == 0 else 26,
        hidden_size_list=[64, 64, 128],
        update_per_collect=5,
        obs_norm=True,
        obs_norm_clamp_min=-1,
        obs_norm_clamp_max=1,
        intrinsic_reward_weight=0.01,
        extrinsic_reward_norm=True,
        extrinsic_reward_norm_max=1,
    )
    model = RndRewardModel(config, device=str(device), tb_logger=writer)
    return RndCuriosity(model, minimum_samples=15)


def build_planu_components(args, envs, device, rnd_writer, config=None):
    scorer = OvercookedActionScorer(
        args.base_model,
        normalization_mode=args.normalization_mode,
        temperature=args.temperature,
        device=str(device),
    )
    adapter = OvercookedAdapter(
        envs,
        task=args.task,
        stochastic_probability=args.stochastic,
        rng=np.random.default_rng(args.seed),
    )
    if config is None:
        config = overcooked_config(
            task=args.task,
            rnd=args.rnd,
            max_iterations=args.maxiterations,
            max_depth=args.depth,
        )
    curiosity = (
        _build_curiosity(args, device, rnd_writer) if args.rnd else None
    )
    search = PlanUSearch(adapter, scorer, config, curiosity)
    return search, scorer, config


def run(args) -> None:
    validate_args(args)

    config = overcooked_config(
        task=args.task,
        rnd=args.rnd,
        max_iterations=args.maxiterations,
        max_depth=args.depth,
    )
    effective_config = _build_effective_config(args, config)
    config_hash = _config_hash(effective_config)
    run_metadata = _build_run_metadata()

    import gym
    import torch
    from torch.utils.tensorboard import SummaryWriter

    time_str = time.strftime("%Y%m%d_%H_%M_%S", time.localtime(time.time()))
    run_name = (
        f"{args.env_id}__{args.exp_name}__{args.seed}__{time_str}__SAMCTS"
    )
    if args.task == 0:
        run_name = "tomato_salad_temeprature"
    else:
        run_name = "tomato_lettuce_salad_temperature"

    result_path, rnd_path = _build_run_paths(
        args,
        run_name,
        config_hash,
    )
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
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        device = torch.device(
            "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
        )

        reward_list = {
            "subtask finished": args.env_reward[0],
            "correct delivery": args.env_reward[1],
            "wrong delivery": -args.env_reward[2],
            "step penalty": -args.env_reward[3],
        }
        task_list = [
            "tomato salad",
            "lettuce salad",
            "onion salad",
            "lettuce-tomato salad",
            "onion-tomato salad",
            "lettuce-onion salad",
            "lettuce-onion-tomato salad",
        ]
        env_params = {
            "grid_dim": args.grid_dim,
            "task": task_list[args.task],
            "rewardList": reward_list,
            "map_type": args.map_type,
            "n_agent": args.n_agent,
            "obs_radius": args.obs_radius,
            "mode": args.mode,
            "debug": args.debug,
        }
        envs = gym.vector.SyncVectorEnv(
            [
                make_env(
                    args.env_id,
                    args.seed + index,
                    index,
                    args.capture_video,
                    run_name,
                    env_params,
                )
                for index in range(args.num_envs)
            ]
        )
        assert isinstance(
            envs.single_action_space,
            gym.spaces.Discrete,
        ), "only discrete action space is supported"

        search, scorer, config = build_planu_components(
            args,
            envs,
            device,
            rnd_writer,
            config,
        )

        trajectory_rewards = []
        global_step = 0
        for iteration in range(args.maxiterations):
            result = search.run_iteration(
                iteration,
                np.random.default_rng(args.seed + iteration),
            )
            episodic_return = float(sum(result.rewards))
            episodic_length = len(result.rewards)
            trajectory_rewards.append(episodic_return)
            global_step += episodic_length
            print(
                f"global_step={global_step}, num_path = {iteration}, "
                f"episodic_return={episodic_return}, "
                f"episodic_length={episodic_length}"
            )
            writer.add_scalar(
                "charts/episodic_return",
                episodic_return,
                global_step,
            )
            writer.add_scalar(
                "charts/episodic_length",
                episodic_length,
                global_step,
            )

        num_success = sum(reward > 1 for reward in trajectory_rewards)
        print(num_success)
        writer.add_text("scalrs/num_success", str(num_success), global_step=0)
        writer.add_text(
            "scalrs/consumed_tokens",
            str(scorer.total_llm_tokenizer_token),
            global_step=0,
        )
        writer.add_text(
            "scalrs/query_times",
            str(scorer.total_llm_tokenizer_call),
            global_step=0,
        )
        with open("token_consumed.txt", "a") as token_file:
            token_file.write("--------------------------------------\n")
            token_file.write(
                f"./results_new/Model={args.base_model}/{run_name}/PlanU/"
                f"seed={args.seed}/stochastic={args.stochastic}/"
                f"rnd={args.rnd}\n"
            )
            token_file.write(
                "consumed_tokens="
                f"{scorer.total_llm_tokenizer_token}\n"
            )
            token_file.write(
                f"query_times={scorer.total_llm_tokenizer_call}\n"
            )
            token_file.write("--------------------------------------\n")
        print(
            f"./results_new/Model={args.base_model}/{run_name}/PlanU/"
            f"seed={args.seed}/stochastic={args.stochastic}/rnd={args.rnd}"
        )
        print(scorer.total_llm_tokenizer_token)
    finally:
        if envs is not None:
            envs.close()
        rnd_writer.close()
        writer.close()


def main(argv: Optional[Sequence[str]] = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
