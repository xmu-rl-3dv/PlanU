import os
import sys
import pathlib

root = str(pathlib.Path(__file__).parents[2])
sys.path.append(root)

import argparse
import os
import random
import time
from distutils.util import strtobool
import copy
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import virtual_home
from mcts.virtualhome.PlanU_v1 import LLMAgent, LanguageNode, ActionNode
from datetime import datetime
import logging
import json
from torch.utils.tensorboard import SummaryWriter
from rnd import RndRewardModel
log_dir = 'log/food_preparation/mcts+'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

now = datetime.now()
time_str = now.strftime('%Y%m%d_%H%M%S')
# run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{time_str}"
# if args.track:
#     import wandb

#     wandb.init(
#         project=args.wandb_project_name,
#         entity=args.wandb_entity,
#         sync_tensorboard=True,
#         config=vars(args),
#         name=run_name,
#         monitor_gym=True,
#         save_code=True,
#     )
# writer = SummaryWriter(f"{args.record_path}/{run_name}")



def make_env(env_id, seed, idx, capture_video, run_name, env_params):
    def thunk():

        env = gym.make(env_id, **env_params)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        return env

    return thunk

def main():

    device = torch.device("cuda")
    parser = argparse.ArgumentParser()
    parser.add_argument('--stochastic', type=float,default=0.2,action= "store",help='stochatic')
    parser.add_argument('--valueweight', type=float, default=0.5, action= "store",help='value_weight')
    parser.add_argument('--maxiterations', type=int, default=1000,action= "store", help='max_iteration')
    parser.add_argument('--transpositions', action='store',type = bool,default = False, help = 'if use transopositions')
    parser.add_argument('--rnd', action='store',type = bool,default = False, help = 'if use rnd')
    parser.add_argument('--base-model',     action='store',        type=str,             default="meta-llama/Meta-Llama-3-8B-Instruct",               help='''select a base model from below:"meta-llama/Meta-Llama-3-8B-Instruct",
                    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"''')
    parser.add_argument('--seed', type=int, default=100,action= "store", help='random seed')  
    args= parser.parse_args()

    log_file = os.path.join(log_dir, f'stochastc={args.stochastic}_vw={args.valueweight}_{time_str}.txt')
    logging.basicConfig(filename=log_file, level=logging.INFO, 
                        format='%(asctime)s - %(message)s')

    seed = args.seed
    env_params = {
        'seed': seed,
        'debug': False,
    }
    rnd_writer = SummaryWriter(f"./rnd_results/Model={args.base_model}/food/PlanU_nollm/seed={args.seed}/stochastic={args.stochastic}/rnd={args.rnd}")
    writer = SummaryWriter(f"./results/Model={args.base_model}/food/PlanU_nollm/seed={args.seed}/stochastic={args.stochastic}/rnd={args.rnd}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    print("play virtual home v1")

    envs = gym.vector.SyncVectorEnv(
        [make_env("VirtualHome-v1", seed, 0, False, "tmp", env_params) for i in
         range(1)]
    )

    print("play virtual home v1")
    # load_path = os.path.join(root, "checkpoints", "food_preparation", "lora")



    agent = LLMAgent(normalization_mode="word", rnd = args.rnd,tb_logger=rnd_writer,base_model = args.base_model
                     )

    success_rate = 0
    traj = []
    traj_rewards = []
    rewards_list = []
    step_list = []
    obs = envs.reset()
    root = LanguageNode(state=obs,initial_value=torch.tensor([1]))
    _ = agent.expand(obs,root,envs)
    for i in range(args.maxiterations):  #100
        logging.info(f"New round : {i} -----------------------------------------------------------------------------------------------------\n")
        steps = 0
        done = False
        rewards = 0
        reward_list = []
        reward_list = torch.tensor(reward_list,device=device)
        discount = 1
        # root._visit_count +=1
        node = root
        node_path = []
        obs = envs.reset()
        action_list = []
        stoc =True
        train_data_cnt = 0
        while not done and steps < 15:
            assert(type(node)==LanguageNode)
            node._visit_count += 1
            if node.is_leaf():
                agent.expand(obs, node,envs,False,value_weight= args.valueweight)
            steps += 1

            obs_temp = obs
            envs_temp = copy.deepcopy(envs)
            done_temp = done    

            action, value, next_node, action_name = agent.select(obs, node)
            node_path.append(next_node)
            action = action.cpu().numpy()
            action_list.append(action_name)
            print("action", action, 'action name', action_name)


            obs, reward, done, info = envs.step(action)
            if reward <= 0:
                reward = np.array([-0.001])
            if stoc and 'open' in action_name and random.random()<args.stochastic: # state transfer uncertainty. Stochastic situation
                obs = obs_temp
                envs = envs_temp
                reward = np.array([-0.001])
                done = done_temp
                # stoc = False
            
            logging.info(f"action : {action_name}  reward : {reward}")
            if args.rnd:
                agent.collect_data(obs)
            rewards += reward * discount
            reward_list=torch.cat((reward_list, torch.tensor(reward, device=device)))
            next_node = agent.expand(obs,next_node,envs, type(next_node) == ActionNode,value_weight = args.valueweight)
            discount *= 0.99
            node = next_node
        agent.mcts_update(node_path, reward_list)
        step_list.append(steps)
        if rewards > 0:
            success_rate += 1
        if args.rnd and train_data_cnt > 15:
            agent.train()
    
        print(i, steps, rewards)
        logging.info(f"steps : {steps}, rewards : {rewards}")
        writer.add_scalar("charts/episodic_return", rewards, i)
        writer.add_scalar("charts/episodic_length", steps, i)
    writer.add_text("scalrs/consumed_tokens",str(agent.total_llm_tokenizer_token),global_step = 0)
    writer.add_text("scalrs/query_times",str(agent.total_llm_tokenizer_call),global_step = 0)
    with open("token_consumed.txt", "a") as f:
        f.write("--------------------------------------\n")
        f.write(f"./results_new/Model={args.base_model}/fp/PlanU/seed={args.seed}/stochastic={args.stochastic}/rnd={args.rnd}")
        f.write(f"consumed_tokens={agent.total_llm_tokenizer_token}\n")
        f.write(f"query_times={agent.total_llm_tokenizer_call}\n")
        f.write("--------------------------------------\n")
    # print(np.mean(reward_list), np.std(reward_list))
    # print(np.mean(step_list), np.std(step_list))
    # print(success_rate)
    # logging.info(f"mean reward : {np.mean(reward_list)}, std reward : {np.std(reward_list)}")
    # logging.info(f"mean step : {np.mean(step_list)}, std step : {np.std(step_list)}")
    # logging.info(f"success rate : {success_rate}\n")


if __name__ == '__main__':
    main()

