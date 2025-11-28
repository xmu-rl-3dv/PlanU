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
from emcts_policy_v2 import LLMAgent, LanguageNode
from datetime import datetime
import logging
import json
from torch.utils.tensorboard import SummaryWriter
log_dir = 'log/food_preparation/mcts'
results_file = 'results/entertainment/mcts'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
if not os.path.exists(results_file):
    os.makedirs(results_file)
now = datetime.now()
time_str = now.strftime('%Y%m%d_%H%M%S')
value_weight = 0.2 ## hyperpara
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


log_file = os.path.join(log_dir, f'prompt_log_{time_str}_entertainment_deterministic_vw={value_weight}.txt')
logging.basicConfig(filename=log_file, level=logging.INFO, 
                    format='%(asctime)s - %(message)s')

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

    print("play agent")
    # load_path = os.path.join(root, "checkpoints", "food_preparation", "lora")

    parser = argparse.ArgumentParser()
    parser.add_argument('--stochastic', type=float,default=0.2,action= "store",help='stochatic')
    parser.add_argument('--valueweight', type=float, default=0.5, action= "store",help='value_weight')
    parser.add_argument('--maxiterations', type=int, default=1000,action= "store", help='max_iteration')

    parser.add_argument('--base-model',     action='store',        type=str,             default="meta-llama/Meta-Llama-3-8B-Instruct",               help='''select a base model from below:"meta-llama/Meta-Llama-3-8B-Instruct",
                        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"''')
    parser.add_argument('--seed', type=int, default=100,action= "store", help='random seed')  
    args= parser.parse_args()
    env_params = {
        'seed': args.seed,
        'debug': False,
    }
    writer = SummaryWriter(f"./results_grab/Model={args.base_model}/entertainment/emcts/seed={args.seed}/stochastic={args.stochastic}/")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    print("play virtual home v1")

    envs = gym.vector.SyncVectorEnv(
        [make_env("VirtualHome-v2", args.seed, 0, False, "tmp", env_params) for i in
         range(1)]
    )
    agent = LLMAgent(normalization_mode="word",base_model = args.base_model
                     )

    success_rate = 0
    traj = []
    traj_rewards = []
    step_list = []
    obs = envs.reset()
    root = LanguageNode(state=obs)
    _ = agent.expand(obs,root,envs,value_weight)
    traj_action_list = []
    rewards_list = []
    for i in range(args.maxiterations):  #100
        logging.info(f"New round : {i} -----------------------------------------------------------------------------------------------------\n")
        steps = 0
        done = False
        rewards = 0
        reward_list = []
        reward_list = torch.tensor(reward_list,device=device)
        discount = 1
        node = root
        node_path = []
        obs = envs.reset()
        action_list = []
        root._visit_count+=1
        while not done and steps <15:
            steps += 1
            action, value, next_node, action_name = agent.select(obs, node)
            node_path.append(next_node)
            action = action.cpu().numpy()
            print("action", action, 'action name', action_name)

            action_list.append(action_name)
            obs_temp = obs
            envs_temp = copy.deepcopy(envs)
            done_temp = done    

            obs, reward, done, info = envs.step(action)
            if reward<=0:
                reward = np.array([-0.001])            
            logging.info(f"action : {action_name}  reward : {reward}")
            # if random.random()>0.5:
            #     obs = obs_temp
            
            if args.stochastic>0 and 'grab' in action_name and random.random()<args.stochastic: # state transfer uncertainty. Stochastic situation
                obs = obs_temp
                envs = envs_temp
                reward = np.array([-0.001])
                done = done_temp
            
            

            rewards += reward * discount
            reward_list=torch.cat((reward_list, torch.tensor(reward, device=device)))
            if not done and next_node.is_leaf():
                agent.expand(obs,next_node,envs)
            discount *= 0.99
            node = next_node
        agent.mcts_update(node_path, reward_list)
        traj_rewards.append(reward_list)
        traj.append(node_path)
        step_list.append(steps)
        traj_action_list.append(action_list)
        if rewards > 1:
            success_rate += 1
    
        print(steps, rewards)
        logging.info(f"steps : {steps}, rewards : {rewards}")
        rewards_list.append(rewards.item())
        writer.add_scalar("charts/episodic_return", rewards, i)
    # print(np.mean(reward_list), np.std(reward_list))
    # print(np.mean(step_list), np.std(step_list))
    # print(success_rate)
    # logging.info(f"mean reward : {np.mean(reward_list)}, std reward : {np.std(reward_list)}")
    # logging.info(f"mean step : {np.mean(step_list)}, std step : {np.std(step_list)}")
    logging.info(f"success rate : {success_rate}\n")
    # with open('rewards.json', 'w') as json_file:
    #     json.dump(rewards_list, json_file)
    print("success rate : ", success_rate)
if __name__ == '__main__':
    main()

