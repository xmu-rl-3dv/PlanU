# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import argparse
import os
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from gym_macro_overcooked.macActEnvWrapper import MacEnvWrapper
# from twosome_mcts.inference.lm_call import LMCallingConfig, VLLMRemoteCaller,LanguageModelCallingFunction
from mcts.overcooked.PlanU_mcts import ActionNode, LanguageNode,LLMAgent
import copy
from collections import defaultdict

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=10,
        help="seed of the experiment")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument("--total-timesteps", type=int, default=1000000,
        help="total timesteps of the experiments")
    parser.add_argument("--num-steps", type=int, default=32,
        help="the number of steps to run in each environment per policy rollout")
    # Algorithm specific arguments
    parser.add_argument("--depth", type=int, default=15,
        help="the depth of the MCTS")
    parser.add_argument('--env-id',                 action='store',        type=str,             default='Overcooked-LLMA-v3',  help='Domain name')
    parser.add_argument('--n-agent',                action='store',        type=int,             default=1,                     help='Number of agents')
    parser.add_argument('--grid-dim',               action='store',        type=int,   nargs=2,  default=[7,7],                 help='Grid world size')
    parser.add_argument('--task',                   action='store',        type=int,             default=3,                     help='The receipt agent cooks')
    parser.add_argument('--map-type',               action='store',        type=str,             default="A",                   help='The type of map')
    parser.add_argument('--obs-radius',             action='store',        type=int,             default=2,                     help='The radius of the agents')
    parser.add_argument('--env-reward',             action='store',        type=float, nargs=4,  default=[0.1, 1, 0, 0.001],    help='The reward list of the env')
    parser.add_argument('--mode',                   action='store',        type=str,             default="vector",              help='The type of the observation(vector/image)')    
    parser.add_argument('--debug',                  action='store',        type=bool,            default=False,                 help='Whehter print the debug information and render') 
    
    
    parser.add_argument('--save-path',              action='store',        type=str,             default="saved_models",        help='The path to save the checkpoint')
    parser.add_argument('--save-interval',          action='store',        type=int,             default=10,                    help='The interval for saving model for certain num_updates')
 
    parser.add_argument('--record-path',            action='store',        type=str,             default="llm5_runs",           help='The path to save the tensorboard results')    

    parser.add_argument('--normalization-mode',     action='store',        type=str,             default="token",               help='The normalization mode of how to deal with the logits of each token') 
    parser.add_argument('--value-weight' , action='store',type = float,default = 0.5, help = 'The exploration rate of the agent') 
    parser.add_argument('--stochastic', action='store',type = float,default = 0.2, help = 'Whether the dynamics is stochastic or deterministic')
    parser.add_argument('--transpositions', action='store',type = bool,default = False, help = 'if use transopositions')
    parser.add_argument("--num-envs", type=int, default=4,
    help="the number of parallel game environments")
    parser.add_argument('--maxiterations', action='store',type = int,default = 1000, help = 'num of iterations')
    parser.add_argument('--rnd', action='store',type = bool,default = False, help = 'if use rnd')
    parser.add_argument('--init_dist', action='store',type = bool,default = False, help = 'if use rnd')
    parser.add_argument('--base-model',     action='store',        type=str,             default="meta-llama/Meta-Llama-3-8B-Instruct",               help='''select a base model from below:"meta-llama/Meta-Llama-3-8B-Instruct",
                        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"''')  
    parser.add_argument('--temperature' , action='store',type = float,default = 1, help = 'The temperature rate of the agent') 
    args = parser.parse_args()
    
    
    # fmt: on
    return args


def make_env(env_id, seed, idx, capture_video, run_name, env_params):
    def thunk():

        env = gym.make(env_id, **env_params)
        env = MacEnvWrapper(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk

if __name__ == "__main__":
    args = parse_args()
    time_str = time.strftime("%Y%m%d_%H_%M_%S", time.localtime(time.time()))
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{time_str}__SAMCTS"
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
    if args.task==0:
        run_name = 'tomato_salad_temeprature'
    else:
        run_name = 'tomato_lettuce_salad_temperature'
    if args.transpositions:
        t = 'transpositions'
    else:
        t = 'no_transpositions'
    print(args.rnd)
    method_name = "PlanU" if not args.init_dist else "PlanU_init_dist"
    print(method_name)
    # input('...')
    writer = SummaryWriter(f"./results/Model={args.base_model}/{run_name}/PlanU/seed={args.seed}/stochastic={args.stochastic}/rnd={args.rnd}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    rnd_writer = SummaryWriter(f"./rnd_reward/Model={args.base_model}/{run_name}/PlanU/seed={args.seed}/{args.rnd}/transpositions={args.transpositions}")

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # If you're using CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    rewardList = {"subtask finished": args.env_reward[0], "correct delivery": args.env_reward[1], "wrong delivery": -args.env_reward[2], "step penalty": -args.env_reward[3]}
    TASKLIST = ["tomato salad", "lettuce salad", "onion salad", "lettuce-tomato salad", "onion-tomato salad", "lettuce-onion salad", "lettuce-onion-tomato salad"]
    env_params = {'grid_dim': args.grid_dim,
                    'task': TASKLIST[args.task],
                    'rewardList': rewardList,
                    'map_type': args.map_type,
                    'n_agent': args.n_agent,
                    'obs_radius': args.obs_radius,
                    'mode': args.mode,
                    'debug': args.debug
                }

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name, env_params) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"


        
    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    steps = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    pre_global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)


    #lm call initialization
        # 根据语言模型名称设置步骤标签

    # 初始化语言模型生成函数
    # gen_config = LMCallingConfig(
    #     n=args.num_sequence,
    #     temperature=args.temperature,
    #     top_k=args.top_k,
    #     top_p=args.top_p,
    #     max_new_tokens=args.max_new_tokens,
    # )
    # llm_gen_fn = VLLMRemoteCaller(
    #     args.LM, args.controller_addr,
    # )
    # lmcallingconfig = LMCallingConfig(n=5, temperature=0.7, max_new_tokens=100,
    #                                   stop_str=['\n\n'], 
    #                                   include_stop_str_in_output=True)
    traj =[]
    traj_rewards = []
    agent = LLMAgent(task=args.task,rnd=args.rnd,tb_logger=rnd_writer,base_model = args.base_model, init_distribution=args.init_dist, temperature = args.temperature)
    done = next_done
    root = LanguageNode(state=next_obs,initial_value=torch.tensor([1],device=device),task=args.env_id)
    agent.expand(next_obs,root,envs)
    tree_node = defaultdict(list)
    stoc = True
    train_data_cnt = 0
    act = [0,1,2,3,4]
    for path_num in range(0, args.maxiterations):
        node_path = []
        # root._visit_count += 1
        node = root
        done = torch.zeros(args.num_envs).to(device)
        rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
        step = 0
        next_obs = torch.Tensor(envs.reset()).to(device)
        envs.reset()
        
        # node_path.append(node) 
        while not done and step < args.depth:
            if node.is_leaf():
                _ = agent.expand(next_obs,node,envs,False)
            assert(type(node)==LanguageNode)
            node._visit_count+=1

            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            next_obs_temp = next_obs
            envs_temp = copy.deepcopy(envs)
            # ALGO LOGIC: action logic
            action, value, next_node, action_name = agent.select(next_obs, node, path_num) # Select an action node
            next_action_node = next_node
            # next_node._visit_count += 1
            node_path.append(next_node)
            values[step] = value.flatten()
            actions[step] = action
            # logprobs[step] = logprob

            
            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, info = envs.step(action.cpu().numpy())
            if stoc and 'chop' in action_name and random.random()<args.stochastic: # state transfer uncertainty. Stochastic situation
                next_obs = next_obs_temp
                envs = envs_temp
                reward = torch.tensor([-0.001],device=device)
                done = next_done
                # stoc = False

            next_node = agent.expand(next_obs,next_node,envs, type(next_node) == ActionNode, value_weight = args.value_weight) # expand and select a state node
            if args.rnd:
                agent.collect_data(next_obs)
                train_data_cnt += 1


            #print("info", info)
            # rewards[step] = torch.tensor(reward).to(device).view(-1)
            rewards[step] = reward.item()
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)
            steps[step] = torch.Tensor([item['macro_action_steps'] for item in info]).to(device)

            # for item in info:
            #     if "episode" in item.keys():
            #         print(f"global_step={global_step}, episodic_return={item['episode']['r']}, episodic_length={item['episode']['l']}")
            #         writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
            #         writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
            #         break
            if args.transpositions:
                if next_action_node not in tree_node[action_name]:
                    tree_node[action_name].append(next_action_node)








            # TD update
            # agent.update(node, next_node, reward)


            # node_path.append(next_node)
            node=next_node
            step += 1
            print(action_name)

        # MC update
        if args.transpositions:
            agent.transpositions_update(node_path,rewards[:step],tree_node)
        agent.mcts_update(node_path, rewards[:step])

        traj.append(node_path)
        traj_rewards.append(rewards[:step])
        print(f"global_step={global_step}, num_path = {path_num}, episodic_return={rewards.sum()}, episodic_length={step}")
        if args.rnd and train_data_cnt > 15:
            agent.train()
        writer.add_scalar("charts/episodic_return", rewards.sum(), global_step)
        writer.add_scalar("charts/episodic_length", step, global_step)

    rewards_sum = torch.tensor([i.sum() for i in traj_rewards]).view(-1)
    num_success =   (rewards_sum>1).sum().item()
    print(num_success)
    writer.add_text("scalrs/num_success",str(num_success),global_step = 0)
    writer.add_text("scalrs/consumed_tokens",str(agent.total_llm_tokenizer_token),global_step = 0)
    writer.add_text("scalrs/query_times",str(agent.total_llm_tokenizer_call),global_step=0)
    with open("token_consumed.txt", "a") as f:
        f.write("--------------------------------------\n")
        f.write(f"./results_new/Model={args.base_model}/{run_name}/PlanU/seed={args.seed}/stochastic={args.stochastic}/rnd={args.rnd}")
        f.write(f"consumed_tokens={agent.total_llm_tokenizer_token}\n")
        f.write(f"query_times={agent.total_llm_tokenizer_call}\n")
        f.write("--------------------------------------\n")
    print(f"./results_new/Model={args.base_model}/{run_name}/PlanU/seed={args.seed}/stochastic={args.stochastic}/rnd={args.rnd}")
    print(agent.total_llm_tokenizer_token)
    envs.close()
    writer.close()
        



