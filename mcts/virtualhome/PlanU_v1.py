from datetime import datetime
import copy
import json
import math
# import jsonlines
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Tuple, Union, Callable, Type
import pdb
from tqdm import tqdm
import heapq
import logging
from collections import defaultdict
from torch.distributions.categorical import Categorical
import random
import transformers
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from rnd import RndRewardModel
from easydict import EasyDict
import torch.nn.functional as F 
# from prompt import DISTRIBUTION_PROMPT
LLM = False

class Node(object):
    """
    Overview:
        The node base class for tree_search.
    """

    def __init__(
        self, parent: "Node" = None, prior_p: float = 1.0, initial_value=torch.tensor([1]),
        calc_q: Callable[[list[float]], float] = np.mean,n_atoms: int = 51, v_min: float = -1, v_max: float = 1,action_id = None
    ) -> None:
        self._parent = parent
        self._children = {}
        self._visit_count = 0
        self._value_sum = 0
        self.prior_p = prior_p
        self.prior_p_ori = prior_p
        # self.value = initial_value
        self._initial_value = initial_value
        self._terminated = False
        self.action_id = action_id
        # RAVE Q,V
        self.Q_RAVE = 0
        self.N_RAVE = 0
        self.cum_rewards: list[float] = []
        # 修改为QRDQN的分位数表示
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        # 分位数位置是固定的，均匀分布在[0,1]区间
        self.quantile_probs = np.linspace(1/(2*n_atoms), 1-1/(2*n_atoms), n_atoms)
        # 分位数值初始化为v_min到v_max之间的均匀分布
        # self.quantile_values = np.linspace(v_min, v_max, n_atoms)
        self.distribution_history = []
        
        # 如果有快速奖励，将所有分位数值设置为该奖励值

        if len(initial_value) == 1:
            self.quantile_values = np.ones(n_atoms) * initial_value.item()
        elif len(initial_value) >1 :
            self.quantile_values = np.zeros(n_atoms)
            initial_v = [0.1,0.3,0.5,0.7,0.9]
            j=0
            for i in range(5):
                while j<int(50*sum(initial_value[:i+1])) and j<len(self.quantile_values):
                    self.quantile_values[j] = initial_v[i]
                    j =j+1
        self.distribution_history.append(self.quantile_values.copy())
                


    def __lt__(self, other):
        return self._initial_value < other._initial_value

    @property
    def terminated(self):
        return self._terminated

    def set_as_terminate_node(self):
        self._terminated = True

    # @property
    # def value(self) -> float:
    #     """
    #     Overview:
    #         The value of the current node.
    #         Q/N
    #     Returns:
    #         - output (:obj:`Int`): Current value, used to compute ucb score.
    #     """
    #     if self._visit_count == 0:
    #         # if not visited, return the initial value
    #         return self._initial_value
    #     return self._value_sum / self._visit_count
    
    @property
    def amaf_value(self) -> float:
        """
        Overview:
            The AMAF value of the current node.
        """
        return self.Q_RAVE / self.N_RAVE if self.N_RAVE > 0 else 0

    def update(self, value: float) -> None:
        """
        Overview:
            Updata the current node information, such as visit_count and value_sum.
        Arguments:
            - value (:obj:`Int`): The value of the node.
        """
        self._visit_count += 1
        self._value_sum += value
        # TODO: add the value to the AMAF value

    def update_recursive(self, leaf_value: float, mcts_mode: str) -> None:
        """
        Overview:
            Update node information recursively.
        Arguments:
            - leaf_value (:obj:`Int`): The value of the node.
        """
        if mcts_mode == "self_play_mode":
            self.update(leaf_value)
            if self.is_root():
                return
            self._parent.update_recursive(-leaf_value, mcts_mode)
        if mcts_mode == "play_with_bot_mode":
            self.update(leaf_value)
            if self.is_root():
                return
            self._parent.update_recursive(leaf_value, mcts_mode)

    def is_leaf(self) -> Dict:
        """
        Overview:
            Check if the current node is a leaf node or not.
        Returns:
            - output (:obj:`Dict`): Dict type children node.
        """
        return self._children == {}

    def is_root(self) -> bool:
        """
        Overview:
            Check if the current node is a root node or not.
        Returns:
            - output (:obj:`Bool`): Whether it is the parent node.
        """
        return self._parent is None

    @property
    def parent(self) -> None:
        return self._parent

    @property
    def children(self) -> None:
        return self._children

    @property
    def visit_count(self) -> None:
        return self._visit_count

    def get_info(self):
        # return [
        #     "visit_cnt: {}, value: {:.6f}, prior: {:.6f}".format(
        #         self.visit_count, self.value, self.prior_p)
        # ]
        return {
            "visit_cnt": self.visit_count,
            "value": self.value,
            "prior_p": float(self.prior_p_ori),
            "initial_value": self._initial_value,
            "terminated": self.terminated,
        }

    def clear(self):
        self._visit_count = 0
        self._value_sum = 0
        self.prior_p = self.prior_p_ori

    def to_json(self):
        childrens = {}
        for name, child_node in self.children.items():
            childrens[name] = child_node.to_json()

        rets = {"children": childrens, "info": self.get_info()}
        return rets


class LanguageNode(Node):
    text_state: Optional[str] = None
    last_action: Optional[str] = None
    num_generated_token: Optional[int] = None

    def __init__(
        self,
        parent: Node = None,
        prior_p: float = 1.0,
        state = None,
        initial_value: float = 0.0,
        num_generated_token: Optional[int] = None,
        task=1,
        last_action=None,
        path_value = 0,
    ) -> None:
        super().__init__(parent, prior_p, initial_value)
        self.state = state, # vector
        self.task = task
        self.num_generated_token = num_generated_token
        self.has_collected_token_num = False
        self.last_action = last_action
        self._path_value = path_value

    def get_path(self):
        ans = []
        node = self
        while not node.is_root():
            ans.append(node.last_action)
            node = node.parent
        return "\n".join(reversed(ans))

    def get_info(self):
        info_dict = super().get_info()
        if not self.is_root():
            info_dict["last_action"] = self.last_action
        else:
            info_dict["text_state"] = self.text_state
        return info_dict

    def obs2text(self, obs):

        text = ""

        in_kitchen = obs[0]
        in_bathroom = obs[1]
        in_bedroom = obs[2]
        in_livingroom = obs[3]

        see_pancake = obs[4]
        close_to_pancake = obs[5]
        hold_pancake = obs[6]

        see_microwave = obs[7]
        close_to_microwave = obs[8]
        is_microwave_open = obs[9]

        pancake_in_microwave = obs[10]

        assert in_kitchen + in_bathroom + in_bedroom + in_livingroom == 1, "Only one room can be true at a time"

        # template for room
        in_room_teplate = "There are four rooms: the kitchen, bathroom, bedroom, and living room. You are in the {}. "
        if in_kitchen:
            text += in_room_teplate.format("kitchen")
        elif in_bathroom:
            text += in_room_teplate.format("bathroom")
        elif in_bedroom:
            text += in_room_teplate.format("bedroom")
        elif in_livingroom:
            text += in_room_teplate.format("living room")

        object_text = ""
        action_list = []

        if in_kitchen:

            if not see_pancake:
                object_text += "The pancake is in the microwave. "
            else:
                object_text += "You notice pancake and microwave. "

            if hold_pancake:
                object_text += "Currently, you have grabbed the pancake in hand. "
                if close_to_microwave:
                    object_text += "The microwave is close to you. "
                    action_list = [0, 2, 3, 4, 7, 8, 9]
                else:
                    object_text += "The microwave is not close to you. "
                    action_list = [0, 2, 3, 4, 5]
            else:
                if close_to_pancake and not close_to_microwave:
                    object_text += "Currently, you are not grabbing anything in hand. The pancake is close to you. "
                    action_list = [0, 2, 3, 5, 6]
                elif close_to_microwave and not close_to_pancake:
                    object_text += "Currently, you are not grabbing anything in hand. The microwave is close to you. "
                    action_list = [0, 2, 3, 4, 8, 9]
                elif not close_to_pancake and not close_to_microwave:
                    object_text += "Currently, you are not grabbing anything in hand. The pancake and the microwave are not close to you. "
                    action_list = [0, 2, 3, 4, 5]
                else:
                    if is_microwave_open:
                        action_list = [0, 2, 3, 8, 9]
                    else:
                        action_list = [0, 2, 3, 9]

            if see_pancake and is_microwave_open:
                object_text += "The microwave is opened. "
            elif see_pancake and not is_microwave_open:
                object_text += "The microwave is not opend. "
            else:
                object_text += "The microwave is closed. "
                action_list = [0, 2, 3]

        elif in_bathroom:

            if hold_pancake:
                object_text += "and notice nothing useful. Currently, you have grabbed the pancake in hand. "
            else:
                object_text += "and notice nothing useful. Currently, you are not grabbing anything in hand. "

            action_list = [0, 1, 3]
        elif in_bedroom:

            if hold_pancake:
                object_text += "and notice nothing useful. Currently, you have grabbed the pancake in hand. "
            else:
                object_text += "and notice nothing useful. Currently, you are not grabbing anything in hand. "

            action_list = [0, 1, 2]
        elif in_livingroom:

            if hold_pancake:
                object_text += "and notice nothing useful. Currently, you have grabbed the pancake in hand. "
            else:
                object_text += "and notice nothing useful. Currently, you are not grabbing anything in hand. "

            action_list = [1, 2, 3]

        text += object_text

        target_template = "In order to heat up the pancake in the microwave, "
        text += target_template

        # template for next step
        next_step_text = "your next step is to"
        text += next_step_text

        self.action_template = [
            "walk to the living room",  # 0
            "walk to the kitchen",  # 1
            "walk to the bathroom",  # 2
            "walk to the bedroom",  # 3

            "walk to the pancake",  # 4
            "walk to the microwave",  # 5

            "grab the pancake",  # 6

            "put the pancake in the microwave",  # 7

            'open the microwave',  # 8
            'close the microwave',  # 9
        ]

        self.template2action = {
            k:i for i,k in enumerate(self.action_template)
        }

        actions = [self.action_template[i] for i in action_list]

        return {"prompt": text, "action": actions}

    def get_state(self,obs):
        return self.obs2text(obs)
    
    def add_child(self, state, action, initial_value):
        child = LanguageNode(state = state, last_action=action, initial_value = initial_value)
        self._children[action] = child
        child.parent = self

    @property
    def value(self,):
        return self._initial_value if self._visit_count ==0 else self._visit_count / self.parent._visit_count  # 0.8 for tomato_salad

    @property
    def ucb_score(self,):
        return self.value if self._visit_count==0 else self.value / self._visit_count + math.sqrt(2 * math.log(self.parent._visit_count) / self._visit_count) 

    


def get_root(node: Node):
    while not node.is_root():
        node = node.parent
    return node



class ActionNode(Node):
    text_state: Optional[str] = None
    last_action: Optional[str] = None
    num_generated_token: Optional[int] = None
    def __init__(
        self,
        parent: Node = None,
        prior_p: float = 1.0,
        state = None,
        initial_value: float = 0.0,
        num_generated_token: Optional[int] = None,
        task=1,
        last_action=None,
        path_value = 0,
        value_weight = 0.5,
        trans_value = 0,
        trans_visit_count=0,
        action_id = None,
    ) -> None:
        super().__init__(parent, prior_p, initial_value)
        self.state = state, # vector
        self.task = task
        self.num_generated_token = num_generated_token
        self.has_collected_token_num = False
        self.last_action = last_action
        self._path_value = path_value
        self.value_weight = value_weight
        self.trans_value = trans_value
        self.trans_visit_count = trans_visit_count
        self.action_id = action_id

    def get_distorted_q(self, distortion_fn: Callable[[np.ndarray, np.ndarray], float]) -> float:
        """
        使用扭曲函数将值分布转换为标量值
        
        :param distortion_fn: 扭曲函数，接收分位数概率和分位数值作为输入，返回标量值
        :return: 扭曲后的Q值
        """
        if self.state is None:
            return self.fast_reward
        return distortion_fn(self.quantile_probs, self.quantile_values)
   

    def add_child(self, state, action, initial_value):
        child = LanguageNode(state = state, last_action=action, initial_value = initial_value)
        self._children[action] = child
        child.parent = self

    @property
    def value(self) -> float:
        """返回分位数的平均值作为Q值"""
        if self.state is None:
            return self.fast_reward
        elif len(self.cum_rewards) == 0:
            return np.mean(self.quantile_values)
        else:
            return self.calc_q(self.cum_rewards)
    
    @property
    def UCC_value(self,):
        return self._initial_value if self._visit_count ==0 else self._path_value/self._visit_count + self._initial_value / math.log(self._visit_count+1) # UCC  
   
   
    @property
    def trans_v(self,):
        return 0 if self.trans_visit_count ==0 else self.trans_value / self.trans_visit_count

    @property
    def ucb_score(self,):
        return self.value if self._visit_count==0 else self.value / self._visit_count + math.sqrt(2 * math.log(self.parent._visit_count) / self._visit_count) 

    def get_distorted_q(self, distortion_fn: Callable[[np.ndarray, np.ndarray], float]) -> float:
        """
        使用扭曲函数将值分布转换为标量值
        
        :param distortion_fn: 扭曲函数，接收分位数概率和分位数值作为输入，返回标量值
        :return: 扭曲后的Q值
        """
        if self.state is None:
            return self.fast_reward
        return distortion_fn(self.quantile_probs, self.quantile_values)
    


def get_root(node: Node):
    while not node.is_root():
        node = node.parent
    return node


class LLMAgent:
    def __init__(self, task = 3,rnd=False,tb_logger=None,base_model = None,
         output_trace_in_each_iter: bool = False,
         w_exp: float = 0.5,
         depth_limit: int = 8,
         n_iters: int = 10,
         cum_reward: Callable[[list[float]], float] = sum,
         calc_q: Callable[[list[float]], float] = np.mean,
         uct_with_fast_reward: bool = True,
         n_atoms: int = 51,
         v_min: float = -10.0,
         v_max: float = 10.0,
         distortion_fn: Callable[[np.ndarray, np.ndarray], float] = None,
         risk_distortion: float = 0.0,
         init_distribution: bool = False,
         normalization_mode = 'word'):
        """
        :param n_atoms: 分位数的数量
        :param v_min: 值分布的最小值
        :param v_max: 值分布的最大值
        :param distortion_fn: 扭曲函数，用于将值分布转换为标量值。默认为期望值计算
        :param risk_distortion: 风险敏感参数，正值表示风险寻求，负值表示风险规避，0表示风险中性
        :param log_distributions: 是否记录分布变化的日志
        :param distribution_log_path: 分布日志保存路径
        :param visualize_key_nodes: 是否可视化关键节点的分布
        :param init_distribution:是否通过LLM初始化动作的值分布
        """
        super().__init__()
        self.init_distribution = init_distribution
        self.world_model = None
        self.search_config = None
        self.output_trace_in_each_iter = output_trace_in_each_iter
        self.w_exp = w_exp
        self.depth_limit = depth_limit
        self.n_iters = n_iters
        self.cum_reward = cum_reward
        self.calc_q = calc_q
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.risk_distortion = risk_distortion

        self.epsilon = 0.1
        if task == 0:
            self.obs_shape =18
        else:
            self.obs_shape = 11
        self.task = task
        self.base_model = 'Neko-Institute-of-Science/LLaMA-7B-HF'
        # self.model_path = "/data/dengziwei/llm/TWOSOME-MCTS/hf_models/meta-llama/Llama-3.1-8B-Instruct"
        self.base_model = base_model
        assert (
            self.base_model
        ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        self.tokenizer.pad_token_id = (
            0  # unk. we want this to be different from the eos token
        )
        self.model = self._init_llama()
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.normalization_mode = "token"
        self.alpha = 0.99 # value reward ratio
        self.gamma = 0.99 # decay rate
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.rnd = rnd
        if self.rnd:
            self.tb_logger = tb_logger
            self.rnd_model = self.init_rnd()
        self.total_llm_model_token = 0          # llm_model 调用 token 总消耗
        self.total_llm_tokenizer_token = 0      # llm_tokenizer 调用 token 总消耗
        self.total_llm_model_call = 0           # llm_model 调用次数
        self.total_llm_tokenizer_call = 0       # llm_tokenizer 调用次数
        if distortion_fn is None:
            if risk_distortion == 0.0:
                # 风险中性：使用期望值
                self.distortion_fn = lambda probs, values: np.sum(values/len(probs))
            else:
                # 风险敏感：使用CVaR (Conditional Value at Risk)
                def cvar_distortion(probs, values):
                    if risk_distortion > 0:  # 风险寻求
                        # 使用较高分位数的平均值
                        threshold = 1.0 - risk_distortion
                        mask = probs >= threshold
                        if np.any(mask):
                            return np.sum(probs[mask] * values[mask]) / np.sum(probs[mask])
                        return np.max(values)
                    else:  # 风险规避
                        # 使用较低分位数的平均值
                        threshold = -risk_distortion
                        mask = probs <= threshold
                        if np.any(mask):
                            return np.sum(probs[mask] * values[mask]) / np.sum(probs[mask])
                        return np.min(values)
                self.distortion_fn = cvar_distortion
        else:
            self.distortion_fn = distortion_fn

    def init_rnd(self,):
        config = dict(
        # (str) Reward model register name, refer to registry ``REWARD_MODEL_REGISTRY``.
        type='rnd',
        intrinsic_reward_type='assign',
        learning_rate=1e-5,
        batch_size=15,
        obs_shape = self.obs_shape,
        hidden_size_list=[64, 64, 128],
        update_per_collect=20,
        obs_norm=True,
        obs_norm_clamp_min=-1,
        obs_norm_clamp_max=1,
        intrinsic_reward_weight=0.01,
        extrinsic_reward_norm=True,
        extrinsic_reward_norm_max=1,
        )
        model = RndRewardModel(EasyDict(config),tb_logger=self.tb_logger)
        return model
    
    def rnd_reward(self,data):
        return self.rnd_model.estimate(data)
    
    def collect_data(self, data):
        data = torch.tensor(data,device=self.device).view(-1)
        self.rnd_model.collect_data(data)
    
    def train(self,):
        self.rnd_model.train()

    def _init_llama(self):
        model = LlamaForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=torch.float16,
            # load_in_8bit=self.load_8bit,
            device_map="auto",
            #cache_dir='weights/llama'
        )

        return model
    
    def rnd_reward(self,data):
        return self.rnd_model.estimate(data)
    
    def collect_data(self, data):
        data = torch.tensor(data,device=self.device).view(-1)
        self.rnd_model.collect_data(data)
    
    def train(self,):
        self.rnd_model.train()



    def _uct_select(self, node) :
        if self.uct_with_fast_reward or all(x.state is not None for x in node.children):
            return max(node.children, key=self._uct)
        else:
            unvisited_children = filter(lambda x: x.state is None, node.children)
            return max(unvisited_children, key=lambda x: x.fast_reward)
        
    def _uct(self, node) -> float:
        # 使用扭曲函数计算Q值
        distorted_q = node.get_distorted_q(self.distortion_fn)
        if not self.rnd:
            node_value = distorted_q + self.w_exp * np.sqrt(np.log(len(node.parent.cum_rewards)+1) / max(1, len(node.cum_rewards)))
        else:
            node_value = distorted_q
        return node_value


    def select(self, obs, node):

        action = []
        value = []
        obs = node.state[0]
        text_obs = [self.obs2text(o) for o in obs]
        prompt = [o["prompt"] for o in text_obs]
        action_list = [o["action"] for o in text_obs]
        initial_value = []

        # epsilon greedy
        for action_tmp, child_tmp in node.children.items():
            action.append(action_tmp)
            if self.rnd:
                node_value = self._uct(child_tmp)
            else:
                node_value = self._uct(child_tmp)
            # assert(node_value.shape == torch.Size([1]))
            # value.append(child_tmp.ucb_score) ## uct
            value.append(node_value)
            # initial_value.append(child_tmp._initial_value)
        values = torch.tensor(value,device=self.device)
        # initial_value = torch.stack(value).view(-1)
        # initial_value = initial_value/initial_value.sum()
        if self.rnd:
            ex_reward = [self.rnd_reward(data = child_tmp.state[0]) for child_tmp in node.children.values()]
            ex_reward = torch.stack(ex_reward).view(-1)
            ex_reward = F.normalize(ex_reward, p=1, dim=0)
            ex_reward = 0.25 * torch.div(ex_reward,  max(1, len(node.cum_rewards)))
            values = values + ex_reward # weight between value and extrinsic reward. Add a weight in ex_reward

        # sampled_index = torch.multinomial(values, 1).item()
        argmax = torch.argmax(values).item()
        action_id_argmax = node.children[action[argmax]].action_id
        # action_id_sampled = node.children[action[sampled_index]].action_id
        if False:
            return torch.tensor([action_id_sampled]),values[sampled_index], node.children[action[sampled_index]],action[sampled_index]
        else:
            return torch.tensor([action_id_argmax]),values[torch.argmax(values)], node.children[action[argmax]], action[argmax]

    def distribution_initialization(self, action_list, prompt):
        action_distribution = []
        criteria = ['very low', 'somewhat low', 'medium level', 'somewhat high', 'very high']
        self.action_list_ids = self.tokenizer(criteria, return_tensors="pt", padding=True)
        for action in action_list[0]:
            sequence = []
            sequence = [prompt + action + "Evaluation: " + c for c in criteria] 
            inputs = self.tokenizer(sequence, return_tensors="pt", padding=True)
            input_ids = inputs["input_ids"].to(self.device)
            
            attention_mask = inputs["attention_mask"].to(self.device)
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=attention_mask)
                
            # self.action_list_ids = self.tokenizer(criteria, return_tensors="pt", padding=True)

            self.action_list_length = torch.sum(self.action_list_ids["attention_mask"], dim = -1) - 1 #delete first token
            sequence_length = torch.sum(attention_mask, dim = -1)
            action_index = [[end - start, end] for start, end in zip(self.action_list_length, sequence_length)]

            # maybe no need to use it, directly use logits
            logits = torch.log_softmax(outputs.logits, dim=-1)

            logits = logits[:, :-1, :]
            input_ids = input_ids[:, 1:]
            gen_logits = torch.gather(logits, 2, input_ids[:, :, None]).squeeze(-1)

            slices = [gen_logits[i, start-1:end-1] for i, (start, end) in enumerate(action_index)]
            
            action_logits = torch.stack([torch.sum(s) for s in slices])
            if self.normalization_mode == 'token':
                action_logits = action_logits / self.action_list_length.to(self.device)
            elif self.normalization_mode == 'word':
                action_word_num = torch.tensor([len(action.split()) for action in criteria]).to(self.device)
                action_logits = action_logits / action_word_num
            elif self.normalization_mode == 'sum':
                action_logits = action_logits
            else:
                assert 1==2

            action_logits = action_logits.reshape(-1, len(criteria)).float()
            probs = Categorical(logits=action_logits)
            action_distribution.append(probs.probs[0])
        return action_distribution


    def expand(self, obs, node, simulate_envs, if_action_node=False, value_weight = 0.5):
        # expand node
        # input:node,obs. Add children to node
        # obs: vector
        # 将动作列表添加到子节点中，每个子结点初始化value为prob
        if if_action_node:
            for key, node_tmp in node.children.items():
                temp= torch.tensor(node_tmp.state[0],device=self.device)
                obs_tmp = torch.tensor(obs,device=self.device)
                assert(type(temp)==type(obs_tmp))
                last_action = node_tmp.last_action
                if torch.equal(temp,obs_tmp):
                    node_tmp._visit_count += 1
                    return node_tmp

            child = LanguageNode(state = obs, parent = node, initial_value = torch.tensor([1],device=self.device), last_action=last_action)
            node.children[str(len(node.children))] = child
            assert(len(node.children)<=2)
            child._visit_count += 1
            return child

        if not node.is_leaf():
            return node
        if self.init_distribution:
            text_obs = [self.obs2text(o) for o in obs]
            prompt = [o["prompt"] for o in text_obs]
            action_list = [o["action"] for o in text_obs]
            prompt = DISTRIBUTION_PROMPT+prompt
            prompt = "".join(prompt)
            action_distribution = self.distribution_initialization(action_list, prompt)
            for id, action in enumerate(action_list[0]):
                envs = copy.deepcopy(simulate_envs)
                id_tmp = torch.tensor([id],device=self.device)
                next_obs, reward, done, info = envs.step(id_tmp.cpu().numpy())
                n = np.array(reward)
                re = torch.tensor(n,device=self.device).view(-1)
                child = ActionNode(parent=node, state = next_obs, last_action=action, initial_value = action_distribution[id]+re,value_weight=value_weight) #修改，添加了reward

                child_child = LanguageNode(parent = child, state = next_obs, last_action = action, initial_value = torch.tensor([1],device=self.device))
                index = str(len(child.children))
                node.children[action] = child
                child.children[index] = child_child
            return node
        text_obs = [self.obs2text(o) for o in obs]
        prompt = [o["prompt"] for o in text_obs]
        action_list = [o["action"] for o in text_obs] # available actions
        action_ids = [[self.template2action[item] for item in env] for env in action_list]
        prompt_nums = len(prompt)
        action_num = len(action_list[0])

        if not LLM:
            action_list = [item for sublist in action_list for item in sublist]
            sequence = []
            for i in range(prompt_nums):
                for p, ac in zip(prompt, action_list):
                    sequence += [p + " " + a for a in ac]

                    for id, action in enumerate(action_list):
                                
                        envs = copy.deepcopy(simulate_envs)
                        id_tmp = torch.tensor([action_ids[i][id]],device=self.device)
                        id_for_prob = torch.tensor([id],device= self.device)
                        next_obs, reward, done, info = envs.step(id_tmp.cpu().numpy())
                        child = ActionNode(parent=node, state = next_obs, last_action=action, initial_value = torch.tensor([reward],device=self.device).view(-1),action_id = action_ids[i][id]) ##修改，添加了reward项到initial value
                        node.children[action] = child
                        child_child = LanguageNode(parent=child, state = next_obs, last_action=action, initial_value = torch.tensor([1],device=self.device))
                        index = str(len(child.children))
                        node.children[action] = child
                        child.children[index] = child_child
            return node


        inputs = self.tokenizer(sequence, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(self.device)
        
        attention_mask = inputs["attention_mask"].to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
        self.total_llm_tokenizer_token += len(outputs.logits)
        self.total_llm_tokenizer_call += 1
        action_list = [item for sublist in action_list for item in sublist]
        self.action_list_ids = self.tokenizer(action_list, return_tensors="pt", padding=True)

        self.action_list_length = torch.sum(self.action_list_ids["attention_mask"], dim = -1) - 1 #delete first token
        sequence_length = torch.sum(attention_mask, dim = -1)
        action_index = [[end - start, end] for start, end in zip(self.action_list_length, sequence_length)]

        # maybe no need to use it, directly use logits
        logits = torch.log_softmax(outputs.logits, dim=-1)

        logits = logits[:, :-1, :]
        input_ids = input_ids[:, 1:]
        gen_logits = torch.gather(logits, 2, input_ids[:, :, None]).squeeze(-1)

        slices = [gen_logits[i, start-1:end-1] for i, (start, end) in enumerate(action_index)]
        
        action_logits = torch.stack([torch.sum(s) for s in slices])
        if self.normalization_mode == 'token':
            action_logits = action_logits / self.action_list_length.to(self.device)
        elif self.normalization_mode == 'word':
            action_word_num = torch.tensor([len(action.split()) for action in action_list]).to(self.device)
            action_logits = action_logits / action_word_num
        elif self.normalization_mode == 'sum':
            action_logits = action_logits
        else:
            assert 1==2

        action_logits = action_logits.reshape(-1, action_num).float()
        probs = Categorical(logits=action_logits)

        # build children node

        for i in range(prompt_nums):
            # logits = action_logits[sum(action_nums[:i]):sum(action_nums[:i+1])].reshape(-1, action_nums[i]).float()

            # probs = Categorical(logits=logits)

            for id, action in enumerate(action_list):
                        
                envs = copy.deepcopy(simulate_envs)
                id_tmp = torch.tensor([action_ids[i][id]],device=self.device)
                id_for_prob = torch.tensor([id],device= self.device)
                next_obs, reward, done, info = envs.step(id_tmp.cpu().numpy())
                child = ActionNode(parent=node, state = next_obs, last_action=action, initial_value = probs.log_prob(id_for_prob).exp()+torch.tensor([reward],device=self.device).view(-1),action_id = action_ids[i][id]) ##修改，添加了reward项到initial value
                node.children[action] = child
                child_child = LanguageNode(parent=child, state = next_obs, last_action=action, initial_value = torch.tensor([1],device=self.device))
                index = str(len(child.children))
                node.children[action] = child
                child.children[index] = child_child
        return node



    def get_value(self, x):
        # get value of state
        # input:state
        # output:value
        pass


    def update(self, node, next_node, reward):
        #backup
        reward = torch.tensor(reward,device = self.device)
        delta = next_node._path_value - node._path_value
        node._path_value = node._path_value + self.alpha * (reward + self.gamma * delta)


    def mcts_update(self, path, rewards):
        cum_reward = -math.inf
        
        # 计算累积奖励并更新分布
        for i,node in enumerate(reversed(path)):
            # rewards.append(node.reward)
            cum_reward = self.cum_reward(rewards[-i-1:])
            node.cum_rewards.append(cum_reward)
            
            if node.state is not None:
                alpha = 0.7
                mu = 0.9
                
                for i, tau in enumerate(node.quantile_probs):

                    current_value = node.quantile_values[i]
                    delta = cum_reward - current_value
                    gradient = tau if delta > 0 else tau - 1.0
                    
                    node.quantile_values[i] += alpha * gradient * np.abs(delta.cpu())
                    # node.quantile_values[i] = mu * node.quantile_values[i] + (1-mu) * cum_reward
                
                node.quantile_values = np.clip(node.quantile_values, node.v_min, node.v_max)
                
                node.distribution_history.append(node.quantile_values.copy())
                
        
        return cum_reward


    def transpositions_update(self,node_path,rewards,tree_node):
        rewards = rewards.view(-1)
        assert len(node_path) == len(rewards)
        a= 0
        cum_rewards = torch.zeros(len(rewards),device = self.device)
        for i in range(len(rewards)):
            cum_rewards[i] = sum(rewards[i:])
        for i,node in enumerate(node_path):
            # node_path[i]._path_value += cum_rewards[i]
            # node_path[i]._visit_count +=1 
            for n in tree_node[node.last_action]:
                temp1= torch.tensor(node.state[0],device=self.device)
                temp2 = torch.tensor(n.state[0],device=self.device)
                assert temp1.size() == temp2.size()
                if torch.equal(temp1,temp2) and n != node:
                    n.trans_value += cum_rewards[i]
                    n.trans_visit_count +=1 
                    a+=1
        print(a)



    def simulate(self, node, simulate_envs):
        envs = copy.deepcopy(simulate_envs)

        pass

    def obs2text(self, obs):

        text = ""

        in_kitchen = obs[0]
        in_bathroom = obs[1]
        in_bedroom = obs[2]
        in_livingroom = obs[3]

        see_pancake = obs[4]
        close_to_pancake = obs[5]
        hold_pancake = obs[6]

        see_microwave = obs[7]
        close_to_microwave = obs[8]
        is_microwave_open = obs[9]

        pancake_in_microwave = obs[10]

        assert in_kitchen + in_bathroom + in_bedroom + in_livingroom == 1, "Only one room can be true at a time"

        # template for room
        in_room_teplate = "There are four rooms: the kitchen, bathroom, bedroom, and living room. You are in the {}. "
        if in_kitchen:
            text += in_room_teplate.format("kitchen")
        elif in_bathroom:
            text += in_room_teplate.format("bathroom")
        elif in_bedroom:
            text += in_room_teplate.format("bedroom")
        elif in_livingroom:
            text += in_room_teplate.format("living room")

        object_text = ""
        action_list = []

        if in_kitchen:

            if not see_pancake:
                object_text += "The pancake is in the microwave. "
            else:
                object_text += "You notice pancake and microwave. "

            if hold_pancake:
                object_text += "Currently, you have grabbed the pancake in hand. "
                if close_to_microwave:
                    object_text += "The microwave is close to you. "
                    action_list = [0, 2, 3, 4, 7, 8, 9]
                else:
                    object_text += "The microwave is not close to you. "
                    action_list = [0, 2, 3, 4, 5]
            else:
                if close_to_pancake and not close_to_microwave:
                    object_text += "Currently, you are not grabbing anything in hand. The pancake is close to you. "
                    action_list = [0, 2, 3, 5, 6]
                elif close_to_microwave and not close_to_pancake:
                    object_text += "Currently, you are not grabbing anything in hand. The microwave is close to you. "
                    action_list = [0, 2, 3, 4, 8, 9]
                elif not close_to_pancake and not close_to_microwave:
                    object_text += "Currently, you are not grabbing anything in hand. The pancake and the microwave are not close to you. "
                    action_list = [0, 2, 3, 4, 5]
                else:
                    if is_microwave_open:
                        action_list = [0, 2, 3, 8, 9]
                    else:
                        action_list = [0, 2, 3, 9]

            if see_pancake and is_microwave_open:
                object_text += "The microwave is opened. "
            elif see_pancake and not is_microwave_open:
                object_text += "The microwave is not opend. "
            else:
                object_text += "The microwave is closed. "
                action_list = [0, 2, 3]

        elif in_bathroom:

            if hold_pancake:
                object_text += "and notice nothing useful. Currently, you have grabbed the pancake in hand. "
            else:
                object_text += "and notice nothing useful. Currently, you are not grabbing anything in hand. "

            action_list = [0, 1, 3]
        elif in_bedroom:

            if hold_pancake:
                object_text += "and notice nothing useful. Currently, you have grabbed the pancake in hand. "
            else:
                object_text += "and notice nothing useful. Currently, you are not grabbing anything in hand. "

            action_list = [0, 1, 2]
        elif in_livingroom:

            if hold_pancake:
                object_text += "and notice nothing useful. Currently, you have grabbed the pancake in hand. "
            else:
                object_text += "and notice nothing useful. Currently, you are not grabbing anything in hand. "

            action_list = [1, 2, 3]

        text += object_text

        target_template = "In order to heat up the pancake in the microwave, "
        text += target_template

        # template for next step
        next_step_text = "your next step is to"
        text += next_step_text

        self.action_template = [
            "walk to the living room",  # 0
            "walk to the kitchen",  # 1
            "walk to the bathroom",  # 2
            "walk to the bedroom",  # 3

            "walk to the pancake",  # 4
            "walk to the microwave",  # 5

            "grab the pancake",  # 6

            "put the pancake in the microwave",  # 7

            'open the microwave',  # 8
            'close the microwave',  # 9
        ]

        self.template2action = {
            k:i for i,k in enumerate(self.action_template)
        }

        actions = [self.action_template[i] for i in action_list]

        return {"prompt": text, "action": actions}
