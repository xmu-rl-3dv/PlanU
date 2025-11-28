import sys

import fire
import gradio as gr
import torch
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer
from peft import PeftModel
import math
import torch.nn.functional as F
import os
import torch.nn as nn
import numpy as np
from torch.distributions.categorical import Categorical
import copy 
import random

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root)

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from typing import List, Dict, Any, Optional, Tuple, Union, Callable, Type


class Node(object):
    """
    Overview:
        The node base class for tree_search.
    """

    def __init__(
        self, parent: "Node" = None, prior_p: float = 1.0, initial_value: float = 0.0,
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

        self.value_estimate = 0.0
        self.reward_variance = 0.0  # V[R(s_k, a_k)]
        self.value_variance = 0.0   # V[V(s_k+1)]
        self.cum_rewards: list[float] = []
    
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
    last_action = None
    num_generated_token = None

    def __init__(
        self,
        parent: Node = None,
        prior_p: float = 1.0,
        state: Optional[str] = None,
        initial_value: float = 0.0,
        num_generated_token: Optional[int] = None,
        task=1,
        last_action=None,
        path_value = 0,
        action_id = None,
        value_weight = 0.5
    ) -> None:
        super().__init__(parent, prior_p, initial_value)
        self.state = state, # vector
        self.task = task
        self.num_generated_token = num_generated_token
        self.has_collected_token_num = False
        self.last_action = last_action
        self._path_value = path_value
        self.action_id = action_id
        self.initial_value = initial_value
        self.value_weight = value_weight

    def get_uct_value(self, c_uct: float = 1.0, beta: float = 0.0) -> float:
        """
        计算UCT值，包括探索项
        
        :param c_uct: UCT探索常数
        :param beta: 控制探索-利用平衡的参数，beta=0表示标准MCTS
        :return: UCT值
        """
        if self.parent is None:
            return 0.0
        
        # 计算q_M^beta(s_k, a)项
        q_value = math.pow(self.Q, beta)
        
        # 计算探索项
        parent_visit = max(1, self.parent._visit_count)
        self_visit = max(1, self._visit_count)
        exploration = c_uct * math.sqrt(2 * math.log(parent_visit) / self_visit)
        
        return q_value + exploration


    @property
    def Q(self) -> float:
        """返回分位数的平均值作为Q值"""
        if len(self.cum_rewards)==0:
            return self.initial_value
        else:
            # 保持原有的计算方式作为备选
            return np.mean(self.cum_rewards)

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
        
        see_chips = obs[4]
        close_to_chips = obs[5]
        hold_chips = obs[6]
        chips_on_coffeetable = obs[7]
        
        see_milk = obs[8]
        close_to_milk = obs[9]
        hold_milk = obs[10]
        milk_on_coffeetable = obs[11]

        see_tv = obs[12]
        close_to_tv = obs[13]
        is_face_tv = obs[14]
        is_tv_on = obs[15]

        see_sofa = obs[16]
        close_to_sofa = obs[17]
        is_sit_sofa = obs[18]

        see_coffeetable = obs[19]
        close_to_coffeetable = obs[20]
        assert in_kitchen + in_bathroom + in_bedroom + in_livingroom == 1, "Only one room can be true at a time"

        # template for room
        in_room_teplate = "There are four rooms: the kitchen, bathroom, bedroom, and living room. You are in the {} "
        if in_kitchen:
            text += in_room_teplate.format("kitchen")
        elif in_bathroom:
            text += in_room_teplate.format("bathroom")
        elif in_bedroom:
            text += in_room_teplate.format("bedroom")
        elif in_livingroom:
            text += in_room_teplate.format("living room")

        ########################################template2####################################
        # template for kitchen
        object_text = ""

        action_list = []

        if in_kitchen:

            if see_chips and see_milk:
                object_text += "and notice chips and milk. "

                if hold_chips and hold_milk:
                    object_text += "Currently, you have grabbed the chips and the milk in hand. "

                    action_list = [
                        0,
                        2,
                        3,
                    ]

                elif hold_chips and not hold_milk:
                    if close_to_milk:
                        object_text += "The milk is close to you. But you have not grabbed the milk. Currently, you have grabbed the chips in hand. "

                        action_list = [
                            0,
                            2,
                            3,
                            10
                        ]
                    else:
                        object_text += "The milk is not close to you. Currently, you have grabbed the chips in hand. "

                        action_list = [
                            0,
                            2,
                            3,
                            5
                        ]
                elif not hold_chips and hold_milk:
                    if close_to_chips:
                        object_text += "The chips are close to you. But you have not grabbed the chips. Currently, you have grabbed the milk in hand. "

                        action_list = [
                            0,
                            2,
                            3,
                            9
                        ]
                    else:
                        object_text += "The chips are not close to you. Currently, you have grabbed the milk in hand. "

                        action_list = [
                            0,
                            2,
                            3,
                            4
                        ]
                else:
                    if close_to_chips and close_to_milk:
                        object_text += "They are close to you. But you have not grabbed the them. "

                        action_list = [
                            0,
                            2,
                            3,
                            9,
                            10
                        ]

                    elif close_to_chips and not close_to_milk:
                        object_text += "The chips are close to you. But you have not grabbed the chips. "

                        action_list = [
                            0,
                            2,
                            3,
                            5,
                            9,
                        ]

                    elif not close_to_chips and close_to_milk:
                        object_text += "The milk is close to you. But you have not grabbed the milk. "

                        action_list = [
                            0,
                            2,
                            3,
                            4,
                            10,
                        ]

                    else:
                        object_text += "But they are not close to you. "

                        action_list = [
                            0,
                            2,
                            3,
                            4,
                            5,
                        ]

                    object_text += "Currently, you are not grabbing anything in hand. "

            elif see_chips and not see_milk:
                object_text += "and only notice chips. "

                if hold_chips:
                    object_text += "Currently, you have grabbed the chips in hand. "

                    action_list = [
                        0,
                        2,
                        3,
                    ]

                else:
                    if close_to_chips:
                        object_text += "The chips are close to you. But you have not grabbed the chips. "

                        action_list = [
                            0,
                            2,
                            3,
                            9,
                        ]
                    else:
                        object_text += "The chips are not close to you. "

                        action_list = [
                            0,
                            2,
                            3,
                            5,
                        ]

            elif not see_chips and see_milk:
                object_text += "and notice milk. "

                if hold_milk:
                    object_text += "Currently, you have grabbed the milk in hand. "

                    action_list = [
                        0,
                        2,
                        3,
                    ]

                else:
                    if close_to_milk:
                        object_text += "The milk is close to you. But you have not grabbed the milk. "

                        action_list = [
                            0,
                            2,
                            3,
                            10,
                        ]
                    else:
                        object_text += "The milk is not close to you. "

                        action_list = [
                            0,
                            2,
                            3,
                            4,
                        ]

            else:
                object_text += "and notice nothing. "

                action_list = [
                    0,
                    2,
                    3,
                ]

        elif in_livingroom:

            object_text += "and you notice a coffee table, a TV and a sofa. "

            assert close_to_coffeetable + close_to_tv + close_to_sofa <= 1, "You are next to more than one object from coffee table, TV and sofa."
            assert see_coffeetable + see_tv + see_sofa >= 3, "You don't see coffee table, TV and sofa."

            if not close_to_coffeetable and not close_to_tv and not close_to_sofa:
                object_text += "They are not close to you. "

                if hold_chips and hold_milk:
                    object_text += "Currently, you have grabbed the chips and the milk in hand. "
                elif not hold_chips and hold_milk:
                    object_text += "Currently, you have grabbed the milk in hand. "
                elif hold_chips and not hold_milk:
                    object_text += "Currently, you have grabbed the chips in hand. "
                else:
                    object_text += "Currently, you are not grabbing anything in hand. "

                action_list = [
                    1,
                    2,
                    3,
                    6,
                    7,
                    8
                ]

            if close_to_coffeetable:

                if (chips_on_coffeetable and hold_milk) or (milk_on_coffeetable and hold_chips):
                    object_text += "The TV is not close to you. "
                else:
                    object_text += "The coffee table is close to you. "

                if hold_chips and hold_milk:
                    object_text += "Currently, you have grabbed the chips and the milk in hand. "

                    action_list = [
                        1,
                        2,
                        3,
                        7,
                        8,
                        11,
                        12
                    ]
                elif not hold_chips and hold_milk:
                    if not chips_on_coffeetable:
                        object_text += "Currently, you have grabbed the milk in hand. "

                        action_list = [
                            1,
                            2,
                            3,
                            7,
                            8,
                            12
                        ]

                    else:
                        object_text += "Currently, you have the chips on the coffee table and the milk in your hand. "

                        action_list = [
                            1,
                            2,
                            3,
                            7,
                            8,
                        ]

                elif hold_chips and not hold_milk:
                    object_text += "Currently, you have grabbed the chips in hand. "

                    if not milk_on_coffeetable:
                        object_text += "Currently, you have grabbed the chips in hand. "

                        action_list = [
                            1,
                            2,
                            3,
                            7,
                            8,
                            11
                        ]

                    else:
                        object_text += "Currently, you have the milk on the coffee table and the chips in your hand. "

                        action_list = [
                            1,
                            2,
                            3,
                            7,
                            8,
                        ]

                else:
                    object_text += "Currently, you are not grabbing anything in hand. "

                    action_list = [
                        1,
                        2,
                        3,
                    ]

            if close_to_tv:
                if is_tv_on:
                    object_text += "The sofa is not close to you. "

                    if hold_chips and hold_milk:
                        object_text += "Currently, the TV is turned on, you have grabbed the chips and the milk in hand. "

                    elif not hold_chips and hold_milk:
                        if not chips_on_coffeetable:
                            object_text += "Currently, the TV is turned on, you have grabbed the milk in hand. "
                        else:
                            object_text += "Currently, the TV is turned on, you have the chips on the coffee table and the milk in your hand. "
                    elif hold_chips and not hold_milk:
                        object_text += "Currently, the TV is turned on, you have grabbed the chips in hand. "
                        if not milk_on_coffeetable:
                            object_text += "Currently, the TV is turned on, you have grabbed the chips in hand. "
                        else:
                            object_text += "Currently, the TV is turned on, you have the milk on the coffee table and the chips in your hand. "

                    action_list = [
                        1,
                        2,
                        3,
                        6,
                        8,
                    ]

                else:
                    object_text += "The TV is close to you. "

                    if hold_chips and hold_milk:
                        object_text += "Currently, you have grabbed the chips and the milk in hand. "

                    elif not hold_chips and hold_milk:
                        if not chips_on_coffeetable:
                            object_text += "Currently, you have grabbed the milk in hand. "
                        else:
                            object_text += "Currently, you have the chips on the coffee table and the milk in your hand. "
                    elif hold_chips and not hold_milk:
                        object_text += "Currently, you have grabbed the chips in hand. "
                        if not milk_on_coffeetable:
                            object_text += "Currently, you have grabbed the chips in hand. "
                        else:
                            object_text += "Currently, you have the milk on the coffee table and the chips in your hand. "

                    action_list = [
                        1,
                        2,
                        3,
                        6,
                        8,
                        13,
                        14
                    ]

            if close_to_sofa:

                if not is_sit_sofa:
                    object_text += "The sofa is close to you. "

                    if is_tv_on:
                        if hold_chips and hold_milk:
                            object_text += "Currently, the TV is turned on, you have grabbed the chips and the milk in hand. "

                        elif not hold_chips and hold_milk:
                            if not chips_on_coffeetable:
                                object_text += "Currently, the TV is turned on, you have grabbed the milk in hand. "
                            else:
                                object_text += "Currently, the TV is turned on, you have the chips on the coffee table and the milk in your hand. "
                        elif hold_chips and not hold_milk:
                            object_text += "Currently, the TV is turned on, you have grabbed the chips in hand. "
                            if not milk_on_coffeetable:
                                object_text += "Currently, the TV is turned on, you have grabbed the chips in hand. "
                            else:
                                object_text += "Currently, the TV is turned on, you have the milk on the coffee table and the chips in your hand. "

                        action_list = [
                            1,
                            2,
                            3,
                            6,
                            7,
                            15,
                            16
                        ]
                    else:
                        if hold_chips and hold_milk:
                            object_text += "Currently, you have grabbed the chips and the milk in hand. "

                        elif not hold_chips and hold_milk:
                            if not chips_on_coffeetable:
                                object_text += "Currently, you have grabbed the milk in hand. "
                            else:
                                object_text += "Currently, you have the chips on the coffee table and the milk in your hand. "
                        elif hold_chips and not hold_milk:
                            object_text += "Currently, you have grabbed the chips in hand. "
                            if not milk_on_coffeetable:
                                object_text += "Currently, you have grabbed the chips in hand. "
                            else:
                                object_text += "Currently, you have the milk on the coffee table and the chips in your hand. "

                        action_list = [
                            1,
                            2,
                            3,
                            6,
                            7,
                        ]

                else:
                    object_text += "You are sitting on the sofa. "

                    if is_tv_on:
                        if hold_chips and hold_milk:
                            object_text += "Currently, the TV is turned on, you have grabbed the chips and the milk in hand. "

                        elif not hold_chips and hold_milk:
                            if not chips_on_coffeetable:
                                object_text += "Currently, the TV is turned on, you have grabbed the milk in hand. "
                            else:
                                object_text += "Currently, the TV is turned on, you have the chips on the coffee table and the milk in your hand. "
                        elif hold_chips and not hold_milk:
                            object_text += "Currently, the TV is turned on, you have grabbed the chips in hand. "
                            if not milk_on_coffeetable:
                                object_text += "Currently, the TV is turned on, you have grabbed the chips in hand. "
                            else:
                                object_text += "Currently, the TV is turned on, you have the milk on the coffee table and the chips in your hand. "

                        action_list = [1, 2, 3]
                    else:
                        if hold_chips and hold_milk:
                            object_text += "Currently, you have grabbed the chips and the milk in hand. "

                        elif not hold_chips and hold_milk:
                            if not chips_on_coffeetable:
                                object_text += "Currently, you have grabbed the milk in hand. "
                            else:
                                object_text += "Currently, you have the chips on the coffee table and the milk in your hand. "
                        elif hold_chips and not hold_milk:
                            object_text += "Currently, you have grabbed the chips in hand. "
                            if not milk_on_coffeetable:
                                object_text += "Currently, you have grabbed the chips in hand. "
                            else:
                                object_text += "Currently, you have the milk on the coffee table and the chips in your hand. "

                        action_list = [1, 2, 3]

        elif in_bedroom:

            if hold_chips and hold_milk:
                object_text += "and notice nothing. Currently, you have grabbed the chips and the milk in hand. "
            elif hold_chips and not hold_milk:
                object_text += "and notice nothing. Currently, you have grabbed the chips in hand. "
            elif not hold_chips and hold_milk:
                object_text += "and notice nothing. Currently, you have grabbed the milk in hand. "
            else:
                object_text += "and notice nothing. Currently, you are not grabbing anything in hand. "

            action_list = [0, 1, 2]

        elif in_bathroom:

            if hold_chips and hold_milk:
                object_text += "and notice nothing. Currently, you have grabbed the chips and the milk in hand. "
            elif hold_chips and not hold_milk:
                object_text += "and notice nothing. Currently, you have grabbed the chips in hand. "
            elif not hold_chips and hold_milk:
                object_text += "and notice nothing. Currently, you have grabbed the milk in hand. "
            else:
                object_text += "and notice nothing. Currently, you are not grabbing anything in hand. "

            action_list = [0, 1, 3]

        text += object_text

        # template for target
        target_template = "In order to enjoy the chips and the milk while watching TV, "
        text += target_template

        # template for next step
        next_step_text = "your next step is to"
        text += next_step_text

        self.action_template = [
            "walk to the living room", # 0
            "walk to the kitchen", # 1
            "walk to the bathroom", # 2
            "walk to the bedroom", # 3

            "walk to the chips", # 4
            "walk to the milk", # 5
            'walk to the coffee table', # 6
            'walk to the TV', # 7
            'walk to the sofa', # 8

            "grab the chips", # 9
            "grab the milk", # 10

            'put the chips on the coffee table', # 11
            'put the milk on the coffee table', # 12

            "turn on the TV", # 13
            "turn off the TV", # 14

            "sit on the sofa", # 15
            "stand up from the sofa" # 16
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
        # return self._initial_value if self._visit_count ==0 else 1 / ((self._visit_count)**self.value_weight +1) * self._initial_value + self._path_value/self._visit_count  # 0.8 for tomato_salad
        # return self._initial_value if self._visit_count ==0 else self._path_value/self._visit_count + self._initial_value/(math.log(self._visit_count+1)) #UCC
        return self._initial_value if self._visit_count ==0 else self._path_value/self._visit_count + 0.5*self._initial_value * math.sqrt(math.log(self.parent._visit_count)/self._visit_count) # UCT
   
    @property
    def ucb_score(self,):
        return self.value if self._visit_count==0 else self.value / self._visit_count + math.sqrt(2 * math.log(self.parent._visit_count) / self._visit_count) 

    


def get_root(node: Node):
    while not node.is_root():
        node = node.parent
    return node


class LLMAgent(nn.Module):
    def __init__(self, normalization_mode = 'token', load_path = None, load_8bit = True, base_model = None,
        beta: float = 5.0,  # 控制探索-利用平衡的参数
         c_uct: float = 1.0,  # UCT探索常数
         gamma: float = 0.95):
        super().__init__()
        self.load_8bit = load_8bit
        self.epsilon = 0.2
        self.base_model = base_model
        assert (
            self.base_model
        ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
        if self.base_model == "meta-llama/Meta-Llama-3.1-8B-Instruct":
            self.base_model = "/home/data/dzw/llm/Decision_AI_Agent/MCTSPP/hf_models/meta-llama/Llama-3.1-8B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        self.tokenizer.pad_token_id = (
            0  # unk. we want this to be different from the eos token
        )
        self.model = self._init_llama()
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.normalization_mode = normalization_mode
        self.alpha = 0.99 # value reward ratio
        self.gamma = 0.99 # decay rate
        self.beta = beta
        self.c_uct = c_uct
        self.gamma = gamma

    def _init_llama(self):
        model = LlamaForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=torch.float16,
            load_in_4bit=False,
            device_map="auto",
            #cache_dir='weights/llama'
        )

        return model
    

    def get_value(self, x):
        if type(x) != list:
            x = [self.obs2text(o)["prompt"] for o in x]
            
        inputs = self.tokenizer(x, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        with self.actor.disable_adapter():
            value = self.critic(input_ids, attention_mask=attention_mask)
        return value

    def expand(self, obs, node, simulate_envs, value_weight = 0.5):
        text_obs = [self.obs2text(o) for o in obs]
        prompt = [o["prompt"] for o in text_obs]

        action_list = [o["action"] for o in text_obs]
        action_ids = [[self.template2action[item] for item in env] for env in action_list]
        
        prompt_nums = len(prompt)
        action_nums = [len(item) for item in action_list]

        sequence = []
        for p, ac in zip(prompt, action_list):
            sequence += [p + " " + a for a in ac]

        inputs = self.tokenizer(sequence, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(self.device)
        
        attention_mask = inputs["attention_mask"].to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
        
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

        actions = []
        log_probs = []
        entroy = []

        for i in range(prompt_nums):
            logits = action_logits[sum(action_nums[:i]):sum(action_nums[:i+1])].reshape(-1, action_nums[i]).float()

            probs = Categorical(logits=logits)

            for id, action in enumerate(action_list):
                        
                envs = copy.deepcopy(simulate_envs)
                id_tmp = torch.tensor([action_ids[i][id]],device=self.device)
                id_for_prob = torch.tensor([id],device= self.device)
                next_obs, reward, done, info = envs.step(id_tmp.cpu().numpy())
                re = torch.tensor([reward],device=self.device).view(-1) #添加reward到initial value
                child = LanguageNode(parent=node, state = next_obs, last_action=action, initial_value = probs.log_prob(id_for_prob).exp()+re,action_id = action_ids[i][id],value_weight=value_weight)
                node.children[action] = child

    
    def select(self, obs, node):
        # TODO: 添加RND
        # input:obs output:node
        obs = node.state[0]
        action = []
        value = []
        text_obs = [self.obs2text(o) for o in obs]
        prompt = [o["prompt"] for o in text_obs]
        action_list = [o["action"] for o in text_obs]
        # epsilon greedy
        for action_tmp, child_tmp in node.children.items():
            action.append(action_tmp)
            v = child_tmp.get_uct_value(self.c_uct,self.beta)
            value.append(v)
        values = torch.tensor(value,device=self.device)
        argmax = torch.argmax(values).item()
        action_id_argmax = node.children[action[argmax]].action_id

        if False:
            return torch.tensor([action_id_sampled]),values[sampled_index], node.children[action[sampled_index]],action[sampled_index]
        else:
            return torch.tensor([action_id_argmax]),values[torch.argmax(values)], node.children[action[argmax]], action[argmax]

    def mcts_update(self, node_path, rewards):
        rewards = rewards.view(-1)
        assert len(node_path) == len(rewards)

        cum_rewards = torch.zeros(len(rewards),device = self.device)
        for i in range(len(rewards)):
            cum_rewards[i] = sum(rewards[i:])
        for i,node in enumerate(node_path):
            node_path[i].cum_rewards.append(cum_rewards[i].cpu())
                # 如果是叶子节点，初始化方差
            if i == len(node_path) - 1:
                # 估计奖励的方差
                if len(node.cum_rewards) > 1:
                    node.reward_variance = np.var(node.cum_rewards)
                
            # 如果不是叶子节点，更新方差
            elif i < len(node_path) - 1:
                child = node_path[i + 1]
                # V[ν(sk, ak)] ← V[R̂(sk, ak)] + γ²V[ν(sk + 1, ak + 1)]
                node.value_variance = node.reward_variance + (self.gamma ** 2) * child.value_variance
                
                # 更新不确定性估计
                if node.visit_count > 1:
                    sigma_q = math.sqrt(node.value_variance - node.reward_variance) / node.visit_count
                    node.value_variance += sigma_q


    def obs2text(self, obs):

        text = ""

        in_kitchen = obs[0]
        in_bathroom = obs[1]
        in_bedroom = obs[2]
        in_livingroom = obs[3]
        
        see_chips = obs[4]
        close_to_chips = obs[5]
        hold_chips = obs[6]
        chips_on_coffeetable = obs[7]
        
        see_milk = obs[8]
        close_to_milk = obs[9]
        hold_milk = obs[10]
        milk_on_coffeetable = obs[11]

        see_tv = obs[12]
        close_to_tv = obs[13]
        is_face_tv = obs[14]
        is_tv_on = obs[15]

        see_sofa = obs[16]
        close_to_sofa = obs[17]
        is_sit_sofa = obs[18]

        see_coffeetable = obs[19]
        close_to_coffeetable = obs[20]
        assert in_kitchen + in_bathroom + in_bedroom + in_livingroom == 1, "Only one room can be true at a time"

        # template for room
        in_room_teplate = "There are four rooms: the kitchen, bathroom, bedroom, and living room. You are in the {} "
        if in_kitchen:
            text += in_room_teplate.format("kitchen")
        elif in_bathroom:
            text += in_room_teplate.format("bathroom")
        elif in_bedroom:
            text += in_room_teplate.format("bedroom")
        elif in_livingroom:
            text += in_room_teplate.format("living room")

        ########################################template2####################################
        # template for kitchen
        object_text = ""

        action_list = []

        if in_kitchen:

            if see_chips and see_milk:
                object_text += "and notice chips and milk. "

                if hold_chips and hold_milk:
                    object_text += "Currently, you have grabbed the chips and the milk in hand. "

                    action_list = [
                        0,
                        2,
                        3,
                    ]

                elif hold_chips and not hold_milk:
                    if close_to_milk:
                        object_text += "The milk is close to you. But you have not grabbed the milk. Currently, you have grabbed the chips in hand. "

                        action_list = [
                            0,
                            2,
                            3,
                            10
                        ]
                    else:
                        object_text += "The milk is not close to you. Currently, you have grabbed the chips in hand. "

                        action_list = [
                            0,
                            2,
                            3,
                            5
                        ]
                elif not hold_chips and hold_milk:
                    if close_to_chips:
                        object_text += "The chips are close to you. But you have not grabbed the chips. Currently, you have grabbed the milk in hand. "

                        action_list = [
                            0,
                            2,
                            3,
                            9
                        ]
                    else:
                        object_text += "The chips are not close to you. Currently, you have grabbed the milk in hand. "

                        action_list = [
                            0,
                            2,
                            3,
                            4
                        ]
                else:
                    if close_to_chips and close_to_milk:
                        object_text += "They are close to you. But you have not grabbed the them. "

                        action_list = [
                            0,
                            2,
                            3,
                            9,
                            10
                        ]

                    elif close_to_chips and not close_to_milk:
                        object_text += "The chips are close to you. But you have not grabbed the chips. "

                        action_list = [
                            0,
                            2,
                            3,
                            5,
                            9,
                        ]

                    elif not close_to_chips and close_to_milk:
                        object_text += "The milk is close to you. But you have not grabbed the milk. "

                        action_list = [
                            0,
                            2,
                            3,
                            4,
                            10,
                        ]

                    else:
                        object_text += "But they are not close to you. "

                        action_list = [
                            0,
                            2,
                            3,
                            4,
                            5,
                        ]

                    object_text += "Currently, you are not grabbing anything in hand. "

            elif see_chips and not see_milk:
                object_text += "and only notice chips. "

                if hold_chips:
                    object_text += "Currently, you have grabbed the chips in hand. "

                    action_list = [
                        0,
                        2,
                        3,
                    ]

                else:
                    if close_to_chips:
                        object_text += "The chips are close to you. But you have not grabbed the chips. "

                        action_list = [
                            0,
                            2,
                            3,
                            9,
                        ]
                    else:
                        object_text += "The chips are not close to you. "

                        action_list = [
                            0,
                            2,
                            3,
                            5,
                        ]

            elif not see_chips and see_milk:
                object_text += "and notice milk. "

                if hold_milk:
                    object_text += "Currently, you have grabbed the milk in hand. "

                    action_list = [
                        0,
                        2,
                        3,
                    ]

                else:
                    if close_to_milk:
                        object_text += "The milk is close to you. But you have not grabbed the milk. "

                        action_list = [
                            0,
                            2,
                            3,
                            10,
                        ]
                    else:
                        object_text += "The milk is not close to you. "

                        action_list = [
                            0,
                            2,
                            3,
                            4,
                        ]

            else:
                object_text += "and notice nothing. "

                action_list = [
                    0,
                    2,
                    3,
                ]

        elif in_livingroom:

            object_text += "and you notice a coffee table, a TV and a sofa. "

            assert close_to_coffeetable + close_to_tv + close_to_sofa <= 1, "You are next to more than one object from coffee table, TV and sofa."
            assert see_coffeetable + see_tv + see_sofa >= 3, "You don't see coffee table, TV and sofa."

            if not close_to_coffeetable and not close_to_tv and not close_to_sofa:
                object_text += "They are not close to you. "

                if hold_chips and hold_milk:
                    object_text += "Currently, you have grabbed the chips and the milk in hand. "
                elif not hold_chips and hold_milk:
                    object_text += "Currently, you have grabbed the milk in hand. "
                elif hold_chips and not hold_milk:
                    object_text += "Currently, you have grabbed the chips in hand. "
                else:
                    object_text += "Currently, you are not grabbing anything in hand. "

                action_list = [
                    1,
                    2,
                    3,
                    6,
                    7,
                    8
                ]

            if close_to_coffeetable:

                if (chips_on_coffeetable and hold_milk) or (milk_on_coffeetable and hold_chips):
                    object_text += "The TV is not close to you. "
                else:
                    object_text += "The coffee table is close to you. "

                if hold_chips and hold_milk:
                    object_text += "Currently, you have grabbed the chips and the milk in hand. "

                    action_list = [
                        1,
                        2,
                        3,
                        7,
                        8,
                        11,
                        12
                    ]
                elif not hold_chips and hold_milk:
                    if not chips_on_coffeetable:
                        object_text += "Currently, you have grabbed the milk in hand. "

                        action_list = [
                            1,
                            2,
                            3,
                            7,
                            8,
                            12
                        ]

                    else:
                        object_text += "Currently, you have the chips on the coffee table and the milk in your hand. "

                        action_list = [
                            1,
                            2,
                            3,
                            7,
                            8,
                        ]

                elif hold_chips and not hold_milk:
                    object_text += "Currently, you have grabbed the chips in hand. "

                    if not milk_on_coffeetable:
                        object_text += "Currently, you have grabbed the chips in hand. "

                        action_list = [
                            1,
                            2,
                            3,
                            7,
                            8,
                            11
                        ]

                    else:
                        object_text += "Currently, you have the milk on the coffee table and the chips in your hand. "

                        action_list = [
                            1,
                            2,
                            3,
                            7,
                            8,
                        ]

                else:
                    object_text += "Currently, you are not grabbing anything in hand. "

                    action_list = [
                        1,
                        2,
                        3,
                    ]

            if close_to_tv:
                if is_tv_on:
                    object_text += "The sofa is not close to you. "

                    if hold_chips and hold_milk:
                        object_text += "Currently, the TV is turned on, you have grabbed the chips and the milk in hand. "

                    elif not hold_chips and hold_milk:
                        if not chips_on_coffeetable:
                            object_text += "Currently, the TV is turned on, you have grabbed the milk in hand. "
                        else:
                            object_text += "Currently, the TV is turned on, you have the chips on the coffee table and the milk in your hand. "
                    elif hold_chips and not hold_milk:
                        object_text += "Currently, the TV is turned on, you have grabbed the chips in hand. "
                        if not milk_on_coffeetable:
                            object_text += "Currently, the TV is turned on, you have grabbed the chips in hand. "
                        else:
                            object_text += "Currently, the TV is turned on, you have the milk on the coffee table and the chips in your hand. "

                    action_list = [
                        1,
                        2,
                        3,
                        6,
                        8,
                    ]

                else:
                    object_text += "The TV is close to you. "

                    if hold_chips and hold_milk:
                        object_text += "Currently, you have grabbed the chips and the milk in hand. "

                    elif not hold_chips and hold_milk:
                        if not chips_on_coffeetable:
                            object_text += "Currently, you have grabbed the milk in hand. "
                        else:
                            object_text += "Currently, you have the chips on the coffee table and the milk in your hand. "
                    elif hold_chips and not hold_milk:
                        object_text += "Currently, you have grabbed the chips in hand. "
                        if not milk_on_coffeetable:
                            object_text += "Currently, you have grabbed the chips in hand. "
                        else:
                            object_text += "Currently, you have the milk on the coffee table and the chips in your hand. "

                    action_list = [
                        1,
                        2,
                        3,
                        6,
                        8,
                        13,
                        14
                    ]

            if close_to_sofa:

                if not is_sit_sofa:
                    object_text += "The sofa is close to you. "

                    if is_tv_on:
                        if hold_chips and hold_milk:
                            object_text += "Currently, the TV is turned on, you have grabbed the chips and the milk in hand. "

                        elif not hold_chips and hold_milk:
                            if not chips_on_coffeetable:
                                object_text += "Currently, the TV is turned on, you have grabbed the milk in hand. "
                            else:
                                object_text += "Currently, the TV is turned on, you have the chips on the coffee table and the milk in your hand. "
                        elif hold_chips and not hold_milk:
                            object_text += "Currently, the TV is turned on, you have grabbed the chips in hand. "
                            if not milk_on_coffeetable:
                                object_text += "Currently, the TV is turned on, you have grabbed the chips in hand. "
                            else:
                                object_text += "Currently, the TV is turned on, you have the milk on the coffee table and the chips in your hand. "

                        action_list = [
                            1,
                            2,
                            3,
                            6,
                            7,
                            15,
                            16
                        ]
                    else:
                        if hold_chips and hold_milk:
                            object_text += "Currently, you have grabbed the chips and the milk in hand. "

                        elif not hold_chips and hold_milk:
                            if not chips_on_coffeetable:
                                object_text += "Currently, you have grabbed the milk in hand. "
                            else:
                                object_text += "Currently, you have the chips on the coffee table and the milk in your hand. "
                        elif hold_chips and not hold_milk:
                            object_text += "Currently, you have grabbed the chips in hand. "
                            if not milk_on_coffeetable:
                                object_text += "Currently, you have grabbed the chips in hand. "
                            else:
                                object_text += "Currently, you have the milk on the coffee table and the chips in your hand. "

                        action_list = [
                            1,
                            2,
                            3,
                            6,
                            7,
                        ]

                else:
                    object_text += "You are sitting on the sofa. "

                    if is_tv_on:
                        if hold_chips and hold_milk:
                            object_text += "Currently, the TV is turned on, you have grabbed the chips and the milk in hand. "

                        elif not hold_chips and hold_milk:
                            if not chips_on_coffeetable:
                                object_text += "Currently, the TV is turned on, you have grabbed the milk in hand. "
                            else:
                                object_text += "Currently, the TV is turned on, you have the chips on the coffee table and the milk in your hand. "
                        elif hold_chips and not hold_milk:
                            object_text += "Currently, the TV is turned on, you have grabbed the chips in hand. "
                            if not milk_on_coffeetable:
                                object_text += "Currently, the TV is turned on, you have grabbed the chips in hand. "
                            else:
                                object_text += "Currently, the TV is turned on, you have the milk on the coffee table and the chips in your hand. "

                        action_list = [1, 2, 3]
                    else:
                        if hold_chips and hold_milk:
                            object_text += "Currently, you have grabbed the chips and the milk in hand. "

                        elif not hold_chips and hold_milk:
                            if not chips_on_coffeetable:
                                object_text += "Currently, you have grabbed the milk in hand. "
                            else:
                                object_text += "Currently, you have the chips on the coffee table and the milk in your hand. "
                        elif hold_chips and not hold_milk:
                            object_text += "Currently, you have grabbed the chips in hand. "
                            if not milk_on_coffeetable:
                                object_text += "Currently, you have grabbed the chips in hand. "
                            else:
                                object_text += "Currently, you have the milk on the coffee table and the chips in your hand. "

                        action_list = [1, 2, 3]

        elif in_bedroom:

            if hold_chips and hold_milk:
                object_text += "and notice nothing. Currently, you have grabbed the chips and the milk in hand. "
            elif hold_chips and not hold_milk:
                object_text += "and notice nothing. Currently, you have grabbed the chips in hand. "
            elif not hold_chips and hold_milk:
                object_text += "and notice nothing. Currently, you have grabbed the milk in hand. "
            else:
                object_text += "and notice nothing. Currently, you are not grabbing anything in hand. "

            action_list = [0, 1, 2]

        elif in_bathroom:

            if hold_chips and hold_milk:
                object_text += "and notice nothing. Currently, you have grabbed the chips and the milk in hand. "
            elif hold_chips and not hold_milk:
                object_text += "and notice nothing. Currently, you have grabbed the chips in hand. "
            elif not hold_chips and hold_milk:
                object_text += "and notice nothing. Currently, you have grabbed the milk in hand. "
            else:
                object_text += "and notice nothing. Currently, you are not grabbing anything in hand. "

            action_list = [0, 1, 3]

        text += object_text

        # template for target
        target_template = "In order to enjoy the chips and the milk while watching TV, "
        text += target_template

        # template for next step
        next_step_text = "your next step is to"
        text += next_step_text

        self.action_template = [
            "walk to the living room", # 0
            "walk to the kitchen", # 1
            "walk to the bathroom", # 2
            "walk to the bedroom", # 3

            "walk to the chips", # 4
            "walk to the milk", # 5
            'walk to the coffee table', # 6
            'walk to the TV', # 7
            'walk to the sofa', # 8

            "grab the chips", # 9
            "grab the milk", # 10

            'put the chips on the coffee table', # 11
            'put the milk on the coffee table', # 12

            "turn on the TV", # 13
            "turn off the TV", # 14

            "sit on the sofa", # 15
            "stand up from the sofa" # 16
        ]

        self.template2action = {
            k:i for i,k in enumerate(self.action_template)
        }

        actions = [self.action_template[i] for i in action_list]

        return {"prompt": text, "action": actions}