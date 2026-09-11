import datasets
import json
from tqdm import tqdm
import torch
import os, pickle
from datetime import datetime
import sys
import random
from reasoners import Evaluator
import copy
import re

import reasoners.benchmark.bw_utils as bw_utils

def rap_bw_extractor(algo_output):
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    # to make sure the plan is saved before evaluation in multi-process setting
    try:
        if algo_output.trace is None:
            print("No plan found")
            return ""
        else:
            return "\n".join(algo_output.trace[1])
    except Exception as e:
        print("Error in output extraction,", e)
        return ""

def get_icl(init_prompt, examples):
    icl = init_prompt["intro"] + \
        "\n".join([
            "[STATEMENT]\nAs initial conditions I have that, " + \
            example["init"] + \
            ".\nMy goal is to have that " +\
            example["goal"] + \
            ".\n\nMy plan is as follows:\n\n[PLAN]" + \
            example["plan"]
            for example in examples
        ])
    icl += "\n[STATEMENT]\nAs initial conditions I have that, <init_state>\nMy goal is to <goals>\n\nMy plan is as follows:\n\n[PLAN]\n<action>"
    return icl

class BWEvaluator(Evaluator):
    def __init__(self,
                 config_file,
                 domain_file,
                 data_path,
                 init_prompt,
                 disable_log=False,
                 disable_tqdm=False,
                 output_extractor=rap_bw_extractor,
                 answer_extractor=lambda x:x,
                 sample_prompt_type="rap") -> None:

        self.init_prompt = init_prompt
        self.output_extractor = output_extractor
        self.answer_extractor = answer_extractor
        self.input_processor = lambda x: x
        self.full_dataset = bw_utils.load_blocksworld(config_file, domain_file, data_path)  # [{"goal": str, "init": str}]
        self._dataset_name = 'blocksworld'
        self.disable_log = disable_log
        self.disable_tqdm = disable_tqdm
        self.sample_prompt_type = sample_prompt_type

        self.lm_plan_file = "tmp_plan.txt"
        self.config_file = config_file
        self.domain_file = domain_file

    def sample_prompt(self,
                      shuffle_prompt=True,
                      num_shot=4):

        sample_prompt_type = self.sample_prompt_type
        if sample_prompt_type == "rap":
            if shuffle_prompt:
                examples = random.sample(self.init_prompt["example_pool"], num_shot)
            else:
                examples = self.init_prompt["example_pool"][:num_shot]

            icl = get_icl(self.init_prompt, examples)

            prompt = copy.deepcopy(self.init_prompt)
            prompt["icl"] = icl
            prompt["icl_list"] = [icl]
            examples = copy.deepcopy(examples)
            for i in range(5):
                new_examples = []
                for example in examples:
                    if len(example["states"]) > 1:
                        new_examples.append({
                            "init": example["states"][0],
                            "goal": example["goal"],
                            "plan": "\n" + "\n".join(example["plan"].split("\n")[3:]),
                            "states": example["states"][1:]
                        })
                    else:
                        new_examples.append(example)
                examples = copy.deepcopy(new_examples)
                icl = get_icl(self.init_prompt, examples)
                prompt["icl_list"].append(icl)
        else:
            raise NotImplementedError
        # print("prompt:",  prompt)
        return prompt

    def eval_output(self, answer, output):
        bw_utils.text_to_plan_blocksworld(output, answer["instance_file"], self.config_file, self.domain_file, self.lm_plan_file)
        correct = bw_utils.validate_plan_new(answer, output)
        return correct

    def eval_stochastic_ouput(self,final_state, goal):
        return goal in final_state

    def parse_conditions(self, condition_str, bd):
        conditions = []
        parts = re.split(r',\s*|\s+and\s+', condition_str)
        for part in parts:
            part = re.sub(r'^\s*(and)?\s*', '', part.strip()).rstrip('.')
            if not part:
                continue
            cond = self.parse_condition(part, bd)
            if cond is None:
                return None
            conditions.append(cond)
        return conditions

    def parse_condition(self, cond_str, bd):
        cond_str = cond_str.strip().lower().rstrip('.')
        clear_pattern = r'^the (\w+) block is (clear|not covered)\.?$'
        on_pattern = r'^the (\w+) block is (on top of|on|over) the (\w+) block\.?$'
        on_table_pattern = r'^the (\w+) block is (on the table|ontable|placed on table)\.?$'
        hand_empty_pattern = r'^the hand is (empty|free|holding nothing)\.?$'

        match = re.match(clear_pattern, cond_str)
        if match:
            color = match.group(1)
            block_name = f"{color} block"
            if block_name not in bd:
                print(f"Error: Block {block_name} not found in BD mapping!")
                return None
            return ('clear', bd[block_name])

        match = re.match(on_pattern, cond_str)
        if match:
            color1 = match.group(1)
            color2 = match.group(3)
            block1 = f"{color1} block"
            block2 = f"{color2} block"
            if block1 not in bd or block2 not in bd:
                print(f"Error: Blocks {block1} or {block2} not found in BD mapping!")
                return None
            return ('on', bd[block1], bd[block2])

        match = re.match(on_table_pattern, cond_str)
        if match:
            color = match.group(1)
            block = f"{color} block"
            if block not in bd:
                print(f"Error: Block {block} not found in BD mapping!")
                return None
            return ('on_table', bd[block])

        match = re.match(hand_empty_pattern, cond_str)
        if match:
            return ('hand_empty',)

        print(f"Error: Unrecognized condition format: {cond_str}")
        return None

    def eval_output_new(self, final_state, goal):

        data = bw_utils.read_config(self.config_file)
        LD = data['encoded_objects']
        BD = {v: k for k, v in LD.items()}

        final_conds = self.parse_conditions(final_state, BD)
        goal_conds = self.parse_conditions(goal, BD)

        if final_conds is None or goal_conds is None:
            return False

        for g_cond in goal_conds:
            if g_cond not in final_conds:
                return False
        return True
