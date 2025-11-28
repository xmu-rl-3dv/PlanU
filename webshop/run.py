import os
import json
import argparse
import logging

from models import gpt_usage
from lats import lats_search
from webshop import WebShopTask
from planu import planu_search

# Configuring the logging
# Configuring the logging

class ResultFilter(logging.Filter):
    def filter(self, record):
        return "RESULTS:" in record.getMessage()

def setup_logging(main_log_path, result_log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 主日志文件 handler
    file_handler = logging.FileHandler(main_log_path, mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    # 结果日志文件 handler（带过滤器）
    result_handler = logging.FileHandler(result_log_path, mode='a')
    result_handler.setLevel(logging.INFO)
    result_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    result_handler.addFilter(ResultFilter())

    # 清除旧的 handlers（防止重复）
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.addHandler(file_handler)
    logger.addHandler(result_handler)
    
def run(args):
    task = WebShopTask()
    print(task)
    logs, cnt_avg, cnt_any = [], 0, 0
    
    # logging.basicConfig(filename=args.log, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filemode='a')
    setup_logging(args.log, "logs/results_only.log")
    count = 0
    task_accs = []
    info = []
    n = args.task_end_index

    for i in range(args.task_start_index, args.task_end_index):
        # solve
        if args.method == 'planu':
            state, value, reward, em, failure_times = planu_search(args, task, f'fixed_{i}', args.iterations, True)
        elif args.method == 'lats':
            state, value, reward, em, failure_times = lats_search(args, task, f'fixed_{i}', args.iterations, True)
        
         # log main metric
        # task_accs.append(em)
        print("best reward", reward)
        # cnt_avg = sum(task_accs) / len(task_accs)
        # print(i, 'len(task_accs)', len(task_accs), 'cnt_avg', cnt_avg, '\n')
        task_accs.append(reward)
        if (i+1) % 1 == 0:
            r, sr, fr = sum(task_accs) / len(task_accs), len([_ for _ in task_accs if _ == 1]) / len(task_accs), count / len(task_accs)
            print(i+1, r, sr, fr)
            print('-------------')
        r, sr, fr = sum(task_accs) / len(task_accs), len([_ for _ in task_accs if _ == 1]) / n, count / n
        print(r, sr, fr)

        logging.info(f"RESULTS: fixed_{i}: {reward}, {r}, {sr}, {fr}, failure_times: {failure_times}")

    n = args.task_end_index - args.task_start_index
    print('usage_so_far', gpt_usage(args.backend))

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--method', type=str, choices=["planu", "lats"], default='lats')
    args.add_argument('--backend', type=str, choices=['gpt-4', 'gpt-3.5-turbo', 'gpt-3.5-turbo-16k', 'llama2', "text-davinci-002", "qwen-plus", "qwen-max", "Qwen/Qwen3-8B"], default='gpt-3.5-turbo-16k')
    args.add_argument('--temperature', type=float, default=1.0)
    args.add_argument('--task_start_index', type=int, default=900)
    args.add_argument('--task_end_index', type=int, default=1000)
    args.add_argument('--prompt_sample', type=str, choices=['standard', 'cot'])  
    args.add_argument('--n_generate_sample', type=int, default=1)  # only thing needed if naive_run
    args.add_argument('--n_evaluate_sample', type=int, default=1)
    args.add_argument('--iterations', type=int, default=30)
    args.add_argument('--log', type=str)

    args = args.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print(args)
    run(args)