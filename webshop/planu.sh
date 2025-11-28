python run.py \
    --method planu\
    --backend qwen-plus \
    --task_start_index 1 \
    --task_end_index 50 \
    --n_generate_sample 5 \
    --n_evaluate_sample 1 \
    --prompt_sample cot \
    --temperature 0.8 \
    --iterations 10 \
    --log logs/lats_1.log \
    ${@}
