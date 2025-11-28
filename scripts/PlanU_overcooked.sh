for seed in 1 10 20 30 40; do
CUDA_VISIBLE_DEVICES=0,1,2,3 python mcts/overcooked/PlanU_inference.py \
  --exp-name "tomato_salad_llm"\
  --num-envs 1 \
  --depth 15 \
  --env-reward 0.2 1 0.1 0.001 \
  --task 0 \
  --env-id "Overcooked-LLMA-v4" \
  --record-path "workdir" \
  --normalization-mode "word"\
  --maxiterations 1000\
  --stochastic 0.5\
  --base-model "meta-llama/Meta-Llama-3.1-8B-Instruct"\
  --seed $seed \
  --rnd "True"
done
