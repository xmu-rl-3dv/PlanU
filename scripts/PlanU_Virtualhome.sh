CUDA_VISIBLE_DEVICES=0,1,2,3
for seed in 1 10 20 30 40; do
python mcts/virtualhome/PlanU_inference_entertainment.py \
    --stochastic 0.2 \
    --maxiterations 1000\
    --base-model "meta-llama/Meta-Llama-3.1-8B-Instruct"\
    --stochastic 0.2\
    --rnd "True"\
    --seed $seed\
done
