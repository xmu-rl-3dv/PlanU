# NeurIPS 2025 PlanU: Large Language Model Reasoning through Planning under Uncertainty

![Python 3.8](https://img.shields.io/badge/Python-3.8-blue)
![Code style](https://img.shields.io/badge/code%20style-black-000000.svg)
![MIT](https://img.shields.io/badge/license-MIT-blue)

This is the official implementation of **"PlanU: Large Language Model Reasoning through Planning under Uncertainty"** accepted at **NeurIPS 2025**.

## 📌 Project Overview

**PlanU** introduces a novel **planning-under-uncertainty framework** that equips Large Language Models (LLMs) with explicit **Monte-Carlo Tree Search (MCTS)** reasoning for long-horizon decision making in stochastic environments.  

## 🎯 Quick Start

### ⚙️ Environment Setup

```bash
# 1. Clone repo
git clone https://github.com/xmu-rl-3dv/PlanU.git
cd PlanU

# 2. Install dependencies
pip install -r requirements.txt
```

### 📂 Overcooked & VirtualHome

1. Set **local LLM path** in  
   `mcts/overcooked/PlanU_mcts.py#L827`  
   `mcts/virtualhome/PlanU_v1.py#L539`  
   `mcts/virtualhome/PlanU_v2.py#L965`

2. Run scripts  
```bash
sh scripts/PlanU_overcooked.sh
sh scripts/PlanU_Virtualhome.sh
```

### 📂 WebShop

1. Install WebShop locally ([guide](https://github.com/princeton-nlp/WebShop))  
2. Export your OpenAI key  
   - Linux/macOS: `export OPENAI_API_KEY=<your_key>`  
   - Windows PowerShell: `$env:OPENAI_API_KEY="your_key"`  
   - Windows CMD: `setx OPENAI_API_KEY "your_key"`  
3. Update `localhost` in `lats.py` to your WebShop port  
4. Launch experiments  
```bash
sh planu.sh
```

**CLI flags**  
- `--n_generate_sample` : # prompts during expansion  
- `--n_evaluate_sample` : # prompts for state evaluation  
- `--iterations`        : max trajectories to sample  

### 📊 Trajectories & Logs

All runs save to `programming/root/`.  
Use `python get_acc.py --log_path <dir>` to compute final scores.  
(HotPotQA & WebShop logs are too large for Git; email us if needed.)

## 📚 Citation

```bibtex
@inproceedings{planu2025,
  title={PlanU: Large Language Model Reasoning through Planning under Uncertainty},
  author={Ziwei Deng, Mian Deng, Chenjing Liang, Zeming Gao, Chennan Ma, Chenxing Lin, Haipeng Zhang, Songzhu Mei, Cheng Wang, Siqi Shen},
  booktitle={NeurIPS},
  year={2025}
}
```

## Acknowledgements

Code adapted from [LATS](https://github.com/lapisrocks/LanguageAgentTreeSearch) and [Overcooked-AI](https://github.com/HumanCompatibleAI/overcooked_ai).
