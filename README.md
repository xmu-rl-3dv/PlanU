<<<<<<< HEAD
### Overcooked and VirtualHome

#### Setup

To get started:

1. Clone this repo and install the requirements:
```bash
pip install -r requirements.txt
```

2. Set the path of your local model in 'mcts\overcooked\PlanU_mcts.py' line 827
   'mcts\virtualhome\PlanU_v1.py' line 539
    'mcts\virtualhome\PlanU_v2.py' line 965

3. Run the scripts in folder scripts
```bash
sh scripts\PlanU_overcooked.sh
sh scripts\PlanU_Virtualhome.sh
```

### WebShop

#### Setup
For Webshop, our code is adapted from LATS, we really appreciate their efforts to the community.(https://github.com/lapisrocks/LanguageAgentTreeSearch)
To get started:

1. Clone this repo and move the folder webshop to the WebShop directory:

2. Install WebShop from source and run environment instance locally. Follow the instructions here (https://github.com/princeton-nlp/WebShop)

3. Install the module dependencies into your environment:
```bash
pip install -r requirements.txt
```

4. Set `OPENAI_API_KEY` environment variable to your OpenAI API key:
```bash
export OPENAI_API_KEY=<your key>
```

5. Change localhost in lats.py to your local port running WebShop

6. Set the scripts and run paper experiments
```bash
sh planu.sh
```

- ``--n_generate_sample``: number of times to prompt during expansion/sampling
- ``--n_evaluate_sample``: number of times to prompt for state evaluation
- ``--iterations``: maximum number of trajectories to sample

## Trajectories
``programming/root/`` contains all the trajectories from the paper's experiments on programming. Please use get_acc.py with the log path to get the actual accuracy. HotPotQA and WebShop logs were too large to upload, feel free to email if interested.

=======
### Overcooked and VirtualHome

#### Setup

To get started:

1. Clone this repo and install the requirements:
```bash
pip install -r requirements.txt
```

2. Set the path of your local model in 'mcts\overcooked\PlanU_mcts.py' line 827
   'mcts\virtualhome\PlanU_v1.py' line 539
    'mcts\virtualhome\PlanU_v2.py' line 965

3. Run the scripts in folder scripts
```bash
sh scripts\PlanU_overcooked.sh
sh scripts\PlanU_Virtualhome.sh
```

### WebShop

#### Setup
For Webshop, our code is adapted from LATS, we really appreciate their efforts to the community.(https://github.com/lapisrocks/LanguageAgentTreeSearch)
To get started:

1. Clone this repo and move the folder webshop to the WebShop directory:

2. Install WebShop from source and run environment instance locally. Follow the instructions here (https://github.com/princeton-nlp/WebShop)

3. Install the module dependencies into your environment:
```bash
pip install -r requirements.txt
```

4. Set `OPENAI_API_KEY` environment variable to your OpenAI API key:
```bash
export OPENAI_API_KEY=<your key>
```

5. Change localhost in lats.py to your local port running WebShop

6. Set the scripts and run paper experiments
```bash
sh planu.sh
```

- ``--n_generate_sample``: number of times to prompt during expansion/sampling
- ``--n_evaluate_sample``: number of times to prompt for state evaluation
- ``--iterations``: maximum number of trajectories to sample

## Trajectories
``programming/root/`` contains all the trajectories from the paper's experiments on programming. Please use get_acc.py with the log path to get the actual accuracy. HotPotQA and WebShop logs were too large to upload, feel free to email if interested.

>>>>>>> 6cec7b1d07aa73bec9af55f1804bd30d7598685a
