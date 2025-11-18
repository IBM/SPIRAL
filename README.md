# SPIRAL: Symbolic LLM Planning via Grounded and Reflective Search

This repository contains the code and analysis notebooks for **SPIRAL**, our framework that embeds a tri-agent cognitive architecture into an MCTS loop to enable more robust, grounded, and reflective planning with large language models. The full details are described in our paper and appendix.

## Table of Contents

- [Repository Structure](#repository-structure)  
- [Getting Started](#getting-started)  
- [Dependencies](#dependencies)  
- [Data](#data)  
- [Running Experiments](#running-experiments)  
- [Scripts & Agents](#scripts--agents)  
- [Analysis Notebooks](#analysis-notebooks)  
- [Configuration & Hyperparameters](#configuration--hyperparameters)  
- [License](#license)  

---

## Repository Structure

Note: Enter SPIRAL folder to follow the next information.

```
├── analysis/
│   ├── ablation/
│   │   └── analysis_ablations.ipynb
│   ├── baseline/
│   │   ├── cot_k1/
│   │   ├── cot_k3/
│   │   ├── cot_k5/
│   │   ├── spiral/
│   │   ├── analysis_baseline_performance.ipynb
│   │   ├── analysis_cost_benefit.ipynb
│   │   ├── cost_comparison_api_calls.pdf
│   │   └── cost_comparison_tokens.pdf
│   └── sota/
│       ├── sota_performance/
│       └── tot_hyper_params_performance/
│
├── scripts/
│   ├── taskbench_ablation.py
│   ├── taskbench_cot_baseline.py
│   ├── taskbench_lats_baseline.py
│   ├── taskbench_rafa_baseline.py
│   ├── taskbench_react_baseline.py
│   ├── taskbench_react_rafa_baseline.py
│   ├── taskbench_spiral.py
│   └── taskbench_tot_baseline.py
│
├── Taskbench/
│   ├── data_dailylifeapis/
│   └── data_huggingface/
│
├── utils/
│   └── generic_client.py
│
├── environment.yml
├── run_all_baseline_experiments.sh
├── run_all_ablation_experiments.sh
├── run_all_sota_experiments.sh
└── LICENSE
```

---

## Getting Started

1. **Clone the repository**  
   ```bash
   git clone https://github.com/<your-org-or-username>/SPIRAL.git
   cd SPIRAL
   ```

2. **Create the Conda environment**  
   ```bash
   conda create -n spiral python=3.11
   conda activate spiral
   pip install -r requirements.txt
   ```

3. **Download TaskBench datasets**  
   Place the `dailylifeapis` and `huggingface` benchmark data under `Taskbench/data_dailylifeapis/` and `Taskbench/data_huggingface/`, respectively.
  ```
   cd Taskbench/
   huggingface-cli download microsoft/Taskbench --local-dir . --repo-type dataset
  ```
---

## Dependencies
 

All required packages are listed in requirements.txt

---

## Data

We evaluate on two TaskBench tool-use benchmarks:

- **DailyLifeAPIs** (`Taskbench/data_dailylifeapis/`)  
- **HuggingFace** (`Taskbench/data_huggingface/`)

Each dataset should be organized as in the original TaskBench release:

```
Taskbench/data_dailylifeapis/
└── problems.jsonl

Taskbench/data_huggingface/
└── problems.jsonl
```

---

## Running Experiments

### 1. Baseline Methods
NOTE: these won't work, they require a different library structure

```bash
./run_all_baseline_experiments.sh
```

This will run Chain-of-Thought (k=1,3,5), ReAct, RAFA, ToT, LATS, etc., via the corresponding `taskbench_*_baseline.py` scripts.

### 2. SPIRAL Agent
NOTE: relative imports need to be fixed with a proper refactor. For now, this will work.
```bash
cd scripts/
python taskbench_spiral.py --run_name test --api_family dailylifeapis num_problems 10 --seed 50 model_namet mistral --debug_llm_output
```

Or run both benchmarks end-to-end:

```bash
./run_all_sota_experiments.sh
```

### 3. Ablation Studies

```bash
./run_all_ablation_experiments.sh
```

This will sweep over standard MCTS budgets and disable components (Planner, Simulator, Critic) to quantify their impact.

---

## Scripts & Agents

- **`scripts/taskbench_spiral.py`**  
  Implements the SPIRAL agent:
  - **Planner**: proposes actions via LLM prompts  
  - **Simulator**: predicts next observation  
  - **Critic**: scores plan progress  

- **Baseline scripts** (`taskbench_cot_baseline.py`, `taskbench_react_baseline.py`, etc.)  
  Wrap existing state-of-the-art methods for fair comparison.

- **`utils/generic_client.py`**  
  A drop-in replacement for internal APIs, providing a `HuggingFaceChatClient` to interface with any HF LLM.

---

## Analysis Notebooks

All result aggregation, tables, and figures are in `analysis/`:

- **`analysis_baseline_performance.ipynb`**  
- **`analysis_cost_benefit.ipynb`**  
- **`analysis_ablations.ipynb`**  
- **`analysis/sota_*`**  

Use these notebooks to reproduce the tables and plots in the paper and appendix.

---

## Configuration & Hyperparameters

Detailed hyperparameters are in Appendix B of the paper:

| Component               | Default Value            |
|-------------------------|--------------------------|
| MCTS Budget (K)         | 50 iterations            |
| Exploration Constant C  | 1.5                      |
| Planner Temperature     | 0.1                      |
| Simulator Temperature   | 0.0                      |
| Critic Temperature      | 0.0                      |
| Reward Weight α         | 0.5                      |
| Random Seeds            | [42, 101, 1234, 2024, 12345] |

See **Appendix B** (`SPIRAL_Technical_Appendix.pdf`) for full details.

---

## License

This project is released under the **MIT License**. See [LICENSE](LICENSE) for details.
