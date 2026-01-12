# SPIRAL: Symbolic LLM Planning via Grounded and Reflective Search

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2512.23167)
[![Conference](https://img.shields.io/badge/AAAI'26-Main%20Track-blue)](https://aaai.org/conference/aaai/aaai-26/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

*Accepted to AAAI 2026 Main Technical Track*

This repository contains the code and analysis notebooks for **SPIRAL**, our framework that embeds a tri-agent cognitive architecture into an MCTS loop to enable more robust, grounded, and reflective planning with large language models.

📄 **Paper**: [SPIRAL: Symbolic LLM Planning via Grounded and Reflective Search](https://arxiv.org/abs/2512.23167)  
📑 **Technical Appendix**: Included in the [arXiv paper](https://arxiv.org/abs/2512.23167)

> **Note**: The experiments in the paper were conducted using IBM's internal infrastructure (WatsonX/RITS). For public use, we provide a Hugging Face-based implementation in `utils/generic_client.py` that allows you to run SPIRAL with open-source models. Results may vary from those reported in the paper due to differences in model versions and inference infrastructure.

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

### Prerequisites

- Python 3.10 or 3.11
- CUDA-compatible GPU (recommended for faster inference)
- [Conda](https://docs.conda.io/en/latest/miniconda.html) or [virtualenv](https://virtualenv.pypa.io/)

### Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/IBM/SPIRAL.git
   cd SPIRAL
   ```

2. **Create and activate the environment**  
   
   Using Conda (recommended):
   ```bash
   conda create -n spiral python=3.11
   conda activate spiral
   pip install -r requirements.txt
   ```
   
   Or using virtualenv:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Download TaskBench datasets**  
   ```bash
   cd scripts/
   huggingface-cli download microsoft/Taskbench --local-dir Taskbench --repo-type dataset
   ```
   
   This will place the `dailylifeapis` and `huggingface` benchmark data under `scripts/Taskbench/`.

4. **Set up environment variables** (optional, for specific model APIs)
   ```bash
   cp .env.example .env
   # Edit .env with your API keys if using external model providers
   ```
---

## Dependencies

All required packages are listed in `requirements.txt`. Key dependencies include:

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥2.0.0 | Deep learning framework |
| `transformers` | ≥4.35.0 | Hugging Face model implementations |
| `datasets` | ≥2.14.0 | Dataset loading and processing |
| `langchain` | ≥0.1.0 | LLM orchestration framework |
| `litellm` | ≥1.0.0 | Unified LLM API interface |
| `numpy` | ≥1.24.0 | Numerical operations |
| `tqdm` | ≥4.65.0 | Progress bars |

For analysis notebooks:
- `pandas`, `matplotlib`, `seaborn`, `jupyter`

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

### Quick Start: Run SPIRAL

```bash
cd scripts/
python taskbench_spiral_method_final.py \
    --run_name my_experiment \
    --api_family dailylifeapis \
    --num_problems 10 \
    --seed 42 \
    --model_name mistral \
    --debug_llm_output
```

### 1. Baseline Methods
NOTE: these won't work, they require a different library structure

```bash
./run_all_baseline_experiments.sh
```

This will run Chain-of-Thought (k=1,3,5), ReAct, RAFA, ToT, LATS, etc., via the corresponding `taskbench_*_baseline.py` scripts.

### 2. SPIRAL Agent
```bash
cd scripts/
python taskbench_spiral_method_final.py \
    --run_name test \
    --api_family dailylifeapis \
    --num_problems 10 \
    --seed 50 \
    --model_name mistral \
    --debug_llm_output
```

Available arguments:
- `--run_name`: Name for the experiment run
- `--api_family`: Dataset to use (`dailylifeapis` or `huggingface`)
- `--num_problems`: Number of problems to evaluate
- `--seed`: Random seed for reproducibility
- `--model_name`: Model to use (e.g., `mistral`, `llama_3`, `phi`)
- `--debug_llm_output`: Enable verbose LLM output logging

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
  **For public use**: A Hugging Face-based implementation providing a `HuggingFaceChatClient` to interface with open-source LLMs. Use this for running experiments without IBM infrastructure.

- **`utils/ritz_client.py`**  
  IBM internal client for RITS/WatsonX endpoints (used in paper experiments).

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

| Component               | Default Value            | CLI Argument |
|-------------------------|--------------------------|--------------|
| MCTS Budget (K)         | 50 iterations            | `--mcts_iterations` |
| Max Tree Depth          | 8                        | `--max_depth` |
| Exploration Constant C  | 1.0 (UCT)                | — |
| Planner Temperature     | 0.0                      | — |
| Simulator Temperature   | 0.2                      | — |
| Random Seeds            | 42 (default)             | `--seed` |
| Max Workers             | CPU count                | `--max_workers` |

### Available Models

For public use, models are accessed via Hugging Face Transformers. See `utils/generic_client.py` for the implementation.

| Model Name | Hugging Face Model ID |
|------------|----------------------|
| `llama_3` | `meta-llama/Meta-Llama-3-70B-Instruct` |
| `mistral` | `mistralai/Mistral-7B-Instruct-v0.3` |
| `phi` | `microsoft/Phi-3-mini-4k-instruct` |
| `deepseek_v2_5` | `deepseek-ai/DeepSeek-V2-Lite` |
| `qwen2_5_72b_instruct` | `Qwen/Qwen2-72B-Instruct` |

> **Paper experiments**: Results reported in the paper used IBM's internal RITS/WatsonX infrastructure with models including Llama 4 Maverick 17B, Mistral Large, and other proprietary endpoints.

See **Appendix B** in the [arXiv paper](https://arxiv.org/abs/2512.23167) for full details.

---

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{zhang2025spiral,
  title={SPIRAL: Symbolic LLM Planning via Grounded and Reflective Search},
  author={Zhang, Yifan and Ganapavarapu, Giridhar and Jayaraman, Srideepika and Agrawal, Bhavna and Patel, Dhaval and Fokoue, Achille},
  journal={arXiv preprint arXiv:2512.23167},
  year={2025}
}
```

---

## License

This project is released under the **MIT License**. See [LICENSE](LICENSE) for details.
