#!/bin/bash

# ─────────────────────────────────────────────────────────────────────────────
# Experiment Configuration
# ─────────────────────────────────────────────────────────────────────────────

# An array of the API families (subtasks) to run.
API_FAMILIES=(
    # "huggingface"
    "dailylifeapis"
)

# An array of the model checkpoints to evaluate.
MODELS_TO_RUN=(
    "llama_4"
    "llama_3_3_70b_instruct"
    # "deepseek_v2_5"
    # "qwen2_5_72b_instruct"
    # "phi"
)

# Seeds for reproducibility.
SEEDS=(42 101 1234 2024 12345)

# Baseline MCTS configurations (light, medium, heavy search budget).
BASELINE_CONFIGS=(
    "light"
    "medium"
    "heavy"
)

# Ablation modes for the main method.
ABLATION_MODES=(
    "no_mcts"
    "no_sim_feedback"
    "no_plan_history"
    "uniform_rewards"
    "no_validator"
)

# ─────────────────────────────────────────────────────────────────────────────
# Main Execution Logic
# ─────────────────────────────────────────────────────────────────────────────

echo "✅ Starting comprehensive Baseline and Ablation experiment batch."
start_time=$(date +%s)

# Iterate through each API family.
for api_family in "${API_FAMILIES[@]}"; do
    echo "=================================================================="
    echo "📦 Starting Subtask: $api_family"
    echo "=================================================================="

    # Dynamically set the number of problems for the dataset.
    if [ "$api_family" == "huggingface" ]; then
        NUM_PROBLEMS=500
    elif [ "$api_family" == "dailylifeapis" ]; then
        NUM_PROBLEMS=121
    elif [ "$api_family" == "multimedia" ]; then
        NUM_PROBLEMS=222
    else
        echo "⚠️ Warning: Unknown API family '$api_family'. Defaulting to 50 problems."
        NUM_PROBLEMS=50
    fi
    echo "  (Dataset size: $NUM_PROBLEMS problems)"

    # Loop through each model checkpoint.
    for model in "${MODELS_TO_RUN[@]}"; do
        echo "  🚀 Processing Model: $model"
        echo "  ----------------------------------------------------------------"

        # --- GROUP 1: BASELINE MCTS EXPERIMENTS ---
        echo "    📊 Processing Baseline MCTS Group"
        for config in "${BASELINE_CONFIGS[@]}"; do
            echo "      - Configuration: $config"
            for seed in "${SEEDS[@]}"; do
                RUN_NAME="baselines/${api_family}/${model}/${config}/run_seed_${seed}"
                echo "        - Starting Run (Seed: $seed)..."
                python run_taskbench_experiments.py \
                    --api_family "$api_family" \
                    --model_name "$model" \
                    --baseline_mcts_config "$config" \
                    --num_problems "$NUM_PROBLEMS" \
                    --seed "$seed" \
                    --run_name "$RUN_NAME"
                echo "        - Run complete."
            done
        done
        echo "    ✅ Finished Baseline MCTS group for model: $model"
        
        # --- GROUP 2: ABLATION EXPERIMENTS ---
        echo "    🔬 Processing Ablation Group"
        for ablation in "${ABLATION_MODES[@]}"; do
            echo "      - Ablation: $ablation"
            for seed in "${SEEDS[@]}"; do
                RUN_NAME="ablations/${api_family}/${model}/${ablation}/run_seed_${seed}"
                echo "        - Starting Run (Seed: $seed)..."
                python run_taskbench_experiments.py \
                    --api_family "$api_family" \
                    --model_name "$model" \
                    --ablation_mode "$ablation" \
                    --num_problems "$NUM_PROBLEMS" \
                    --seed "$seed" \
                    --run_name "$RUN_NAME"
                echo "        - Run complete."
            done
        done
        echo "    ✅ Finished Ablation group for model: $model"
        echo "  ----------------------------------------------------------------"
    done
done

end_time=$(date +%s)
duration=$((end_time - start_time))

echo "🎉 All baseline and ablation experiments completed in $(($duration / 3600))h $(($duration % 3600 / 60))m $(($duration % 60))s."