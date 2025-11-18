#!/bin/bash

# ─────────────────────────────────────────────────────────────────────────────
# Experiment Configuration
# ─────────────────────────────────────────────────────────────────────────────

# An array of the API families (subtasks) to run.
API_FAMILIES=(
    #
    "dailylifeapis"
    "huggingface" 
    # "multimedia"
)

# An array of the model checkpoints to evaluate.
MODELS_TO_RUN=(
    # "llama_4"
    # "llama_3"
    # "llama_4_scout_17b_16e_instruct"
    # "deepseek_v3_h200"
    # "qwen3_8b" # Not working
    #
    "llama_3_3_70b_instruct"
    "qwen2_5_72b_instruct"
    "phi"
    "deepseek_v2_5"
)

# An array of specific, common seeds for reproducibility.
SEEDS=(42 101 1234 2024 12345)


# ─────────────────────────────────────────────────────────────────────────────
# Main Execution Logic
# ─────────────────────────────────────────────────────────────────────────────

echo "✅ Starting comprehensive experiment batch for all API families."
start_time=$(date +%s)

# NEW: Outermost loop to iterate through each API family.
for api_family in "${API_FAMILIES[@]}"; do
    echo "=================================================================="
    echo "📦 Starting Subtask: $api_family"
    echo "=================================================================="

    # Dynamically set the number of problems for the full dataset of each subtask.
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
    echo "  (Full dataset size: $NUM_PROBLEMS problems)"

    # Define the main output directory for this subtask.
    MAIN_OUTPUT_DIR="predictions/${api_family}_experiments"
    mkdir -p "$MAIN_OUTPUT_DIR"

    # Loop through each model checkpoint.
    for model in "${MODELS_TO_RUN[@]}"; do
        echo "  🚀 Processing Model: $model"
        MODEL_DIR="${MAIN_OUTPUT_DIR}/${model}"
        mkdir -p "$MODEL_DIR"

        # Loop through the predefined array of seeds.
        for seed in "${SEEDS[@]}"; do
            RUN_NAME="${api_family}_experiments/${model}/run_seed_${seed}"

            echo "    - Starting Run (Seed: $seed)..."

            # Execute the Python script with all the specified arguments.
            python taskbench_smriv_mcts_revised_final.py \
                --api_family "$api_family" \
                --model_name "$model" \
                --num_problems "$NUM_PROBLEMS" \
                --seed "$seed" \
                --run_name "$RUN_NAME" \
	        --max_workers 16 \
		--debug_llm_output
	    
            echo "    - Run with seed $seed complete."
        done
        echo "  ✅ Finished all runs for model: $model"
        echo "  ----------------------------------------------------------------"
    done
done

end_time=$(date +%s)
duration=$((end_time - start_time))

echo "🎉 All experiments for all subtasks completed in $(($duration / 3600))h $(($duration % 3600 / 60))m $(($duration % 60))s."
