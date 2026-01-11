#!/bin/bash
set -e
set -o pipefail

# ==============================================================================
# MASTER SCRIPT FOR RESIDUAL LEARNING EXPERIMENTS (USING PRE-COMPUTED CoT)
# ==============================================================================
# This script uses existing CoT (k=1) results to create a residual dataset,
# then runs ToT, LATS, ReAct, RAFA, the ReAct+RAFA Hybrid, and SPIRAL on the 
# problems that CoT failed.
# ==============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# Global Experiment Configuration
# ─────────────────────────────────────────────────────────────────────────────

# An array of the API families (subtasks) to run.
API_FAMILIES=(
    "huggingface"
    "dailylifeapis"
)

# An array of the model checkpoints to evaluate.
MODELS_TO_RUN=(
    "llama_4"
    "llama_3_3_70b_instruct"
    "deepseek_v2_5"
    "qwen2_5_72b_instruct"
    "phi"
)

# An array of specific, common seeds for reproducibility.
SEEDS=(42 101 1234 2024 12345)

# Path to the pre-computed CoT k=1 results
COT_BACKUP_DIR="predictions/predictions_cot_k1_backup"

# ─────────────────────────────────────────────────────────────────────────────
# Method-Specific Hyperparameters
# ─────────────────────────────────────────────────────────────────────────────
# Note: CoT params are now only used to find the correct results path.
COT_K=1
COT_TEMPERATURE=0.7

# 2. ToT (Runs on CoT's failures)
TOT_MAX_STEPS=4
TOT_SEARCH_BREADTH=3
TOT_CANDIDATES_PER_STATE=2

# 3. LATS (Runs on CoT's failures)
LATS_MCTS_ITERATIONS=25
LATS_EXPLORATION_WEIGHT=1.0
LATS_CANDIDATES_PER_STATE=2

# 4. ReAct (Runs on CoT's failures)
REACT_MAX_STEPS=8

# 5. RAFA (Runs on CoT's failures)
RAFA_MAX_REAL_STEPS=4
RAFA_SEARCH_BREADTH=3
RAFA_SEARCH_DEPTH=2

# 6. ReAct+RAFA Hybrid (Runs on CoT's failures)
HYBRID_MAX_REAL_STEPS=4
HYBRID_SEARCH_BREADTH=3
HYBRID_SEARCH_DEPTH=2

# 7. SPIRAL (Your Method, runs on CoT's failures)
SPIRAL_MCTS_ITERATIONS=50
SPIRAL_MAX_DEPTH=8

# ==============================================================================
# Main Execution Logic
# ==============================================================================

echo "✅ Starting comprehensive experiment batch using pre-computed CoT results from '${COT_BACKUP_DIR}'."
main_start_time=$(date +%s)

for api_family in "${API_FAMILIES[@]}"; do
    echo "##################################################################"
    echo "📦 API Family: $api_family"

    for model in "${MODELS_TO_RUN[@]}"; do
        echo "=================================================================="
        echo "🚀 Model: $model"
        
        for seed in "${SEEDS[@]}"; do
            echo "  ----------------------------------------------------------------"
            echo "  🌱 Starting run for Seed: $seed"
            
            # --- STEP 1: Locate Pre-computed CoT results and Create Residual Dataset ---
            cot_extra_args="--consistency_level ${COT_K} --temperature ${COT_TEMPERATURE}"
            cot_run_name_suffix=$(echo "$cot_extra_args" | tr -d '[:space:]' | tr -c '[:alnum:]' '_')
            cot_results_path="${COT_BACKUP_DIR}/${api_family}_experiments/${model}/run${cot_run_name_suffix}_seed_${seed}/results.json"
            
            residual_api_family="${api_family}_residual"
            residual_data_dir="Taskbench/data_${residual_api_family}"
            residual_dataset_file="${residual_data_dir}/user_requests.jsonl"

            echo "    [1/7] Creating residual dataset from CoT failures..."
            if [ ! -f "$cot_results_path" ]; then
                echo "    ⚠️ Pre-computed CoT results not found at '$cot_results_path'. Skipping this run."
                continue
            fi
            
            python create_residual_dataset.py --api_family "$api_family" --results_path "$cot_results_path"

            if [ ! -f "$residual_dataset_file" ] || [ ! -s "$residual_dataset_file" ]; then
                echo "    ✅ CoT solved all problems in the pre-computed run. No residual experiments needed."
                rm -rf "$residual_data_dir"
                continue
            fi
            
            residual_problems=$(wc -l < "$residual_dataset_file")
            echo "    -> CoT failed on ${residual_problems} problems. Continuing with advanced methods."

            # # --- STEP 2: Run ToT on the RESIDUAL dataset ---
            # tot_run_name="predictions_tot_residual/${model}/${api_family}_seed${seed}"
            # echo "    [2/7] Running Tree of Thoughts (ToT) on residual dataset..."
            # python taskbench_tot_baseline.py \
            #     --api_family "$residual_api_family" --model_name "$model" --num_problems "$residual_problems" --seed "$seed" \
            #     --run_name "$tot_run_name" --max_steps ${TOT_MAX_STEPS} \
            #     --search_breadth ${TOT_SEARCH_BREADTH} --candidates_per_state ${TOT_CANDIDATES_PER_STATE}

            # # --- STEP 3: Run LATS on the RESIDUAL dataset ---
            # lats_run_name="predictions_lats_residual/${model}/${api_family}_seed${seed}"
            # echo "    [3/7] Running Language Agent Tree Search (LATS) on residual dataset..."
            # python taskbench_lats_baseline.py \
            #     --api_family "$residual_api_family" --model_name "$model" --num_problems "$residual_problems" --seed "$seed" \
            #     --run_name "$lats_run_name" --mcts_iterations ${LATS_MCTS_ITERATIONS} \
            #     --exploration_weight ${LATS_EXPLORATION_WEIGHT} --candidates_per_state ${LATS_CANDIDATES_PER_STATE}
            
            # # --- STEP 4: Run ReAct on the RESIDUAL dataset ---
            # react_run_name="predictions_react_residual/${model}/${api_family}_seed${seed}"
            # echo "    [4/7] Running ReAct on residual dataset..."
            # python taskbench_react_baseline.py \
            #     --api_family "$residual_api_family" --model_name "$model" --num_problems "$residual_problems" --seed "$seed" \
            #     --run_name "$react_run_name" --max_steps ${REACT_MAX_STEPS}

            # # --- STEP 5: Run RAFA on the RESIDUAL dataset ---
            # rafa_run_name="predictions_rafa_residual/${model}/${api_family}_seed${seed}"
            # echo "    [5/7] Running RAFA on residual dataset..."
            # python taskbench_rafa_baseline.py \
            #     --api_family "$residual_api_family" --model_name "$model" --num_problems "$residual_problems" --seed "$seed" \
            #     --run_name "$rafa_run_name" --max_real_steps ${RAFA_MAX_REAL_STEPS} \
            #     --search_breadth ${RAFA_SEARCH_BREADTH} --search_depth ${RAFA_SEARCH_DEPTH}

            # # --- STEP 6: Run ReAct+RAFA Hybrid on the RESIDUAL dataset ---
            # hybrid_run_name="predictions_react_rafa_residual/${model}/${api_family}_seed${seed}"
            # echo "    [6/7] Running ReAct+RAFA Hybrid on residual dataset..."
            # python taskbench_react_rafa_baseline.py \
            #     --api_family "$residual_api_family" --model_name "$model" --num_problems "$residual_problems" --seed "$seed" \
            #     --run_name "$hybrid_run_name" --max_real_steps ${HYBRID_MAX_REAL_STEPS} \
            #     --search_breadth ${HYBRID_SEARCH_BREADTH} --search_depth ${HYBRID_SEARCH_DEPTH}

            # --- STEP 7: Run SPIRAL (Your Method) on the RESIDUAL dataset ---
            spiral_run_name="predictions_spiral_residual/${model}/${api_family}_seed${seed}"
            echo "    [7/7] Running SPIRAL on residual dataset..."
            python taskbench_spiral_method_final.py \
                --api_family "$residual_api_family" --model_name "$model" --num_problems "$residual_problems" --seed "$seed" \
                --run_name "$spiral_run_name" --mcts_iterations ${SPIRAL_MCTS_ITERATIONS} --max_depth ${SPIRAL_MAX_DEPTH}
            
            # --- Clean up this run's residual data ---
            rm -rf "$residual_data_dir"
            echo "    -> Cleaned up residual data for seed $seed."
            echo "  ----------------------------------------------------------------"
        done # seed loop
    done # model loop
done # api_family loop

# --- Final Summary ---
main_end_time=$(date +%s)
duration=$((main_end_time - main_start_time))

echo ""
echo "##################################################################"
echo "🎉 ALL RESIDUAL EXPERIMENTS COMPLETED!"
echo "Total execution time: $(($duration / 3600))h $(($duration % 3600 / 60))m $(($duration % 60))s."
echo "Check the 'predictions_*' directories for results."
echo "##################################################################"