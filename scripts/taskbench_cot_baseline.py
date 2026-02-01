#!/usr/bin/env python3
# taskbench_cot_baseline.py

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager
import collections
import ast
import re
from datetime import datetime
import random
import numpy as np
import torch

from datasets import load_dataset
from tqdm import tqdm
from SPIRAL.scripts.utils.ritz_client import RitsChatClient, MODELMAP, MODEL_ID_MAP

# ─────────────────────────────────────────────────────────────────────────────
# NOTE: The following helper functions and constants are copied from the
# MCTS script to ensure a fair and consistent experimental setup.
# ─────────────────────────────────────────────────────────────────────────────

CORRECTED_TOOL_PARAMETERS = {
    "Token Classification": {"text": "string"}, "Translation": {"text": "string", "source_lang": "string", "target_lang": "string"}, "Summarization": {"text": "string"}, "Question Answering": {"context": "string", "question": "string"}, "Conversational": {"prompt": "string", "history": "list"}, "Text Generation": {"prompt": "string"}, "Sentence Similarity": {"sentence1": "string", "sentence2": "string"}, "Tabular Classification": {"table_image_path": "string"}, "Object Detection": {"image_path": "string"}, "Image Classification": {"image_path": "string"}, "Image-to-Image": {"image_path": "string", "target_image_path": "string"}, "Image-to-Text": {"image_path": "string"}, "Text-to-Image": {"prompt": "string"}, "Text-to-Video": {"prompt": "string"}, "Visual Question Answering": {"image_path": "string", "question": "string"}, "Document Question Answering": {"document_image_path": "string", "question": "string"}, "Image Segmentation": {"image_path": "string"}, "Depth Estimation": {"image_path": "string"}, "Text-to-Speech": {"text": "string"}, "Automatic Speech Recognition": {"audio_path": "string"}, "Audio-to-Audio": {"audio_path": "string"}, "Audio Classification": {"audio_path": "string"}, "Image Editing": {"image_path": "string", "edits": "dict"}, "get_weather": {"location": "string", "date": "string"}, "get_news_for_topic": {"topic": "string"}, "stock_operation": {"stock": "string", "operation": "string"}, "book_flight": {"date": "string", "from": "string", "to": "string"}, "book_hotel": {"date": "string", "name": "string"}, "book_restaurant": {"date": "string", "name": "string"}, "book_car": {"date": "string", "location": "string"}, "online_shopping": {"website": "string", "product": "string"}, "send_email": {"email_address": "string", "content": "string"}, "send_sms": {"phone_number": "string", "content": "string"}, "share_by_social_network": {"content": "string", "social_network": "string"}, "search_by_engine": {"query": "string", "engine": "string"}, "apply_for_job": {"job": "string"}, "see_doctor_online": {"disease": "string", "doctor": "string"}, "consult_lawyer_online": {"issue": "string", "lawyer": "string"}, "enroll_in_course": {"course": "string", "university": "string"}, "buy_insurance": {"insurance": "string", "company": "string"}, "online_banking": {"instruction": "string", "bank": "string"}, "daily_bill_payment": {"bill": "string"}, "sell_item_online": {"item": "string", "store": "string"}, "do_tax_return": {"year": "string"}, "apply_for_passport": {"country": "string"}, "pay_for_credit_card": {"credit_card": "string"}, "auto_housework_by_robot": {"instruction": "string"}, "auto_driving_to_destination": {"destination": "string"}, "deliver_package": {"package": "string", "destination": "string"}, "order_food_delivery": {"food": "string", "location": "string", "platform": "string"}, "order_taxi": {"location": "string", "platform": "string"}, "play_music_by_title": {"title": "string"}, "play_movie_by_title": {"title": "string"}, "take_note": {"content": "string"}, "borrow_book_online": {"book": "string", "library": "string"}, "recording_audio": {"content": "string"}, "make_video_call": {"phone_number": "string"}, "make_voice_call": {"phone_number": "string"}, "organize_meeting_online": {"topic": "string"}, "attend_meeting_online": {"topic": "string"}, "software_management": {"software": "string", "instruction": "string"}, "print_document": {"document": "string"}, "set_alarm": {"time": "string"},
}

def load_tool_descriptions_from_file(api_family_data_dir: Path) -> str:
    tool_desc_path = api_family_data_dir / "tool_desc.json"
    if not tool_desc_path.exists():
        raise FileNotFoundError(f"Tool description file not found: {tool_desc_path}.")
    try:
        with open(tool_desc_path, 'r', encoding='utf-8') as f: tool_data_root = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {tool_desc_path}: {e}") from e
    description_parts = ["Available tools (use the `api_call` function to invoke them):"]
    if not isinstance(tool_data_root, dict) or "nodes" not in tool_data_root:
        raise ValueError("Expected tool_desc.json to have a root dict with a 'nodes' key.")
    tool_nodes = tool_data_root["nodes"]
    for tool_node in tool_nodes:
        if not isinstance(tool_node, dict): continue
        tool_id, tool_desc = tool_node.get("id"), tool_node.get("desc")
        parameters = tool_node.get("parameters", [])
        if not tool_id or not tool_desc: continue
        args_list, example_args_dict = [], {}
        effective_parameters = [{"name": n, "type": t} for n, t in CORRECTED_TOOL_PARAMETERS.get(tool_id, {}).items()] or parameters
        for param in effective_parameters:
            param_name, param_type = param.get("name"), param.get("type", "Any")
            if param_name:
                args_list.append(f"`{param_name}` ({param_type})")
                example_args_dict[param_name] = f"<{param_name}_value>"
        example_call_str = f"api_call(\"{tool_id}\", {json.dumps(example_args_dict)})"
        description_parts.append(f"\n`{example_call_str}`")
        description_parts.append(f"  Description: {tool_desc}")
        if args_list: description_parts.append(f"  Parameters: {'; '.join(args_list)}")
    return "\n".join(description_parts)

def load_graph_descriptions_from_file(api_family_data_dir: Path) -> str:
    graph_desc_path = api_family_data_dir / "graph_desc.json"
    if not graph_desc_path.exists(): return ""
    try:
        with open(graph_desc_path, 'r', encoding='utf-8') as f: graph_data = json.load(f)
    except (json.JSONDecodeError, Exception) as e:
        print(f"Warning: Could not read {graph_desc_path}: {e}", file=sys.stderr); return ""
    description_parts = ["\n--- Tool Dependencies ---"]
    for dep_type, deps in graph_data.items():
        if isinstance(deps, list) and deps:
            description_parts.append(f"{dep_type.replace('_', ' ').title()}:")
            for dep in deps:
                if not isinstance(dep, dict): continue
                pre, post = dep.get("pre_tool"), dep.get("post_tool")
                if "resource" in dep_type:
                    res = ", ".join(dep.get("resources", [])); description_parts.append(f"  - `{post}` requires resource(s) `{res}` from `{pre}`.")
                elif "temporal" in dep_type:
                    cond = dep.get("condition", "completion"); description_parts.append(f"  - `{post}` can only be called after `{pre}` upon its {cond}.")
    return "\n".join(description_parts) if len(description_parts) > 1 else ""

# ─────────────────────────────────────────────────────────────────────────────
# NEW: Core CoT Logic
# ─────────────────────────────────────────────────────────────────────────────

def parse_plan_from_response(text: str) -> List[str]:
    """Extracts a Python list of strings from a markdown code block."""
    match = re.search(r"```(?:python)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if not match:
        return []
    try:
        plan = ast.literal_eval(match.group(1).strip())
        if isinstance(plan, list) and all(isinstance(item, str) for item in plan):
            return plan
    except (ValueError, SyntaxError):
        return []
    return []

def make_plan_hashable(plan: List[str]) -> Tuple[str, ...]:
    """Converts a list of strings to a hashable tuple for voting."""
    return tuple(plan)

def process_problem_with_cot(problem_info: Dict) -> Optional[Dict]:
    """
    Generates and evaluates a plan for a given problem using a Chain-of-Thought
    approach with optional self-consistency.
    """
    idx, example, api_family, log_path, log_lock, consistency_level, temperature = (
        problem_info['dataset_index'], problem_info['example'], problem_info['api_family_for_tools'],
        problem_info['log_path'], problem_info['log_lock'],
        problem_info['consistency_level'], problem_info['temperature']
    )

    def write_log(message: str):
        with log_lock:
            with log_path.open("a", encoding="utf-8") as f:
                f.write(f"--- Problem {idx} ({example['id']}) ---\n{message}\n" + "="*80 + "\n\n")

    user_request_text = example['instruction']
    
    # Use a RitsChatClient with the specified temperature for diverse sampling
    client = RitsChatClient(temperature=temperature, max_tokens=2048)

    # Load tool descriptions for the prompt
    try:
        tools_description = load_tool_descriptions_from_file(Path("Taskbench") / f"data_{api_family}")
        graph_description = load_graph_descriptions_from_file(Path("Taskbench") / f"data_{api_family}")
    except (FileNotFoundError, ValueError) as e:
        write_log(f"CRITICAL ERROR: Could not load descriptions. Error: {e}"); return None

    # Construct the CoT prompt
    prompt_template = """You are an expert planner. Your task is to create a complete step-by-step plan to solve the user's request using the available tools.

### RULES
1.  **Think Step-by-Step**: First, write your reasoning within the 'Thought' section. Analyze the request, break it down, and formulate a high-level plan.
2.  **Generate Final Plan**: After your reasoning, provide the final, complete plan as a Python list of strings inside a python markdown block.
3.  **Tool Calls**: Each string in the list must be a valid `api_call(...)` for one of the available tools.
4.  **Finish Call**: The last step in your plan MUST be `finish(reason=\"<brief summary of the result>\")`.

### AVAILABLE TOOLS
{tools_description}
{graph_description}

### USER REQUEST
{user_request}

### YOUR RESPONSE

#### Thought
(Your step-by-step reasoning and logic goes here. Break down the problem and map it to the available tools.)

#### Plan
```python
[
    "api_call(\"tool_name_1\", {{\"param1\": \"value1\"}})",
    "api_call(\"tool_name_2\", {{\"param1\": \"value2\"}})",
    "finish(reason=\"The plan is complete.\")"
]
```"""
    prompt = prompt_template.format(
        tools_description=tools_description,
        graph_description=graph_description,
        user_request=user_request_text
    )

    start_time = time.time()
    generated_plans = []
    total_llm_tokens = 0
    
    # Generate 'k' plans for self-consistency
    for _ in range(consistency_level):
        response, tokens_used = client.send(prompt)
        total_llm_tokens += tokens_used
        plan = parse_plan_from_response(response)
        if plan:
            generated_plans.append(plan)

    if not generated_plans:
        write_log("Failed to generate any valid plans from the LLM.")
        return None

    # Self-consistency: vote for the most frequent plan
    plan_counts = collections.Counter(make_plan_hashable(p) for p in generated_plans)
    best_plan_tuple = plan_counts.most_common(1)[0][0]
    final_plan = list(best_plan_tuple)

    generation_time_seconds = time.time() - start_time

    # Evaluate the final plan using the same LLM-based evaluator
    final_reward_score = 0.0
    EVALUATION_PROMPT = """Did the 'Generated Plan' successfully solve the 'User Request'? Answer with only "Yes" or "No".\n[User Request]:\n{user_request}\n\n[Generated Plan]:\n{generated_plan}\n\n[Answer (Yes/No)]:"""
    try:
        eval_client = RitsChatClient(temperature=0.0, max_tokens=10)
        eval_prompt = EVALUATION_PROMPT.format(user_request=user_request_text, generated_plan="\n".join(final_plan))
        verdict, _ = eval_client.send(eval_prompt)
        if verdict.strip().lower().startswith("yes"): final_reward_score = 1.0
    except Exception as e:
        write_log(f"Warning: LLM-based evaluation failed. Error: {e}")

    # Structure the final output with comparable metrics
    final_output = {
        "id": example['id'],
        "result": {
            "task_steps": final_plan
        },
        "metrics": {
            "accuracy": final_reward_score,
            "generation_time_seconds": round(generation_time_seconds, 2),
            "plan_length": sum(1 for step in final_plan if step.startswith("api_call")),
            "reasoning_cost": {
                "llm_calls": consistency_level,
                "total_llm_tokens": total_llm_tokens,
            }
        }
    }
    return {"record": final_output}

# --- Data loading helper ---
def load_hf(config_name: str):
    try:
        ds = load_dataset('microsoft/Taskbench', name=config_name, split='test')
        for ex in ds:
            yield {'id': ex['id'], 'instruction': ex['instruction'], 'input': ex.get('input',''), 'tool_steps': ex.get('tool_steps',[])}
    except Exception as e:
        print(f"\n❌ Failed to load '{config_name}' from Hugging Face.", file=sys.stderr)
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# Main Orchestrator
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Run CoT Baseline on TaskBench.")
    
    # --- Experiment Configuration ---
    ap.add_argument('--run_name', type=str, default=None, help="Optional name for the output directory.")
    ap.add_argument('--api_family', type=str, default='huggingface', help="API family to test.")
    ap.add_argument('--num_problems', type=int, default=50, help="Number of problems to sample.")
    ap.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")
    ap.add_argument('--model_name', type=str, default='llama_4', help="The model checkpoint to use.")

    # --- CoT Hyperparameters ---
    ap.add_argument('--consistency_level', type=int, default=1, choices=[1, 3, 5], help="Number of plans for self-consistency (k). 1=Standard CoT.")
    ap.add_argument('--temperature', type=float, default=0.7, help="Temperature for LLM sampling. Should be >0 for self-consistency.")

    # --- Execution Settings ---
    ap.add_argument('--max_workers', type=int, default=os.cpu_count(), help="Maximum parallel processes.")
    args = ap.parse_args()

    # Validate model name
    valid_rits_models = list(MODEL_ID_MAP["rits"].keys())
    if args.model_name not in valid_rits_models:
        print(f"❌ Error: Invalid model name '{args.model_name}'. Choose from: {valid_rits_models}", file=sys.stderr)
        sys.exit(1)
        
    MODELMAP.set_model('generate_model', args.model_name)
    print(f"✅ Configured to use model: {MODELMAP.generate_model}")

    # Set seeds
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)

    # Setup run directory
    run_name = args.run_name
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"cot_k{args.consistency_level}_{args.api_family}_{args.model_name}_{timestamp}"
        print(f"✅ No run name provided. Using auto-generated name: {run_name}")

    run_dir = Path('predictions') / run_name; run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / 'debug_log.txt'
    if log_path.exists(): log_path.unlink()
    print(f"✅ Outputs will be saved in: {run_dir}")

    # Load and sample data
    all_records = list(load_hf(config_name=args.api_family))
    random.shuffle(all_records)
    records_to_process = all_records[:args.num_problems]
    print(f"✅ Loaded and sampled {len(records_to_process)} problems.")

    # Process problems in parallel
    with Manager() as manager:
        log_lock = manager.Lock()
        
        problems_to_submit = [{
            "dataset_index": j, "example": ex, "api_family_for_tools": args.api_family,
            "log_path": log_path, "log_lock": log_lock,
            "consistency_level": args.consistency_level, "temperature": args.temperature
        } for j, ex in enumerate(records_to_process)]
        
        run_results = []
        desc = f"CoT (k={args.consistency_level}) on {args.api_family}"
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {executor.submit(process_problem_with_cot, prob): prob['dataset_index'] for prob in problems_to_submit}
            for future in tqdm(as_completed(futures), total=len(records_to_process), desc=desc):
                try:
                    result = future.result()
                    if result: run_results.append(result['record'])
                except Exception as e:
                    print(f"Problem {futures[future]} failed: {e}", file=sys.stderr)

        # Save results
        run_output_path = run_dir / 'results.json'
        with run_output_path.open("w", encoding="utf-8") as f: json.dump(run_results, f, indent=2)
        
        total_correct = sum(1 for r in run_results if r.get('metrics', {}).get('accuracy', 0.0) > 0.9)
        total_problems = len(run_results)
        accuracy = (total_correct / total_problems) * 100 if total_problems > 0 else 0
        print(f"📈 Accuracy for this run: {accuracy:.2f}% | Results saved to {run_output_path}")

    # Save summary
    summary = {
        "run_name": run_name, "model_name": args.model_name, "api_family": args.api_family,
        "num_problems_processed": len(records_to_process), "seed": args.seed,
        "consistency_level": args.consistency_level, "temperature": args.temperature,
        "final_accuracy": f"{accuracy:.2f}%"
    }
    summary_path = run_dir / 'summary.json'
    with summary_path.open("w", encoding="utf-8") as f: json.dump(summary, f, indent=2)

    print(f"📊 Final Accuracy: {accuracy:.2f}%")
    print(f"✅ Final summary saved to {summary_path}")

if __name__ == '__main__':
    main()