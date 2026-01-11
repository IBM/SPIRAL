#!/usr/bin/env python3
# taskbench_tot_baseline.py

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager
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
# other scripts to ensure a fair and consistent experimental setup.
# ─────────────────────────────────────────────────────────────────────────────

CORRECTED_TOOL_PARAMETERS = {
    "Token Classification": {"text": "string"}, "Translation": {"text": "string", "source_lang": "string", "target_lang": "string"}, "Summarization": {"text": "string"}, "Question Answering": {"context": "string", "question": "string"}, "Conversational": {"prompt": "string", "history": "list"}, "Text Generation": {"prompt": "string"}, "Sentence Similarity": {"sentence1": "string", "sentence2": "string"}, "Tabular Classification": {"table_image_path": "string"}, "Object Detection": {"image_path": "string"}, "Image Classification": {"image_path": "string"}, "Image-to-Image": {"image_path": "string", "target_image_path": "string"}, "Image-to-Text": {"image_path": "string"}, "Text-to-Image": {"prompt": "string"}, "Text-to-Video": {"prompt": "string"}, "Visual Question Answering": {"image_path": "string", "question": "string"}, "Document Question Answering": {"document_image_path": "string", "question": "string"}, "Image Segmentation": {"image_path": "string"}, "Depth Estimation": {"image_path": "string"}, "Text-to-Speech": {"text": "string"}, "Automatic Speech Recognition": {"audio_path": "string"}, "Audio-to-Audio": {"audio_path": "string"}, "Audio Classification": {"audio_path": "string"}, "Image Editing": {"image_path": "string", "edits": "dict"}, "get_weather": {"location": "string", "date": "string"}, "get_news_for_topic": {"topic": "string"}, "stock_operation": {"stock": "string", "operation": "string"}, "book_flight": {"date": "string", "from": "string", "to": "string"}, "book_hotel": {"date": "string", "name": "string"}, "book_restaurant": {"date": "string", "name": "string"}, "book_car": {"date": "string", "location": "string"}, "online_shopping": {"website": "string", "product": "string"}, "send_email": {"email_address": "string", "content": "string"}, "send_sms": {"phone_number": "string", "content": "string"}, "share_by_social_network": {"content": "string", "social_network": "string"}, "search_by_engine": {"query": "string", "engine": "string"}, "apply_for_job": {"job": "string"}, "see_doctor_online": {"disease": "string", "doctor": "string"}, "consult_lawyer_online": {"issue": "string", "lawyer": "string"}, "enroll_in_course": {"course": "string", "university": "string"}, "buy_insurance": {"insurance": "string", "company": "string"}, "online_banking": {"instruction": "string", "bank": "string"}, "daily_bill_payment": {"bill": "string"}, "sell_item_online": {"item": "string", "store": "string"}, "do_tax_return": {"year": "string"}, "apply_for_passport": {"country": "string"}, "pay_for_credit_card": {"credit_card": "string"}, "auto_housework_by_robot": {"instruction": "string"}, "auto_driving_to_destination": {"destination": "string"}, "deliver_package": {"package": "string", "destination": "string"}, "order_food_delivery": {"food": "string", "location": "string", "platform": "string"}, "order_taxi": {"location": "string", "platform": "string"}, "play_music_by_title": {"title": "string"}, "play_movie_by_title": {"title": "string"}, "take_note": {"content": "string"}, "borrow_book_online": {"book": "string", "library": "string"}, "recording_audio": {"content": "string"}, "make_video_call": {"phone_number": "string"}, "make_voice_call": {"phone_number": "string"}, "organize_meeting_online": {"topic": "string"}, "attend_meeting_online": {"topic": "string"}, "software_management": {"software": "string", "instruction": "string"}, "print_document": {"document": "string"}, "set_alarm": {"time": "string"},
}

def load_tool_descriptions_from_file(api_family_data_dir: Path) -> str:
    tool_desc_path = api_family_data_dir / "tool_desc.json"
    if not tool_desc_path.exists():
        raise FileNotFoundError(f"Tool description file not found: {tool_desc_path}.")
    with open(tool_desc_path, 'r', encoding='utf-8') as f:
        tool_data_root = json.load(f)
    description_parts = ["Available tools (use the `api_call` function to invoke them):"]
    tool_nodes = tool_data_root.get("nodes", [])
    for tool_node in tool_nodes:
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
        description_parts.append(f"\n`{example_call_str}`\n  Description: {tool_desc}")
        if args_list: description_parts.append(f"  Parameters: {'; '.join(args_list)}")
    return "\n".join(description_parts)

def load_graph_descriptions_from_file(api_family_data_dir: Path) -> str:
    graph_desc_path = api_family_data_dir / "graph_desc.json"
    if not graph_desc_path.exists(): return ""
    with open(graph_desc_path, 'r', encoding='utf-8') as f:
        graph_data = json.load(f)
    description_parts = ["\n--- Tool Dependencies ---"]
    for dep_type, deps in graph_data.items():
        if isinstance(deps, list) and deps:
            description_parts.append(f"{dep_type.replace('_', ' ').title()}:")
            for dep in deps:
                pre, post = dep.get("pre_tool"), dep.get("post_tool")
                if "resource" in dep_type:
                    res = ", ".join(dep.get("resources", [])); description_parts.append(f"  - `{post}` requires resource(s) `{res}` from `{pre}`.")
                elif "temporal" in dep_type:
                    cond = dep.get("condition", "completion"); description_parts.append(f"  - `{post}` can only be called after `{pre}` upon its {cond}.")
    return "\n".join(description_parts) if len(description_parts) > 1 else ""

# ─────────────────────────────────────────────────────────────────────────────
# Core ToT Logic
# ─────────────────────────────────────────────────────────────────────────────

class ToT_Proposer:
    """Generates candidate thoughts for the ToT search."""
    def __init__(self, user_request: str, tools_description: str, graph_description: str, num_candidates: int):
        self.client = RitsChatClient(temperature=0.5, max_tokens=1024)
        self.user_request = user_request
        self.tools_desc = tools_description
        self.graph_desc = graph_description
        self.num_candidates = num_candidates
        self.prompt_template = """As an expert planner, your goal is to generate diverse and promising next steps to solve a user's request.

### AVAILABLE TOOLS
{tools_description}
{graph_description}

### TASK
User Request: {user_request}

### CURRENT PLAN
{plan_history}

### INSTRUCTION
Based on the current plan, generate a Python list of {num_candidates} distinct `api_call(...)` or `finish(...)` actions to take next.
Focus on variety and relevance. Your response MUST be ONLY a Python list of strings in a markdown block.
```python
[
    "action_string_1",
    "action_string_2",
    ...
]
```"""

    def propose(self, plan_history_str: str) -> Tuple[List[str], int]:
        prompt = self.prompt_template.format(
            tools_description=self.tools_desc,
            graph_description=self.graph_desc,
            user_request=self.user_request,
            plan_history=plan_history_str,
            num_candidates=self.num_candidates
        )
        response, tokens = self.client.send(prompt)
        match = re.search(r"```(?:python)?\s*(\[.*?\])\s*```", response, re.DOTALL)
        if match:
            try:
                # Use literal_eval for safe evaluation of the list string
                proposals = ast.literal_eval(match.group(1).strip())
                if isinstance(proposals, list) and all(isinstance(p, str) for p in proposals):
                    return proposals, tokens
            except (ValueError, SyntaxError):
                # Fallback if parsing fails
                return [], tokens
        return [], tokens

class ToT_Evaluator:
    """Evaluates the quality of a partial plan (a state in the ToT)."""
    def __init__(self, user_request: str):
        self.client = RitsChatClient(temperature=0.0, max_tokens=150)
        self.user_request = user_request
        self.prompt_template = """As a meticulous evaluator, assess the following partial plan for its potential to solve the user's request.

### TASK
User Request: {user_request}

### PARTIAL PLAN
{plan_to_evaluate}

### INSTRUCTION
Evaluate the plan's progress and likelihood of success. Is it on a good path? Is it coherent and logical?
Respond with ONLY a single line in the format: `Score: <float_from_0.0_to_1.0> | Justification: <brief_explanation>`"""

    def evaluate(self, plan_str: str) -> Tuple[float, int]:
        prompt = self.prompt_template.format(user_request=self.user_request, plan_to_evaluate=plan_str)
        response, tokens = self.client.send(prompt)
        score_match = re.search(r"Score:\s*([0-9.]+)", response)
        score = float(score_match.group(1)) if score_match else 0.0
        return score, tokens

def process_problem_with_tot(problem_info: Dict) -> Optional[Dict]:
    """
    Generates and evaluates a plan for a given problem using the Tree of Thoughts (ToT)
    methodology with Breadth-First Search (BFS).
    """
    idx, example, api_family, log_path, log_lock, args = (
        problem_info['dataset_index'], problem_info['example'], problem_info['api_family_for_tools'],
        problem_info['log_path'], problem_info['log_lock'], problem_info['args']
    )

    def write_log(message: str):
        with log_lock:
            with log_path.open("a", encoding="utf-8") as f:
                f.write(f"--- Problem {idx} ({example['id']}) ---\n{message}\n" + "="*80 + "\n\n")

    user_request_text = example['instruction']
    try:
        tools_description = load_tool_descriptions_from_file(Path("Taskbench") / f"data_{api_family}")
        graph_description = load_graph_descriptions_from_file(Path("Taskbench") / f"data_{api_family}")
    except (FileNotFoundError, ValueError) as e:
        write_log(f"CRITICAL ERROR: Could not load descriptions. Error: {e}"); return None

    # Initialize ToT components
    proposer = ToT_Proposer(user_request_text, tools_description, graph_description, args.candidates_per_state)
    evaluator = ToT_Evaluator(user_request_text)

    start_time = time.time()
    total_llm_tokens = 0
    llm_calls = 0

    # ToT with BFS state representation: A list of (plan_steps, score) tuples
    # Start with an empty plan
    active_states = [ ([], 1.0) ]

    for step in range(args.max_steps):
        all_new_candidates = []
        for plan_steps, _ in active_states:
            plan_history_str = "\n".join(plan_steps) if plan_steps else "No steps taken yet."
            
            # 1. GENERATE thoughts for the current state
            proposals, prop_tokens = proposer.propose(plan_history_str)
            total_llm_tokens += prop_tokens
            llm_calls += 1
            
            for p in proposals:
                # A new candidate is the old plan plus the new proposal
                new_plan = plan_steps + [p]
                all_new_candidates.append(new_plan)
                if p.startswith("finish("): # If a plan is finished, keep it for evaluation
                    continue

        if not all_new_candidates:
            break # Stop if no new ideas are generated

        # 2. EVALUATE all generated candidate plans
        evaluated_candidates = []
        for plan in all_new_candidates:
            plan_str = "\n".join(plan)
            score, eval_tokens = evaluator.evaluate(plan_str)
            total_llm_tokens += eval_tokens
            llm_calls += 1
            evaluated_candidates.append( (plan, score) )

        # 3. SELECT the best `b` (breadth) states for the next step
        evaluated_candidates.sort(key=lambda x: x[1], reverse=True)
        active_states = evaluated_candidates[:args.search_breadth]

        # Check if the top state is a finished plan
        if active_states and active_states[0][0][-1].startswith("finish("):
            break

    generation_time_seconds = time.time() - start_time
    
    # Final selection: choose the best plan from the final set of active states
    final_plan_steps = active_states[0][0] if active_states else []

    # Final evaluation of the chosen plan
    final_reward_score = 0.0
    EVALUATION_PROMPT = """Did the 'Generated Plan' successfully solve the 'User Request'? Answer with only "Yes" or "No".\n[User Request]:\n{user_request}\n\n[Generated Plan]:\n{generated_plan}\n\n[Answer (Yes/No)]:"""
    try:
        eval_client = RitsChatClient(temperature=0.0, max_tokens=10)
        eval_prompt = EVALUATION_PROMPT.format(user_request=user_request_text, generated_plan="\n".join(final_plan_steps))
        verdict, eval_tokens = eval_client.send(eval_prompt)
        total_llm_tokens += eval_tokens
        llm_calls +=1
        if verdict.strip().lower().startswith("yes"): final_reward_score = 1.0
    except Exception as e:
        write_log(f"Warning: LLM-based evaluation failed. Error: {e}")

    final_output = {
        "id": example['id'],
        "result": {
            "task_steps": final_plan_steps
        },
        "metrics": {
            "accuracy": final_reward_score,
            "generation_time_seconds": round(generation_time_seconds, 2),
            "plan_length": sum(1 for s in final_plan_steps if s.startswith("api_call")),
            "reasoning_cost": {
                "llm_calls": llm_calls,
                "total_llm_tokens": total_llm_tokens,
            }
        }
    }
    return {"record": final_output}


def load_hf(config_name: str):
    try:
        ds = load_dataset('microsoft/Taskbench', name=config_name, split='test')
        for ex in ds:
            yield {'id': ex['id'], 'instruction': ex['instruction'], 'input': ex.get('input',''), 'tool_steps': ex.get('tool_steps',[])}
    except Exception as e:
        print(f"\n❌ Failed to load '{config_name}' from Hugging Face.", file=sys.stderr); sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# Main Orchestrator
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Run Tree of Thoughts (ToT) Baseline on TaskBench.")
    
    ap.add_argument('--run_name', type=str, default=None, help="Optional name for the output directory.")
    ap.add_argument('--api_family', type=str, default='huggingface', help="API family to test.")
    ap.add_argument('--num_problems', type=int, default=50, help="Number of problems to sample.")
    ap.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")
    ap.add_argument('--model_name', type=str, default='llama_4', help="The model checkpoint to use.")
    ap.add_argument('--max_steps', type=int, default=5, help="Maximum number of steps (depth) in the ToT search.")
    ap.add_argument('--search_breadth', type=int, default=5, help="Beam width for BFS (b).")
    ap.add_argument('--candidates_per_state', type=int, default=3, help="Number of new thoughts to propose per state (k).")
    ap.add_argument('--max_workers', type=int, default=os.cpu_count(), help="Maximum parallel processes.")
    args = ap.parse_args()

    valid_rits_models = list(MODEL_ID_MAP["rits"].keys())
    if args.model_name not in valid_rits_models:
        print(f"❌ Error: Invalid model name '{args.model_name}'. Choose from: {valid_rits_models}", file=sys.stderr); sys.exit(1)
        
    MODELMAP.set_model('generate_model', args.model_name)
    print(f"✅ Configured to use model: {MODELMAP.generate_model}")

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)

    run_name = args.run_name
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"tot_b{args.search_breadth}_{args.api_family}_{args.model_name}_{timestamp}"
        print(f"✅ No run name provided. Using auto-generated name: {run_name}")

    run_dir = Path('predictions') / run_name; run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / 'debug_log.txt'
    if log_path.exists(): log_path.unlink()
    print(f"✅ Outputs will be saved in: {run_dir}")

    # --- Load Data (with local override) ---
    local_data_dir = Path("Taskbench") / f"data_{args.api_family}"
    all_records = []
    
    def load_local(data_dir: Path):
        path = data_dir / 'user_requests.jsonl'
        if not path.exists(): path = data_dir / 'user_requests.json'
        with path.open('r', encoding='utf-8') as f:
            for line in f:
                yield json.loads(line)

    if local_data_dir.is_dir():
        print(f"✅ Found local dataset at '{local_data_dir}'. Loading...")
        all_records = list(load_local(local_data_dir))
    else:
        print(f"✅ No local dataset found. Loading '{args.api_family}' from Hugging Face...")
        all_records = list(load_hf(config_name=args.api_family))

    if not all_records:
        print(f"❌ No problems loaded for API family '{args.api_family}'. Exiting.", file=sys.stderr)
        sys.exit(1)
        
    num_to_process = min(args.num_problems, len(all_records))
    random.shuffle(all_records)
    records_to_process = all_records[:num_to_process]
    print(f"✅ Loaded {len(all_records)} problems, processing {len(records_to_process)}.")

    with Manager() as manager:
        log_lock = manager.Lock()
        
        problems_to_submit = [{
            "dataset_index": j, "example": ex, "api_family_for_tools": args.api_family,
            "log_path": log_path, "log_lock": log_lock, "args": args
        } for j, ex in enumerate(records_to_process)]
        
        run_results = []
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {executor.submit(process_problem_with_tot, prob): prob['dataset_index'] for prob in problems_to_submit}
            for future in tqdm(as_completed(futures), total=len(records_to_process), desc=f"ToT on {args.api_family}"):
                try:
                    result = future.result()
                    if result: run_results.append(result['record'])
                except Exception as e:
                    print(f"Problem {futures[future]} failed: {e}", file=sys.stderr)

        run_output_path = run_dir / 'results.json'
        with run_output_path.open("w", encoding="utf-8") as f: json.dump(run_results, f, indent=2)
        
        total_correct = sum(1 for r in run_results if r.get('metrics', {}).get('accuracy', 0.0) > 0.9)
        total_problems = len(run_results)
        accuracy = (total_correct / total_problems) * 100 if total_problems > 0 else 0
        
        summary = {
            "run_name": run_name, "model_name": args.model_name, "api_family": args.api_family,
            "num_problems_processed": len(records_to_process), "seed": args.seed,
            "search_breadth": args.search_breadth,
            "candidates_per_state": args.candidates_per_state,
            "max_steps": args.max_steps,
            "final_accuracy": f"{accuracy:.2f}%"
        }
        summary_path = run_dir / 'summary.json'
        with summary_path.open("w", encoding="utf-8") as f: json.dump(summary, f, indent=2)

        print(f"\n{'='*25} Experiment Complete {'='*25}")
        print(f"📊 Final Accuracy: {accuracy:.2f}%")
        print(f"✅ Results saved to {run_output_path}")
        print(f"✅ Final summary saved to {summary_path}")

if __name__ == '__main__':
    main()