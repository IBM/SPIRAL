#!/usr/bin/env python3
# taskbench_react_rafa_baseline.py

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
    # (Implementation is identical to previous scripts)
    tool_desc_path = api_family_data_dir / "tool_desc.json"
    if not tool_desc_path.exists():
        raise FileNotFoundError(f"Tool description file not found: {tool_desc_path}.")
    with open(tool_desc_path, 'r', encoding='utf-8') as f: tool_data_root = json.load(f)
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
    # (Implementation is identical to previous scripts)
    graph_desc_path = api_family_data_dir / "graph_desc.json"
    if not graph_desc_path.exists(): return ""
    with open(graph_desc_path, 'r', encoding='utf-8') as f: graph_data = json.load(f)
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

class SimulatedToolExecutor:
    # (Implementation is identical to previous scripts)
    def __init__(self, user_request: str):
        self.client = RitsChatClient(temperature=0.2, max_tokens=150)
        self.user_request = user_request

    def execute(self, api_call_str: str) -> Tuple[str, int]:
        prompt_template = """You are a simulated API tool. Provide a realistic, one-line observation for the given tool call.
### User's Goal:
"{user_request}"
### Tool Call to Simulate:
`{api_call_str}`
### Your Single-Line Response (must start with `Observation: tool_output = `):
"""
        prompt = prompt_template.format(user_request=self.user_request, api_call_str=api_call_str)
        try:
            response_text, tokens_used = self.client.send(prompt)
            if response_text and response_text.strip().startswith("Observation: tool_output ="):
                return response_text.strip().split('\n')[0], tokens_used
            return 'Observation: tool_output = "Error: Tool simulation failed."', tokens_used
        except Exception:
            return 'Observation: tool_output = "Error: Tool simulation encountered an exception."', 0

# ─────────────────────────────────────────────────────────────────────────────
# Core ReAct+RAFA Logic
# ─────────────────────────────────────────────────────────────────────────────

class HybridElite:
    def __init__(self, user_request: str, tools_description: str, graph_description: str, breadth: int):
        self.client = RitsChatClient(temperature=0.4, max_tokens=1024)
        self.prompt_template = """As an Elite Planner, propose diverse next actions to solve the user's request, based on the history.

### TOOLS
{tools_description}
{graph_description}

### TASK
User Request: {user_request}

### CURRENT TRAJECTORY
{history}

### INSTRUCTION
Generate a Python list of {breadth} distinct `api_call(...)` or `finish(...)` actions to take next.
Respond with ONLY a Python list of strings in a markdown block.
```python
["action_1", "action_2"]
```"""
        self.user_request, self.tools_desc, self.graph_desc, self.breadth = user_request, tools_description, graph_description, breadth
    
    def propose(self, history: str) -> Tuple[List[str], int]:
        prompt = self.prompt_template.format(tools_description=self.tools_desc, graph_description=self.graph_desc, user_request=self.user_request, history=history, breadth=self.breadth)
        response, tokens = self.client.send(prompt)
        match = re.search(r"```(?:python)?\s*(\[.*?\])\s*```", response, re.DOTALL)
        if match:
            try:
                plan = ast.literal_eval(match.group(1).strip())
                if isinstance(plan, list) and all(isinstance(p, str) for p in plan): return plan, tokens
            except: pass
        return [], tokens

class HybridCritic:
    def __init__(self, user_request: str):
        self.client = RitsChatClient(temperature=0.0, max_tokens=100)
        self.prompt_template = """As a Critic, evaluate the following plan's likelihood of success.

### TASK
User Request: {user_request}

### PROPOSED PLAN
{trajectory}

### INSTRUCTION
Respond with ONLY a single line: `Score: <float_0.0_to_1.0> | Justification: <brief_explanation>`"""
        self.user_request = user_request

    def evaluate(self, trajectory: str) -> Tuple[float, str, int]:
        prompt = self.prompt_template.format(user_request=self.user_request, trajectory=trajectory)
        response, tokens = self.client.send(prompt)
        score_match = re.search(r"Score:\s*([0-9.]+)", response)
        just_match = re.search(r"Justification:\s*(.*)", response)
        score = float(score_match.group(1)) if score_match else 0.0
        justification = just_match.group(1).strip() if just_match else "No justification provided."
        return score, justification, tokens

def process_problem_with_react_rafa(problem_info: Dict) -> Optional[Dict]:
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
        tools_desc = load_tool_descriptions_from_file(Path("Taskbench") / f"data_{api_family}")
        graph_desc = load_graph_descriptions_from_file(Path("Taskbench") / f"data_{api_family}")
    except Exception as e:
        write_log(f"CRITICAL ERROR: Could not load descriptions. Error: {e}"); return None

    elite_agent = HybridElite(user_request_text, tools_desc, graph_desc, args.search_breadth)
    model_agent = SimulatedToolExecutor(user_request_text) # Internal model for planning
    critic_agent = HybridCritic(user_request_text)
    real_environment = SimulatedToolExecutor(user_request_text) # External environment

    start_time = time.time()
    # Memory buffer now stores the full ReAct-style trajectory
    memory_buffer = ["User Request: " + user_request_text]
    final_plan_steps = []
    total_tokens = 0
    
    for real_step in range(args.max_real_steps):
        # --- 1. REASON FOR FUTURE (RAFA Planning) ---
        history_str = "\n".join(memory_buffer)
        candidate_actions, elite_tokens = elite_agent.propose(history_str)
        total_tokens += elite_tokens
        
        planned_trajectories = []
        for action in candidate_actions:
            trajectory = [action]
            sim_history_list = memory_buffer + [f"Action: {action}"] # Use a temporary history for simulation
            if not action.startswith("finish("):
                obs, model_tokens = model_agent.execute(action)
                total_tokens += model_tokens
                sim_history_list.append(obs)

            # Lookahead for `search_depth` steps
            current_beam = [ (trajectory, sim_history_list) ]
            for depth in range(args.search_depth - 1):
                next_beam = []
                for traj, hist in current_beam:
                    next_actions, next_elite_tokens = elite_agent.propose("\n".join(hist))
                    total_tokens += next_elite_tokens
                    if not next_actions: continue
                    
                    next_action = next_actions[0] # Single beam for simplicity
                    new_traj = traj + [next_action]
                    new_hist = hist + [f"Action: {next_action}"]
                    if next_action.startswith("finish("):
                        next_beam.append( (new_traj, new_hist) )
                        break 
                    
                    obs, model_tokens = model_agent.execute(next_action)
                    total_tokens += model_tokens
                    new_hist.append(obs)
                    next_beam.append( (new_traj, new_hist) )
                current_beam = next_beam
                if not current_beam: break
            
            if current_beam:
                planned_trajectories.extend([traj for traj, hist in current_beam])

        # Evaluate trajectories and select the best one
        best_trajectory, best_justification, max_score = None, "No valid plan was found.", -1.0
        for trajectory in planned_trajectories:
            score, justification, critic_tokens = critic_agent.evaluate("\n".join(trajectory))
            total_tokens += critic_tokens
            if score > max_score:
                max_score, best_trajectory, best_justification = score, trajectory, justification
        
        if not best_trajectory:
            write_log("ReAct+RAFA planning failed to produce a trajectory.")
            break

        # --- 2. ACT FOR NOW (ReAct-style Execution) ---
        thought = best_justification
        action_to_execute = best_trajectory[0]
        
        final_plan_steps.append(action_to_execute)
        memory_buffer.append(f"Thought: {thought}")
        memory_buffer.append(f"Action: {action_to_execute}")
        
        if action_to_execute.startswith("finish("):
            break

        observation, env_tokens = real_environment.execute(action_to_execute)
        total_tokens += env_tokens
        memory_buffer.append(observation)

    generation_time_seconds = time.time() - start_time

    # Evaluate the final executed plan
    final_reward_score = 0.0
    EVALUATION_PROMPT = """Did the 'Generated Plan' successfully solve the 'User Request'? Answer with only "Yes" or "No".\n[User Request]:\n{user_request}\n\n[Generated Plan]:\n{generated_plan}\n\n[Answer (Yes/No)]:"""
    try:
        eval_client = RitsChatClient(temperature=0.0, max_tokens=10)
        eval_prompt = EVALUATION_PROMPT.format(user_request=user_request_text, generated_plan="\n".join(final_plan_steps))
        verdict, _ = eval_client.send(eval_prompt)
        if verdict.strip().lower().startswith("yes"): final_reward_score = 1.0
    except Exception as e:
        write_log(f"Warning: LLM-based evaluation failed. Error: {e}")

    final_output = {
        "id": example['id'],
        "result": {"task_steps": final_plan_steps},
        "metrics": {
            "accuracy": final_reward_score,
            "generation_time_seconds": round(generation_time_seconds, 2),
            "plan_length": sum(1 for s in final_plan_steps if s.startswith("api_call")),
            "reasoning_cost": {"total_llm_tokens": total_tokens}
        }
    }
    return {"record": final_output}

def load_hf(config_name: str):
    try:
        ds = load_dataset('microsoft/Taskbench', name=config_name, split='test')
        for ex in ds: yield {'id': ex['id'], 'instruction': ex['instruction'], 'input': ex.get('input',''), 'tool_steps': ex.get('tool_steps',[])}
    except Exception as e:
        print(f"\n❌ Failed to load '{config_name}' from Hugging Face.", file=sys.stderr); sys.exit(1)

def main():
    ap = argparse.ArgumentParser(description="Run ReAct+RAFA Hybrid Baseline on TaskBench.")
    
    ap.add_argument('--run_name', type=str, default=None)
    ap.add_argument('--api_family', type=str, default='huggingface')
    ap.add_argument('--num_problems', type=int, default=50)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--model_name', type=str, default='llama_4')
    ap.add_argument('--max_real_steps', type=int, default=8, help="Max steps in the ReAct loop.")
    ap.add_argument('--search_breadth', type=int, default=3, help="RAFA planner breadth (B).")
    ap.add_argument('--search_depth', type=int, default=2, help="RAFA planner depth (U).")
    ap.add_argument('--max_workers', type=int, default=os.cpu_count())
    args = ap.parse_args()

    valid_rits_models = list(MODEL_ID_MAP["rits"].keys())
    if args.model_name not in valid_rits_models:
        print(f"❌ Error: Invalid model name '{args.model_name}'. Choose from: {valid_rits_models}", file=sys.stderr); sys.exit(1)
        
    MODELMAP.set_model('generate_model', args.model_name)
    print(f"✅ Configured to use model: {MODELMAP.generate_model}")

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)

    run_name = args.run_name or f"react_rafa_b{args.search_breadth}d{args.search_depth}_{args.api_family}_{args.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path('predictions') / run_name; run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / 'debug_log.txt'
    if log_path.exists(): log_path.unlink()
    print(f"✅ Outputs will be saved in: {run_dir}")

    # --- Load Data (with local override) ---
    local_data_dir = Path("Taskbench") / f"data_{args.api_family}"
    all_records = []
    
    # The 'load_local' function needs to be present in each script
    def load_local(data_dir: Path):
        path = data_dir / 'user_requests.jsonl'
        if not path.exists(): path = data_dir / 'user_requests.json'
        with path.open('r', encoding='utf-8') as f:
            for line in f:
                yield json.loads(line)

    # Check if a local directory for the api_family exists
    if local_data_dir.is_dir():
        print(f"✅ Found local dataset at '{local_data_dir}'. Loading...")
        all_records = list(load_local(local_data_dir))
    else:
        # Fallback to Hugging Face if no local data is found
        print(f"✅ No local dataset found. Loading '{args.api_family}' from Hugging Face...")
        all_records = list(load_hf(config_name=args.api_family))

    if not all_records:
        print(f"❌ No problems loaded for API family '{args.api_family}'. Exiting.", file=sys.stderr)
        sys.exit(1)
        
    # Shuffle and select the specified number of problems
    num_to_process = min(args.num_problems, len(all_records))
    random.shuffle(all_records)
    records_to_process = all_records[:num_to_process]
    print(f"✅ Loaded {len(all_records)} problems, processing {len(records_to_process)}.")

    with Manager() as manager:
        log_lock = manager.Lock()
        
        problems_to_submit = [{"dataset_index": j, "example": ex, "api_family_for_tools": args.api_family, "log_path": log_path, "log_lock": log_lock, "args": args} for j, ex in enumerate(records_to_process)]
        
        run_results = []
        desc = f"ReAct+RAFA (B={args.search_breadth}, D={args.search_depth}) on {args.api_family}"
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {executor.submit(process_problem_with_react_rafa, prob): prob['dataset_index'] for prob in problems_to_submit}
            for future in tqdm(as_completed(futures), total=len(records_to_process), desc=desc):
                try:
                    result = future.result()
                    if result: run_results.append(result['record'])
                except Exception as e:
                    print(f"Problem {futures[future]} failed: {e}", file=sys.stderr)

        run_output_path = run_dir / 'results.json'
        with run_output_path.open("w", encoding="utf-8") as f: json.dump(run_results, f, indent=2)
        
        total_correct = sum(1 for r in run_results if r.get('metrics', {}).get('accuracy', 0.0) > 0.9)
        accuracy = (total_correct / len(run_results)) * 100 if run_results else 0
        
        summary = {
            "run_name": run_name, "model_name": args.model_name, "api_family": args.api_family,
            "num_problems_processed": len(records_to_process), "seed": args.seed,
            "search_breadth": args.search_breadth, "search_depth": args.search_depth,
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