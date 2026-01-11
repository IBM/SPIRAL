#!/usr/bin/env python3
# taskbench_lats_baseline.py

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
import math

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
    # (Implementation is identical to other scripts)
    tool_desc_path = api_family_data_dir / "tool_desc.json"
    if not tool_desc_path.exists(): raise FileNotFoundError(f"Tool description file not found: {tool_desc_path}.")
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
            if param_name: args_list.append(f"`{param_name}` ({param_type})"); example_args_dict[param_name] = f"<{param_name}_value>"
        example_call_str = f"api_call(\"{tool_id}\", {json.dumps(example_args_dict)})"
        description_parts.append(f"\n`{example_call_str}`\n  Description: {tool_desc}")
        if args_list: description_parts.append(f"  Parameters: {'; '.join(args_list)}")
    return "\n".join(description_parts)

def load_graph_descriptions_from_file(api_family_data_dir: Path) -> str:
    # (Implementation is identical to other scripts)
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
    # (Implementation is identical to other scripts)
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
# Core LATS Logic
# ─────────────────────────────────────────────────────────────────────────────

class LATS_Node:
    """A node in the MCTS tree for LATS."""
    def __init__(self, state: List[str], parent: Optional['LATS_Node'] = None, action: Optional[str] = None):
        self.state = state  # The sequence of (Thought, Action, Observation) strings
        self.parent = parent
        self.action = action # The action that led to this state
        self.children: List['LATS_Node'] = []
        self.visits = 0
        self.value = 0.0

    def is_terminal(self) -> bool:
        return self.state and self.state[-1].startswith("Action: finish(")

    def is_fully_expanded(self, num_candidates: int) -> bool:
        return len(self.children) >= num_candidates

class LATS_Agent:
    """The core agent for LATS, responsible for proposing actions, evaluating states, and reflecting."""
    def __init__(self, user_request: str, tools_desc: str, graph_desc: str):
        self.user_request = user_request
        self.tools_desc = tools_desc
        self.graph_desc = graph_desc
        self.agent_client = RitsChatClient(temperature=0.5, max_tokens=512)
        self.value_client = RitsChatClient(temperature=0.0, max_tokens=256)
        self.reflect_client = RitsChatClient(temperature=0.1, max_tokens=512)

    def _format_reflections(self, reflections: List[str]) -> str:
        if not reflections: return ""
        formatted = "\n".join(f"- {r}" for r in reflections)
        return f"\n### PREVIOUS MISTAKES (Reflections)\n{formatted}\n"

    def propose_actions(self, state_history: str, num_candidates: int, reflections: List[str]) -> Tuple[List[str], int]:
        prompt = f"""As an expert assistant, you solve tasks by thinking and acting.
### AVAILABLE TOOLS
{self.tools_desc}
{self.graph_desc}
{self._format_reflections(reflections)}
### TASK
User Request: {self.user_request}

### CURRENT TRAJECTORY
{state_history}

### INSTRUCTION
Based on the trajectory, generate a Python list of {num_candidates} diverse and promising `api_call(...)` or `finish(...)` actions to try next.
Your response MUST be ONLY a Python list of strings in a markdown block.
```python
[ ... ]
```"""
        response, tokens = self.agent_client.send(prompt)
        match = re.search(r"```(?:python)?\s*(\[.*?\])\s*```", response, re.DOTALL)
        if match:
            try: return ast.literal_eval(match.group(1).strip()), tokens
            except: pass
        return [], tokens

    def evaluate_state(self, state_history: str, reflections: List[str]) -> Tuple[float, int]:
        prompt = f"""As a state evaluator, assess the potential of the current trajectory to solve the user's request.
### TASK
User Request: {self.user_request}
{self._format_reflections(reflections)}
### CURRENT TRAJECTORY
{state_history}

### INSTRUCTION
Evaluate the trajectory's progress and likelihood of success.
Respond with ONLY a single line: `Score: <float_from_0.0_to_1.0> | Justification: <brief_explanation>`"""
        response, tokens = self.value_client.send(prompt)
        match = re.search(r"Score:\s*([0-9.]+)", response)
        return (float(match.group(1)) if match else 0.0), tokens

    def reflect(self, failed_trajectory: str) -> Tuple[str, int]:
        prompt = f"""You are a reasoning agent reflecting on a failed attempt.
### TASK
User Request: {self.user_request}

### FAILED TRAJECTORY
{failed_trajectory}

### INSTRUCTION
You were unsuccessful. In a few sentences, diagnose the primary reason for failure and devise a concise, high-level plan to mitigate this specific failure in the future.
Your response must be a short paragraph."""
        reflection, tokens = self.reflect_client.send(prompt)
        return reflection.strip(), tokens

def process_problem_with_lats(problem_info: Dict) -> Optional[Dict]:
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
    except (FileNotFoundError, ValueError) as e:
        write_log(f"CRITICAL ERROR: Could not load descriptions. Error: {e}"); return None

    # Initialize LATS components
    agent = LATS_Agent(user_request_text, tools_desc, graph_desc)
    sim_executor = SimulatedToolExecutor(user_request_text)
    
    start_time = time.time()
    total_llm_tokens = 0
    reflections: List[str] = []
    root = LATS_Node(state=[f"User Request: {user_request_text}"])

    for i in range(args.mcts_iterations):
        # 1. SELECTION
        leaf = root
        while leaf.children:
            leaf = max(leaf.children, key=lambda n: (n.value / (n.visits + 1e-6)) + args.exploration_weight * math.sqrt(math.log(leaf.visits + 1) / (n.visits + 1e-6)))

        # 2. EXPANSION
        state_history_str = "\n".join(leaf.state)
        if not leaf.is_terminal():
            actions, prop_tokens = agent.propose_actions(state_history_str, args.candidates_per_state, reflections)
            total_llm_tokens += prop_tokens
            for act in actions:
                # Add action and observation to create child state
                new_state = leaf.state + [f"Action: {act}"]
                if not act.startswith("finish("):
                    obs, exec_tokens = sim_executor.execute(act)
                    total_llm_tokens += exec_tokens
                    new_state.append(obs)
                child_node = LATS_Node(state=new_state, parent=leaf, action=act)
                leaf.children.append(child_node)

        # 3. EVALUATION & 4. SIMULATION
        node_to_simulate = leaf.children[0] if leaf.children else leaf
        
        # Simple simulation: just evaluate the current node's potential
        sim_state_str = "\n".join(node_to_simulate.state)
        sim_score, eval_tokens = agent.evaluate_state(sim_state_str, reflections)
        total_llm_tokens += eval_tokens

        # A more complete simulation would run a greedy rollout here.
        # For simplicity in this baseline, we use the evaluated score as the simulation result.
        reward = sim_score

        # 5. BACKPROPAGATION
        temp_node = node_to_simulate
        while temp_node is not None:
            temp_node.visits += 1
            temp_node.value = ((temp_node.value * (temp_node.visits - 1)) + reward) / temp_node.visits
            temp_node = temp_node.parent

        # 6. REFLECTION (if simulation ended in failure)
        if node_to_simulate.is_terminal() and reward < 0.5: # Heuristic for failure
            reflection, reflect_tokens = agent.reflect(sim_state_str)
            total_llm_tokens += reflect_tokens
            if reflection: reflections.append(reflection)
            
    generation_time_seconds = time.time() - start_time

    # Final plan selection: traverse from root, choosing the most visited child
    best_plan_node = root
    final_plan_steps = []
    while best_plan_node.children:
        best_plan_node = max(best_plan_node.children, key=lambda n: n.visits)
        if best_plan_node.action:
            final_plan_steps.append(best_plan_node.action)

    # Final evaluation of the chosen plan
    final_reward_score = 0.0
    EVALUATION_PROMPT = """Did the 'Generated Plan' successfully solve the 'User Request'? Answer with only "Yes" or "No".\n[User Request]:\n{user_request}\n\n[Generated Plan]:\n{generated_plan}\n\n[Answer (Yes/No)]:"""
    try:
        eval_client = RitsChatClient(temperature=0.0, max_tokens=10)
        eval_prompt = EVALUATION_PROMPT.format(user_request=user_request_text, generated_plan="\n".join(final_plan_steps))
        verdict, eval_tokens = eval_client.send(eval_prompt)
        total_llm_tokens += eval_tokens
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
            "reasoning_cost": {"total_llm_tokens": total_llm_tokens}
        }
    }
    return {"record": final_output}

def load_hf(config_name: str):
    # (Implementation is identical to other scripts)
    try:
        ds = load_dataset('microsoft/Taskbench', name=config_name, split='test')
        for ex in ds: yield {'id': ex['id'], 'instruction': ex['instruction'], 'input': ex.get('input',''), 'tool_steps': ex.get('tool_steps',[])}
    except Exception as e:
        print(f"\n❌ Failed to load '{config_name}' from Hugging Face.", file=sys.stderr); sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# Main Orchestrator
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Run Language Agent Tree Search (LATS) Baseline on TaskBench.")
    
    ap.add_argument('--run_name', type=str, default=None)
    ap.add_argument('--api_family', type=str, default='huggingface')
    ap.add_argument('--num_problems', type=int, default=50)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--model_name', type=str, default='llama_4')
    ap.add_argument('--mcts_iterations', type=int, default=10, help="Number of iterations for the MCTS loop (k).")
    ap.add_argument('--exploration_weight', type=float, default=1.0, help="Exploration weight (w) for UCT.")
    ap.add_argument('--candidates_per_state', type=int, default=2, help="Number of actions to expand from a node (n).")
    ap.add_argument('--max_workers', type=int, default=os.cpu_count())
    args = ap.parse_args()

    valid_rits_models = list(MODEL_ID_MAP["rits"].keys())
    if args.model_name not in valid_rits_models:
        print(f"❌ Error: Invalid model name '{args.model_name}'. Choose from: {valid_rits_models}", file=sys.stderr); sys.exit(1)
        
    MODELMAP.set_model('generate_model', args.model_name)
    print(f"✅ Configured to use model: {MODELMAP.generate_model}")

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)

    run_name = args.run_name or f"lats_k{args.mcts_iterations}_{args.api_family}_{args.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path('predictions') / run_name; run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / 'debug_log.txt'
    if log_path.exists(): log_path.unlink()
    print(f"✅ Outputs will be saved in: {run_dir}")

    # --- Load Data (with local override) ---
    local_data_dir = Path("Taskbench") / f"data_{args.api_family}"
    all_records = []
    
    def load_local(data_dir: Path):
        path = data_dir / 'user_requests.jsonl';
        if not path.exists(): path = data_dir / 'user_requests.json'
        with path.open('r', encoding='utf-8') as f:
            for line in f: yield json.loads(line)

    if local_data_dir.is_dir():
        print(f"✅ Found local dataset at '{local_data_dir}'. Loading...")
        all_records = list(load_local(local_data_dir))
    else:
        print(f"✅ No local dataset found. Loading '{args.api_family}' from Hugging Face...")
        all_records = list(load_hf(config_name=args.api_family))

    if not all_records:
        print(f"❌ No problems loaded for API family '{args.api_family}'. Exiting.", file=sys.stderr); sys.exit(1)
        
    num_to_process = min(args.num_problems, len(all_records))
    random.shuffle(all_records)
    records_to_process = all_records[:num_to_process]
    print(f"✅ Loaded {len(all_records)} problems, processing {len(records_to_process)}.")

    with Manager() as manager:
        log_lock = manager.Lock()
        
        problems_to_submit = [{"dataset_index": j, "example": ex, "api_family_for_tools": args.api_family, "log_path": log_path, "log_lock": log_lock, "args": args} for j, ex in enumerate(records_to_process)]
        
        run_results = []
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {executor.submit(process_problem_with_lats, prob): prob['dataset_index'] for prob in problems_to_submit}
            for future in tqdm(as_completed(futures), total=len(records_to_process), desc=f"LATS on {args.api_family}"):
                try:
                    result = future.result()
                    if result: run_results.append(result['record'])
                except Exception as e:
                    print(f"\nProblem {futures[future]} failed: {e}", file=sys.stderr)

        run_output_path = run_dir / 'results.json'
        with run_output_path.open("w", encoding="utf-8") as f: json.dump(run_results, f, indent=2)
        
        total_correct = sum(1 for r in run_results if r.get('metrics', {}).get('accuracy', 0.0) > 0.9)
        accuracy = (total_correct / len(run_results)) * 100 if run_results else 0
        
        summary = {
            "run_name": run_name, "model_name": args.model_name, "api_family": args.api_family,
            "num_problems_processed": len(records_to_process), "seed": args.seed,
            "mcts_iterations": args.mcts_iterations,
            "exploration_weight": args.exploration_weight,
            "candidates_per_state": args.candidates_per_state,
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