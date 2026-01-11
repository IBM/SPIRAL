#!/usr/bin/env python3
# run_taskbench_experiments.py

import os
import sys
import json
import time
import math
import shutil
import tempfile
import argparse
import subprocess
from pathlib import Path
from typing import List, Optional, Dict, Any, Set, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager
import collections
import ast
import re
from datetime import datetime

import random

from SPIRAL.scripts.utils.ritz_client import MODELMAP, MODEL_ID_MAP

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util

from datasets import load_dataset
from tqdm import tqdm
from SPIRAL.scripts.utils.ritz_client import RitsChatClient, MODELMAP, MODEL_ID_MAP

# --- Unchanged Helper functions and constants from original script ---
def make_value_hashable(value: Any) -> Any:
    if isinstance(value, dict): return frozenset((k, make_value_hashable(v)) for k, v in value.items())
    if isinstance(value, list): return tuple(make_value_hashable(v) for v in value)
    return value
CORRECTED_TOOL_PARAMETERS = { "Token Classification": {"text": "string"}, "Translation": {"text": "string", "source_lang": "string", "target_lang": "string"}, "Summarization": {"text": "string"}, "Question Answering": {"context": "string", "question": "string"}, "Conversational": {"prompt": "string", "history": "list"}, "Text Generation": {"prompt": "string"}, "Sentence Similarity": {"sentence1": "string", "sentence2": "string"}, "Tabular Classification": {"table_image_path": "string"}, "Object Detection": {"image_path": "string"}, "Image Classification": {"image_path": "string"}, "Image-to-Image": {"image_path": "string", "target_image_path": "string"}, "Image-to-Text": {"image_path": "string"}, "Text-to-Image": {"prompt": "string"}, "Text-to-Video": {"prompt": "string"}, "Visual Question Answering": {"image_path": "string", "question": "string"}, "Document Question Answering": {"document_image_path": "string", "question": "string"}, "Image Segmentation": {"image_path": "string"}, "Depth Estimation": {"image_path": "string"}, "Text-to-Speech": {"text": "string"}, "Automatic Speech Recognition": {"audio_path": "string"}, "Audio-to-Audio": {"audio_path": "string"}, "Audio Classification": {"audio_path": "string"}, "Image Editing": {"image_path": "string", "edits": "dict"}, "get_weather": {"location": "string", "date": "string"}, "get_news_for_topic": {"topic": "string"}, "stock_operation": {"stock": "string", "operation": "string"}, "book_flight": {"date": "string", "from": "string", "to": "string"}, "book_hotel": {"date": "string", "name": "string"}, "book_restaurant": {"date": "string", "name": "string"}, "book_car": {"date": "string", "location": "string"}, "online_shopping": {"website": "string", "product": "string"}, "send_email": {"email_address": "string", "content": "string"}, "send_sms": {"phone_number": "string", "content": "string"}, "share_by_social_network": {"content": "string", "social_network": "string"}, "search_by_engine": {"query": "string", "engine": "string"}, "apply_for_job": {"job": "string"}, "see_doctor_online": {"disease": "string", "doctor": "string"}, "consult_lawyer_online": {"issue": "string", "lawyer": "string"}, "enroll_in_course": {"course": "string", "university": "string"}, "buy_insurance": {"insurance": "string", "company": "string"}, "online_banking": {"instruction": "string", "bank": "string"}, "daily_bill_payment": {"bill": "string"}, "sell_item_online": {"item": "string", "store": "string"}, "do_tax_return": {"year": "string"}, "apply_for_passport": {"country": "string"}, "pay_for_credit_card": {"credit_card": "string"}, "auto_housework_by_robot": {"instruction": "string"}, "auto_driving_to_destination": {"destination": "string"}, "deliver_package": {"package": "string", "destination": "string"}, "order_food_delivery": {"food": "string", "location": "string", "platform": "string"}, "order_taxi": {"location": "string", "platform": "string"}, "play_music_by_title": {"title": "string"}, "play_movie_by_title": {"title": "string"}, "take_note": {"content": "string"}, "borrow_book_online": {"book": "string", "library": "string"}, "recording_audio": {"content": "string"}, "make_video_call": {"phone_number": "string"}, "make_voice_call": {"phone_number": "string"}, "organize_meeting_online": {"topic": "string"}, "attend_meeting_online": {"topic": "string"}, "software_management": {"software": "string", "instruction": "string"}, "print_document": {"document": "string"}, "set_alarm": {"time": "string"}, }
SENTENCE_MODEL = None
def get_sentence_model():
    global SENTENCE_MODEL
    if SENTENCE_MODEL is None: SENTENCE_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    return SENTENCE_MODEL
def parse_tool_code(text: str) -> str:
    match = re.search(r"```(?:python\n)?(.*?)\n?```", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()
def load_tool_descriptions_from_file(api_family_data_dir: Path) -> str:
    tool_desc_path = api_family_data_dir / "tool_desc.json"
    if not tool_desc_path.exists(): raise FileNotFoundError(f"Tool description file not found: {tool_desc_path}.")
    with open(tool_desc_path, 'r', encoding='utf-8') as f: tool_data_root = json.load(f)
    description_parts = ["Available tools (use the `api_call` function to invoke them):"]
    tool_nodes = tool_data_root.get("nodes", [])
    for tool_node in tool_nodes:
        if not isinstance(tool_node, dict): continue
        tool_id, tool_desc, params = tool_node.get("id"), tool_node.get("desc"), tool_node.get("parameters", [])
        if not tool_id or not tool_desc: continue
        args_list, example_args_dict = [], {}
        effective_params = [{"name": n, "type": t} for n, t in CORRECTED_TOOL_PARAMETERS.get(tool_id, {}).items()] or params
        for param in effective_params:
            param_name, param_type = param.get("name"), param.get("type", "Any")
            if param_name: args_list.append(f"`{param_name}` ({param_type})"); example_args_dict[param_name] = f"<{param_name}_value>"
        example_call_str = f"api_call(\"{tool_id}\", {json.dumps(example_args_dict)})"
        description_parts.extend([f"\n`{example_call_str}`", f"  Description: {tool_desc}"])
        if args_list: description_parts.append(f"  Parameters: {'; '.join(args_list)}")
    return "\n".join(description_parts)
def load_graph_descriptions_from_file(api_family_data_dir: Path) -> str:
    graph_desc_path = api_family_data_dir / "graph_desc.json"
    if not graph_desc_path.exists(): return ""
    with open(graph_desc_path, 'r', encoding='utf-8') as f: graph_data = json.load(f)
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
class ToolValidator:
    def __init__(self, parsed_tool_data_root: Dict, debug_llm_output: bool = False):
        self.tool_signatures = collections.defaultdict(dict)
        if not isinstance(parsed_tool_data_root, dict) or "nodes" not in parsed_tool_data_root: return
        for tool_node in parsed_tool_data_root["nodes"]:
            if not isinstance(tool_node, dict): continue
            tool_id, parameters = tool_node.get("id"), tool_node.get("parameters", [])
            if tool_id:
                effective_params = [{"name": n, "type": t} for n, t in CORRECTED_TOOL_PARAMETERS.get(tool_id, {}).items()] or parameters
                self.tool_signatures[tool_id] = {"parameters": {p.get("name"): p.get("type") for p in effective_params if isinstance(p, dict)}}
    def validate_api_call(self, code_str: str) -> bool:
        match = re.search(r'api_call\("([^"]+)",\s*({.*?})\)', code_str, re.DOTALL)
        if not match: return False
        tool_id, args_str = match.group(1), match.group(2)
        if tool_id not in self.tool_signatures: return False
        try:
            parsed_args = ast.literal_eval(args_str)
            if not isinstance(parsed_args, dict): return False
            for arg_name in parsed_args:
                if arg_name not in self.tool_signatures[tool_id]["parameters"]: return False
            return True
        except (ValueError, SyntaxError): return False
class SimulatedToolExecutor:
    def __init__(self, user_request: str, debug_llm_output: bool = False):
        self.client = RitsChatClient(temperature=0.2, max_tokens=150); self.user_request = user_request; self.debug_llm_output = debug_llm_output
    def execute(self, api_call_str: str, ablation_mode: str) -> Tuple[str, int]:
        if ablation_mode == 'no_sim_feedback': return 'Observation: tool_output = "OK"', 0
        prompt_template = """You are a simulated API tool. Your role is to provide a realistic, one-line observation for the given tool call, based on the user's overall goal.
        ### Rules:
        1.  Your entire response MUST be a single line starting with `Observation: tool_output = `.
        2.  The value part should be a plausible result. For tools that create files (like image editing or generation), the value should be a new, unique filename string (e.g., `"edited_image.png"`). For analysis tools, it should be a short, descriptive string or the direct answer (e.g., `"a red sports car"`).
        3.  The observation must be grounded in the user's request.
        ### User's Goal:
        "{user_request}"
        ### Tool Call to Simulate:
        `{api_call_str}`
        ### Your Single-Line Response:
        """
        prompt = prompt_template.format(user_request=self.user_request, api_call_str=api_call_str)
        try:
            response_text, tokens_used = self.client.send(prompt)
            if response_text and response_text.strip().startswith("Observation: tool_output ="): return response_text.strip().split('\n')[0], tokens_used
            if self.debug_llm_output: print(f"  Executor LLM failed format. Response: {response_text}", file=sys.stderr)
            return 'Observation: tool_output = "Error: Tool simulation failed."', tokens_used
        except Exception as e:
            if self.debug_llm_output: print(f"  Executor LLM call failed: {e}", file=sys.stderr)
            return 'Observation: tool_output = "Error: Tool simulation encountered an exception."', 0
@dataclass
class Node:
    chain: List[str]; parent: Optional["Node"] = None; children: List["Node"] = field(default_factory=list)
    visits: int = 0; value_sum: float = 0.0; _id: int = field(default_factory=lambda: id(Node))
    def __post_init__(self): self._id = id(self)
    def __hash__(self): return hash(self._id)
    def __eq__(self, other): return isinstance(other, Node) and self._id == other._id
    @property
    def depth(self) -> int: return 0 if self.parent is None else self.parent.depth + 1
    def backpropagate(self, reward: float):
        current = self
        while current is not None: current.visits += 1; current.value_sum += reward; current = current.parent
    def uct_score(self, exploration_constant: float = 1.0) -> float:
        if self.visits == 0: return float('inf')
        if self.parent is None or self.parent.visits == 0: return self.value_sum / self.visits
        exploitation = self.value_sum / self.visits
        exploration = exploration_constant * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration

def process_taskbench_problem(problem_info: Dict) -> Optional[Dict]:
    # Unpack all arguments
    idx, example, api_family, debug_llm, parsed_tool_data, log_path, log_lock, ablation_mode, baseline_config = (
        problem_info['dataset_index'], problem_info['example'], problem_info['api_family_for_tools'],
        problem_info['debug_llm_output'], problem_info['parsed_tool_data'], problem_info['log_path'], problem_info['log_lock'],
        problem_info['ablation_mode'], problem_info['baseline_mcts_config']
    )

    run_mode_str = ablation_mode if ablation_mode != 'none' else f"baseline_mcts_{baseline_config}"
    def write_log(message: str):
        with log_lock:
            with log_path.open("a", encoding="utf-8") as f: f.write(f"--- P{idx} ({example['id']}) | Mode: {run_mode_str} ---\n{message}\n" + "="*40 + "\n\n")

    # --- Experiment Configuration ---
    user_request_text = example['instruction']
    tool_validator = ToolValidator(parsed_tool_data, debug_llm)
    simulated_executor = SimulatedToolExecutor(user_request=user_request_text, debug_llm_output=debug_llm)
    planner_client = RitsChatClient(temperature=0.0, max_tokens=1024)
    
    # Define search budget based on experiment type
    MAX_DEPTH = 8
    BUDGET_MAP = {'light': 15, 'medium': 30, 'heavy': 50}
    BUDGET_ITERATIONS = BUDGET_MAP[baseline_config] if baseline_config != 'none' else 50

    is_uniform_rewards = (ablation_mode == 'uniform_rewards' or baseline_config != 'none')
    is_no_mcts = (ablation_mode == 'no_mcts')

    # --- Metric counters ---
    start_time = time.time(); expansion_llm_calls, expansion_llm_tokens = 0, 0
    simulation_llm_calls, simulation_llm_tokens = 0, 0; invalid_steps_generated = 0
    
    try:
        tools_description = load_tool_descriptions_from_file(Path("Taskbench") / f"data_{api_family}")
        graph_description = load_graph_descriptions_from_file(Path("Taskbench") / f"data_{api_family}")
    except (FileNotFoundError, ValueError) as e:
        write_log(f"CRITICAL ERROR: Could not load descriptions. Error: {e}"); return None

    base_prompt_parts = [ "You are an expert assistant...", "## RULES:", "1. Generate ONLY the single next `api_call(...)`...", "\n## TOOLS:", tools_description, graph_description, '## FINISH ACTION:\n`finish(reason="<explanation>")`...'] # Truncated for brevity

    final_chain = []; final_best_node = None; total_nodes_explored = 0

    # --- Main Logic: Greedy Search or MCTS ---
    if is_no_mcts:
        current_chain = [f"Instruction: {example['instruction']}"]
        for _ in range(MAX_DEPTH):
            prompt_list = list(base_prompt_parts)
            if ablation_mode != 'no_plan_history': prompt_list.append(f"## CURRENT PLAN:\n" + "\n".join(current_chain))
            prompt_list.append("\nRespond with ONLY the next line of code:")
            prompt_expand = "\n".join(filter(None, prompt_list))
            response, tokens_used = planner_client.send(prompt_expand); expansion_llm_calls += 1; expansion_llm_tokens += tokens_used
            extracted_code = parse_tool_code(response.strip())
            if extracted_code.startswith("finish("): current_chain.append(extracted_code); break
            if ablation_mode == 'no_validator' or tool_validator.validate_api_call(extracted_code):
                observation, sim_tokens = simulated_executor.execute(extracted_code, ablation_mode)
                simulation_llm_calls += 1; simulation_llm_tokens += sim_tokens; current_chain.extend([extracted_code, observation])
            else: invalid_steps_generated += 1; break
        final_chain = current_chain; total_nodes_explored = 1
    else: # MCTS Run (Baseline or Ablation)
        root = Node(chain=[f"Instruction: {example['instruction']}"]); terminal_nodes = []
        try:
            for i in range(BUDGET_ITERATIONS):
                current_node = root
                while current_node.children: current_node = max(current_node.children, key=lambda n: n.uct_score())
                if current_node.depth >= MAX_DEPTH or any("finish(" in step for step in current_node.chain):
                    current_node.backpropagate(-0.5); continue
                prompt_list = list(base_prompt_parts)
                if ablation_mode != 'no_plan_history': prompt_list.append(f"## CURRENT PLAN:\n" + "\n".join(current_node.chain))
                prompt_list.append("\nRespond with ONLY the next line of code:")
                prompt_expand = "\n".join(filter(None, prompt_list))
                response, tokens_used = planner_client.send(prompt_expand); expansion_llm_calls += 1; expansion_llm_tokens += tokens_used
                extracted_code = parse_tool_code(response.strip())
                if extracted_code.startswith("finish("):
                    new_node = Node(chain=current_node.chain + [extracted_code], parent=current_node)
                    current_node.children.append(new_node); terminal_nodes.append(new_node); new_node.backpropagate(1.0)
                elif ablation_mode == 'no_validator' or tool_validator.validate_api_call(extracted_code):
                    observation, sim_tokens = simulated_executor.execute(extracted_code, ablation_mode)
                    simulation_llm_calls += 1; simulation_llm_tokens += sim_tokens
                    reward = 0.0 if is_uniform_rewards else 0.1
                    new_node = Node(chain=current_node.chain + [extracted_code, observation], parent=current_node)
                    current_node.children.append(new_node); new_node.backpropagate(reward)
                else: invalid_steps_generated += 1; current_node.backpropagate(-1.0)
        except Exception as e: import traceback; write_log(f"MCTS ERROR: {e}\n{traceback.format_exc()}")
        
        final_best_node = root
        if terminal_nodes: final_best_node = max(terminal_nodes, key=lambda n: n.value_sum / n.visits if n.visits > 0 else -1)
        else:
            q, all_nodes = collections.deque([root]), {root}
            while q:
                n = q.popleft()
                for child in n.children:
                    if child not in all_nodes: all_nodes.add(child); q.append(child)
            if all_nodes: final_best_node = max(list(all_nodes), key=lambda n: (n.value_sum / n.visits if n.visits > 0 else -1, n.depth))
        q_explore, explored = collections.deque([root]), {root}
        while q_explore:
            n = q_explore.popleft()
            for child in n.children:
                if child not in explored: explored.add(child); q_explore.append(child)
        total_nodes_explored, final_chain = len(explored), final_best_node.chain

    search_time_seconds = time.time() - start_time
    task_steps = [parse_tool_code(s) for s in final_chain[1:]]
    plan_length = sum(1 for s in task_steps if s.startswith("api_call"))
    final_reward_score = 0.0
    EVALUATION_PROMPT = """Did the 'Generated Plan' successfully solve the 'User Request'? Answer with only "Yes" or "No".\n[User Request]:\n{user_request}\n\n[Generated Plan]:\n{generated_plan}\n\n[Answer (Yes/No)]:"""
    try:
        eval_client = RitsChatClient(temperature=0.0, max_tokens=10)
        eval_prompt = EVALUATION_PROMPT.format(user_request=user_request_text, generated_plan="\n".join(task_steps))
        verdict, _ = eval_client.send(eval_prompt)
        if verdict.strip().lower().startswith("yes"): final_reward_score = 1.0
    except Exception as e: write_log(f"Warning: LLM-based evaluation failed. Error: {e}")

    return {"record": { "id": example['id'], "result": {"task_steps": task_steps}, "metrics": { "accuracy": final_reward_score, "final_plan_reward": (final_best_node.value_sum / final_best_node.visits if final_best_node and final_best_node.visits > 0 else 0), "search_time_seconds": round(search_time_seconds, 2), "plan_length": plan_length, "search_process": { "total_nodes_explored": total_nodes_explored, "mcts_iterations": BUDGET_ITERATIONS if not is_no_mcts else 0, "expansion_llm_calls": expansion_llm_calls, "expansion_llm_tokens": expansion_llm_tokens, "simulation_llm_calls": simulation_llm_calls, "simulation_llm_tokens": simulation_llm_tokens, }, "robustness": {"invalid_steps_generated": invalid_steps_generated} } } }

# --- Data loading helpers ---
def load_local(data_dir: Path, split: str):
    path = data_dir / 'user_requests.json'
    if not path.exists(): path = data_dir / 'user_requests.jsonl'
    if not path.exists(): raise FileNotFoundError(f"Missing {data_dir}/user_requests.json or .jsonl")
    with path.open() as f:
        for ln in f:
            data = json.loads(ln)
            yield {'id': data['id'], 'instruction': data.get('user_request',''), 'input': data.get('input',''), 'tool_steps': data.get('tool_steps',[])}
def load_hf(config_name: str):
    try:
        ds = load_dataset('microsoft/Taskbench', name=config_name, split='test')
        for ex in ds: yield {'id': ex['id'], 'instruction': ex['instruction'], 'input': ex.get('input',''), 'tool_steps': ex.get('tool_steps',[])}
    except Exception as e: print(f"\n❌ HF Load Error: {e}", file=sys.stderr); sys.exit(1)

def main():
    ap = argparse.ArgumentParser(description="Run Baseline and Ablation Experiments on TaskBench.")
    ap.add_argument('--run_name', type=str, default=None); ap.add_argument('--api_family', type=str, default='huggingface')
    ap.add_argument('--num_problems', type=int, default=50); ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--model_name', type=str, default='llama_4'); ap.add_argument('--max_workers', type=int, default=os.cpu_count())
    ap.add_argument('--debug_llm_output', action='store_true')
    
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--ablation_mode', type=str, default='none', choices=['none', 'no_mcts', 'no_sim_feedback', 'no_plan_history', 'uniform_rewards', 'no_validator'])
    group.add_argument('--baseline_mcts_config', type=str, default='none', choices=['none', 'light', 'medium', 'heavy'])
    args = ap.parse_args()

    run_mode = args.ablation_mode if args.ablation_mode != 'none' else f"baseline_{args.baseline_mcts_config}"
    MODELMAP.set_model('generate_model', args.model_name)
    print(f"✅ Model: {MODELMAP.generate_model} | Experiment: {run_mode}")

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)

    run_name = args.run_name or f"{run_mode}_{args.api_family}_{args.model_name}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = Path('predictions') / run_name; run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / 'debug_log.txt'; log_path.unlink(missing_ok=True)
    print(f"✅ Outputs -> {run_dir}")

    api_data_path = Path("Taskbench") / f"data_{args.api_family}"
    if not api_data_path.is_dir(): print(f"❌ Error: '{api_data_path}' not found.", file=sys.stderr); sys.exit(1)
    with open(api_data_path / "tool_desc.json", 'r', encoding='utf-8') as f: parsed_tool_data = json.load(f)

    all_records = list(load_hf(config_name=args.api_family)); random.shuffle(all_records)
    records_to_process = all_records[:args.num_problems]

    with Manager() as manager:
        log_lock = manager.Lock()
        problems = [{"dataset_index": j, "example": ex, "api_family_for_tools": args.api_family, "debug_llm_output": args.debug_llm_output,
                     "parsed_tool_data": parsed_tool_data, "log_path": log_path, "log_lock": log_lock, 
                     "ablation_mode": args.ablation_mode, "baseline_mcts_config": args.baseline_mcts_config}
                    for j, ex in enumerate(records_to_process)]
        
        results = []
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {executor.submit(process_taskbench_problem, p): p['dataset_index'] for p in problems}
            desc = f"Running '{run_mode}' on {args.api_family}"
            for future in tqdm(as_completed(futures), total=len(problems), desc=desc):
                try:
                    if res := future.result(): results.append(res['record'])
                except Exception as e: print(f"Problem {futures[future]} failed: {e}", file=sys.stderr)

    with (run_dir / 'results.json').open("w", encoding="utf-8") as f: json.dump(results, f, indent=2)
    
    total_correct = sum(1 for r in results if r.get('metrics', {}).get('accuracy', 0.0) > 0.9)
    accuracy = (total_correct / len(results)) * 100 if results else 0
    
    summary = {"run_name": run_name, "model_name": args.model_name, "api_family": args.api_family, "experiment_mode": run_mode, 
               "num_problems": len(records_to_process), "seed": args.seed, "final_accuracy": f"{accuracy:.2f}%"}
    with (run_dir / 'summary.json').open("w", encoding="utf-8") as f: json.dump(summary, f, indent=2)

    print(f"📊 Final Accuracy for '{run_mode}': {accuracy:.2f}%")
    print(f"✅ Summary saved to {run_dir / 'summary.json'}")

if __name__ == '__main__':
    main()