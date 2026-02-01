#!/usr/bin/env python3
# taskbench_smriv_mcts_ablations.py

import os
import sys
import json
import time
import math
import argparse
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager
import collections
import ast
import re
from datetime import datetime

import random

# New dependencies for semantic matching and rule evolution
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from datasets import load_dataset
from tqdm import tqdm
from SPIRAL.scripts.utils.ritz_client import RitsChatClient, MODELMAP, MODEL_ID_MAP

def make_value_hashable(value: Any) -> Any:
    """Recursively converts lists to tuples and dicts to frozensets of items."""
    if isinstance(value, dict):
        return frozenset((k, make_value_hashable(v)) for k, v in value.items())
    if isinstance(value, list):
        return tuple(make_value_hashable(v) for v in value)
    return value

# ─────────────────────────────────────────────────────────────────────────────
# Manually corrected/enriched parameter definitions for common tools
# ─────────────────────────────────────────────────────────────────────────────
CORRECTED_TOOL_PARAMETERS = {
    "Token Classification": {"text": "string"}, "Translation": {"text": "string", "source_lang": "string", "target_lang": "string"}, "Summarization": {"text": "string"}, "Question Answering": {"context": "string", "question": "string"}, "Conversational": {"prompt": "string", "history": "list"}, "Text Generation": {"prompt": "string"}, "Sentence Similarity": {"sentence1": "string", "sentence2": "string"}, "Tabular Classification": {"table_image_path": "string"}, "Object Detection": {"image_path": "string"}, "Image Classification": {"image_path": "string"}, "Image-to-Image": {"image_path": "string", "target_image_path": "string"}, "Image-to-Text": {"image_path": "string"}, "Text-to-Image": {"prompt": "string"}, "Text-to-Video": {"prompt": "string"}, "Visual Question Answering": {"image_path": "string", "question": "string"}, "Document Question Answering": {"document_image_path": "string", "question": "string"}, "Image Segmentation": {"image_path": "string"}, "Depth Estimation": {"image_path": "string"}, "Text-to-Speech": {"text": "string"}, "Automatic Speech Recognition": {"audio_path": "string"}, "Audio-to-Audio": {"audio_path": "string"}, "Audio Classification": {"audio_path": "string"}, "Image Editing": {"image_path": "string", "edits": "dict"}, "get_weather": {"location": "string", "date": "string"}, "get_news_for_topic": {"topic": "string"}, "stock_operation": {"stock": "string", "operation": "string"}, "book_flight": {"date": "string", "from": "string", "to": "string"}, "book_hotel": {"date": "string", "name": "string"}, "book_restaurant": {"date": "string", "name": "string"}, "book_car": {"date": "string", "location": "string"}, "online_shopping": {"website": "string", "product": "string"}, "send_email": {"email_address": "string", "content": "string"}, "send_sms": {"phone_number": "string", "content": "string"}, "share_by_social_network": {"content": "string", "social_network": "string"}, "search_by_engine": {"query": "string", "engine": "string"}, "apply_for_job": {"job": "string"}, "see_doctor_online": {"disease": "string", "doctor": "string"}, "consult_lawyer_online": {"issue": "string", "lawyer": "string"}, "enroll_in_course": {"course": "string", "university": "string"}, "buy_insurance": {"insurance": "string", "company": "string"}, "online_banking": {"instruction": "string", "bank": "string"}, "daily_bill_payment": {"bill": "string"}, "sell_item_online": {"item": "string", "store": "string"}, "do_tax_return": {"year": "string"}, "apply_for_passport": {"country": "string"}, "pay_for_credit_card": {"credit_card": "string"}, "auto_housework_by_robot": {"instruction": "string"}, "auto_driving_to_destination": {"destination": "string"}, "deliver_package": {"package": "string", "destination": "string"}, "order_food_delivery": {"food": "string", "location": "string", "platform": "string"}, "order_taxi": {"location": "string", "platform": "string"}, "play_music_by_title": {"title": "string"}, "play_movie_by_title": {"title": "string"}, "take_note": {"content": "string"}, "borrow_book_online": {"book": "string", "library": "string"}, "recording_audio": {"content": "string"}, "make_video_call": {"phone_number": "string"}, "make_voice_call": {"phone_number": "string"}, "organize_meeting_online": {"topic": "string"}, "attend_meeting_online": {"topic": "string"}, "software_management": {"software": "string", "instruction": "string"}, "print_document": {"document": "string"}, "set_alarm": {"time": "string"},
}

# --- Global Sentence Transformer Model (Preserved from original) ---
SENTENCE_MODEL = None
def get_sentence_model():
    """Initializes and returns the sentence transformer model as a singleton."""
    global SENTENCE_MODEL
    if SENTENCE_MODEL is None:
        SENTENCE_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    return SENTENCE_MODEL

def parse_tool_code(text: str) -> str:
    """Extracts Python code from a markdown block."""
    match = re.search(r"```(?:python\n)?(.*?)\n?```", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()

# ─────────────────────────────────────────────────────────────────────────────
# Functions to load and format tool & graph descriptions dynamically
# ─────────────────────────────────────────────────────────────────────────────
def load_tool_descriptions_from_file(api_family_data_dir: Path) -> str:
    """Loads and formats tool descriptions from the specified tool_desc.json file."""
    tool_desc_path = api_family_data_dir / "tool_desc.json"
    if not tool_desc_path.exists():
        raise FileNotFoundError(f"Tool description file not found: {tool_desc_path}.")
    try:
        with open(tool_desc_path, 'r', encoding='utf-8') as f:
            tool_data_root = json.load(f)
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
    """Loads and formats graph (dependency) descriptions for the LLM prompt."""
    graph_desc_path = api_family_data_dir / "graph_desc.json"
    if not graph_desc_path.exists(): return ""
    try:
        with open(graph_desc_path, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)
    except (json.JSONDecodeError, Exception) as e:
        print(f"Warning: Could not read {graph_desc_path}: {e}", file=sys.stderr)
        return ""
    
    description_parts = ["\n--- Tool Dependencies ---"]
    for dep_type, deps in graph_data.items():
        if isinstance(deps, list) and deps:
            description_parts.append(f"{dep_type.replace('_', ' ').title()}:")
            for dep in deps:
                if not isinstance(dep, dict): continue
                pre, post = dep.get("pre_tool"), dep.get("post_tool")
                if "resource" in dep_type:
                    res = ", ".join(dep.get("resources", []))
                    description_parts.append(f"  - `{post}` requires resource(s) `{res}` from `{pre}`.")
                elif "temporal" in dep_type:
                    cond = dep.get("condition", "completion")
                    description_parts.append(f"  - `{post}` can only be called after `{pre}` upon its {cond}.")
    
    return "\n".join(description_parts) if len(description_parts) > 1 else ""

# ─────────────────────────────────────────────────────────────────────────────
# Tool Validator (Frontend Compiler)
# ─────────────────────────────────────────────────────────────────────────────
class ToolValidator:
    def __init__(self, parsed_tool_data_root: Dict, debug_llm_output: bool = False):
        self.tool_signatures = collections.defaultdict(dict)
        self.debug_llm_output = debug_llm_output
        if not isinstance(parsed_tool_data_root, dict) or "nodes" not in parsed_tool_data_root:
            return
        tool_nodes = parsed_tool_data_root["nodes"]
        for tool_node in tool_nodes:
            if not isinstance(tool_node, dict): continue
            tool_id, parameters = tool_node.get("id"), tool_node.get("parameters", [])
            if tool_id:
                effective_params = [{"name": n, "type": t} for n, t in CORRECTED_TOOL_PARAMETERS.get(tool_id, {}).items()] or parameters
                self.tool_signatures[tool_id] = {
                    "parameters": {p.get("name"): p.get("type") for p in effective_params if isinstance(p, dict)}
                }

    def validate_api_call(self, code_str: str) -> bool:
        """Performs basic syntax and semantic validation of an api_call string."""
        match = re.search(r'api_call\("([^"]+)",\s*({.*?})\)', code_str, re.DOTALL)
        if not match: return False
        tool_id, args_str = match.group(1), match.group(2)
        if tool_id not in self.tool_signatures: return False
        
        expected_params = self.tool_signatures[tool_id]["parameters"]
        try:
            parsed_args = ast.literal_eval(args_str)
            if not isinstance(parsed_args, dict): return False
            for arg_name in parsed_args:
                if arg_name not in expected_params: return False
            return True
        except (ValueError, SyntaxError):
            return False

# ─────────────────────────────────────────────────────────────────────────────
# Simulated Tool Executor
# ─────────────────────────────────────────────────────────────────────────────
class SimulatedToolExecutor:
    def __init__(self, user_request: str, debug_llm_output: bool = False):
        self.client = RitsChatClient(temperature=0.2, max_tokens=150)
        self.user_request = user_request
        self.debug_llm_output = debug_llm_output

    def execute(self, api_call_str: str, ablation_mode: str) -> Tuple[str, int]:
        # ABLATION: No Rich Simulation Feedback
        if ablation_mode == 'no_sim_feedback':
            return 'Observation: tool_output = "OK"', 0
            
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
            if response_text and response_text.strip().startswith("Observation: tool_output ="):
                return response_text.strip().split('\n')[0], tokens_used
            else:
                if self.debug_llm_output: print(f"  Executor LLM failed format. Response: {response_text}", file=sys.stderr)
                return 'Observation: tool_output = "Error: Tool simulation failed."', tokens_used
        except Exception as e:
            if self.debug_llm_output: print(f"  Executor LLM call failed: {e}", file=sys.stderr)
            return 'Observation: tool_output = "Error: Tool simulation encountered an exception."', 0

# ─────────────────────────────────────────────────────────────────────────────
# MCTS Node
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Node:
    chain: List[str]
    parent: Optional["Node"] = None
    children: List["Node"] = field(default_factory=list)
    visits: int = 0
    value_sum: float = 0.0
    _id: int = field(default_factory=lambda: id(Node))

    def __post_init__(self): self._id = id(self)
    def __hash__(self): return hash(self._id)
    def __eq__(self, other):
        if not isinstance(other, Node): return NotImplemented
        return self._id == other._id
    @property
    def depth(self) -> int: return 0 if self.parent is None else self.parent.depth + 1
    def backpropagate(self, reward: float):
        current = self
        while current is not None:
            current.visits += 1; current.value_sum += reward; current = current.parent
    def uct_score(self, exploration_constant: float = 1.0) -> float:
        if self.visits == 0: return float('inf')
        if self.parent is None or self.parent.visits == 0: return self.value_sum / self.visits
        exploitation = self.value_sum / self.visits
        exploration = exploration_constant * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration

# ─────────────────────────────────────────────────────────────────────────────
# Core Logic for a Single Problem
# ─────────────────────────────────────────────────────────────────────────────
def process_taskbench_problem(problem_info: Dict) -> Optional[Dict]:
    idx, example, api_family, debug_llm, parsed_tool_data, log_path, log_lock, ablation_mode = (
        problem_info['dataset_index'], problem_info['example'], problem_info['api_family_for_tools'],
        problem_info['debug_llm_output'], problem_info['parsed_tool_data'],
        problem_info['log_path'], problem_info['log_lock'], problem_info['ablation_mode']
    )

    def write_log(message: str):
        with log_lock:
            with log_path.open("a", encoding="utf-8") as f:
                f.write(f"--- Problem {idx} ({example['id']}) | Ablation: {ablation_mode} ---\n{message}\n" + "="*40 + "\n\n")

    user_request_text = example['instruction']
    tool_validator = ToolValidator(parsed_tool_data, debug_llm)
    simulated_executor = SimulatedToolExecutor(user_request=user_request_text, debug_llm_output=debug_llm)
    planner_client = RitsChatClient(temperature=0.0, max_tokens=1024)
    
    initial_prompt = f"Instruction: {example['instruction']}" + (f" | Input: {example['input']}" if example.get('input') else "")
    BUDGET_ITERATIONS, MAX_DEPTH = 50, 8

    # --- Initialize metric counters
    start_time = time.time()
    expansion_llm_calls, expansion_llm_tokens = 0, 0
    simulation_llm_calls, simulation_llm_tokens = 0, 0
    invalid_steps_generated = 0

    try:
        tools_description = load_tool_descriptions_from_file(Path("Taskbench") / f"data_{api_family}")
        graph_description = load_graph_descriptions_from_file(Path("Taskbench") / f"data_{api_family}")
    except (FileNotFoundError, ValueError) as e:
        write_log(f"CRITICAL ERROR: Could not load descriptions. Error: {e}"); return None

    # --- Base Prompt Construction ---
    base_prompt_parts = [
        "You are an expert assistant that only responds with code.", "Your task is to create a plan to solve the user's request by generating a sequence of tool calls.", "## RULES:",
        "1. Generate ONLY the single next `api_call(...)` or the final `finish(...)` call.",
        "2. If a previous step produced an observation `tool_output = <value>`, you MUST use that exact `<value>` in the arguments of the next tool.",
        "3. When the user's request is fully satisfied, you MUST call `finish(reason=\"<final answer and summary>\")`.",
        "\n## TOOLS:", tools_description, graph_description, '## FINISH ACTION:\n`finish(reason="<explanation>")`: Call this ONLY when the task is complete.',
    ]

    final_chain = []
    final_best_node = None 
    total_nodes_explored = 0

    # --- ABLATION 1: No MCTS (Greedy Search) ---
    if ablation_mode == 'no_mcts':
        current_chain = [initial_prompt]
        for _ in range(MAX_DEPTH):
            prompt_list = list(base_prompt_parts)
            # ABLATION 3 (Planner): Conditionally add plan history
            if ablation_mode != 'no_plan_history':
                prompt_list.append(f"## CURRENT PLAN:\n" + "\n".join(current_chain))
            prompt_list.append("\nRespond with ONLY the next line of code:")
            prompt_expand = "\n".join(filter(None, prompt_list))
            
            response, tokens_used = planner_client.send(prompt_expand); expansion_llm_calls += 1; expansion_llm_tokens += tokens_used
            extracted_code = parse_tool_code(response.strip())
            
            if extracted_code.startswith("finish("):
                current_chain.append(extracted_code); break
            
            # ABLATION 5 (Validator): Bypass validator if mode is 'no_validator'
            if ablation_mode == 'no_validator' or tool_validator.validate_api_call(extracted_code):
                observation, sim_tokens = simulated_executor.execute(extracted_code, ablation_mode)
                simulation_llm_calls += 1; simulation_llm_tokens += sim_tokens
                current_chain.extend([extracted_code, observation])
            else:
                invalid_steps_generated += 1; break # End greedy search on invalid step
        final_chain = current_chain; total_nodes_explored = 1
    
    # --- DEFAULT: Run MCTS (with other potential ablations) ---
    else:
        root = Node(chain=[initial_prompt]); terminal_nodes = []
        try:
            for _ in range(BUDGET_ITERATIONS):
                current_node = root
                while current_node.children:
                    current_node = max(current_node.children, key=lambda n: n.uct_score())
                if current_node.depth >= MAX_DEPTH or any("finish(" in step for step in current_node.chain):
                    current_node.backpropagate(-0.5); continue

                prompt_list = list(base_prompt_parts)
                # ABLATION 3 (Planner): Conditionally add plan history
                if ablation_mode != 'no_plan_history':
                    prompt_list.append(f"## CURRENT PLAN:\n" + "\n".join(current_node.chain))
                prompt_list.append("\nRespond with ONLY the next line of code:")
                prompt_expand = "\n".join(filter(None, prompt_list))
                
                response, tokens_used = planner_client.send(prompt_expand); expansion_llm_calls += 1; expansion_llm_tokens += tokens_used
                extracted_code = parse_tool_code(response.strip())

                if extracted_code.startswith("finish("):
                    new_node = Node(chain=current_node.chain + [extracted_code], parent=current_node)
                    current_node.children.append(new_node); terminal_nodes.append(new_node)
                    new_node.backpropagate(1.0)
                # ABLATION 5 (Validator): Bypass validator if mode is 'no_validator'
                elif ablation_mode == 'no_validator' or tool_validator.validate_api_call(extracted_code):
                    observation, sim_tokens = simulated_executor.execute(extracted_code, ablation_mode)
                    simulation_llm_calls += 1; simulation_llm_tokens += sim_tokens
                    
                    # ABLATION 4 (Critic): Use uniform rewards
                    reward = 0.0 if ablation_mode == 'uniform_rewards' else 0.1
                    
                    new_node = Node(chain=current_node.chain + [extracted_code, observation], parent=current_node)
                    current_node.children.append(new_node); new_node.backpropagate(reward)
                else:
                    invalid_steps_generated += 1; current_node.backpropagate(-1.0)
        except Exception as e:
            import traceback; write_log(f"CRITICAL ERROR in MCTS loop: {e}\n{traceback.format_exc()}")

        # Find best node from MCTS
        final_best_node = root
        if terminal_nodes:
            final_best_node = max(terminal_nodes, key=lambda n: n.value_sum / n.visits if n.visits > 0 else -1)
        else:
            all_nodes_q, all_nodes_set = collections.deque([root]), {root}
            while all_nodes_q:
                node = all_nodes_q.popleft()
                for child in node.children:
                    if child not in all_nodes_set: all_nodes_set.add(child); all_nodes_q.append(child)
            if all_nodes_set:
                final_best_node = max(list(all_nodes_set), key=lambda n: (n.value_sum / n.visits if n.visits > 0 else -1, n.depth))
        
        nodes_q, explored_nodes = collections.deque([root]), {root}
        while nodes_q:
            node = nodes_q.popleft()
            for child in node.children:
                if child not in explored_nodes: explored_nodes.add(child); nodes_q.append(child)
        total_nodes_explored, final_chain = len(explored_nodes), final_best_node.chain

    search_time_seconds = time.time() - start_time
    task_steps = [parse_tool_code(step) for step in final_chain[1:]]
    plan_length = sum(1 for step in task_steps if step.startswith("api_call"))
    
    final_reward_score = 0.0
    EVALUATION_PROMPT = """Did the 'Generated Plan' successfully solve the 'User Request'? Answer with only "Yes" or "No".\n[User Request]:\n{user_request}\n\n[Generated Plan]:\n{generated_plan}\n\n[Answer (Yes/No)]:"""
    try:
        eval_client = RitsChatClient(temperature=0.0, max_tokens=10)
        eval_prompt = EVALUATION_PROMPT.format(user_request=user_request_text, generated_plan="\n".join(task_steps))
        verdict, _ = eval_client.send(eval_prompt)
        if verdict.strip().lower().startswith("yes"): final_reward_score = 1.0
    except Exception as e:
        write_log(f"Warning: LLM-based evaluation failed. Error: {e}")

    final_output = {
        "id": example['id'], "result": {"task_steps": task_steps},
        "metrics": {
            "accuracy": final_reward_score,
            "final_plan_reward": (final_best_node.value_sum / final_best_node.visits if final_best_node and final_best_node.visits > 0 else 0),
            "search_time_seconds": round(search_time_seconds, 2), "plan_length": plan_length,
            "search_process": { "total_nodes_explored": total_nodes_explored, "mcts_iterations": BUDGET_ITERATIONS if ablation_mode != 'no_mcts' else 0,
                "expansion_llm_calls": expansion_llm_calls, "expansion_llm_tokens": expansion_llm_tokens,
                "simulation_llm_calls": simulation_llm_calls, "simulation_llm_tokens": simulation_llm_tokens,
            }, "robustness": {"invalid_steps_generated": invalid_steps_generated}
        }
    }
    return {"record": final_output}

# --- Data loading helpers (Preserved from original) ---
def load_local(data_dir: Path, split: str):
    path = data_dir / 'user_requests.json'
    if not path.exists(): path = data_dir / 'user_requests.jsonl'
    if not path.exists(): raise FileNotFoundError(f"Missing {data_dir}/user_requests.json or .jsonl")
    with path.open() as f:
        for ln in f:
            data = json.loads(ln)
            yield {'id': data['id'], 'instruction': data.get('user_request',''), 'input': data.get('input',''), 'tool_steps': data.get('tool_steps',[])}

def load_hf(config_name: str):
    """Loads a specific configuration from the microsoft/Taskbench dataset."""
    try:
        ds = load_dataset('microsoft/Taskbench', name=config_name, split='test')
        for ex in ds:
            yield {'id': ex['id'], 'instruction': ex['instruction'], 'input': ex.get('input',''), 'tool_steps': ex.get('tool_steps',[])}
    except Exception as e:
        print(f"\n❌ Failed to load '{config_name}' from Hugging Face.", file=sys.stderr)
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    ap = argparse.ArgumentParser(description="Run Ablation Studies for SMR-IV MCTS on TaskBench.")
    ap.add_argument('--run_name', type=str, default=None, help="Optional name for the output directory.")
    ap.add_argument('--api_family', type=str, default='huggingface', help="API family to test.")
    ap.add_argument('--num_problems', type=int, default=50, help="Number of problems to sample.")
    ap.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")
    ap.add_argument('--model_name', type=str, default='llama_4', help="The model checkpoint to use.")
    
    # REVISED: Argument for ablation mode selection
    ap.add_argument('--ablation_mode', type=str, required=True, 
                        choices=['no_mcts', 'no_sim_feedback', 'no_plan_history', 'uniform_rewards', 'no_validator'],
                        help="Specify the ablation mode to run.")

    ap.add_argument('--max_workers', type=int, default=os.cpu_count(), help="Max parallel processes.")
    ap.add_argument('--debug_llm_output', action='store_true', help="Print detailed LLM IO for debugging.")
    args = ap.parse_args()

    valid_rits_models = list(MODEL_ID_MAP["rits"].keys())
    if args.model_name not in valid_rits_models:
        print(f"❌ Error: Invalid model name '{args.model_name}'. Choose from: {valid_rits_models}", file=sys.stderr); sys.exit(1)
    MODELMAP.set_model('generate_model', args.model_name)
    print(f"✅ Configured to use model: {MODELMAP.generate_model} | Ablation Mode: {args.ablation_mode}")

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)

    run_name = args.run_name or f"{args.ablation_mode}_{args.api_family}_{args.model_name}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = Path('predictions') / run_name; run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / 'debug_log.txt'; log_path.unlink(missing_ok=True)
    print(f"✅ Outputs will be saved in: {run_dir}")

    api_family_data_path = Path("Taskbench") / f"data_{args.api_family}"
    if not api_family_data_path.is_dir(): print(f"❌ Error: 'Taskbench/data_{args.api_family}' not found.", file=sys.stderr); sys.exit(1)
    with open(api_family_data_path / "tool_desc.json", 'r', encoding='utf-8') as f: parsed_tool_data = json.load(f)

    all_records = list(load_hf(config_name=args.api_family)); random.shuffle(all_records)
    records_to_process = all_records[:args.num_problems]

    with Manager() as manager:
        log_lock = manager.Lock()
        problems_to_submit = [{"dataset_index": j, "example": ex, "api_family_for_tools": args.api_family, "debug_llm_output": args.debug_llm_output,
                               "parsed_tool_data": parsed_tool_data, "log_path": log_path, "log_lock": log_lock, "ablation_mode": args.ablation_mode}
                              for j, ex in enumerate(records_to_process)]
        
        run_results = []
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {executor.submit(process_taskbench_problem, p): p['dataset_index'] for p in problems_to_submit}
            desc = f"Ablating '{args.ablation_mode}' on {args.api_family}"
            for future in tqdm(as_completed(futures), total=len(problems_to_submit), desc=desc):
                try:
                    if result := future.result(): run_results.append(result['record'])
                except Exception as e: print(f"Problem {futures[future]} failed: {e}", file=sys.stderr)

    with (run_dir / 'results.json').open("w", encoding="utf-8") as f: json.dump(run_results, f, indent=2)
    
    total_correct = sum(1 for r in run_results if r.get('metrics', {}).get('accuracy', 0.0) > 0.9)
    total_problems = len(run_results)
    accuracy = (total_correct / total_problems) * 100 if total_problems > 0 else 0
    
    summary = {"run_name": run_name, "model_name": args.model_name, "api_family": args.api_family, "ablation_mode": args.ablation_mode, 
               "num_problems_processed": len(records_to_process), "seed": args.seed, "final_accuracy": f"{accuracy:.2f}%"}
    with (run_dir / 'summary.json').open("w", encoding="utf-8") as f: json.dump(summary, f, indent=2)

    print(f"📊 Final Accuracy for '{args.ablation_mode}': {accuracy:.2f}%")
    print(f"✅ Summary saved to {run_dir / 'summary.json'}")

if __name__ == '__main__':
    main()