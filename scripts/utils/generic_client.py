#!/usr/bin/env python3
# utils/generic_client.py
#
# Refactored for public submission with generic Hugging Face implementations.
# This file provides all necessary components and helper functions used by the
# TaskBench experiment scripts, using only publicly available names.

import re
import json
import torch
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, Union
import ast
import collections

# Hugging Face Transformers for generic LLM implementation
from transformers import pipeline, AutoTokenizer

from .ritz_client import RitsChatClient
from .watsonx_client import WatsonxChatClient

# ─────────────────────────────────────────────────────────────────────────────
# 1. GENERIC MODEL CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

MODEL_ID_MAP = {
    # This key is used by the experiment scripts to validate model names.
    "taskbench_models": {

        # Meta Llama Models (Public versions as proxies)
        "llama_3": "meta-llama/Meta-Llama-3-8B-Instruct",
        # Using a powerful model as a proxy for non-public Llama-4
        "llama_4": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "llama_3_3_70b_instruct": "meta-llama/Meta-Llama-3-70B-Instruct",

        # Mistral Models
        "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
        "codestral": "mistralai/Codestral-22B-v0.1",
        "mixtral_8_22b": "mistralai/Mixtral-8x22B-v0.1",

        # Other Models (Public versions as proxies)
        "phi": "microsoft/Phi-3-mini-4k-instruct",
        "deepseek_v2_5": "deepseek-ai/DeepSeek-V2-Lite",
        "qwen2_5_72b_instruct": "Qwen/Qwen2-72B-Instruct",
    }
}


class MODELMAP:
    """
    Class to configure which model to use for different task types.
    """
    er_model = "llama_4"
    generate_model = "llama_4"
    review_model = "llama_4"
    explain_model = "phi"

    @classmethod
    def set_model(cls, model_type: str, model_name: str):
        VALID_TYPES = ["er_model", "generate_model", "review_model", "explain_model"]
        VALID_MODELS = list(MODEL_ID_MAP["taskbench_models"].keys())
        if model_name not in VALID_MODELS:
            raise ValueError(f"Invalid model: {model_name}. Choose from {VALID_MODELS}")
        if model_type not in VALID_TYPES:
            raise ValueError(f"Invalid model type: {model_type}. Choose from {VALID_TYPES}")
        setattr(cls, model_type, model_name)

    @classmethod
    def get_model_id(cls, model_type: str) -> str:
        model_name = getattr(cls, model_type)
        return MODEL_ID_MAP["taskbench_models"][model_name]

# ─────────────────────────────────────────────────────────────────────────────
# 2. CORE LLM CLIENT (Hugging Face Implementation)
# ─────────────────────────────────────────────────────────────────────────────

class HFPipelineManager:
    """
    A generic wrapper for Hugging Face models using the transformers pipeline.
    """
    def __init__(self, model_id: str, temperature: float, max_new_tokens: int):
        device = 0 if torch.cuda.is_available() else -1
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.pipe = pipeline(
            "text-generation",
            model=model_id,
            tokenizer=self.tokenizer,
            device=device,
            torch_dtype=torch.bfloat16,
        )
        self.default_params = {
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "do_sample": True if temperature > 0 else False,
            "return_full_text": False,
            "pad_token_id": self.pipe.tokenizer.eos_token_id,
        }

    def generate(self, prompt: str, **kwargs) -> str:
        gen_params = self.default_params.copy()
        if "temperature" in kwargs:
            gen_params["do_sample"] = True if kwargs["temperature"] > 0 else False
        gen_params.update(kwargs)
        
        response = self.pipe(prompt, **gen_params)
        return response[0]['generated_text'].strip()


class HuggingFaceChatClient:
    """
    Generic chat client that maintains conversation history.
    """
    def __init__(self, temperature: float = 0.5, max_tokens: int = 1024):
        model_id = MODELMAP.get_model_id("generate_model")
        self.manager = HFPipelineManager(
            model_id=model_id,
            temperature=temperature,
            max_new_tokens=max_tokens,
        )

    def send(self, user_message: str, **kwargs) -> Tuple[str, int]:
        """
        Sends a message to the LLM and returns the response and token count.
        """
        prompt = user_message
        input_tokens = len(self.manager.tokenizer.encode(prompt))
        response_text = self.manager.generate(prompt, **kwargs)
        generated_tokens = len(self.manager.tokenizer.encode(response_text))
        total_tokens = input_tokens + generated_tokens
        return response_text, total_tokens

# ─────────────────────────────────────────────────────────────────────────────
# 3. TASKBENCH HELPER FUNCTIONS & CLASSES
# ─────────────────────────────────────────────────────────────────────────────

CORRECTED_TOOL_PARAMETERS = {
    "Token Classification": {"text": "string"}, "Translation": {"text": "string", "source_lang": "string", "target_lang": "string"}, "Summarization": {"text": "string"}, "Question Answering": {"context": "string", "question": "string"}, "Conversational": {"prompt": "string", "history": "list"}, "Text Generation": {"prompt": "string"}, "Sentence Similarity": {"sentence1": "string", "sentence2": "string"}, "Tabular Classification": {"table_image_path": "string"}, "Object Detection": {"image_path": "string"}, "Image Classification": {"image_path": "string"}, "Image-to-Image": {"image_path": "string", "target_image_path": "string"}, "Image-to-Text": {"image_path": "string"}, "Text-to-Image": {"prompt": "string"}, "Text-to-Video": {"prompt": "string"}, "Visual Question Answering": {"image_path": "string", "question": "string"}, "Document Question Answering": {"document_image_path": "string", "question": "string"}, "Image Segmentation": {"image_path": "string"}, "Depth Estimation": {"image_path": "string"}, "Text-to-Speech": {"text": "string"}, "Automatic Speech Recognition": {"audio_path": "string"}, "Audio-to-Audio": {"audio_path": "string"}, "Audio Classification": {"audio_path": "string"}, "Image Editing": {"image_path": "string", "edits": "dict"}, "get_weather": {"location": "string", "date": "string"}, "get_news_for_topic": {"topic": "string"}, "stock_operation": {"stock": "string", "operation": "string"}, "book_flight": {"date": "string", "from": "string", "to": "string"}, "book_hotel": {"date": "string", "name": "string"}, "book_restaurant": {"date": "string", "name": "string"}, "book_car": {"date": "string", "location": "string"}, "online_shopping": {"website": "string", "product": "string"}, "send_email": {"email_address": "string", "content": "string"}, "send_sms": {"phone_number": "string", "content": "string"}, "share_by_social_network": {"content": "string", "social_network": "string"}, "search_by_engine": {"query": "string", "engine": "string"}, "apply_for_job": {"job": "string"}, "see_doctor_online": {"disease": "string", "doctor": "string"}, "consult_lawyer_online": {"issue": "string", "lawyer": "string"}, "enroll_in_course": {"course": "string", "university": "string"}, "buy_insurance": {"insurance": "string", "company": "string"}, "online_banking": {"instruction": "string", "bank": "string"}, "daily_bill_payment": {"bill": "string"}, "sell_item_online": {"item": "string", "store": "string"}, "do_tax_return": {"year": "string"}, "apply_for_passport": {"country": "string"}, "pay_for_credit_card": {"credit_card": "string"}, "auto_housework_by_robot": {"instruction": "string"}, "auto_driving_to_destination": {"destination": "string"}, "deliver_package": {"package": "string", "destination": "string"}, "order_food_delivery": {"food": "string", "location": "string", "platform": "string"}, "order_taxi": {"location": "string", "platform": "string"}, "play_music_by_title": {"title": "string"}, "play_movie_by_title": {"title": "string"}, "take_note": {"content": "string"}, "borrow_book_online": {"book": "string", "library": "string"}, "recording_audio": {"content": "string"}, "make_video_call": {"phone_number": "string"}, "make_voice_call": {"phone_number": "string"}, "organize_meeting_online": {"topic": "string"}, "attend_meeting_online": {"topic": "string"}, "software_management": {"software": "string", "instruction": "string"}, "print_document": {"document": "string"}, "set_alarm": {"time": "string"},
}

def parse_tool_code(text: str) -> str:
    match = re.search(r"```(?:python\n)?(.*?)\n?```", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()

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

class ToolValidator:
    def __init__(self, parsed_tool_data_root: Dict):
        self.tool_signatures = collections.defaultdict(dict)
        tool_nodes = parsed_tool_data_root.get("nodes", [])
        for tool_node in tool_nodes:
            tool_id, parameters = tool_node.get("id"), tool_node.get("parameters", [])
            if tool_id:
                effective_params = [{"name": n, "type": t} for n, t in CORRECTED_TOOL_PARAMETERS.get(tool_id, {}).items()] or parameters
                self.tool_signatures[tool_id] = {"parameters": {p.get("name"): p.get("type") for p in effective_params if isinstance(p, dict)}}

    def validate_api_call(self, code_str: str) -> bool:
        match = re.search(r'api_call\("([^"]+)",\s*({.*?})\)', code_str, re.DOTALL)
        if not match: return False
        tool_id, args_str = match.group(1), match.group(2)
        if tool_id not in self.tool_signatures: return False
        expected_params = self.tool_signatures[tool_id]["parameters"]
        try:
            parsed_args = ast.literal_eval(args_str)
            return isinstance(parsed_args, dict) and all(arg_name in expected_params for arg_name in parsed_args)
        except (ValueError, SyntaxError):
            return False

class SimulatedToolExecutor:
    def __init__(self, user_request: str):
        self.client = HuggingFaceChatClient(temperature=0.2, max_tokens=150)
        self.user_request = user_request

    def execute(self, api_call_str: str) -> Tuple[str, int]:
        prompt_template = """You are a simulated API tool. Provide a realistic, one-line observation for the given tool call.
### User's Goal: "{user_request}"
### Tool Call: `{api_call_str}`
### Your Response (one line starting with `Observation: tool_output = `):
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
# 4. CORE LLM CLIENT (Hugging, RITS, or WATSONX )
# ─────────────────────────────────────────────────────────────────────────────

def getLLMChatClient(llm_platform:str, **kwargs) -> Union[RitsChatClient, WatsonxChatClient, HuggingFaceChatClient]:
    """
    llm_platform: the llm platform to use. It must be "watsonx", "rits" or "hf"
    """
    llm_platform = llm_platform.lower()
    if llm_platform == "watsonx".lower():
        model_id = MODELMAP.get_model_id("generate_model")
        return WatsonxChatClient(model_id, **kwargs)
    elif llm_platform == "rits".lower():
        return RitsChatClient(**kwargs)
    elif llm_platform == "hf".lower():
        return HuggingFaceChatClient(**kwargs)
    else:
        raise Exception(f"Unknown or Unsupported LLM Platform: {llm_platform}.")