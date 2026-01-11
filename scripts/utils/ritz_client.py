import os
import re
import time
import logging
from typing import List, Optional, Any
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.outputs import LLMResult
from langchain_core.messages import BaseMessage
from langchain_ibm import WatsonxLLM
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods


# ─────────────────────────────────────────────────────────────────────────────
# DISABLE LITELLM LOGGING ENTIRELY
# ─────────────────────────────────────────────────────────────────────────────
logging.disable(logging.CRITICAL)

import litellm
from langchain_community.chat_models import ChatLiteLLM
from langchain_community.chat_models.litellm import _create_retry_decorator

# ─────────────────────────────────────────────────────────────────────────────
# MODEL ID MAPS
# ─────────────────────────────────────────────────────────────────────────────
MODEL_ID_MAP = {
    "watsonx": {
        "granite": {"model_id": "ibm/granite-3-2-8b-instruct", "model_url_id": None},
        "llama_4": {"model_id": "meta-llama/llama-4-maverick-17b-128e-instruct-fp8", "model_url_id": None},
        "mistral": {"model_id": "mistralai/mistral-large", "model_url_id": None},
    },
    "rits": {
        "granite": {"model_id": "ibm-granite/granite-3.3-8b-instruct", "model_url_id": "granite-3-3-8b-instruct"},
        "llama_3": {"model_id": "meta-llama/llama-3-1-405b-instruct-fp8", "model_url_id": "llama-3-1-405b-instruct-fp8"},
        "llama_4": {"model_id": "meta-llama/llama-4-maverick-17b-128e-instruct-fp8", "model_url_id": "llama-4-mvk-17b-128e-fp8"},
        "phi": {"model_id": "microsoft/phi-4", "model_url_id": "microsoft-phi-4"},
        "codestral": {"model_id": "mistralai/Codestral-22B-v0.1", "model_url_id": "codestral-22b-v01"},
        "mixtral_8_22b": {"model_id": "mistralai/mixtral-8x22B-instruct-v0.1", "model_url_id": "mixtral-8x22b-instruct-a100"},
        "granite_34b": {"model_id": "ibm-granite/granite-34b-code-instruct-8k", "model_url_id": "granite-34b-code-instruct-8k"},
        "granite_20b": {"model_id": "ibm-granite/granite-20b-code-instruct-8k", "model_url_id": "granite-20b-code-instruct-8k"},
        "deepseek_v3_h200": {
            "model_id": "deepseek-ai/DeepSeek-V3",
            "model_url_id": "deepseek-v3-h200"
        },
        "deepseek_v2_5": {
            "model_id": "deepseek-ai/DeepSeek-V2.5",
            "model_url_id": "deepseek-v2-5"
        },
        "llama_4_scout_17b_16e_instruct": {
            "model_id": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
            "model_url_id": "llama-4-scout-17b-16e-instruct"
        },
        "llama_3_3_70b_instruct": {
            "model_id": "meta-llama/llama-3-3-70b-instruct",
            "model_url_id": "llama-3-3-70b-instruct"
        },
        "qwen3_8b": {
            "model_id": "Qwen/Qwen3-8B",
            "model_url_id": "qwen3-8b"
        },
        "qwen2_5_72b_instruct": {
            "model_id": "Qwen/Qwen2.5-72B-Instruct",
            "model_url_id": "qwen2-5-72b-instruct"
        },
    },
}


def parsebool(val):
    bool_map = {"y": True, "true": True, "t": True, "yes": True, "n": False, "false": False, "f": False, "no": False}
    return bool_map.get(str(val).lower(), False)


class MODELMAP:
    er_model = "llama_4"
    generate_model = "llama_4"
    review_model = "llama_4"
    explain_model = "llama_4"

    @classmethod
    def set_model(cls, model_type: str, model_name: str):
        is_watsonx = parsebool(os.environ.get("USE_WATSONX", False))
        VALID_TYPES = ["er_model", "generate_model", "review_model", "explain_model"]
        VALID_MODELS = list(MODEL_ID_MAP["watsonx"].keys())
        if not is_watsonx:
            VALID_MODELS = list(MODEL_ID_MAP["rits"].keys())

        if model_name not in VALID_MODELS:
            raise ValueError(f"Invalid model name: {model_name}. Choose from {VALID_MODELS}")
        if model_type not in VALID_TYPES:
            raise ValueError(f"Invalid model type: {model_type}. Choose from {VALID_TYPES}")
        setattr(cls, model_type, model_name)

    @classmethod
    def get_wpa_model_details(cls):
        is_watsonx = parsebool(os.environ.get("USE_WATSONX", False))
        if is_watsonx:
            return (
                MODEL_ID_MAP["watsonx"][cls.generate_model]["model_id"],
                MODEL_ID_MAP["watsonx"][cls.generate_model]["model_url_id"],
            )
        else:
            return (
                MODEL_ID_MAP["rits"][cls.generate_model]["model_id"],
                MODEL_ID_MAP["rits"][cls.generate_model]["model_url_id"],
            )


class LLMSelector:
    def __init__(
        self,
        model_id="ibm/granite-20b-code-instruct",
        print_prompt=False,
        model_url_id=None,
        temperature=0,
        top_p=None,
        n=1,
        min_tokens=1,
        max_tokens=None,
        max_retries=3,
    ):
        is_watsonx = parsebool(os.environ.get("USE_WATSONX", "False"))
        if is_watsonx:
            api_url = os.environ["WATSONX_URL"]
            api_key = os.environ["WATSONX_APIKEY"]
            project_id = os.environ["WATSONX_PROJECT_ID"]
            decoding_method = DecodingMethods.SAMPLE.value
            if temperature == 0:
                decoding_method = DecodingMethods.GREEDY.value
            parameters = {
                GenParams.DECODING_METHOD: decoding_method,
                GenParams.MAX_NEW_TOKENS: max_tokens,
                GenParams.MIN_NEW_TOKENS: min_tokens,
                GenParams.TEMPERATURE: temperature,
                GenParams.TOP_K: n,
                GenParams.TOP_P: top_p,
            }
            self.model = WatsonxLLM(
                model_id=model_id, url=api_url, apikey=api_key, project_id=project_id, params=parameters
            )
        else:
            self.model = lc_lite_llm(
                model_id=model_id,
                print_prompt=print_prompt,
                model_url_id=model_url_id,
                temperature=temperature,
                top_p=top_p,
                n=n,
                min_tokens=min_tokens,
                max_tokens=max_tokens,
                max_retries=max_retries,
            )

    def generate(self, input: List[Any]) -> dict:
        result = {
            "llm_response": "",
            "token_usage": {"input_token_count": 0, "generated_token_count": 0, "total_token_count": 0},
        }
        is_watsonx = parsebool(os.environ.get("USE_WATSONX", "False"))
        if is_watsonx:
            response = self.model.generate(input)
            result["llm_response"] = response.generations[0][0].text
            result["token_usage"]["input_token_count"] = int(
                response.llm_output["token_usage"]["input_token_count"]
            )
            result["token_usage"]["generated_token_count"] = int(
                response.llm_output["token_usage"]["generated_token_count"]
            )
            result["token_usage"]["total_token_count"] = (
                result["token_usage"]["input_token_count"]
                + result["token_usage"]["generated_token_count"]
            )
        else:
            response = self.model.invoke(input)
            result["llm_response"] = response.content
            result["token_usage"]["input_token_count"] = int(
                response.response_metadata["token_usage"]["prompt_tokens"]
            )
            result["token_usage"]["generated_token_count"] = int(
                response.response_metadata["token_usage"]["completion_tokens"]
            )
            result["token_usage"]["total_token_count"] = int(
                response.response_metadata["token_usage"]["total_tokens"]
            )
        return result


class LCLITELLM(ChatLiteLLM):
    print_mcac_prompt: bool = False

    def completion_with_retry(
        self, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any
    ) -> Any:
        retry_decorator = _create_retry_decorator(self, run_manager=run_manager)

        @retry_decorator
        def _completion_with_retry(**kwargs: Any) -> Any:
            is_watsonx = parsebool(os.environ.get("USE_WATSONX", "False"))
            if is_watsonx:
                if os.environ.get("WATSONX_APIKEY", None) is None:
                    raise Exception("WATSONX_APIKEY must be provided.")
                kwargs["api_key"] = os.environ["WATSONX_APIKEY"]
                kwargs["project_id"] = os.environ["WATSONX_PROJECT_ID"]
            else:
                if os.environ.get("RITS_API_KEY", None) is None:
                    raise Exception("RITS_API_KEY must be provided.")
                kwargs["extra_headers"] = {"RITS_API_KEY": os.environ["RITS_API_KEY"]}
                kwargs["api_key"] = os.environ["RITS_API_KEY"]
            return self.client.completion(**kwargs)

        return _completion_with_retry(**kwargs)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        if self.print_mcac_prompt:
            print("#" * 60)
            print(messages[0].content)
            print("#" * 60)
        return super(LCLITELLM, self)._generate(messages=messages, stop=stop, run_manager=run_manager, **kwargs)


def lc_lite_llm(
    model_id="ibm/granite-20b-code-instruct",
    print_prompt=False,
    model_url_id=None,
    temperature=0,
    top_p=None,
    n=1,
    min_tokens=1,
    max_tokens=None,
    max_retries=2,
):
    is_watsonx = parsebool(os.environ.get("USE_WATSONX", "False"))
    if is_watsonx:
        api_url = os.environ["WATSONX_URL"]
        model = "watsonx/{}".format(model_id)
    else:
        if model_url_id is None:
            model_url_id = model_id.split("/")[-1].replace(".", "-")
        api_url = "{}/{}/v1".format(os.environ["RITS_URL"], model_url_id)
        model = "openai/{}".format(model_id)

    rits_model = LCLITELLM(
        model=model,
        api_base=api_url,
        temperature=temperature,
        top_p=top_p,
        n=n,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        max_retries=max_retries,
    )
    if print_prompt:
        rits_model.print_mcac_prompt = True
    return rits_model


# ─────────────────────────────────────────────────────────────────────────────
# RitsChatClient: preserves conversation history across multiple `send(...)` calls
# ─────────────────────────────────────────────────────────────────────────────
class RitsChatClient:
    def __init__(self, temperature=0.5, top_p=1.0, max_tokens: int = 256):
        model_id, model_url_id = MODELMAP.get_wpa_model_details()
        self.selector = LLMSelector(
            model_id=model_id,
            model_url_id=model_url_id,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        self.history: List[str] = []

    def reset(self):
        """Clear conversation history."""
        self.history = []

    # START OF CORRECTION
    def send(
        self,
        user_message: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> (str, int):
        """
        Sends a message to the LLM, managing history and allowing temporary
        overrides for temperature and max_tokens.
        """
        if len(self.history) > 10:
             self.history = self.history[-10:]

        self.history.append(f"User: {user_message}")
        combined = "\n\n".join(self.history)
        
        # Store original model parameters
        original_max_tokens = self.selector.model.max_tokens
        original_temperature = self.selector.model.temperature

        try:
            # Temporarily override parameters if new values are provided
            if max_tokens is not None:
                self.selector.model.max_tokens = max_tokens
            if temperature is not None:
                self.selector.model.temperature = temperature

            # Generate the response
            out = self.selector.generate([combined])
            text = out["llm_response"]
            tok = out["token_usage"]["total_token_count"]
            self.history.append(f"Assistant: {text}")
            return text, tok
        finally:
            # IMPORTANT: Restore original parameters to avoid affecting subsequent calls
            self.selector.model.max_tokens = original_max_tokens
            self.selector.model.temperature = original_temperature
    # END OF CORRECTION