import os
from typing import Any, List, Optional
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.outputs import LLMResult
from langchain_core.messages import BaseMessage
from langchain_ibm import WatsonxLLM
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods



MODEL_ID_MAP = {
    "watsonx": {
        "granite": {
            "model_id": "ibm/granite-3-2-8b-instruct",
            "model_url_id": None,
        },
        "llama_4": {
            "model_id": "meta-llama/llama-4-maverick-17b-128e-instruct-fp8",
            "model_url_id": None,
        },
        "mistral": {
            "model_id": "mistralai/mistral-large",
            "model_url_id": None,
        },
    },
    "rits": {
        "granite": {
            "model_id": "ibm-granite/granite-3.3-8b-instruct",
            "model_url_id": "granite-3-3-8b-instruct",
        },
        "llama_3": {
            "model_id": "meta-llama/llama-3-1-405b-instruct-fp8",
            "model_url_id": "llama-3-1-405b-instruct-fp8",
        },
        "llama_4": {
            "model_id": "meta-llama/llama-4-maverick-17b-128e-instruct-fp8",
            "model_url_id": "llama-4-mvk-17b-128e-fp8",
        },
        "phi": {"model_id": "microsoft/phi-4", "model_url_id": "microsoft-phi-4"},
    },
}

def parsebool(val):
    bool_map = {
        "y": True,
        "true": True,
        "t": True,
        "yes": True,
        "n": False,
        "false": False,
        "f": False,
        "no": False,
    }

    return bool_map.get(str(val).lower(), False)

class MODELMAP:
    er_model = "granite"
    generate_model = "granite"
    review_model = "granite"
    explain_model = "granite"

    @classmethod
    def set_model(cls, model_type: str, model_name: str):
        is_watsonx = parsebool(os.environ.get("USE_WATSONX", False))
        VALID_TYPES = ["er_model", "generate_model", "review_model", "explain_model"]
        VALID_MODELS = list(MODEL_ID_MAP["watsonx"].keys())
        if not is_watsonx:
            VALID_MODELS = list(MODEL_ID_MAP["rits"].keys())

        if model_name not in VALID_MODELS:
            raise ValueError(
                f"Invalid model name: {model_name}. Choose from {VALID_MODELS}"
            )
        if model_type not in VALID_TYPES:
            raise ValueError(
                f"Invalid model type: {model_type}. Choose from {VALID_TYPES}"
            )
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
        temperature=0,  # default is greedy sampling
        top_p=None,
        n=1,  # this should always be 1 for langchain's chain.
        min_tokens=1,
        max_tokens=None,
        max_retries=2,
    ):
        is_watsonx = parsebool(os.environ.get("USE_WATSONX", "False"))
        if is_watsonx:
            api_url = "{}".format(os.environ.get("WATSONX_URL"))
            api_key = os.environ.get("WATSONX_APIKEY")
            project_id = os.environ.get("WATSONX_PROJECT_ID")

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
                model_id=model_id,
                url=api_url,
                apikey=api_key,
                project_id=project_id,
                params=parameters,
            )
        else:
            self.model = lc_lite_llm(
                model_id=model_id,
                print_prompt=print_prompt,
                model_url_id=model_url_id,
                temperature=temperature,  # default is greedy sampling
                top_p=top_p,
                n=n,  # this should always be 1 for langchain's chain.
                min_tokens=min_tokens,
                max_tokens=max_tokens,
                max_retries=max_retries,
            )

    def generate(self, input):
        result = {
            "llm_response": "",
            "token_usage": {
                "input_token_count": 0,
                "generated_token_count": 0,
                "total_token_count": 0,
            },
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


# if not os.environ.get("USE_WATSONX", True):
import litellm
from langchain_community.chat_models import ChatLiteLLM
from langchain_community.chat_models.litellm import _create_retry_decorator


class LCLITELLM(ChatLiteLLM):
    litellm.set_verbose = True
    print_mcac_prompt: bool = False

    def completion_with_retry(
        self, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any
    ) -> Any:
        """Use tenacity to retry the completion call."""
        retry_decorator = _create_retry_decorator(self, run_manager=run_manager)
        # os.environ["LITELLM_LOG"] = "True"

        @retry_decorator
        def _completion_with_retry(**kwargs: Any) -> Any:
            is_watsonx = parsebool(os.environ.get("USE_WATSONX", "False"))
            if is_watsonx:
                if os.environ.get("WATSONX_APIKEY", None) is None:
                    raise Exception(
                        "env variable WATSONX_APIKEY must be provided to use RITS service."
                    )
                kwargs["api_key"] = os.environ.get("WATSONX_APIKEY")
                kwargs["project_id"] = os.environ.get("WATSONX_PROJECT_ID")

            else:
                if os.environ.get("RITS_API_KEY", None) is None:
                    raise Exception(
                        "env variable RITS_API_KEY must be provided to use RITS service."
                    )
                kwargs["extra_headers"] = {"RITS_API_KEY": os.environ["RITS_API_KEY"]}
                kwargs["api_key"] = os.environ.get("RITS_API_KEY")

            return self.client.completion(**kwargs)

        return _completion_with_retry(**kwargs)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:

        if self.print_mcac_prompt:

            print("#" * 100)
            # print(type(messages[0].content))
            # print(messages[0].content)
            print("#" * 100)

        return super(LCLITELLM, self)._generate(
            messages=messages, stop=stop, run_manager=run_manager, **kwargs
        )


def lc_lite_llm(
    model_id="ibm/granite-20b-code-instruct",
    print_prompt=False,
    model_url_id=None,
    temperature=0,  # default is greedy sampling
    top_p=None,
    n=1,  # this should always be 1 for langchain's chain.
    min_tokens=1,
    max_tokens=None,
    max_retries=2,
):
    is_watsonx = parsebool(os.environ.get("USE_WATSONX", "False"))
    model = "openai/{}".format(model_id)
    if is_watsonx:
        api_url = "{}".format(os.environ.get("WATSONX_URL"))
        model = "watsonx/{}".format(model_id)
    else:
        api_url = "{}/{}/v1".format(os.environ.get("RITS_URL"), model_url_id)
        if model_url_id is None:
            model_url_id = model_id.split("/")[-1]
            model_url_id = model_url_id.replace(".", "-")

    model = LCLITELLM(
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
        model.print_mcac_prompt = True
    return model