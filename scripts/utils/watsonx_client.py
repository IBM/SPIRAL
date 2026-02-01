# File: utilities/watsonx_client.py

import os
from ibm_watsonx_ai.foundation_models.schema import TextChatParameters
from langchain_ibm import ChatWatsonx
from langchain.schema import SystemMessage, HumanMessage, AIMessage


class WatsonxChatClient:
    """
    A simple wrapper around ChatWatsonx that keeps track of conversation history.
    """

    # Hardcoded Watsonx.ai endpoint, project ID, and API key
    _URL = os.getenv("WATSONX_URL")
    _PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")
    _APIKEY = os.getenv("WATSONX_APIKEY")  # Replace with your actual key

    def __init__(
        self,
        model_id: str,
        system_prompt: str = "You are a helpful assistant.",
        max_tokens: int = 200,
        temperature: float = 0.5,
        top_p: float = 1.0,
    ):
        """
        Initializes the WatsonxChatClient.

        Parameters:
        - model_id:      The Watsonx.ai model ID (e.g. "ibm/granite-3-8b-instruct").
        - system_prompt: The initial “system” message for every conversation.
        - max_tokens:    Max tokens to produce per response.
        - temperature:   Sampling temperature.
        - top_p:         Nucleus sampling parameter.
        """
        parameters = TextChatParameters(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        self._chat = ChatWatsonx(
            model_id=model_id,
            url=self._URL,
            project_id=self._PROJECT_ID,
            apikey=self._APIKEY,
            params=parameters,
        )

        self._conversation = [SystemMessage(content=system_prompt)]

    def send(self, user_text: str) -> str:
        """
        Appends the given user_text as a HumanMessage to the conversation,
        calls Watsonx.invoke(...), and returns the assistant’s reply.
        """
        self._conversation.append(HumanMessage(content=user_text))
        ai_message: AIMessage = self._chat.invoke(input=self._conversation)
        self._conversation.append(ai_message)
        return ai_message.content

    def reset(self, system_prompt: str = None):
        """
        Clears the current conversation history. Optionally override the system prompt.
        """
        if system_prompt is not None:
            self._conversation = [SystemMessage(content=system_prompt)]
        else:
            original_system = self._conversation[0].content
            self._conversation = [SystemMessage(content=original_system)]
