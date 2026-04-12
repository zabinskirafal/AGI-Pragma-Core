"""
Azure AI Foundry LLMActor
=========================
LLMActor subclass that targets an Azure AI Foundry endpoint.
Azure AI Foundry exposes an OpenAI-compatible REST interface, so the
implementation uses the ``openai`` SDK with a custom base URL.

Environment variables
---------------------
AZURE_BASE_URL   Base URL of the Azure AI Foundry endpoint (required).
                 Example: https://<resource>.openai.azure.com/openai/deployments/<deployment>
AZURE_MODEL      Model / deployment name to request.
                 Default: gpt-4o
AZURE_API_KEY    API key for the endpoint (required).

Usage
-----
    AZURE_BASE_URL=https://my-resource.openai.azure.com/openai/deployments/my-gpt4o \\
    AZURE_API_KEY=<key> \\
    python -m demos.dic_llm.run --task "Create a project plan"

Or import directly::

    from demos.dic_llm.llm_actor_azure import AzureLLMActor
    actor = AzureLLMActor()
"""

import os
from typing import Optional

from .llm_actor   import LLMActor, _SYSTEM_PROMPT
from .file_action import FileAction

# ── Defaults ─────────────────────────────────────────────────────────────── #

_DEFAULT_MODEL = "gpt-4o"


class AzureLLMActor(LLMActor):
    """
    LLMActor that routes requests to an Azure AI Foundry endpoint.

    The Azure AI Foundry API is OpenAI-compatible, so this class replaces
    the Anthropic client with ``openai.OpenAI`` pointed at the configured
    Azure endpoint.  Conversation history, feedback loop, and JSON parsing
    are inherited unchanged from LLMActor.

    Parameters
    ----------
    model : str | None
        Deployment / model name.  Falls back to ``AZURE_MODEL`` env var,
        then ``gpt-4o``.
    base_url : str | None
        Azure AI Foundry endpoint base URL.  Falls back to ``AZURE_BASE_URL``
        env var.  Raises ``EnvironmentError`` if neither is provided.
    api_key : str | None
        Azure API key.  Falls back to ``AZURE_API_KEY`` env var.  Raises
        ``EnvironmentError`` if neither is provided.
    max_tokens : int
        Maximum tokens in each completion response (default 512).
    """

    def __init__(
        self,
        model:      str | None = None,
        base_url:   str | None = None,
        api_key:    str | None = None,
        max_tokens: int        = 512,
    ) -> None:
        self.model       = model    or os.environ.get("AZURE_MODEL",   _DEFAULT_MODEL)
        self._base_url   = base_url or os.environ.get("AZURE_BASE_URL")
        self._api_key    = api_key  or os.environ.get("AZURE_API_KEY")
        self._max_tokens = max_tokens
        self.messages: list = []

        if not self._base_url:
            raise EnvironmentError(
                "AZURE_BASE_URL is not set.\n"
                "Export the Azure AI Foundry endpoint URL, e.g.:\n"
                "  export AZURE_BASE_URL=https://<resource>.openai.azure.com/"
                "openai/deployments/<deployment>"
            )
        if not self._api_key:
            raise EnvironmentError(
                "AZURE_API_KEY is not set.\n"
                "Export your Azure API key:\n"
                "  export AZURE_API_KEY=<your-key>"
            )

        try:
            import openai
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required for AzureLLMActor.\n"
                "Install it with:  pip install openai"
            ) from exc

        self._openai_client = openai.OpenAI(
            base_url=self._base_url,
            api_key=self._api_key,
        )

    # ── Override ──────────────────────────────────────────────────────── #

    def propose_action(self) -> FileAction:
        """
        Request the next action from the Azure AI Foundry endpoint.

        Sends the full conversation history as an OpenAI-style messages array
        with the system prompt prepended.  Parses the assistant reply with
        the inherited ``_parse()`` method.
        """
        messages = [{"role": "system", "content": _SYSTEM_PROMPT}] + self.messages

        response = self._openai_client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self._max_tokens,
            temperature=0.0,
        )

        raw = response.choices[0].message.content.strip()
        self.messages.append({"role": "assistant", "content": raw})
        return self._parse(raw)

    # start_task(), feedback(), and _parse() inherited from LLMActor.

    def __repr__(self) -> str:
        return (
            f"AzureLLMActor(model={self.model!r}, "
            f"base_url={self._base_url!r})"
        )
