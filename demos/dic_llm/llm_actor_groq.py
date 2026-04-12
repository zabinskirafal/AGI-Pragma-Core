"""
Groq LLMActor
=============
LLMActor subclass that targets the Groq API.
Groq exposes an OpenAI-compatible REST interface, so the implementation
uses the ``openai`` SDK pointed at https://api.groq.com/openai/v1.

Environment variables
---------------------
GROQ_API_KEY   Groq API key (required).
               Obtain from: https://console.groq.com/keys
GROQ_MODEL     Model ID to request.
               Default: llama-3.1-70b-versatile

Usage
-----
    export GROQ_API_KEY=gsk_...
    python -m demos.dic_llm.run --task "Create a project plan"

    # different model
    GROQ_MODEL=mixtral-8x7b-32768 python -m demos.dic_llm.run --task "..."

Or import directly::

    from demos.dic_llm.llm_actor_groq import GroqLLMActor
    actor = GroqLLMActor()
"""

import os

from .llm_actor   import LLMActor, _SYSTEM_PROMPT
from .file_action import FileAction

# ── Constants ─────────────────────────────────────────────────────────────── #

BASE_URL      = "https://api.groq.com/openai/v1"
_DEFAULT_MODEL = "llama-3.1-70b-versatile"


class GroqLLMActor(LLMActor):
    """
    LLMActor that routes requests to the Groq API.

    Uses ``openai.OpenAI`` with Groq's base URL and a required API key.
    Conversation history, feedback loop, and JSON parsing are inherited
    unchanged from LLMActor.

    Parameters
    ----------
    model : str | None
        Groq model ID.  Falls back to ``GROQ_MODEL`` env var, then
        ``llama-3.1-70b-versatile``.
    api_key : str | None
        Groq API key.  Falls back to ``GROQ_API_KEY`` env var.  Raises
        ``EnvironmentError`` if neither is provided.
    max_tokens : int
        Maximum tokens in each completion response (default 512).
    """

    def __init__(
        self,
        model:      str | None = None,
        api_key:    str | None = None,
        max_tokens: int        = 512,
    ) -> None:
        self.model       = model   or os.environ.get("GROQ_MODEL",   _DEFAULT_MODEL)
        self._api_key    = api_key or os.environ.get("GROQ_API_KEY")
        self._max_tokens = max_tokens
        self.messages: list = []

        if not self._api_key:
            raise EnvironmentError(
                "GROQ_API_KEY is not set.\n"
                "Obtain a key at https://console.groq.com/keys and export it:\n"
                "  export GROQ_API_KEY=gsk_..."
            )

        try:
            import openai
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required for GroqLLMActor.\n"
                "Install it with:  pip install openai"
            ) from exc

        self._openai_client = openai.OpenAI(
            base_url=BASE_URL,
            api_key=self._api_key,
        )

    # ── Override ──────────────────────────────────────────────────────── #

    def propose_action(self) -> FileAction:
        """
        Request the next action from the Groq API.

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
        return f"GroqLLMActor(model={self.model!r})"
