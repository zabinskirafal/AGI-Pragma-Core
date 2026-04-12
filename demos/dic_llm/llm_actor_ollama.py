"""
Ollama LLMActor
===============
LLMActor subclass that targets a local Ollama server.
Ollama exposes an OpenAI-compatible REST interface at /v1, so the
implementation uses the ``openai`` SDK with a custom base URL.

No API key is required — Ollama accepts any non-empty string.

Environment variables
---------------------
OLLAMA_BASE_URL   Base URL of the Ollama server.
                  Default: http://localhost:11434/v1
OLLAMA_MODEL      Model tag to request.
                  Default: llama3.1:70b

Usage
-----
    # default model
    python -m demos.dic_llm.run --task "Create a project plan"

    # different model
    OLLAMA_MODEL=mistral:7b python -m demos.dic_llm.run --task "..."

Or import directly::

    from demos.dic_llm.llm_actor_ollama import OllamaLLMActor
    actor = OllamaLLMActor()
    actor = OllamaLLMActor(model="mistral:7b")
"""

import os

from .llm_actor   import LLMActor, _SYSTEM_PROMPT
from .file_action import FileAction

# ── Defaults ─────────────────────────────────────────────────────────────── #

_DEFAULT_BASE_URL = "http://localhost:11434/v1"
_DEFAULT_MODEL    = "llama3.1:70b"
_OLLAMA_API_KEY   = "ollama"   # Ollama ignores the key; openai SDK requires non-empty


class OllamaLLMActor(LLMActor):
    """
    LLMActor that routes requests to a local Ollama server.

    Ollama's OpenAI-compatible endpoint is used via ``openai.OpenAI`` with
    a custom base URL.  No API key is required; the placeholder value
    ``"ollama"`` is sent to satisfy the SDK's non-empty requirement.

    Conversation history, feedback loop, and JSON parsing are inherited
    unchanged from LLMActor.

    Parameters
    ----------
    model : str | None
        Ollama model tag (e.g. ``"llama3.1:70b"``, ``"mistral:7b"``).
        Falls back to ``OLLAMA_MODEL`` env var, then ``llama3.1:70b``.
    base_url : str | None
        Ollama server base URL.  Falls back to ``OLLAMA_BASE_URL`` env var,
        then ``http://localhost:11434/v1``.
    max_tokens : int
        Maximum tokens in each completion response (default 512).
    """

    def __init__(
        self,
        model:      str | None = None,
        base_url:   str | None = None,
        max_tokens: int        = 512,
    ) -> None:
        self.model       = model    or os.environ.get("OLLAMA_MODEL",    _DEFAULT_MODEL)
        self._base_url   = base_url or os.environ.get("OLLAMA_BASE_URL", _DEFAULT_BASE_URL)
        self._max_tokens = max_tokens
        self.messages: list = []

        try:
            import openai
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required for OllamaLLMActor.\n"
                "Install it with:  pip install openai"
            ) from exc

        self._openai_client = openai.OpenAI(
            base_url=self._base_url,
            api_key=_OLLAMA_API_KEY,
        )

    # ── Override ──────────────────────────────────────────────────────── #

    def propose_action(self) -> FileAction:
        """
        Request the next action from the Ollama server.

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
            f"OllamaLLMActor(model={self.model!r}, "
            f"base_url={self._base_url!r})"
        )
