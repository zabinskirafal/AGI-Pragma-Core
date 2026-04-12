"""
NIM LLMActor
============
LLMActor subclass that targets an NVIDIA NIM endpoint instead of the
Anthropic API.  NIM exposes an OpenAI-compatible REST interface, so the
implementation uses the ``openai`` SDK with a custom base URL.

Environment variables
---------------------
NIM_BASE_URL   Base URL of the NIM endpoint.
               Default: http://localhost:8000/v1
NIM_MODEL      Model ID to request from the endpoint.
               Default: nvidia/llama-3.1-nemotron-70b-instruct
NIM_API_KEY    API key forwarded in the Authorization header.
               Default: "nim" (NIM local deployments accept any non-empty value;
               set a real key for hosted endpoints such as api.nvcf.nvidia.com)

Usage
-----
    # local NIM container
    python -m demos.dic_llm.run --task "Create a project plan"

    # hosted NIM (build.nvidia.com / NVCF)
    NIM_BASE_URL=https://integrate.api.nvidia.com/v1 \\
    NIM_API_KEY=nvapi-... \\
    NIM_MODEL=nvidia/llama-3.1-nemotron-70b-instruct \\
    python -m demos.dic_llm.run --task "Create a project plan"

Alternatively, import directly::

    from demos.dic_llm.llm_actor_nim import NIMLLMActor
    actor = NIMLLMActor()
"""

import json
import os
from typing import Optional

from .llm_actor  import LLMActor, _SYSTEM_PROMPT
from .file_action import FileAction, FileOp

# ── Defaults ─────────────────────────────────────────────────────────────── #

_DEFAULT_BASE_URL = "http://localhost:8000/v1"
_DEFAULT_MODEL    = "nvidia/llama-3.1-nemotron-70b-instruct"
_DEFAULT_API_KEY  = "nim"   # NIM local deployments accept any non-empty value


class NIMLLMActor(LLMActor):
    """
    LLMActor that routes requests to an NVIDIA NIM endpoint.

    The NIM API is OpenAI-compatible, so this class replaces the Anthropic
    client with ``openai.OpenAI`` pointed at the NIM base URL.  Everything
    else — conversation history, feedback loop, JSON parsing — is inherited
    unchanged from LLMActor.

    Parameters
    ----------
    model : str | None
        Model ID to request.  Falls back to the ``NIM_MODEL`` env var, then
        the compiled-in default (``nvidia/llama-3.1-nemotron-70b-instruct``).
    base_url : str | None
        NIM endpoint base URL.  Falls back to ``NIM_BASE_URL`` env var, then
        ``http://localhost:8000/v1``.
    api_key : str | None
        API key.  Falls back to ``NIM_API_KEY`` env var, then ``"nim"``.
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
        # Resolve configuration from args → env → defaults
        self.model      = model    or os.environ.get("NIM_MODEL",    _DEFAULT_MODEL)
        self._base_url  = base_url or os.environ.get("NIM_BASE_URL", _DEFAULT_BASE_URL)
        self._api_key   = api_key  or os.environ.get("NIM_API_KEY",  _DEFAULT_API_KEY)
        self._max_tokens = max_tokens
        self.messages: list = []

        # Import openai here so the rest of the codebase has no hard dep on it
        try:
            import openai
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required for NIMLLMActor.\n"
                "Install it with:  pip install openai"
            ) from exc

        self._openai_client = openai.OpenAI(
            base_url=self._base_url,
            api_key=self._api_key,
        )

    # ── Overrides ─────────────────────────────────────────────────────── #

    def propose_action(self) -> FileAction:
        """
        Request the next action from the NIM endpoint.

        Sends the full conversation history as an OpenAI-style messages array
        with the system prompt prepended.  Parses the assistant reply using
        the inherited ``_parse()`` method.
        """
        messages = [{"role": "system", "content": _SYSTEM_PROMPT}] + self.messages

        response = self._openai_client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self._max_tokens,
            temperature=0.0,    # deterministic — matches the Anthropic actor's intent
        )

        raw = response.choices[0].message.content.strip()

        # Append to history so subsequent turns see the full context
        self.messages.append({"role": "assistant", "content": raw})

        return self._parse(raw)

    # start_task() and feedback() are identical to LLMActor — inherited as-is.
    # _parse() is identical — inherited as-is.

    def __repr__(self) -> str:
        return (
            f"NIMLLMActor(model={self.model!r}, "
            f"base_url={self._base_url!r})"
        )
