"""
DIC + LLM Agent Demo
====================
An LLM proposes file operations to complete a task.
DIC evaluates every proposed action through the full 7-stage pipeline
before anything touches the filesystem.

Usage:
    python -m demos.dic_llm.run
    python -m demos.dic_llm.run --task "Create a shopping list and a budget file"
    python -m demos.dic_llm.run --actor nim    --model nvidia/llama-3.1-nemotron-70b-instruct
    python -m demos.dic_llm.run --actor azure  --model gpt-4o
    python -m demos.dic_llm.run --actor ollama --model llama3.1:70b
    python -m demos.dic_llm.run --actor groq   --model llama-3.1-70b-versatile
    python -m demos.dic_llm.run --mock --scenario escalate
"""

import argparse
import json
import sys
import textwrap
import time
from pathlib import Path

from .llm_actor        import LLMActor
from .llm_actor_nim    import NIMLLMActor
from .llm_actor_azure  import AzureLLMActor
from .llm_actor_ollama import OllamaLLMActor
from .llm_actor_groq   import GroqLLMActor
from .mock_actor       import MockActor
from .dic_governor     import DICGovernor, DICDecision
from .executor         import Executor
from .file_action      import FileAction, FileOp

# ── Actor factory ────────────────────────────────────────────────────── #

_ACTORS = ("claude", "nim", "azure", "ollama", "groq")

def _build_actor(actor: str, model: str, scenario: str, mock: bool):
    """Instantiate the correct LLMActor subclass for *actor*."""
    if mock:
        return MockActor(scenario=scenario)
    if actor == "claude":
        return LLMActor(model=model)
    if actor == "nim":
        return NIMLLMActor(model=model or None)
    if actor == "azure":
        return AzureLLMActor(model=model or None)
    if actor == "ollama":
        return OllamaLLMActor(model=model or None)
    if actor == "groq":
        return GroqLLMActor(model=model or None)
    raise ValueError(f"Unknown actor {actor!r}. Choose from: {_ACTORS}")

# ── ANSI colours ────────────────────────────────────────────────────── #
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

W = 62   # box width


# ── Formatting helpers ───────────────────────────────────────────────── #

def _box_line(text: str = "", colour: str = "") -> str:
    padded = f" {text:<{W-2}}"
    return f"║{colour}{padded}{RESET if colour else ''}║"

def _hdr(label: str, colour: str = CYAN) -> str:
    bar = "═" * W
    return f"{colour}╔{bar}╗\n{_box_line(label, BOLD)}\n╠{bar}╣{RESET}"

def _sep() -> str:
    return f"{DIM}╟{'─'*W}╢{RESET}"

def _footer() -> str:
    return f"{DIM}╚{'═'*W}╝{RESET}"


def print_decision(step: int, action: FileAction, d: DICDecision) -> None:
    verdict_colour = GREEN if d.approved else RED
    verdict_label  = f"{'✓ YES' if d.approved else '✗ NO '}  (RPN {d.max_rpn})"

    print()
    print(_hdr(f"Step {step}  │  DIC DECISION", verdict_colour))

    # Action summary
    print(_box_line(f"Op:       {action.op.value.upper()}"))
    print(_box_line(f"Path:     {action.path}"))
    if action.content:
        preview = action.content[:60].replace("\n", "↵")
        print(_box_line(f"Content:  {preview}{'…' if len(action.content)>60 else ''}"))
    print(_box_line(f"Reason:   {action.reason}"))
    print(_sep())

    # Verdict
    verdict_line = f"{BOLD}{verdict_colour}{verdict_label}{RESET}"
    print(f"║ Decision: {verdict_line:<{W+len(BOLD)+len(verdict_colour)+len(RESET)-2}}║")
    if not d.approved:
        wrapped = textwrap.wrap(d.block_reason or "", W - 4)
        for line in wrapped:
            print(_box_line(f"  Reason:  {line}", RED))
    print(_sep())

    # Audit trace — one line per stage
    print(_box_line("AUDIT TRACE", DIM))
    for entry in d.stage_log:
        stage = entry["stage"]

        if stage == "branching":
            icon = "✓" if entry["pass"] else "✗"
            print(_box_line(f"  1. Branching       {icon}  {entry['detail'][:38]}"))

        elif stage == "critical_path":
            rev  = entry["reversibility"].upper()
            p    = entry["p_irreversible"]
            fx   = "exists" if entry["file_exists"] else "new"
            print(_box_line(f"  2. Critical Path      rev={rev:<8} p_irrev={p:.2f}  [{fx}]"))
            for se in entry.get("side_effects", []):
                w = textwrap.wrap(se, W - 10)
                for ln in w:
                    print(_box_line(f"       ↳ {ln}", DIM))

        elif stage == "fmea":
            print(_box_line(f"  3. FMEA               max_rpn={entry['max_rpn']}"))
            for fname, fdata in entry.get("table", {}).items():
                rpn = fdata.get("rpn", "?")
                s   = fdata.get("severity", "?")
                o   = fdata.get("occurrence", "?")
                det = fdata.get("detection", "?")
                r   = fdata.get("reversibility", "?")
                print(_box_line(
                    f"       {fname:<24} S={s} O={o} D={det} R={r}  RPN={rpn}", DIM
                ))

        elif stage == "decision_gate":
            icon = "✗ BLOCKED" if entry["blocked"] else "✓ PASS"
            print(_box_line(f"  4. Decision Gate   {icon}  ({entry['max_rpn']} vs {entry['threshold']})"))

        elif stage == "circuit_breaker":
            state = entry["state"].upper()
            if state == "OK":
                colour = GREEN
            elif state in ("WARN", "SLOW"):
                colour = YELLOW
            else:                          # STOP or ESCALATE
                colour = RED
            escalate_marker = f"  {BOLD}⚠ HUMAN CONFIRMATION REQUIRED{RESET}" if state == "ESCALATE" else ""
            print(_box_line(f"  5. Circuit Breaker    {colour}{state}{RESET}  {entry['reason']}"))
            if escalate_marker:
                print(_box_line(f"     {BOLD}{RED}⚠ ESCALATE — human confirmation required{RESET}", RED))

        elif stage == "utility":
            print(_box_line(f"  6. Utility            score={entry['score']:.3f}"))

        elif stage == "belief_update":
            mean = entry["llm_risk_mean"]
            bar  = int(mean * 20) * "▓" + (20 - int(mean * 20)) * "░"
            print(_box_line(f"  7. Belief Update      llm_risk_mean={mean:.3f}  {bar}"))

    print(_footer())


def print_sandbox_listing(sandbox: Path) -> None:
    files = sorted(sandbox.rglob("*"))
    if not files:
        print(f"\n{DIM}  (sandbox is empty){RESET}")
        return
    print(f"\n{CYAN}  Sandbox contents:{RESET}")
    for f in files:
        if f.is_file():
            size = f.stat().st_size
            rel  = f.relative_to(sandbox)
            print(f"    {rel}  {DIM}({size} bytes){RESET}")


# ── Main loop ────────────────────────────────────────────────────────── #

def run(task: str, max_steps: int = 15, model: str = "claude-haiku-4-5-20251001",
        mock: bool = False, scenario: str = "default",
        actor: str = "claude") -> None:
    sandbox  = Path(__file__).parent / "sandbox"
    sandbox.mkdir(exist_ok=True)

    actor_label = f"[mock:{scenario}]" if mock else actor
    print(f"\n{BOLD}{CYAN}{'═'*64}{RESET}")
    print(f"{BOLD}{CYAN}  DIC + LLM Agent Demo{RESET}")
    print(f"{BOLD}{CYAN}{'═'*64}{RESET}")
    print(f"  Task:      {task}")
    print(f"  Actor:     {actor_label}")
    print(f"  Sandbox:   {sandbox}")
    print(f"  Max steps: {max_steps}")

    actor_obj = _build_actor(actor, model, scenario, mock)
    if not mock:
        print(f"  Model:     {getattr(actor_obj, 'model', model)}")
    governor  = DICGovernor()
    executor  = Executor(sandbox_root=sandbox)

    actor_obj.start_task(task)

    for step in range(1, max_steps + 1):
        # ── LLM proposes next action ──────────────────────────────── #
        try:
            action = actor_obj.propose_action()
        except ValueError as e:
            print(f"\n{RED}  LLM parse error: {e}{RESET}")
            print(f"  {DIM}Sending error feedback and retrying…{RESET}")
            actor_obj.feedback(
                FileAction(FileOp.READ, "", None, ""),
                approved=False,
                result=None,
                block_reason=f"Malformed JSON response: {e}",
            )
            continue

        # ── DONE signal ───────────────────────────────────────────── #
        if action.op == FileOp.DONE:
            print(f"\n{BOLD}{GREEN}  ✓ Task complete (LLM signalled DONE at step {step}){RESET}")
            print_sandbox_listing(sandbox)
            return

        # ── DIC evaluates ─────────────────────────────────────────── #
        decision = governor.evaluate(action)
        print_decision(step, action, decision)

        # ── Execute if approved ───────────────────────────────────── #
        result: str | None = None
        if decision.approved:
            try:
                result = executor.execute(action)
            except Exception as e:
                print(f"{RED}  Executor error: {e}{RESET}")
                result = None

        # ── Feed result back to LLM ───────────────────────────────── #
        actor_obj.feedback(
            action,
            approved=decision.approved,
            result=result,
            block_reason=decision.block_reason,
        )

        # Small delay so output is readable in terminals
        time.sleep(0.3)

    print(f"\n{YELLOW}  Max steps ({max_steps}) reached without DONE signal.{RESET}")
    print_sandbox_listing(sandbox)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DIC + LLM agent demo")
    parser.add_argument(
        "--task",
        default=(
            "Create a file called 'notes.txt' with three bullet points about AI safety. "
            "Then create a file called 'summary.txt' that summarises notes.txt in one sentence."
        ),
    )
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--model",               default="claude-haiku-4-5-20251001",
                        help="Model ID passed to the actor (overrides actor default)")
    parser.add_argument("--actor",               default="claude",
                        choices=_ACTORS,
                        help="LLM backend: claude (default), nim, azure, ollama, groq")
    parser.add_argument("--mock",      action="store_true",
                        help="Use scripted mock actor (no API key required)")
    parser.add_argument("--scenario",  default="default",
                        choices=["default", "escalate"],
                        help="Mock scenario: 'default' or 'escalate'")
    args = parser.parse_args()

    run(task=args.task, max_steps=args.max_steps, model=args.model,
        mock=args.mock, scenario=args.scenario, actor=args.actor)
