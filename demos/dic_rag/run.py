"""
DIC + RAG Demo
==============
Runs the standard mock scenario (READ → WRITE → WRITE → DELETE → DONE)
using RAGGovernor instead of plain DICGovernor.

For each action the output shows:
  • Retrieved policy chunks (section, source, similarity score)
  • RAG override applied (severity_delta, detection_delta, notes)
  • Full DIC audit trace including the adjusted FMEA RPNs

Usage:
    python -m demos.dic_rag.run
    python -m demos.dic_rag.run --scenario escalate
    python -m demos.dic_rag.run --rebuild-index
"""

import argparse
import textwrap
import time
from pathlib import Path

from .rag_indexer  import build_index
from .rag_governor import RAGGovernor
from ..dic_llm.mock_actor  import MockActor
from ..dic_llm.executor    import Executor
from ..dic_llm.file_action import FileAction, FileOp
from ..dic_llm.dic_governor import DICDecision

# ── ANSI ─────────────────────────────────────────────────────────────────── #
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
MAGENTA= "\033[95m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

W = 66


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


# ── RAG context printer ───────────────────────────────────────────────────── #

def _print_rag_stage(entry: dict) -> None:
    chunks   = entry.get("chunks", [])
    override = entry.get("override", {})

    print(_box_line("RAG CONTEXT", MAGENTA))

    if not chunks:
        print(_box_line("  (no chunks retrieved — DONE signal)", DIM))
    else:
        for i, c in enumerate(chunks, 1):
            score   = c.get("score", 0)
            section = c.get("section", "")[:44]
            source  = c.get("source", "")
            preview = c.get("text_preview", "")[:54]
            score_colour = GREEN if score >= 0.4 else YELLOW if score >= 0.2 else DIM
            print(_box_line(
                f"  [{i}] {score_colour}{score:.3f}{RESET}  [{source}]  {section}"
            ))
            print(_box_line(f"       {DIM}{preview}…{RESET}", ""))

    s_d  = override.get("severity_delta", 0)
    d_d  = override.get("detection_delta", 0)
    notes = override.get("notes", [])

    delta_colour = RED if (s_d > 0 or d_d > 0) else (GREEN if (s_d < 0 or d_d < 0) else DIM)
    print(_box_line(
        f"  Override: ΔSeverity={s_d:+d}  ΔDetection={d_d:+d}",
        delta_colour,
    ))
    for note in notes:
        wrapped = textwrap.wrap(note, W - 12)
        for ln in wrapped:
            print(_box_line(f"    ↳ {ln}", DIM))


# ── Block justification printer ──────────────────────────────────────────── #

def _print_justification(entry: dict) -> None:
    lines = entry.get("lines", [])
    if not lines:
        return

    bar = "═" * W
    print(f"  {RED}╔{bar}╗{RESET}")
    print(f"  {RED}║{BOLD} POLICY BLOCK JUSTIFICATION{' ' * (W - 26)}{RESET}{RED}║{RESET}")
    print(f"  {RED}╠{bar}╣{RESET}")
    for line in lines:
        # Wrap each justification line to fit the box
        wrapped = textwrap.wrap(line, W - 4)
        for i, ln in enumerate(wrapped):
            indent = "  " if i == 0 else "    "
            print(f"  {RED}║{RESET} {indent}{ln:<{W - len(indent) - 1}}{RED}║{RESET}")
        print(f"  {RED}║{' ' * W}║{RESET}")
    print(f"  {RED}╚{bar}╝{RESET}")


# ── Full decision printer ─────────────────────────────────────────────────── #

def print_decision(step: int, action: FileAction, d: DICDecision) -> None:
    verdict_colour = GREEN if d.approved else RED
    verdict_label  = f"{'✓ YES' if d.approved else '✗ NO '}  (RPN {d.max_rpn})"

    print()
    print(_hdr(f"Step {step}  │  DIC+RAG DECISION", verdict_colour))

    print(_box_line(f"Op:       {action.op.value.upper()}"))
    print(_box_line(f"Path:     {action.path}"))
    if action.content:
        preview = action.content[:60].replace("\n", "↵")
        print(_box_line(f"Content:  {preview}{'…' if len(action.content)>60 else ''}"))
    print(_box_line(f"Reason:   {action.reason}"))
    print(_sep())

    verdict_line = f"{BOLD}{verdict_colour}{verdict_label}{RESET}"
    print(f"║ Decision: {verdict_line:<{W+len(BOLD)+len(verdict_colour)+len(RESET)-2}}║")
    if not d.approved:
        for line in textwrap.wrap(d.block_reason or "", W - 4):
            print(_box_line(f"  Reason:  {line}", RED))
    print(_sep())

    print(_box_line("AUDIT TRACE", DIM))
    for entry in d.stage_log:
        stage = entry["stage"]

        if stage == "branching":
            icon = "✓" if entry["pass"] else "✗"
            print(_box_line(f"  1. Branching        {icon}  {entry['detail'][:40]}"))

        elif stage == "critical_path":
            rev = entry["reversibility"].upper()
            p   = entry["p_irreversible"]
            fx  = "exists" if entry["file_exists"] else "new"
            print(_box_line(f"  2. Critical Path       rev={rev:<8} p_irrev={p:.2f}  [{fx}]"))

        elif stage == "rag_context":
            _print_rag_stage(entry)

        elif stage == "fmea":
            print(_box_line(f"  3. FMEA (RAG-adjusted) max_rpn={entry['max_rpn']}"))
            for fname, fdata in entry.get("table", {}).items():
                rpn = fdata.get("rpn", "?")
                s   = fdata.get("severity", "?")
                o   = fdata.get("occurrence", "?")
                det = fdata.get("detection", "?")
                r   = fdata.get("reversibility", "?")
                print(_box_line(
                    f"       {fname:<26} S={s} O={o} D={det} R={r}  RPN={rpn}", DIM
                ))

        elif stage == "decision_gate":
            icon = "✗ BLOCKED" if entry["blocked"] else "✓ PASS"
            print(_box_line(f"  4. Decision Gate    {icon}  ({entry['max_rpn']} vs {entry['threshold']})"))

        elif stage == "circuit_breaker":
            state = entry["state"].upper()
            col   = GREEN if state == "OK" else (YELLOW if state in ("WARN", "SLOW") else RED)
            print(_box_line(f"  5. Circuit Breaker     {col}{state}{RESET}  {entry['reason']}"))
            if state == "ESCALATE":
                print(_box_line(f"     {BOLD}{RED}⚠ ESCALATE — human confirmation required{RESET}", RED))

        elif stage == "utility":
            print(_box_line(f"  6. Utility             score={entry['score']:.3f}"))

        elif stage == "belief_update":
            mean = entry["llm_risk_mean"]
            bar  = int(mean * 20) * "▓" + (20 - int(mean * 20)) * "░"
            print(_box_line(f"  7. Belief Update       llm_risk_mean={mean:.3f}  {bar}"))

        elif stage == "block_justification":
            _print_justification(entry)

    print(_footer())


# ── Main ──────────────────────────────────────────────────────────────────── #

def run(scenario: str = "default", rebuild_index: bool = False) -> None:
    sandbox = Path(__file__).parent.parent / "dic_llm" / "sandbox"
    sandbox.mkdir(exist_ok=True)

    print(f"\n{BOLD}{CYAN}{'═'*68}{RESET}")
    print(f"{BOLD}{CYAN}  DIC + RAG Demo{RESET}")
    print(f"{BOLD}{CYAN}{'═'*68}{RESET}")
    print(f"  Scenario:  {scenario}")
    print(f"  Sandbox:   {sandbox}")

    # Build / reuse Chroma index
    print(f"\n{DIM}  Building RAG index…{RESET}")
    build_index(rebuild=rebuild_index)

    actor    = MockActor(scenario=scenario)
    governor = RAGGovernor()
    executor = Executor(sandbox_root=sandbox)

    task = "Demonstrate RAG-augmented DIC governance"
    actor.start_task(task)

    for step in range(1, 20):
        action = actor.propose_action()

        if action.op == FileOp.DONE:
            print(f"\n{BOLD}{GREEN}  ✓ DONE at step {step}{RESET}")
            break

        decision = governor.evaluate(action)
        print_decision(step, action, decision)

        result = None
        if decision.approved:
            try:
                result = executor.execute(action)
            except Exception as e:
                print(f"{RED}  Executor error: {e}{RESET}")

        time.sleep(0.1)

    print(f"\n{DIM}  Escalation count this session: {governor.escalation_count}{RESET}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DIC + RAG demo")
    parser.add_argument("--scenario", default="default",
                        choices=["default", "escalate"])
    parser.add_argument("--rebuild-index", action="store_true",
                        help="Force Chroma index rebuild")
    args = parser.parse_args()
    run(scenario=args.scenario, rebuild_index=args.rebuild_index)
