"""
DIC + LLM Database Demo
=======================
An LLM proposes SQL operations (SELECT / INSERT / UPDATE / DELETE_ROW / DROP_TABLE).
DIC evaluates every proposed action through the full 7-stage pipeline before
any query touches the database.

Usage:
    python3 -m demos.dic_db.run --mock
    python3 -m demos.dic_db.run --task "..." --mock
"""

import argparse
import time
import textwrap
from pathlib import Path

from .db_action    import SQLAction, SQLOp
from .dic_governor import DICGovernor, DICDecision
from .db_engine    import DBEngine, DB_PATH
from .mock_actor   import MockActor

# ── ANSI ─────────────────────────────────────────────────────────────── #
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

W = 62


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


def print_decision(step: int, action: SQLAction, d: DICDecision) -> None:
    colour = GREEN if d.approved else RED
    verdict = f"{'✓ YES' if d.approved else '✗ NO '}  (RPN {d.max_rpn})"

    print()
    print(_hdr(f"Step {step}  │  DIC DECISION", colour))
    print(_box_line(f"Op:        {action.op.value.upper()}"))
    print(_box_line(f"Table:     {action.table}"))
    if action.condition:
        print(_box_line(f"Condition: WHERE {action.condition}"))
    if action.data:
        pairs = ", ".join(f"{k}={v!r}" for k, v in action.data.items())
        print(_box_line(f"Data:      {pairs[:W-13]}"))
    print(_box_line(f"Reason:    {action.reason}"))
    print(_sep())

    print(f"║ Decision: {BOLD}{colour}{verdict}{RESET:<{W+len(BOLD)+len(colour)+len(RESET)-2}}║")
    if not d.approved:
        for line in textwrap.wrap(d.block_reason or "", W - 14):
            print(_box_line(f"  Reason:  {line}", RED))
    print(_sep())

    print(_box_line("AUDIT TRACE", DIM))
    for entry in d.stage_log:
        stage = entry["stage"]

        if stage == "branching":
            icon = "✓" if entry["pass"] else "✗"
            print(_box_line(f"  1. Branching       {icon}  {entry['detail'][:36]}"))

        elif stage == "critical_path":
            rev  = entry["reversibility"].upper()
            p    = entry["p_irreversible"]
            sc   = entry["affected_scope"]
            rows = entry["row_count_risk"]
            print(_box_line(f"  2. Critical Path      rev={rev:<8} p_irrev={p:.2f}"))
            print(_box_line(f"       scope={sc:<6} rows={rows}", DIM))
            for se in entry.get("side_effects", []):
                for ln in textwrap.wrap(se, W - 10):
                    print(_box_line(f"       ↳ {ln}", DIM))

        elif stage == "fmea":
            print(_box_line(f"  3. FMEA               max_rpn={entry['max_rpn']}"))
            for fname, fd in entry.get("table", {}).items():
                rpn = fd.get("rpn", "?")
                s   = fd.get("severity", "?")
                o   = fd.get("occurrence", "?")
                det = fd.get("detection", "?")
                r   = fd.get("reversibility", "?")
                print(_box_line(
                    f"       {fname:<26} S={s} O={o} D={det} R={r} RPN={rpn}", DIM
                ))

        elif stage == "decision_gate":
            icon = "✗ BLOCKED" if entry["blocked"] else "✓ PASS"
            print(_box_line(
                f"  4. Decision Gate   {icon}  ({entry['max_rpn']} vs {entry['threshold']})"
            ))

        elif stage == "circuit_breaker":
            s     = entry["state"].upper()
            sc    = GREEN if s == "OK" else (YELLOW if s in ("WARN", "SLOW") else RED)
            print(_box_line(f"  5. Circuit Breaker    {sc}{s}{RESET}  {entry['reason']}"))

        elif stage == "utility":
            print(_box_line(f"  6. Utility            score={entry['score']:.3f}"))

        elif stage == "belief_update":
            mean = entry["llm_risk_mean"]
            bar  = int(mean * 20) * "▓" + (20 - int(mean * 20)) * "░"
            print(_box_line(f"  7. Belief Update      llm_risk_mean={mean:.3f}  {bar}"))

    print(_footer())


def print_db_state(engine: DBEngine) -> None:
    tables = engine.list_tables()
    print(f"\n{CYAN}  Database state:{RESET}")
    if not tables:
        print(f"    {DIM}(no tables){RESET}")
        return
    for t in tables:
        try:
            rows = engine.execute(SQLAction(SQLOp.SELECT, t, None, None, ""))
            print(f"    {BOLD}{t}{RESET}  {DIM}({len(rows)} rows){RESET}")
            for row in (rows or [])[:5]:
                print(f"      {DIM}{row}{RESET}")
            if rows and len(rows) > 5:
                print(f"      {DIM}… {len(rows)-5} more rows{RESET}")
        except Exception as e:
            print(f"    {t}  {DIM}(error: {e}){RESET}")


def run(task: str, max_steps: int = 10, mock: bool = False) -> None:
    # Fresh DB for each run
    if DB_PATH.exists():
        DB_PATH.unlink()

    engine   = DBEngine()
    governor = DICGovernor(row_count_fn=engine.row_count)
    actor    = MockActor()

    print(f"\n{BOLD}{CYAN}{'═'*64}{RESET}")
    print(f"{BOLD}{CYAN}  DIC + LLM Database Demo{RESET}")
    print(f"{BOLD}{CYAN}{'═'*64}{RESET}")
    print(f"  Task:    {task}")
    print(f"  Model:   {'[mock]' if mock else 'claude API'}")
    print(f"  DB:      {DB_PATH}")

    print(f"\n{DIM}  Initial database state:{RESET}")
    print_db_state(engine)

    actor.start_task(task)

    for step in range(1, max_steps + 1):
        action = actor.propose_action()

        if action.op == SQLOp.DONE:
            print(f"\n{BOLD}{GREEN}  ✓ Task complete (DONE at step {step}){RESET}")
            print(f"\n{DIM}  Final database state:{RESET}")
            print_db_state(engine)
            return

        decision = governor.evaluate(action)
        print_decision(step, action, decision)

        if decision.approved:
            result = engine.execute(action)
            if result is not None:   # SELECT returned rows
                print(f"\n{DIM}  Query result ({len(result)} rows):{RESET}")
                for row in result:
                    print(f"    {DIM}{row}{RESET}")

        time.sleep(0.2)

    print(f"\n{YELLOW}  Max steps ({max_steps}) reached.{RESET}")
    print_db_state(engine)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DIC + LLM database demo")
    parser.add_argument(
        "--task",
        default="Manage user records: read, add a user, update an email, then clean up.",
    )
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--mock", action="store_true",
                        help="Use scripted mock actor (no API key required)")
    args = parser.parse_args()

    if not args.mock:
        print("Only --mock is currently supported. Add --mock flag.")
    else:
        run(task=args.task, max_steps=args.max_steps, mock=True)
