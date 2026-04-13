"""
FMEA Calibration Wizard
=======================
Interactive CLI that guides a user through S/O/D/R scoring for a new domain
and exports a ready-to-use fmea_config.json.

Usage:
    python -m demos.dic_llm.fmea_wizard
    python -m demos.dic_llm.fmea_wizard --domain file_ops
    python -m demos.dic_llm.fmea_wizard --domain database --out my_fmea.json
    python -m demos.dic_llm.fmea_wizard --domain custom
"""

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

# ── Colour helpers (no external deps) ─────────────────────────────────── #

RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RED    = "\033[91m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
CYAN   = "\033[96m"


def _c(text: str, *codes: str) -> str:
    return "".join(codes) + text + RESET


# ── Domain definitions ─────────────────────────────────────────────────── #

@dataclass
class ActionTemplate:
    key:         str
    label:       str
    description: str
    # Default scores — user can override
    severity:      int
    occurrence:    int
    detection:     int
    reversibility: int


FILE_OPS_ACTIONS: List[ActionTemplate] = [
    ActionTemplate("read",   "READ",   "Read file contents",                          3, 3, 8, 1),
    ActionTemplate("write",  "WRITE",  "Write / overwrite file",                      7, 5, 5, 6),
    ActionTemplate("delete", "DELETE", "Permanently delete file",                    10, 4, 3, 10),
    ActionTemplate("rename", "RENAME", "Rename or move file",                         5, 4, 6, 5),
    ActionTemplate("chmod",  "CHMOD",  "Change file permissions",                     7, 3, 5, 7),
]

DATABASE_ACTIONS: List[ActionTemplate] = [
    ActionTemplate("select",    "SELECT",     "Read rows from a table",               2, 4, 8, 1),
    ActionTemplate("insert",    "INSERT",     "Insert new rows",                      5, 5, 6, 6),
    ActionTemplate("update",    "UPDATE",     "Update existing rows",                 8, 5, 4, 7),
    ActionTemplate("delete_row","DELETE",     "Delete rows from a table",            10, 4, 3, 9),
    ActionTemplate("drop",      "DROP TABLE", "Drop an entire table",                10, 2, 2, 10),
    ActionTemplate("migrate",   "MIGRATE",    "Schema migration (ALTER / CREATE)",    8, 3, 4, 8),
]

NETWORK_ACTIONS: List[ActionTemplate] = [
    ActionTemplate("http_get",   "HTTP GET",    "Read-only HTTP request",             2, 5, 7, 1),
    ActionTemplate("http_post",  "HTTP POST",   "Send data to external endpoint",     6, 5, 5, 5),
    ActionTemplate("http_delete","HTTP DELETE", "Delete remote resource via API",     9, 3, 4, 9),
    ActionTemplate("ssh_exec",   "SSH EXEC",    "Execute command on remote host",    10, 3, 2, 8),
    ActionTemplate("dns_change", "DNS CHANGE",  "Modify DNS record",                  9, 2, 3, 9),
    ActionTemplate("firewall",   "FIREWALL",    "Modify firewall / security group",  10, 2, 3, 10),
]

DOMAINS: Dict[str, List[ActionTemplate]] = {
    "file_ops":  FILE_OPS_ACTIONS,
    "database":  DATABASE_ACTIONS,
    "network":   NETWORK_ACTIONS,
}

# ── Scoring rubrics ────────────────────────────────────────────────────── #

SEVERITY_RUBRIC = [
    (1,  2,  "Negligible — cosmetic impact, easily noticed and corrected"),
    (3,  4,  "Minor — degraded functionality, recoverable with low effort"),
    (5,  6,  "Moderate — significant disruption, recovery requires effort"),
    (7,  8,  "Major — data loss, service outage, compliance breach possible"),
    (9,  10, "Catastrophic — irreversible harm, legal exposure, or safety risk"),
]

OCCURRENCE_RUBRIC = [
    (1,  2,  "Remote — failure almost never observed (<1% of actions)"),
    (3,  4,  "Low — occasional failures, well-understood conditions (1–10%)"),
    (5,  6,  "Moderate — failures occur under foreseeable load (10–30%)"),
    (7,  8,  "High — failures common; design mitigation needed (30–60%)"),
    (9,  10, "Very high — failure is near-certain without intervention (>60%)"),
]

DETECTION_RUBRIC = [
    (1,  2,  "Almost certain — failure visible immediately before/after execution"),
    (3,  4,  "High — observable through standard monitoring"),
    (5,  6,  "Moderate — detectable with targeted checks"),
    (7,  8,  "Low — silent failure; only caught by deep audit or downstream breakage"),
    (9,  10, "Undetectable — failure propagates invisibly; found only in post-mortem"),
]

REVERSIBILITY_RUBRIC = [
    (1,  2,  "Fully reversible — trivial to undo (e.g. read-only, cached state)"),
    (3,  4,  "Mostly reversible — undo possible with moderate effort"),
    (5,  6,  "Partially reversible — rollback possible but costly"),
    (7,  8,  "Hard to reverse — requires restore from backup or manual repair"),
    (9,  10, "Irreversible — permanent data loss, sent message, real-world effect"),
]

RUBRICS = {
    "Severity (S)":      SEVERITY_RUBRIC,
    "Occurrence (O)":    OCCURRENCE_RUBRIC,
    "Detection (D)":     DETECTION_RUBRIC,
    "Reversibility (R)": REVERSIBILITY_RUBRIC,
}


# ── RPN interpretation ─────────────────────────────────────────────────── #

def rpn_label(rpn: int) -> str:
    if rpn >= 4000:
        return _c("CRITICAL", RED, BOLD)
    if rpn >= 2000:
        return _c("HIGH    ", YELLOW, BOLD)
    if rpn >= 800:
        return _c("MEDIUM  ", CYAN)
    return _c("LOW     ", GREEN)


def suggest_threshold(items: List[dict]) -> int:
    """Suggest tau as the median RPN rounded to nearest 200."""
    rpns = sorted(it["rpn"] for it in items)
    median = rpns[len(rpns) // 2]
    return max(200, round(median / 200) * 200)


# ── Input helpers ──────────────────────────────────────────────────────── #

def _prompt_int(prompt: str, lo: int = 1, hi: int = 10, default: Optional[int] = None) -> int:
    hint = f" [{default}]" if default is not None else f" ({lo}-{hi})"
    while True:
        raw = input(_c(f"  {prompt}{hint}: ", BOLD)).strip()
        if raw == "" and default is not None:
            return default
        try:
            val = int(raw)
            if lo <= val <= hi:
                return val
            print(_c(f"    Enter a number between {lo} and {hi}.", RED))
        except ValueError:
            print(_c("    Not a number — try again.", RED))


def _show_rubric(rubric: List[tuple]) -> None:
    for lo, hi, desc in rubric:
        label = f"{lo}" if lo == hi else f"{lo}-{hi}"
        print(f"    {_c(label, DIM):6s}  {desc}")


def _confirm(prompt: str) -> bool:
    raw = input(_c(f"  {prompt} [y/N]: ", BOLD)).strip().lower()
    return raw in ("y", "yes")


# ── Wizard core ────────────────────────────────────────────────────────── #

def calibrate_action(tmpl: ActionTemplate) -> dict:
    """Walk the user through S/O/D/R for one action. Returns scored dict."""
    print()
    print(_c(f"  Action: {tmpl.label}", BOLD, CYAN))
    print(_c(f"  {tmpl.description}", DIM))
    print()

    scores = {}
    for field, rubric in RUBRICS.items():
        key = field.split()[0].lower().rstrip("(sodr)")  # severity/occurrence/detection/reversibility
        # map readable name to dataclass field
        attr_map = {
            "severity":      "severity",
            "occurrence":    "occurrence",
            "detection":     "detection",
            "reversibility": "reversibility",
        }
        # Extract short key from "Severity (S)" → "severity"
        short = field.split()[0].lower()
        default = getattr(tmpl, short)
        print(_c(f"  {field}", BOLD))
        _show_rubric(rubric)
        val = _prompt_int(field, default=default)
        scores[short] = val
        print()

    s, o, d, r = scores["severity"], scores["occurrence"], scores["detection"], scores["reversibility"]
    rpn = s * o * d * r
    print(f"  RPN = {s} × {o} × {d} × {r} = {_c(str(rpn), BOLD)}  {rpn_label(rpn)}")

    return {
        "key":          tmpl.key,
        "label":        tmpl.label,
        "description":  tmpl.description,
        "severity":     s,
        "occurrence":   o,
        "detection":    d,
        "reversibility": r,
        "rpn":          rpn,
    }


def run_wizard(domain: str, actions: List[ActionTemplate]) -> dict:
    print()
    print(_c("=" * 60, BOLD))
    print(_c(f"  FMEA Calibration Wizard — {domain.upper()}", BOLD))
    print(_c("=" * 60, BOLD))
    print()
    print("  This wizard scores each action on four dimensions:")
    print("   S  Severity      — impact if the failure occurs")
    print("   O  Occurrence    — how often the failure mode arises")
    print("   D  Detection     — how hard it is to catch before harm")
    print("   R  Reversibility — how hard it is to undo")
    print()
    print("  RPN = S × O × D × R   (max 10,000)")
    print("  Press Enter to accept the suggested default score.")
    print()

    calibrated = []
    for i, tmpl in enumerate(actions, 1):
        print(_c(f"── Action {i}/{len(actions)} ──────────────────────────────────", DIM))
        result = calibrate_action(tmpl)
        calibrated.append(result)

    tau = suggest_threshold(calibrated)
    print()
    print(_c("=" * 60, BOLD))
    print(_c("  Threshold Recommendation", BOLD))
    print(_c("=" * 60, BOLD))
    print(f"  Suggested τ (RPN gate) = {_c(str(tau), BOLD, YELLOW)}")
    print("  Actions with RPN > τ will be blocked by DIC.")
    print()
    tau = _prompt_int("Set tau", lo=1, hi=10000, default=tau)

    config = {
        "domain":    domain,
        "threshold": tau,
        "actions":   calibrated,
    }
    return config


def run_custom_wizard() -> dict:
    """Let the user define their own domain and action list."""
    print()
    print(_c("=" * 60, BOLD))
    print(_c("  FMEA Calibration Wizard — CUSTOM DOMAIN", BOLD))
    print(_c("=" * 60, BOLD))
    print()
    domain = input(_c("  Domain name: ", BOLD)).strip() or "custom"
    print()
    print("  Enter action names one per line. Empty line to finish.")
    templates = []
    while True:
        raw = input(_c(f"  Action {len(templates)+1} key (or Enter to stop): ", BOLD)).strip()
        if not raw:
            break
        label = input(_c(f"  Label for '{raw}': ", BOLD)).strip() or raw.upper()
        desc  = input(_c(f"  Description: ", BOLD)).strip() or ""
        templates.append(ActionTemplate(raw, label, desc, 5, 5, 5, 5))

    if not templates:
        print(_c("  No actions defined — exiting.", RED))
        sys.exit(1)

    return run_wizard(domain, templates)


# ── Entry point ────────────────────────────────────────────────────────── #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive FMEA calibration wizard — exports fmea_config.json"
    )
    parser.add_argument(
        "--domain",
        choices=["file_ops", "database", "network", "custom"],
        default=None,
        help="Domain to calibrate (omit to choose interactively)",
    )
    parser.add_argument(
        "--out",
        default="fmea_config.json",
        help="Output file path (default: fmea_config.json)",
    )
    args = parser.parse_args()

    domain = args.domain
    if domain is None:
        print()
        print(_c("  Select domain:", BOLD))
        choices = ["file_ops", "database", "network", "custom"]
        for i, ch in enumerate(choices, 1):
            print(f"    {i}. {ch}")
        idx = _prompt_int("Choice", lo=1, hi=len(choices))
        domain = choices[idx - 1]

    if domain == "custom":
        config = run_custom_wizard()
    else:
        config = run_wizard(domain, DOMAINS[domain])

    out_path = Path(args.out)
    out_path.write_text(json.dumps(config, indent=2))
    print()
    print(_c("=" * 60, BOLD))
    print(_c("  Export complete", GREEN, BOLD))
    print(_c("=" * 60, BOLD))
    print(f"  File   : {out_path.resolve()}")
    print(f"  Domain : {config['domain']}")
    print(f"  τ (tau): {config['threshold']}")
    print(f"  Actions: {len(config['actions'])}")
    print()
    _print_summary(config)


def _print_summary(config: dict) -> None:
    print(_c("  Action summary:", BOLD))
    print()
    header = f"  {'Action':<14} {'S':>3} {'O':>3} {'D':>3} {'R':>3} {'RPN':>6}  Risk"
    print(_c(header, DIM))
    print(_c("  " + "-" * 52, DIM))
    for item in config["actions"]:
        rpn = item["rpn"]
        row = (
            f"  {item['label']:<14}"
            f" {item['severity']:>3}"
            f" {item['occurrence']:>3}"
            f" {item['detection']:>3}"
            f" {item['reversibility']:>3}"
            f" {rpn:>6}  "
        )
        print(row + rpn_label(rpn))
    print()
    tau = config["threshold"]
    blocked = [it for it in config["actions"] if it["rpn"] > tau]
    approved = [it for it in config["actions"] if it["rpn"] <= tau]
    print(f"  At τ={tau}: {_c(str(len(blocked))+' blocked', RED)},"
          f" {_c(str(len(approved))+' approved', GREEN)}")
    print()


if __name__ == "__main__":
    main()
