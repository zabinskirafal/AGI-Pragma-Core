from dataclasses import dataclass
from typing import Dict


@dataclass
class FMEAItem:
    failure_mode: str
    severity: int    # 1..10 (10 = catastrophic)
    occurrence: int  # 1..10 (10 = frequent)
    detection: int   # 1..10 (10 = hard to detect)
    rpn: int


def clamp10(x: float) -> int:
    return max(1, min(10, int(round(x))))


def occ_from_prob(p: float) -> int:
    """Convert probability (0..1) to Occurrence score 1..10."""
    if p <= 0.01: return 1
    if p <= 0.03: return 2
    if p <= 0.07: return 3
    if p <= 0.12: return 4
    if p <= 0.20: return 5
    if p <= 0.32: return 6
    if p <= 0.45: return 7
    if p <= 0.60: return 8
    if p <= 0.80: return 9
    return 10


def fmea_table(
    p_timeout: float,
    p_dead_end: float,
) -> Dict[str, FMEAItem]:
    """
    Build a minimal FMEA table for a candidate maze action.

    Failure modes:
      timeout_failure  — step budget exhausted before reaching goal (catastrophic loss).
                         Parallel to Snake's prob_death: S=9, D=3 (estimable via rollout).
      dead_end_trap    — action leads toward a dead-end corridor (costly backtrack).
                         Parallel to Snake's trap: S=7, D=6 (subtler to detect).

    Note: there is no immediate_collision path in maze — wall hits are no-ops filtered
    at the branching stage. All risk here is probabilistic, not immediate.
    """
    s_timeout = 9
    o_timeout = occ_from_prob(p_timeout)
    d_timeout = 3   # rollouts make this estimable
    rpn_timeout = s_timeout * o_timeout * d_timeout

    s_dead_end = 7
    o_dead_end = occ_from_prob(p_dead_end)
    d_dead_end = 6  # dead ends are structurally subtle — corridor looks passable
    rpn_dead_end = s_dead_end * o_dead_end * d_dead_end

    return {
        "timeout_failure": FMEAItem(
            "Timeout — goal not reached within step budget",
            s_timeout, o_timeout, d_timeout, rpn_timeout,
        ),
        "dead_end_trap": FMEAItem(
            "Dead-end corridor — forced costly backtrack",
            s_dead_end, o_dead_end, d_dead_end, rpn_dead_end,
        ),
    }


def max_rpn(table: Dict[str, FMEAItem]) -> int:
    return max(item.rpn for item in table.values()) if table else 0
