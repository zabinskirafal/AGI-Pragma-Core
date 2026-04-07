from dataclasses import dataclass
from typing import Dict


@dataclass
class FMEAItem:
    failure_mode:  str
    severity:      int  # 1..10 (10 = catastrophic)
    occurrence:    int  # 1..10 (10 = frequent)
    detection:     int  # 1..10 (10 = hard to detect)
    reversibility: int  # 0=fully reversible … 10=fully irreversible
    rpn:           int  # S × O × D × R


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
    p_death:    float,
    p_trap:     float,
    immediate_collision: bool = False,
) -> Dict[str, FMEAItem]:
    """
    FMEA table for a candidate gridworld action.

    Failure modes:

    immediate_collision — agent walks directly into a current hazard position.
        Catastrophic (S=10), certain (O=10), fully observable (D=1).
        RPN = 100. If present, returned alone — gate will block this action.

    collision_death — probabilistic hazard contact within rollout horizon.
        Unlike maze timeout_failure, this signal is genuinely differentiating:
        actions toward hazard clusters score higher p_death than WAIT or evasion.
        S=10, D=2 (collision is observable but hazard paths are stochastic).

    proximity_trap — surrounded state (≥3 neighbours occupied) within horizon.
        Does not kill immediately but collapses future safe actions.
        S=8, D=5 (hazard convergence is subtle across multiple steps).
    """
    # Immediate collision — irreversible (R=10): death cannot be undone
    if immediate_collision:
        return {
            "immediate_collision": FMEAItem(
                "Agent moves directly onto hazard",
                severity=10, occurrence=10, detection=1, reversibility=10, rpn=1000,
            )
        }

    # Hazard collision within horizon — irreversible (R=10): death is permanent
    s_col, o_col, d_col, r_col = 10, occ_from_prob(p_death), 2, 10
    rpn_col = s_col * o_col * d_col * r_col

    # Proximity trap — partially reversible (R=4): WAIT often clears surrounded states
    s_trap, o_trap, d_trap, r_trap = 8, occ_from_prob(p_trap), 5, 4
    rpn_trap = s_trap * o_trap * d_trap * r_trap

    return {
        "collision_death": FMEAItem(
            "Hazard contact within horizon",
            s_col, o_col, d_col, r_col, rpn_col,
        ),
        "proximity_trap": FMEAItem(
            "Surrounded state — ≥3 neighbours occupied by hazards",
            s_trap, o_trap, d_trap, r_trap, rpn_trap,
        ),
    }


def max_rpn(table: Dict[str, FMEAItem]) -> int:
    return max(item.rpn for item in table.values()) if table else 0
