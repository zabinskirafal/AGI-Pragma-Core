from dataclasses import dataclass
from typing import Dict

from .db_action import SQLAction, SQLOp


@dataclass
class FMEAItem:
    failure_mode:  str
    severity:      int  # 1..10
    occurrence:    int  # 1..10 — driven by LLM risk tracker mean
    detection:     int  # 1..10
    reversibility: int  # 0=fully reversible … 10=fully irreversible
    rpn:           int  # S × O × D × R


def occ_from_prob(p: float) -> int:
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
    action:        SQLAction,
    has_condition: bool,    # UPDATE/DELETE_ROW with no WHERE is broader blast radius
    llm_risk_mean: float,   # Beta tracker mean for LLM risk rate
) -> Dict[str, FMEAItem]:
    """
    FMEA table for a candidate SQL operation.

    Reversibility per op:
      DROP_TABLE  R=10 — schema + data gone, no undo
      DELETE_ROW  R=8  — rows deleted; possible with backup only
      UPDATE      R=6  — original values overwritten
      INSERT      R=3  — new row; can be deleted
      SELECT      R=1  — read-only

    DROP_TABLE is always blocked: even at O=1, S=10 × O=1 × D=1 × R=10 = 100 per mode,
    but schema_destruction uses D=1 (trivially detectable) and combined RPN still
    exceeds any reasonable threshold at O≥3.
    """
    o = occ_from_prob(llm_risk_mean)
    table: Dict[str, FMEAItem] = {}

    # ── DROP_TABLE ───────────────────────────────────────────────────── #
    if action.op == SQLOp.DROP_TABLE:
        # Schema destruction — catastrophic, fully observable, fully irreversible
        s, d, r = 10, 1, 10
        table["schema_destruction"] = FMEAItem(
            "Table schema and all data permanently destroyed",
            s, min(10, o + 3), d, r, s * min(10, o + 3) * d * r,
        )
        # Cascade failure — downstream queries / foreign keys break silently
        s2, d2, r2 = 9, 6, 10
        table["cascade_failure"] = FMEAItem(
            "Foreign key / application references silently broken",
            s2, o, d2, r2, s2 * o * d2 * r2,
        )

    # ── DELETE_ROW ───────────────────────────────────────────────────── #
    elif action.op == SQLOp.DELETE_ROW:
        # Permanent row loss
        s, d, r = 8, 3, 8
        table["permanent_row_loss"] = FMEAItem(
            "Rows permanently deleted — unrecoverable without backup",
            s, min(10, o + 1), d, r, s * min(10, o + 1) * d * r,
        )
        if not has_condition:
            # No WHERE → full table wipe
            s2, d2, r2 = 10, 2, 9
            table["full_table_wipe"] = FMEAItem(
                "No WHERE clause — entire table contents deleted",
                s2, min(10, o + 2), d2, r2, s2 * min(10, o + 2) * d2 * r2,
            )
        else:
            # Wrong rows matched — condition error
            s2, d2, r2 = 8, 7, 8
            table["wrong_rows_deleted"] = FMEAItem(
                "Condition error — wrong rows deleted (LLM hallucination risk)",
                s2, o, d2, r2, s2 * o * d2 * r2,
            )

    # ── UPDATE ───────────────────────────────────────────────────────── #
    elif action.op == SQLOp.UPDATE:
        # Data overwrite — original values lost
        s, d, r = 7, 4, 6
        table["data_overwrite"] = FMEAItem(
            "Original column values overwritten — no undo without backup",
            s, o, d, r, s * o * d * r,
        )
        if not has_condition:
            # No WHERE → full table update
            s2, d2, r2 = 9, 2, 7
            table["full_table_update"] = FMEAItem(
                "No WHERE clause — all rows updated simultaneously",
                s2, min(10, o + 2), d2, r2, s2 * min(10, o + 2) * d2 * r2,
            )
        else:
            # Wrong rows updated
            s2, d2, r2 = 7, 6, 6
            table["wrong_rows_updated"] = FMEAItem(
                "Condition error — wrong rows updated (LLM hallucination risk)",
                s2, o, d2, r2, s2 * o * d2 * r2,
            )

    # ── INSERT ───────────────────────────────────────────────────────── #
    elif action.op == SQLOp.INSERT:
        # Duplicate / constraint violation
        s, o_dup, d, r = 4, max(1, o - 2), 6, 3
        table["duplicate_insert"] = FMEAItem(
            "Duplicate row inserted — may violate uniqueness constraints",
            s, o_dup, d, r, s * o_dup * d * r,
        )
        # Incorrect data
        s2, d2, r2 = 5, 5, 3
        table["incorrect_data"] = FMEAItem(
            "Row inserted with incorrect or hallucinated field values",
            s2, o, d2, r2, s2 * o * d2 * r2,
        )

    # ── SELECT ───────────────────────────────────────────────────────── #
    elif action.op == SQLOp.SELECT:
        s, o_sel, d, r = 4, max(1, o - 3), 5, 1
        table["sensitive_exposure"] = FMEAItem(
            "Sensitive table contents exposed to LLM context window",
            s, o_sel, d, r, s * o_sel * d * r,
        )

    return table


def max_rpn(table: Dict[str, FMEAItem]) -> int:
    return max(item.rpn for item in table.values()) if table else 0
