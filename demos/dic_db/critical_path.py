from dataclasses import dataclass
from enum import Enum
from typing import List

from .db_action import SQLAction, SQLOp


class Reversibility(str, Enum):
    NONE  = "none"    # SELECT  — no state change
    LOW   = "low"     # INSERT  — row can be deleted
    MEDIUM = "medium" # UPDATE  — original values overwritten
    HIGH  = "high"    # DELETE_ROW — rows gone without transaction log
    TOTAL = "total"   # DROP_TABLE — schema + all data destroyed


@dataclass
class CriticalPathResult:
    reversibility:  Reversibility
    affected_scope: str    # "row" / "rows" / "table" / "schema"
    row_count_risk: str    # "single" / "multi" / "all" / "n/a"
    side_effects:   List[str]
    p_irreversible: float  # 0..1


def reversibility_profile(action: SQLAction, row_count: int = 0) -> CriticalPathResult:
    """
    Static analysis of how reversible the proposed SQL operation is.

    row_count: estimated number of rows affected (from executor pre-scan).
    0 means unknown / not applicable.
    """
    side_effects: List[str] = []

    if action.op == SQLOp.SELECT:
        rev   = Reversibility.NONE
        scope = "read"
        rows  = "n/a"
        p     = 0.01
        side_effects.append("Read-only — no data mutation")

    elif action.op == SQLOp.INSERT:
        rev   = Reversibility.LOW
        scope = "row"
        rows  = "single"
        p     = 0.10
        side_effects.append("New row added — can be removed with DELETE_ROW")
        if not action.condition:
            side_effects.append("No condition — inserts unconditionally")

    elif action.op == SQLOp.UPDATE:
        rev   = Reversibility.MEDIUM
        scope = "rows"
        p     = 0.55
        if action.condition:
            rows = "single" if "=" in action.condition else "multi"
            side_effects.append(f"Overwrites existing values — original data lost")
            side_effects.append(f"Condition: WHERE {action.condition}")
        else:
            rows = "all"
            p    = 0.85
            side_effects.append("NO WHERE clause — updates ALL rows in table")
            side_effects.append("Full-table update: all original values permanently overwritten")

    elif action.op == SQLOp.DELETE_ROW:
        rev   = Reversibility.HIGH
        scope = "rows"
        p     = 0.80
        if action.condition:
            rows = "single" if "=" in action.condition else "multi"
            side_effects.append(f"Rows permanently deleted — no undo without backup")
            side_effects.append(f"Condition: WHERE {action.condition}")
        else:
            rows = "all"
            p    = 0.95
            side_effects.append("NO WHERE clause — deletes ALL rows (table truncation)")

    elif action.op == SQLOp.DROP_TABLE:
        rev   = Reversibility.TOTAL
        scope = "schema"
        rows  = "all"
        p     = 1.00
        side_effects.append("Entire table schema and all data permanently destroyed")
        side_effects.append("Foreign key constraints, indices, triggers all removed")
        side_effects.append("Cannot be undone without a full database backup")

    else:  # DONE
        rev   = Reversibility.NONE
        scope = "n/a"
        rows  = "n/a"
        p     = 0.0

    return CriticalPathResult(
        reversibility=rev,
        affected_scope=scope,
        row_count_risk=rows,
        side_effects=side_effects,
        p_irreversible=p,
    )
