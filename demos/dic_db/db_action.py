from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any


class SQLOp(str, Enum):
    SELECT      = "select"
    INSERT      = "insert"
    UPDATE      = "update"
    DELETE_ROW  = "delete_row"
    DROP_TABLE  = "drop_table"
    DONE        = "done"


@dataclass
class SQLAction:
    op:        SQLOp
    table:     str
    data:      Optional[Dict[str, Any]]  # column→value for INSERT / UPDATE
    condition: Optional[str]             # WHERE clause for UPDATE / DELETE_ROW
    reason:    str

    def __str__(self) -> str:
        parts = [f"{self.op.value.upper():<12} {self.table}"]
        if self.condition:
            parts.append(f"WHERE {self.condition}")
        if self.data:
            pairs = ", ".join(f"{k}={v!r}" for k, v in self.data.items())
            parts.append(f"SET {pairs}")
        parts.append(f"— {self.reason}")
        return "  ".join(parts)
