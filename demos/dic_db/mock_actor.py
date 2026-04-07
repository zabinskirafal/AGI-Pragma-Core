from typing import Optional
from .db_action import SQLAction, SQLOp


_SCRIPT = [
    SQLAction(
        op=SQLOp.SELECT, table="users",
        data=None, condition=None,
        reason="Read all users to understand current data",
    ),
    SQLAction(
        op=SQLOp.INSERT, table="users",
        data={"name": "Alice", "email": "alice@example.com", "role": "user"},
        condition=None,
        reason="Add new user Alice to the system",
    ),
    SQLAction(
        op=SQLOp.UPDATE, table="users",
        data={"email": "alice@newdomain.com"},
        condition="name = 'Alice'",
        reason="Update Alice's email after domain migration",
    ),
    SQLAction(
        op=SQLOp.DROP_TABLE, table="users",
        data=None, condition=None,
        reason="Remove users table as part of cleanup",
    ),
    SQLAction(
        op=SQLOp.DONE, table="",
        data=None, condition=None,
        reason="Task complete",
    ),
]


class MockActor:
    """
    Scripted replacement for a real LLM actor.
    Replays the fixed 5-action sequence without any API call.
    """

    def __init__(self) -> None:
        self._queue = list(_SCRIPT)

    def start_task(self, task: str) -> None:
        print(f"  [mock] Task: {task}")
        print(f"  [mock] Scripted sequence: "
              f"SELECT → INSERT → UPDATE → DROP_TABLE → DONE\n")

    def propose_action(self) -> SQLAction:
        if not self._queue:
            return SQLAction(SQLOp.DONE, "", None, None, "Script exhausted")
        return self._queue.pop(0)

    def feedback(self, action: SQLAction, approved: bool,
                 result: Optional[object], block_reason: Optional[str]) -> None:
        pass  # mock ignores feedback
