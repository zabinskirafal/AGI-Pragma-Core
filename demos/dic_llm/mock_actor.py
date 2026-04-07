from typing import Optional
from .file_action import FileAction, FileOp


_SCRIPT = [
    FileAction(FileOp.READ,   "readme.txt",  None,
               "Read readme to understand the project structure"),
    FileAction(FileOp.WRITE,  "plan.md",
               "# Project Plan\n\n## Goals\n- Deliver MVP by Q3\n- Onboard 3 pilot customers\n\n## Milestones\n1. Architecture design (week 1-2)\n2. Core implementation (week 3-6)\n3. Testing & QA (week 7-8)\n4. Pilot launch (week 9)\n",
               "Write the project plan with milestones"),
    FileAction(FileOp.WRITE,  "budget.csv",
               "Category,Q1,Q2,Q3\nEngineering,50000,50000,50000\nMarketing,10000,15000,20000\nInfrastructure,5000,5000,8000\nTotal,65000,70000,78000\n",
               "Write the budget spreadsheet with quarterly breakdown"),
    FileAction(FileOp.DELETE, "temp.txt",    None,
               "Clean up temporary scratch file"),
    FileAction(FileOp.DONE,   "",            None,
               "All files created, task complete"),
]


class MockActor:
    """
    Scripted replacement for LLMActor — no API call needed.
    Replays a fixed sequence of FileActions, ignoring DIC feedback.
    Used with --mock flag for offline demo / testing.
    """

    def __init__(self) -> None:
        self._queue = list(_SCRIPT)
        self._step  = 0

    def start_task(self, task: str) -> None:
        print(f"  [mock] Task received: {task}")
        print(f"  [mock] Will propose {len(self._queue)} scripted actions\n")

    def propose_action(self) -> FileAction:
        if not self._queue:
            return FileAction(FileOp.DONE, "", None, "Script exhausted")
        action = self._queue.pop(0)
        self._step += 1
        return action

    def feedback(self, action: FileAction, approved: bool,
                 result: Optional[str], block_reason: Optional[str]) -> None:
        # MockActor ignores feedback — script is fixed
        pass
