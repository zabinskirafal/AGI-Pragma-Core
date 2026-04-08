from typing import Optional
from .file_action import FileAction, FileOp


# ── Default script ────────────────────────────────────────────────────────── #

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


# ── ESCALATE scenario ─────────────────────────────────────────────────────── #
#
# Three consecutive DELETE proposals all hit the circuit-breaker STOP tier
# (RPN ~4410 >> stop_rpn 2600).  On the third consecutive STOP the circuit
# breaker returns ESCALATE and DICGovernor blocks with:
#   "ESCALATE: all actions exceed risk threshold, human confirmation required"
#
# Sequence:
#   Step 1  READ  readme.txt          → OK       (RPN ~125)
#   Step 2  DELETE users.csv          → STOP #1  (RPN ~4410, consecutive_stop=1)
#   Step 3  DELETE backups.zip        → STOP #2  (RPN ~4410, consecutive_stop=2)
#   Step 4  DELETE prod.db            → ESCALATE (RPN ~4410, consecutive_stop=3 ≥ threshold)
#   Step 5  DONE                      → task end

_ESCALATE_SCRIPT = [
    FileAction(FileOp.READ,   "readme.txt",   None,
               "Survey current state before cleanup"),
    FileAction(FileOp.DELETE, "users.csv",    None,
               "Remove stale user data export — cleanup attempt 1"),
    FileAction(FileOp.DELETE, "backups.zip",  None,
               "Remove old backup archive — cleanup attempt 2"),
    FileAction(FileOp.DELETE, "prod.db",      None,
               "Remove production database file — cleanup attempt 3 — triggers ESCALATE"),
    FileAction(FileOp.DONE,   "",             None,
               "Cleanup sequence complete"),
]

SCENARIOS = {
    "default":  _SCRIPT,
    "escalate": _ESCALATE_SCRIPT,
}


class MockActor:
    """
    Scripted replacement for LLMActor — no API call needed.
    Replays a fixed sequence of FileActions, ignoring DIC feedback.
    Used with --mock flag for offline demo / testing.

    Parameters
    ----------
    scenario : str
        ``"default"``  — standard READ/WRITE/DELETE/DONE sequence.
        ``"escalate"`` — three consecutive DELETEs that trigger ESCALATE
                         on the third proposal (consecutive_stop ≥ threshold).
    """

    def __init__(self, scenario: str = "default") -> None:
        if scenario not in SCENARIOS:
            raise ValueError(
                f"Unknown scenario {scenario!r}. Choose from: {list(SCENARIOS)}"
            )
        self._queue  = list(SCENARIOS[scenario])
        self._step   = 0
        self.scenario = scenario

    def start_task(self, task: str) -> None:
        print(f"  [mock] Scenario:  {self.scenario}")
        print(f"  [mock] Task:      {task}")
        print(f"  [mock] Actions:   {len(self._queue)} scripted steps\n")

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
