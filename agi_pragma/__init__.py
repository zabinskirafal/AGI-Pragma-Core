"""
AGI Pragma — AI Action Firewall
================================
Seven-stage Decision Intelligence Core (DIC) that evaluates every proposed
AI agent action for risk before execution.

Quick start
-----------
>>> from agi_pragma import DICGovernor, FileAction, FileOp
>>> gov = DICGovernor()
>>> action = FileAction(op=FileOp.WRITE, path="notes.txt",
...                     content="hello", reason="create notes")
>>> decision = gov.evaluate(action)
>>> print(decision.approved, decision.max_rpn)
True 504
"""

from demos.dic_llm.dic_governor import DICGovernor, DICDecision
from demos.dic_llm.file_action  import FileAction, FileOp
from demos.dic_llm.critical_path import CriticalPathResult, Reversibility
from demos.dic_llm.circuit_breaker import (
    CircuitBreaker, CircuitBreakerConfig, CircuitBreakerResult, CBState,
)
from demos.dic_llm.risk_fmea import FMEAItem, fmea_table, max_rpn
from demos.dic_llm.bayes import BetaTracker

__version__ = "1.0.0"
__author__  = "Rafał Żabiński"

__all__ = [
    # Core pipeline
    "DICGovernor",
    "DICDecision",
    # Actions
    "FileAction",
    "FileOp",
    # Risk assessment
    "FMEAItem",
    "fmea_table",
    "max_rpn",
    # Critical path
    "CriticalPathResult",
    "Reversibility",
    # Circuit breaker
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerResult",
    "CBState",
    # Belief tracking
    "BetaTracker",
]
