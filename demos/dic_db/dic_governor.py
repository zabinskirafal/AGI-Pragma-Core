from dataclasses import dataclass
from typing import Dict, Any, Optional, Set

from .db_action import SQLAction, SQLOp
from .risk_fmea import fmea_table, max_rpn
from .critical_path import reversibility_profile, CriticalPathResult
from .circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitBreakerResult
from .bayes import BetaTracker


# Tables that may never be touched by LLM-proposed actions
_FORBIDDEN_TABLES: Set[str] = {
    "sqlite_master", "sqlite_sequence", "sqlite_stat1",
    "pg_catalog", "information_schema",
}

RPN_THRESHOLD = 2400


@dataclass
class DICDecision:
    approved:        bool
    action:          SQLAction
    block_reason:    Optional[str]
    critical_path:   CriticalPathResult
    fmea:            Dict[str, Any]
    max_rpn:         int
    circuit_breaker: CircuitBreakerResult
    utility:         float
    bayes:           Dict[str, float]
    stage_log:       list


class DICGovernor:
    """
    Full 7-stage DIC pipeline for LLM-proposed SQL operations.

    Stage 1 — Branching:      op whitelist, system table guard
    Stage 2 — Critical Path:  static reversibility analysis
    Stage 3 — FMEA:           S×O×D×R per failure mode
    Stage 4 — Decision Gate:  block if max_rpn ≥ threshold
    Stage 5 — Circuit Breaker: session escalation on repeated risk
    Stage 6 — Utility:        task progress benefit − risk penalty
    Stage 7 — Belief Update:  Beta tracker for LLM risk rate
    """

    def __init__(
        self,
        rpn_threshold:          int = RPN_THRESHOLD,
        circuit_breaker_config: CircuitBreakerConfig | None = None,
        row_count_fn=None,   # optional callable(table, condition) → int
    ):
        self.rpn_threshold    = rpn_threshold
        self.circuit_breaker  = CircuitBreaker(circuit_breaker_config)
        self.llm_risk_tracker = BetaTracker(1.0, 1.0)
        self.row_count_fn     = row_count_fn  # plugged in by executor

    def evaluate(self, action: SQLAction) -> DICDecision:
        stage_log: list = []

        # ── 1. Branching ─────────────────────────────────────────── #
        ok, msg = self._scope_check(action)
        stage_log.append({"stage": "branching", "pass": ok, "detail": msg})
        if not ok:
            return self._block(action, msg, stage_log)

        # ── 2. Critical Path ──────────────────────────────────────── #
        row_count = self.row_count_fn(action) if self.row_count_fn else 0
        cp = reversibility_profile(action, row_count)
        stage_log.append({
            "stage":          "critical_path",
            "reversibility":  cp.reversibility.value,
            "affected_scope": cp.affected_scope,
            "row_count_risk": cp.row_count_risk,
            "p_irreversible": cp.p_irreversible,
            "side_effects":   cp.side_effects,
        })

        # ── 3. FMEA ───────────────────────────────────────────────── #
        has_cond  = bool(action.condition)
        table     = fmea_table(action, has_cond, self.llm_risk_tracker.mean)
        m_rpn     = max_rpn(table)
        fmea_out  = {k: vars(v) for k, v in table.items()}
        stage_log.append({"stage": "fmea", "table": fmea_out, "max_rpn": m_rpn})

        # ── 4. Decision Gate ─────────────────────────────────────── #
        gate_blocked = m_rpn >= self.rpn_threshold
        stage_log.append({
            "stage":     "decision_gate",
            "max_rpn":   m_rpn,
            "threshold": self.rpn_threshold,
            "blocked":   gate_blocked,
        })

        # ── 5. Circuit Breaker ────────────────────────────────────── #
        cb = self.circuit_breaker.evaluate(m_rpn)
        stage_log.append({"stage": "circuit_breaker",
                          "state": cb.state.value, "reason": cb.reason})
        cb_blocked = cb.state.value == "stop"

        # ── 6. Utility ───────────────────────────────────────────── #
        utility = self._utility(action, cp, m_rpn)
        stage_log.append({"stage": "utility", "score": utility})

        # ── 7. Belief Update ──────────────────────────────────────── #
        risky = m_rpn >= self.rpn_threshold // 2
        self.llm_risk_tracker.update(risky)
        stage_log.append({
            "stage":         "belief_update",
            "risky_signal":  risky,
            "llm_risk_mean": self.llm_risk_tracker.mean,
            "beta":          repr(self.llm_risk_tracker),
        })

        if gate_blocked or cb_blocked:
            reason = cb.reason if cb_blocked else f"RPN {m_rpn} ≥ threshold {self.rpn_threshold}"
            return self._build(False, action, reason, cp, fmea_out, m_rpn, cb, utility, stage_log)

        return self._build(True, action, None, cp, fmea_out, m_rpn, cb, utility, stage_log)

    # ---------------------------------------------------------------- #

    def _scope_check(self, action: SQLAction):
        if action.op == SQLOp.DONE:
            return True, "DONE signal — no table to check"

        t = action.table.strip().lower()
        if not t:
            return False, "BLOCKED: empty table name"
        if t in _FORBIDDEN_TABLES:
            return False, f"BLOCKED: system table '{action.table}' is off-limits"
        if any(c in t for c in (";", "--", " ", "\n", "'")):
            return False, f"BLOCKED: suspicious characters in table name '{action.table}'"

        return True, f"PASS — table '{action.table}' is within allowed scope"

    def _utility(self, action: SQLAction, cp: CriticalPathResult, m_rpn: int) -> float:
        benefit = {
            SQLOp.SELECT:     3.0,
            SQLOp.INSERT:     6.0,
            SQLOp.UPDATE:     7.0,
            SQLOp.DELETE_ROW: 2.0,
            SQLOp.DROP_TABLE: 0.5,
            SQLOp.DONE:       10.0,
        }.get(action.op, 0.0)
        risk = (m_rpn / 1000.0) + cp.p_irreversible * 5.0
        return round(benefit - risk, 3)

    def _block(self, action, reason, stage_log) -> DICDecision:
        from .critical_path import Reversibility
        from .circuit_breaker import CBState, CircuitBreakerResult
        dummy_cp = CriticalPathResult(
            reversibility=Reversibility.TOTAL,
            affected_scope="unknown",
            row_count_risk="unknown",
            side_effects=[],
            p_irreversible=1.0,
        )
        dummy_cb = CircuitBreakerResult(CBState.STOP, reason)
        return DICDecision(
            approved=False, action=action, block_reason=reason,
            critical_path=dummy_cp, fmea={}, max_rpn=9999,
            circuit_breaker=dummy_cb, utility=-1e9,
            bayes={"llm_risk_mean": self.llm_risk_tracker.mean},
            stage_log=stage_log,
        )

    def _build(self, approved, action, block_reason, cp, fmea, m_rpn,
               cb, utility, stage_log) -> DICDecision:
        return DICDecision(
            approved=approved, action=action, block_reason=block_reason,
            critical_path=cp, fmea=fmea, max_rpn=m_rpn,
            circuit_breaker=cb, utility=utility,
            bayes={"llm_risk_mean": self.llm_risk_tracker.mean},
            stage_log=stage_log,
        )
