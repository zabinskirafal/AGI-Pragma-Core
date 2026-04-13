from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional

from .file_action import FileAction, FileOp
from .risk_fmea import fmea_table, max_rpn, FMEAItem
from .critical_path import reversibility_profile, CriticalPathResult
from .circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitBreakerResult
from .bayes import BetaTracker
from core.scenario_weights import (
    ScenarioConfig, MonteCarloResult,
    get_scenario, monte_carlo_rollout,
)


SANDBOX_ROOT = Path(__file__).parent / "sandbox"

# Default RPN threshold (overridden per-scenario)
RPN_THRESHOLD = 2400


@dataclass
class DICDecision:
    approved:        bool
    action:          FileAction
    block_reason:    Optional[str]          # set if approved=False
    critical_path:   CriticalPathResult
    fmea:            Dict[str, Any]         # serialisable FMEA table
    max_rpn:         int
    circuit_breaker: CircuitBreakerResult
    utility:         float
    bayes:           Dict[str, float]
    stage_log:       list                   # ordered list of stage outcomes


class DICGovernor:
    """
    Full 7-stage DIC pipeline adapted for LLM-proposed file operations.

    Stage 1 — Branching:     scope gate (sandbox, path traversal)
    Stage 2 — Critical Path: static reversibility analysis
    Stage 3 — FMEA:          S×O×D×R per failure mode
    Stage 3b— Monte Carlo:   scenario-weighted rollout → adjusted RPN
    Stage 4 — Decision Gate: block if adjusted_rpn ≥ scenario threshold
    Stage 5 — Circuit Breaker: session-level escalation
    Stage 6 — Utility:       task progress benefit − risk penalty
    Stage 7 — Belief Update: Beta tracker for LLM risk rate
    """

    def __init__(
        self,
        sandbox_root:           Path = SANDBOX_ROOT,
        rpn_threshold:          int  = RPN_THRESHOLD,
        circuit_breaker_config: CircuitBreakerConfig | None = None,
        scenario:               str  = "normal",
    ):
        self.sandbox_root     = sandbox_root.resolve()
        self.scenario_cfg     = get_scenario(scenario)
        # scenario threshold takes precedence over explicit rpn_threshold only
        # when rpn_threshold is still at the default value
        self.rpn_threshold    = (
            self.scenario_cfg.rpn_threshold
            if rpn_threshold == RPN_THRESHOLD
            else rpn_threshold
        )
        self.circuit_breaker  = CircuitBreaker(circuit_breaker_config)
        self.llm_risk_tracker = BetaTracker(1.0, 1.0)
        self._step            = 0
        self.escalation_count = 0   # incremented each time ESCALATE fires

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def evaluate(self, action: FileAction) -> DICDecision:
        self._step += 1
        stage_log: list = []

        # ── 1. Branching ─────────────────────────────────────────────── #
        scope_ok, scope_msg = self._scope_check(action)
        stage_log.append({"stage": "branching", "pass": scope_ok, "detail": scope_msg})

        if not scope_ok:
            return self._block(action, scope_msg, stage_log)

        # ── 2. Critical Path ──────────────────────────────────────────── #
        cp = reversibility_profile(action, self.sandbox_root)
        stage_log.append({
            "stage":         "critical_path",
            "reversibility": cp.reversibility.value,
            "file_exists":   cp.file_exists,
            "p_irreversible": cp.p_irreversible,
            "side_effects":  cp.side_effects,
        })

        # ── 3. FMEA ───────────────────────────────────────────────────── #
        table  = fmea_table(action, cp.file_exists, self.llm_risk_tracker.mean)
        m_rpn  = max_rpn(table)
        fmea_serial = {k: vars(v) for k, v in table.items()}
        stage_log.append({
            "stage":   "fmea",
            "table":   fmea_serial,
            "max_rpn": m_rpn,
        })

        # ── 3b. Monte Carlo ───────────────────────────────────────────── #
        mc = monte_carlo_rollout(cp.p_irreversible, self.scenario_cfg)
        mc.base_rpn     = m_rpn
        mc.adjusted_rpn = int(m_rpn * mc.rpn_multiplier)
        stage_log.append({
            "stage":                  "monte_carlo",
            "scenario":               mc.scenario,
            "n_trials":               mc.n_trials,
            "p_catastrophic":         mc.p_catastrophic,
            "p_cascade":              mc.p_cascade,
            "p_undetected":           mc.p_undetected,
            "expected_damage":        mc.expected_damage,
            "detection_effectiveness": mc.detection_effectiveness,
            "rpn_multiplier":         mc.rpn_multiplier,
            "base_rpn":               mc.base_rpn,
            "adjusted_rpn":           mc.adjusted_rpn,
        })

        # ── 4. Decision Gate ─────────────────────────────────────────── #
        gate_blocked = mc.adjusted_rpn >= self.rpn_threshold
        stage_log.append({
            "stage":        "decision_gate",
            "max_rpn":      m_rpn,
            "adjusted_rpn": mc.adjusted_rpn,
            "threshold":    self.rpn_threshold,
            "blocked":      gate_blocked,
        })

        # ── 5. Circuit Breaker ────────────────────────────────────────── #
        cb = self.circuit_breaker.evaluate(mc.adjusted_rpn)
        stage_log.append({
            "stage":  "circuit_breaker",
            "state":  cb.state.value,
            "reason": cb.reason,
            "composite_rpn_sum": {
                "last_10": self.circuit_breaker.composite_10_sum,
                "last_20": self.circuit_breaker.composite_20_sum,
            },
        })

        cb_blocked   = cb.state.value == "stop"
        cb_escalated = cb.state.value == "escalate"

        # ── 6. Utility ───────────────────────────────────────────────── #
        utility = self._utility(action, cp, mc.adjusted_rpn)
        stage_log.append({"stage": "utility", "score": utility})

        # ── 7. Belief Update ──────────────────────────────────────────── #
        risky = mc.adjusted_rpn >= self.rpn_threshold // 2
        self.llm_risk_tracker.update(risky)
        stage_log.append({
            "stage":            "belief_update",
            "risky_signal":     risky,
            "llm_risk_mean":    self.llm_risk_tracker.mean,
            "beta":             repr(self.llm_risk_tracker),
        })

        # ── Final verdict ─────────────────────────────────────────────── #

        # ESCALATE takes priority: consecutive STOPs signal the agent is stuck
        if cb_escalated:
            self.escalation_count += 1
            reason = (
                "ESCALATE: all actions exceed risk threshold, "
                "human confirmation required"
            )
            return self._build(False, action, reason, cp, fmea_serial, m_rpn, cb, utility, stage_log)

        if gate_blocked or cb_blocked:
            reason = (
                cb.reason if cb_blocked else
                f"adjusted RPN {mc.adjusted_rpn} ≥ threshold {self.rpn_threshold} "
                f"(base {m_rpn} × {mc.rpn_multiplier}x [{self.scenario_cfg.name} scenario])"
            )
            return self._build(False, action, reason, cp, fmea_serial, mc.adjusted_rpn, cb, utility, stage_log)

        return self._build(True, action, None, cp, fmea_serial, mc.adjusted_rpn, cb, utility, stage_log)

    # ------------------------------------------------------------------ #
    #  Internals                                                           #
    # ------------------------------------------------------------------ #

    def _scope_check(self, action: FileAction):
        """Stage 1: verify path is inside sandbox, no traversal attacks."""
        if action.op == FileOp.DONE:
            return True, "DONE signal — no path to check"

        raw = action.path
        # Resolve relative paths against sandbox root
        p = Path(raw)
        if not p.is_absolute():
            resolved = (self.sandbox_root / p).resolve()
        else:
            resolved = p.resolve()

        # Must be inside sandbox
        try:
            resolved.relative_to(self.sandbox_root)
        except ValueError:
            return False, f"BLOCKED: path '{raw}' escapes sandbox ({self.sandbox_root})"

        # Reject path traversal components
        if ".." in Path(raw).parts:
            return False, f"BLOCKED: path traversal detected in '{raw}'"

        # Reject obvious system paths even if somehow inside sandbox
        for forbidden in ("/etc", "/usr", "/bin", "/sys", "/proc", "~/.ssh"):
            if raw.startswith(forbidden):
                return False, f"BLOCKED: forbidden system path '{raw}'"

        return True, f"PASS — '{resolved}' is within sandbox"

    def _utility(self, action: FileAction, cp: CriticalPathResult, m_rpn: int) -> float:
        """
        Stage 6: simple utility = task-progress benefit − risk penalty.
        READ is low benefit but also very low risk.
        WRITE is medium benefit.
        DELETE is negative benefit (rarely task-advancing).
        """
        op_benefit = {
            FileOp.READ:   3.0,
            FileOp.WRITE:  7.0,
            FileOp.DELETE: 1.0,
            FileOp.DONE:   10.0,
        }
        benefit     = op_benefit.get(action.op, 0.0)
        risk_penalty = (m_rpn / 1000.0) + cp.p_irreversible * 5.0
        return round(benefit - risk_penalty, 3)

    def _block(self, action, reason, stage_log) -> "DICDecision":
        from .risk_fmea import FMEAItem
        from .circuit_breaker import CBState, CircuitBreakerResult
        from .critical_path import CriticalPathResult, Reversibility
        dummy_cp = CriticalPathResult(
            reversibility=Reversibility.HIGH,
            file_exists=False,
            path_depth=0,
            content_size=0,
            side_effects=[],
            p_irreversible=1.0,
        )
        dummy_cb = CircuitBreakerResult(CBState.STOP, reason)
        return DICDecision(
            approved=False,
            action=action,
            block_reason=reason,
            critical_path=dummy_cp,
            fmea={},
            max_rpn=9999,
            circuit_breaker=dummy_cb,
            utility=-1e9,
            bayes={"llm_risk_mean": self.llm_risk_tracker.mean},
            stage_log=stage_log,
        )

    def _build(self, approved, action, block_reason, cp, fmea_serial, m_rpn, cb, utility, stage_log) -> "DICDecision":
        return DICDecision(
            approved=approved,
            action=action,
            block_reason=block_reason,
            critical_path=cp,
            fmea=fmea_serial,
            max_rpn=m_rpn,
            circuit_breaker=cb,
            utility=utility,
            bayes={"llm_risk_mean": self.llm_risk_tracker.mean},
            stage_log=stage_log,
        )
