from dataclasses import dataclass, field
from enum import Enum


class CBState(Enum):
    OK       = "ok"
    WARN     = "warn"
    SLOW     = "slow"
    STOP     = "stop"
    ESCALATE = "escalate"   # consecutive STOPs — human confirmation required


@dataclass
class CircuitBreakerConfig:
    warn_rpn:          int = 1800
    slow_rpn:          int = 2200
    stop_rpn:          int = 2600
    # Session-level escalation: consecutive high-RPN proposals
    consecutive_warn_for_slow: int = 3   # 3 WARN → SLOW
    consecutive_slow_for_stop: int = 2   # 2 SLOW → STOP
    escalate_threshold:        int = 3   # 3 consecutive STOP → ESCALATE


@dataclass
class CircuitBreakerResult:
    state:  CBState
    reason: str


class CircuitBreaker:
    """
    Stateful circuit breaker tracking consecutive risky LLM proposals.

    Per-action: RPN threshold check (same as benchmark pattern).
    Session-level: escalates if LLM repeatedly proposes risky actions,
    even if each individually falls below the stop threshold.

    State ladder:
        OK → WARN → SLOW → STOP → ESCALATE

    ESCALATE fires after ``escalate_threshold`` consecutive STOP-level
    proposals (RPN ≥ stop_rpn).  It signals that the agent is stuck —
    every candidate action is unsafe — and human confirmation is required
    before execution resumes.  The counter resets after ESCALATE fires.
    """

    def __init__(self, config: CircuitBreakerConfig | None = None):
        self.cfg               = config or CircuitBreakerConfig()
        self._consecutive_warn = 0
        self._consecutive_slow = 0
        self._consecutive_stop = 0

    def evaluate(self, max_rpn: int) -> CircuitBreakerResult:
        # ── STOP / ESCALATE tier ─────────────────────────────────────── #
        if max_rpn >= self.cfg.stop_rpn:
            self._consecutive_warn = 0
            self._consecutive_slow = 0
            self._consecutive_stop += 1

            if self._consecutive_stop >= self.cfg.escalate_threshold:
                self._reset_counters()
                return CircuitBreakerResult(
                    CBState.ESCALATE,
                    "ESCALATE: all actions exceed risk threshold, "
                    "human confirmation required",
                )

            return CircuitBreakerResult(
                CBState.STOP,
                f"STOP: RPN {max_rpn} ≥ {self.cfg.stop_rpn}",
            )

        # Any sub-STOP proposal resets the STOP counter
        self._consecutive_stop = 0

        # ── SLOW tier ────────────────────────────────────────────────── #
        if max_rpn >= self.cfg.slow_rpn:
            self._consecutive_warn = 0
            self._consecutive_slow += 1
            if self._consecutive_slow >= self.cfg.consecutive_slow_for_stop:
                return CircuitBreakerResult(
                    CBState.STOP,
                    f"STOP: {self._consecutive_slow} consecutive SLOW proposals",
                )
            return CircuitBreakerResult(
                CBState.SLOW,
                f"SLOW: RPN {max_rpn} ≥ {self.cfg.slow_rpn}",
            )

        # ── WARN tier ────────────────────────────────────────────────── #
        if max_rpn >= self.cfg.warn_rpn:
            self._consecutive_slow = 0
            self._consecutive_warn += 1
            if self._consecutive_warn >= self.cfg.consecutive_warn_for_slow:
                return CircuitBreakerResult(
                    CBState.SLOW,
                    f"SLOW: {self._consecutive_warn} consecutive WARN proposals",
                )
            return CircuitBreakerResult(
                CBState.WARN,
                f"WARN: RPN {max_rpn} ≥ {self.cfg.warn_rpn}",
            )

        # ── Clean proposal — reset all consecutive counters ───────────── #
        self._reset_counters()
        return CircuitBreakerResult(CBState.OK, "OK")

    def _reset_counters(self) -> None:
        self._consecutive_warn = 0
        self._consecutive_slow = 0
        self._consecutive_stop = 0
