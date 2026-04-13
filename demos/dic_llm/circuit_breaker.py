from collections import deque
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
    # Sliding-window composite thresholds
    composite_slow_window:     int = 10  # look-back window for SLOW
    composite_slow_threshold:  int = 5000
    composite_stop_window:     int = 20  # look-back window for STOP
    composite_stop_threshold:  int = 10000


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
        # Sliding window — keeps the last composite_stop_window RPNs
        self._window: deque[int] = deque(maxlen=self.cfg.composite_stop_window)

    @property
    def composite_10_sum(self) -> int:
        """Sum of the last composite_slow_window RPNs."""
        n = self.cfg.composite_slow_window
        return sum(list(self._window)[-n:])

    @property
    def composite_20_sum(self) -> int:
        """Sum of the last composite_stop_window RPNs."""
        return sum(self._window)

    def evaluate(self, max_rpn: int) -> CircuitBreakerResult:
        self._window.append(max_rpn)

        # ── Composite sliding-window checks ───────────────────────────── #
        # Evaluated first so per-action tier can only escalate, never soften.
        w20 = self.composite_20_sum
        w10 = self.composite_10_sum
        if w20 > self.cfg.composite_stop_threshold:
            composite = CircuitBreakerResult(
                CBState.STOP,
                f"STOP: composite RPN sum (last {self.cfg.composite_stop_window}) "
                f"{w20} > {self.cfg.composite_stop_threshold}",
            )
        elif w10 > self.cfg.composite_slow_threshold:
            composite = CircuitBreakerResult(
                CBState.SLOW,
                f"SLOW: composite RPN sum (last {self.cfg.composite_slow_window}) "
                f"{w10} > {self.cfg.composite_slow_threshold}",
            )
        else:
            composite = None

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
            # Composite STOP overrides SLOW tier
            if composite is not None and composite.state == CBState.STOP:
                return composite
            return CircuitBreakerResult(
                CBState.SLOW,
                f"SLOW: RPN {max_rpn} ≥ {self.cfg.slow_rpn}",
            )

        # ── WARN tier ────────────────────────────────────────────────── #
        if max_rpn >= self.cfg.warn_rpn:
            self._consecutive_slow = 0
            self._consecutive_warn += 1
            if self._consecutive_warn >= self.cfg.consecutive_warn_for_slow:
                # Composite STOP overrides consecutive-WARN-promoted SLOW
                if composite is not None and composite.state == CBState.STOP:
                    return composite
                return CircuitBreakerResult(
                    CBState.SLOW,
                    f"SLOW: {self._consecutive_warn} consecutive WARN proposals",
                )
            # Composite can promote WARN → SLOW or STOP
            if composite is not None:
                return composite
            return CircuitBreakerResult(
                CBState.WARN,
                f"WARN: RPN {max_rpn} ≥ {self.cfg.warn_rpn}",
            )

        # ── Clean proposal — reset all consecutive counters ───────────── #
        self._reset_counters()
        if composite is not None:
            return composite
        return CircuitBreakerResult(CBState.OK, "OK")

    def _reset_counters(self) -> None:
        self._consecutive_warn = 0
        self._consecutive_slow = 0
        self._consecutive_stop = 0
