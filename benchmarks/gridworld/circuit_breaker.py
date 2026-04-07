from dataclasses import dataclass
from enum import Enum


class CBState(Enum):
    OK   = "ok"
    WARN = "warn"
    SLOW = "slow"
    STOP = "stop"


@dataclass
class CircuitBreakerConfig:
    warn_rpn: int = 1800
    slow_rpn: int = 2200
    stop_rpn: int = 2600


@dataclass
class CircuitBreakerResult:
    state:  CBState
    reason: str


class CircuitBreaker:
    """
    Autonomy safety circuit breaker.
    Thresholds match Snake and Maze for cross-benchmark comparability.

    - OK   : normal autonomous operation
    - WARN : autonomy allowed, elevated risk flagged
    - SLOW : restrict rollout depth and exploration
    - STOP : block action, force safest fallback
    """

    def __init__(self, config: CircuitBreakerConfig | None = None):
        self.cfg = config or CircuitBreakerConfig()

    def evaluate(self, max_rpn: int) -> CircuitBreakerResult:
        if max_rpn >= self.cfg.stop_rpn:
            return CircuitBreakerResult(
                CBState.STOP, f"STOP: RPN {max_rpn} >= {self.cfg.stop_rpn}"
            )
        if max_rpn >= self.cfg.slow_rpn:
            return CircuitBreakerResult(
                CBState.SLOW, f"SLOW: RPN {max_rpn} >= {self.cfg.slow_rpn}"
            )
        if max_rpn >= self.cfg.warn_rpn:
            return CircuitBreakerResult(
                CBState.WARN, f"WARN: RPN {max_rpn} >= {self.cfg.warn_rpn}"
            )
        return CircuitBreakerResult(CBState.OK, "OK")
