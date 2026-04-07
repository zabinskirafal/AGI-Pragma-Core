from dataclasses import dataclass
from enum import Enum

class CBState(Enum):
    OK = "ok"
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
    state: CBState
    reason: str

class CircuitBreaker:
    """
    Autonomy safety circuit breaker.
    - OK   : normal operation
    - WARN : autonomy allowed, but flagged
    - SLOW : restrict exploration / depth
    - STOP : block action, force no-op / safest fallback
    """
    def __init__(self, config: CircuitBreakerConfig):
        self.cfg = config

    def evaluate(self, max_rpn: int) -> CircuitBreakerResult:
        if max_rpn >= self.cfg.stop_rpn:
            return CircuitBreakerResult(
                CBState.STOP,
                f"STOP: RPN {max_rpn} >= {self.cfg.stop_rpn}"
            )
        if max_rpn >= self.cfg.slow_rpn:
            return CircuitBreakerResult(
                CBState.SLOW,
                f"SLOW: RPN {max_rpn} >= {self.cfg.slow_rpn}"
            )
        if max_rpn >= self.cfg.warn_rpn:
            return CircuitBreakerResult(
                CBState.WARN,
                f"WARN: RPN {max_rpn} >= {self.cfg.warn_rpn}"
            )
        return CircuitBreakerResult(CBState.OK, "OK")
