from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

from .maze_env import MazeEnv, ACTIONS
from .critical_path import critical_path_estimate
from .risk_fmea import fmea_table, max_rpn
from .tornado import tornado_rank
from .bayes import BetaTracker
from .circuit_breaker import CircuitBreaker, CircuitBreakerConfig


@dataclass
class DecisionReport:
    action: str
    blocked_actions: Dict[str, str]
    per_action: Dict[str, Dict[str, Any]]
    tornado: List[Dict[str, Any]]
    bayes: Dict[str, float]


class PragmaMazeAgent:
    """
    DIC-compliant agent for the maze benchmark.

    Pipeline per step:
      1. Branching          — filter wall collisions
      2. Critical Path      — Monte Carlo rollouts: p_death (timeout), p_trap (dead end)
      3. FMEA               — RPN = S × O × D per failure mode
      4. Decision Gate      — block actions exceeding fmea_rpn_threshold
      5. Circuit Breaker    — OK / WARN / SLOW / STOP state
      6. Selection          — utility = survival + goal progress - risk
      7. Belief Update      — Beta trackers for dead-end and timeout rates
    """

    def __init__(
        self,
        fmea_rpn_threshold: int = 240,
        rollouts: int = 200,
        depth: int = 50,
        seed: int = 0,
        circuit_breaker_config: CircuitBreakerConfig | None = None,
    ):
        self.fmea_rpn_threshold = fmea_rpn_threshold
        self.rollouts = rollouts
        self.depth = depth
        self.seed = seed

        self.dead_end_tracker = BetaTracker(1, 1)
        self.timeout_tracker = BetaTracker(1, 1)

        self.circuit_breaker = CircuitBreaker(circuit_breaker_config)

    def choose_action(self, env: MazeEnv) -> Tuple[str, DecisionReport]:
        # 1. Branching — remove wall collisions
        candidates = [a for a in ACTIONS if not env.is_dead_move(a)]

        # If fully enclosed (shouldn't happen in a valid maze), allow all
        if not candidates:
            candidates = list(ACTIONS.keys())

        per_action: Dict[str, Dict[str, Any]] = {}
        blocked: Dict[str, str] = {}

        for a in candidates:
            # 2. Critical Path Estimation
            cp = critical_path_estimate(
                env, a,
                rollouts=self.rollouts,
                depth=self.depth,
                seed_base=self.seed,
            )

            # 3. FMEA
            table = fmea_table(cp.p_death, cp.p_trap)
            m_rpn = max_rpn(table)

            # 5. Circuit Breaker
            cb = self.circuit_breaker.evaluate(m_rpn)

            # 6. Utility
            # - survival:   reward paths less likely to timeout
            # - trap avoid: reward paths less likely to enter dead ends
            # - goal pull:  reward actions that reduce BFS path distance to goal
            # - revisit:    penalise moving to already-visited cells
            # - risk drag:  penalise high RPN
            next_pos = self._next_pos(env, a)
            dist = env.bfs_to_goal(next_pos)
            revisits = env.visit_counts.get(next_pos, 0)
            utility = (
                (1.0 - cp.p_death) * 10.0
                + (1.0 - cp.p_trap) * 3.0
                - dist * 1.5
                - revisits * 2.0
                - (m_rpn / 1000.0)
            )

            per_action[a] = {
                "critical_path": {
                    "p_death": cp.p_death,
                    "p_trap": cp.p_trap,
                    "expected_steps_to_death": cp.expected_steps_to_death,
                },
                "fmea": {k: vars(v) for k, v in table.items()},
                "max_rpn": m_rpn,
                "circuit_breaker": {
                    "state": cb.state.value,
                    "reason": cb.reason,
                },
                "utility": utility,
                "revisits": revisits,
            }

            # 4. Decision Gate
            if m_rpn >= self.fmea_rpn_threshold or cb.state.value == "stop":
                blocked[a] = cb.reason

        allowed = [a for a in candidates if a not in blocked]
        if not allowed:
            # All actions blocked: fall back to least-risk candidate
            allowed = sorted(candidates, key=lambda x: per_action[x]["max_rpn"])

        best = max(allowed, key=lambda x: per_action[x]["utility"])

        # Tornado rank: which risk factor drove this decision most
        factors = {
            "p_death": -per_action[best]["critical_path"]["p_death"],
            "p_trap":  -per_action[best]["critical_path"]["p_trap"],
            "max_rpn": -per_action[best]["max_rpn"] / 100.0,
        }
        ranked = tornado_rank(factors)

        return best, DecisionReport(
            action=best,
            blocked_actions=blocked,
            per_action=per_action,
            tornado=[{"factor": f.name, "impact": f.impact} for f in ranked],
            bayes={
                "dead_end_rate_mean": self.dead_end_tracker.mean,
                "timeout_rate_mean": self.timeout_tracker.mean,
            },
        )

    def update_bayes(self, report: DecisionReport):
        """7. Belief Update — update Beta trackers from observed risk signals."""
        cp = report.per_action[report.action]["critical_path"]
        self.dead_end_tracker.update(cp["p_trap"] > 0.25)
        self.timeout_tracker.update(cp["p_death"] > 0.25)

    def _next_pos(self, env: MazeEnv, action: str):
        from .maze_env import ACTIONS as _A
        r, c = env.agent_pos
        dr, dc = _A[action]
        return (r + dr, c + dc)
