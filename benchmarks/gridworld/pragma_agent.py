from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from .gridworld_env import GridworldEnv, ACTIONS
from .critical_path import critical_path_estimate
from .risk_fmea import fmea_table, max_rpn
from .tornado import tornado_rank
from .bayes import BetaTracker
from .circuit_breaker import CircuitBreaker, CircuitBreakerConfig


@dataclass
class DecisionReport:
    action:          str
    blocked_actions: Dict[str, str]
    per_action:      Dict[str, Dict[str, Any]]
    tornado:         List[Dict[str, Any]]
    bayes:           Dict[str, float]


class PragmaGridworldAgent:
    """
    DIC-compliant agent for the Dynamic Threat Gridworld benchmark.

    Pipeline per step:
      1. Branching      — filter immediate hazard collisions; all 5 actions considered
      2. Critical Path  — Monte Carlo rollouts with hazard movement: p_death, p_trap
      3. FMEA           — RPN = S × O × D per failure mode
      4. Decision Gate  — block actions with RPN ≥ fmea_rpn_threshold
      5. Circuit Breaker — OK / WARN / SLOW / STOP
      6. Selection      — utility: survival + goal pull - proximity - revisits - risk
      7. Belief Update  — Beta trackers for collision_rate and trap_rate

    Unlike Snake and Maze, the p_death signal here is genuinely differentiating
    across actions because hazards create local risk variation. WAIT is a first-class
    action: it can have lower p_death than directional moves when hazards are close.
    """

    def __init__(
        self,
        fmea_rpn_threshold:    int = 2400,
        rollouts:              int = 200,
        depth:                 int = 50,
        seed:                  int = 0,
        circuit_breaker_config: CircuitBreakerConfig | None = None,
        priors: dict | None = None,
    ):
        self.fmea_rpn_threshold = fmea_rpn_threshold
        self.rollouts           = rollouts
        self.depth              = depth
        self.seed               = seed

        p = priors or {}
        collision_a, collision_b = p.get("collision_rate", (1.0, 1.0))
        trap_a,      trap_b      = p.get("trap_rate",      (1.0, 1.0))
        self.collision_tracker = BetaTracker(collision_a, collision_b)
        self.trap_tracker      = BetaTracker(trap_a,      trap_b)

        self.circuit_breaker = CircuitBreaker(circuit_breaker_config)

    def choose_action(self, env: GridworldEnv) -> Tuple[str, DecisionReport]:
        hpos = env.hazard_positions()

        # 1. Branching — immediate collision = certain death, block outright
        per_action: Dict[str, Dict[str, Any]] = {}
        blocked:    Dict[str, str]            = {}

        for a in ACTIONS:
            dr, dc  = ACTIONS[a]
            r, c    = env.agent_pos
            next_pos = (r + dr, c + dc)

            # Out-of-bounds or immediate hazard collision
            immediate = env.is_dead_move(a)

            if immediate:
                table  = fmea_table(1.0, 0.0, immediate_collision=True)
                m_rpn  = max_rpn(table)
                per_action[a] = {
                    "immediate_collision": True,
                    "critical_path": {"p_death": 1.0, "p_trap": 0.0,
                                      "expected_steps_to_death": 1.0},
                    "fmea":          {k: vars(v) for k, v in table.items()},
                    "max_rpn":       m_rpn,
                    "circuit_breaker": {"state": "stop",
                                        "reason": "STOP: immediate hazard collision"},
                    "utility":       -1e9,
                    "proximity":     0,
                    "revisits":      0,
                }
                blocked[a] = "Blocked: immediate hazard collision"
                continue

            # 2. Critical Path Estimation
            cp = critical_path_estimate(
                env, a,
                rollouts=self.rollouts,
                depth=self.depth,
                seed_base=self.seed,
            )

            # Blend MC estimate with Beta tracker mean (episodic memory)
            p_death_adj = 0.7 * cp.p_death + 0.3 * self.collision_tracker.mean
            p_trap_adj  = 0.7 * cp.p_trap  + 0.3 * self.trap_tracker.mean

            # 3. FMEA
            table = fmea_table(p_death_adj, p_trap_adj, immediate_collision=False)
            m_rpn = max_rpn(table)

            # 5. Circuit Breaker
            cb = self.circuit_breaker.evaluate(m_rpn)

            # 6. Utility
            # next_pos for WAIT is the current position
            eval_pos  = next_pos if env._in_bounds(next_pos) else env.agent_pos
            dist      = env.manhattan_to_goal(eval_pos)
            proximity = env.proximity_score(eval_pos)
            revisits  = env.visit_counts.get(eval_pos, 0)

            utility = (
                (1.0 - p_death_adj) * 10.0
                + (1.0 - p_trap_adj) * 3.0
                - dist      * 1.5
                - proximity * 1.0
                - revisits  * 1.0
                - (m_rpn / 1000.0)
            )

            per_action[a] = {
                "critical_path": {
                    "p_death":               p_death_adj,
                    "p_trap":                p_trap_adj,
                    "mc_p_death":            cp.p_death,
                    "mc_p_trap":             cp.p_trap,
                    "expected_steps_to_death": cp.expected_steps_to_death,
                },
                "fmea":          {k: vars(v) for k, v in table.items()},
                "max_rpn":       m_rpn,
                "circuit_breaker": {"state": cb.state.value, "reason": cb.reason},
                "utility":       utility,
                "proximity":     proximity,
                "revisits":      revisits,
            }

            # 4. Decision Gate
            if m_rpn >= self.fmea_rpn_threshold or cb.state.value == "stop":
                blocked[a] = cb.reason

        allowed = [a for a in ACTIONS if a not in blocked]
        if not allowed:
            # All blocked: fall back to least-risk candidate
            allowed = sorted(ACTIONS.keys(), key=lambda x: per_action[x]["max_rpn"])

        best = max(allowed, key=lambda x: per_action[x]["utility"])

        # Tornado rank
        factors = {
            "p_death":   -per_action[best]["critical_path"]["p_death"],
            "p_trap":    -per_action[best]["critical_path"]["p_trap"],
            "proximity": -per_action[best]["proximity"] / 5.0,
            "max_rpn":   -per_action[best]["max_rpn"] / 100.0,
        }
        ranked = tornado_rank(factors)

        return best, DecisionReport(
            action=best,
            blocked_actions=blocked,
            per_action=per_action,
            tornado=[{"factor": f.name, "impact": f.impact} for f in ranked],
            bayes={
                "collision_rate_mean": self.collision_tracker.mean,
                "trap_rate_mean":      self.trap_tracker.mean,
            },
        )

    def update_bayes(self, report: DecisionReport):
        """7. Belief Update — update Beta trackers from observed risk signals."""
        cp = report.per_action[report.action]["critical_path"]
        self.collision_tracker.update(cp["p_death"] > 0.25)
        self.trap_tracker.update(cp["p_trap"] > 0.25)
