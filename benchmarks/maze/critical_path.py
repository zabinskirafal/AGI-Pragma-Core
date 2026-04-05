from dataclasses import dataclass

from .maze_env import MazeEnv, ACTIONS


@dataclass
class CriticalPathResult:
    p_death: float       # fraction of rollouts that timed out before reaching goal
    p_trap: float        # fraction of rollouts that entered a dead end
    expected_steps_to_death: float


def critical_path_estimate(
    env: MazeEnv,
    first_action: str,
    rollouts: int = 200,
    depth: int = 25,
    seed_base: int = 0,
) -> CriticalPathResult:
    """
    Monte Carlo estimate of risk for a candidate action.

    p_death  : probability of timeout (step budget exhausted) within horizon.
               In maze, death = failure to reach goal, not physical collision.
    p_trap   : probability of stepping into a dead-end cell within horizon.
               Dead ends are cells with exactly one open neighbour — forced backtrack.
    expected_steps_to_death : mean steps until timeout within rollout, else depth.

    Each rollout takes first_action then follows a random walk for up to depth steps.
    """
    timeouts = 0
    traps = 0
    steps_sum = 0.0

    for i in range(rollouts):
        sim = env.clone(seed=seed_base + i)
        result = sim.step(first_action)

        # first action hit a wall (no-op) or immediately reached goal
        if result.reached_goal:
            steps_sum += 1
            continue
        if not result.alive:
            # timed out on first step (shouldn't happen at step 0, but guard it)
            timeouts += 1
            steps_sum += 1
            continue

        timed_out = False
        entered_trap = False

        for t in range(1, depth):
            if sim.is_dead_end(sim.agent_pos):
                entered_trap = True

            safe = sim.safe_actions()
            if not safe:
                # fully enclosed — treat as timeout
                timed_out = True
                steps_sum += t
                break

            action = sim.rng.choice(safe)
            r = sim.step(action)

            if r.reached_goal:
                steps_sum += t + 1
                break
            if not r.alive:
                timed_out = True
                steps_sum += t + 1
                break
        else:
            steps_sum += depth

        if timed_out:
            timeouts += 1
        if entered_trap:
            traps += 1

    p_death = timeouts / max(1, rollouts)
    p_trap = traps / max(1, rollouts)
    expected = steps_sum / max(1, rollouts)

    return CriticalPathResult(
        p_death=p_death,
        p_trap=p_trap,
        expected_steps_to_death=expected,
    )
