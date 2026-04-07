import json
from typing import Any, Dict, List

from .snake_env import SnakeEnv
from .pragma_agent import PragmaSnakeAgent
from .artifacts import ArtifactWriter
from core.episodic_memory import EpisodicMemory

MEMORY_PATH = "artifacts/snake/memory.json"


def run_episode(
    seed: int = 0,
    steps: int = 300,
    rollouts: int = 200,
    depth: int = 25,
    rpn_threshold: int = 2400,
    log: bool = False,
    priors: dict | None = None,
) -> Dict[str, Any]:

    env = SnakeEnv(width=10, height=10, seed=seed)
    agent = PragmaSnakeAgent(
        fmea_rpn_threshold=rpn_threshold,
        rollouts=rollouts,
        depth=depth,
        seed=seed,
        priors=priors,
    )

    env.reset()
    writer = ArtifactWriter()
    total_reward = 0.0

    if log:
        writer.write_decision({
            "type": "run_header",
            "seed": seed,
            "config": {
                "steps": steps,
                "rollouts": rollouts,
                "depth": depth,
                "rpn_threshold": rpn_threshold,
            }
        })

    for t in range(steps):
        action, report = agent.choose_action(env)
        res = env.step(action)
        total_reward += res.reward

        agent.update_bayes(report)

        if log:
            writer.write_decision({
                "t": t,
                "action": action,
                "alive": res.alive,
                "ate": res.ate,
                "reward": res.reward,
                "report": {
                    "blocked_actions": report.blocked_actions,
                    "tornado": report.tornado,
                    "bayes": report.bayes,
                },
            })

        if not res.alive:
            break

    summary = {
        "seed": seed,
        "steps": t + 1,
        "score": env.score,
        "alive": env.alive,
        "total_reward": total_reward,
        "final_bayes": {
            "trap_rate_mean":  agent.trap_tracker.mean,
            "death_rate_mean": agent.death_tracker.mean,
        },
        "bayes_state": {
            "trap_rate":  {"a": agent.trap_tracker.a,  "b": agent.trap_tracker.b},
            "death_rate": {"a": agent.death_tracker.a, "b": agent.death_tracker.b},
        },
    }

    writer.write_summary(summary)
    return summary


if __name__ == "__main__":
    memory = EpisodicMemory(MEMORY_PATH)
    priors = memory.load()
    print(f"Episodic memory: {memory.describe(priors)}")

    results: List[Dict[str, Any]] = []
    running_priors = priors
    for i in range(50):
        summary = run_episode(seed=i, log=False, priors=running_priors)
        results.append(summary)
        running_priors = memory.extract(summary["bayes_state"])

    memory.save(running_priors)
    print(f"Memory saved → {MEMORY_PATH}")
    print(f"Final priors:   {memory.describe(running_priors)}")
    print(json.dumps(results, indent=2))
