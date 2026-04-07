import json
from typing import Any, Dict, List

from .maze_env import MazeEnv
from .pragma_agent import PragmaMazeAgent
from .artifacts import ArtifactWriter
from core.episodic_memory import EpisodicMemory

MEMORY_PATH = "artifacts/maze/memory.json"


def run_episode(
    seed: int = 0,
    max_steps: int = 300,
    rollouts: int = 200,
    depth: int = 50,
    rpn_threshold: int = 2400,
    log: bool = False,
    priors: dict | None = None,
) -> Dict[str, Any]:

    env = MazeEnv(seed=seed)
    agent = PragmaMazeAgent(
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
                "max_steps": max_steps,
                "rollouts": rollouts,
                "depth": depth,
                "rpn_threshold": rpn_threshold,
            },
        })

    t = 0
    while env.alive and t < max_steps:
        action, report = agent.choose_action(env)
        res = env.step(action)
        total_reward += res.reward

        agent.update_bayes(report)

        if log:
            writer.write_decision({
                "t": t,
                "action": action,
                "alive": res.alive,
                "reached_goal": res.reached_goal,
                "reward": res.reward,
                "agent_pos": env.agent_pos,
                "report": {
                    "blocked_actions": report.blocked_actions,
                    "bayes": report.bayes,
                },
            })

        t += 1

        if not res.alive:
            break

    summary = {
        "seed": seed,
        "steps": env.steps,
        "score": env.score,
        "reached_goal": env.reached_goal,
        "total_reward": total_reward,
        "final_bayes": {
            "dead_end_rate_mean": agent.dead_end_tracker.mean,
            "timeout_rate_mean":  agent.timeout_tracker.mean,
        },
        "bayes_state": {
            "dead_end_rate": {"a": agent.dead_end_tracker.a, "b": agent.dead_end_tracker.b},
            "timeout_rate":  {"a": agent.timeout_tracker.a,  "b": agent.timeout_tracker.b},
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

    solved = sum(1 for r in results if r["reached_goal"])
    scores = [r["score"] for r in results if r["reached_goal"]]
    steps  = [r["steps"] for r in results]

    print(f"Solved: {solved}/50")
    if scores:
        print(f"Score (steps remaining) — avg: {sum(scores)/len(scores):.1f}  "
              f"min: {min(scores)}  max: {max(scores)}")
    print(f"Steps taken — avg: {sum(steps)/len(steps):.1f}  "
          f"min: {min(steps)}  max: {max(steps)}")

    print(json.dumps(results, indent=2))
