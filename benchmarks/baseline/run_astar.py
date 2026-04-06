"""
Three-way comparison: Random baseline vs A* baseline vs AGI Pragma.

Run:
    python3 -m benchmarks.baseline.run_astar
"""

import random
import statistics
from typing import Any, Dict, List

from benchmarks.snake.snake_env      import SnakeEnv
from benchmarks.maze.maze_env        import MazeEnv
from benchmarks.gridworld.gridworld_env import GridworldEnv

from benchmarks.snake.run      import run_episode as snake_pragma
from benchmarks.maze.run       import run_episode as maze_pragma
from benchmarks.gridworld.run  import run_episode as gw_pragma

from benchmarks.baseline.astar import (
    snake_astar_action,
    maze_astar_action,
    gridworld_astar_action,
)

N = 50


# ══════════════════════════════════════════════════════════════════════════════
# Random baseline runners (identical to run_all.py)
# ══════════════════════════════════════════════════════════════════════════════

def run_snake_random(seed: int) -> Dict[str, Any]:
    rng = random.Random(seed)
    env = SnakeEnv(width=10, height=10, seed=seed)
    env.reset()
    total_reward = 0.0
    t = 0
    for t in range(300):
        safe = env.safe_actions()
        action = rng.choice(safe) if safe else rng.choice(["U", "D", "L", "R"])
        res = env.step(action)
        total_reward += res.reward
        if not res.alive:
            break
    return {"seed": seed, "steps": t + 1, "score": env.score, "total_reward": total_reward}


def run_maze_random(seed: int) -> Dict[str, Any]:
    rng = random.Random(seed)
    env = MazeEnv(seed=seed)
    env.reset()
    total_reward = 0.0
    t = 0
    while env.alive:
        safe = env.safe_actions()
        action = rng.choice(safe) if safe else rng.choice(["U", "D", "L", "R"])
        res = env.step(action)
        total_reward += res.reward
        t += 1
        if not res.alive:
            break
    return {
        "seed": seed, "steps": env.steps, "score": env.score,
        "reached_goal": env.reached_goal, "total_reward": total_reward,
    }


def run_gw_random(seed: int) -> Dict[str, Any]:
    rng = random.Random(seed)
    env = GridworldEnv(seed=seed)
    env.reset()
    total_reward = 0.0
    while env.alive:
        safe = env.safe_actions()
        action = rng.choice(safe) if safe else rng.choice(["U", "D", "L", "R", "WAIT"])
        res = env.step(action)
        total_reward += res.reward
        if not res.alive:
            break
    return {
        "seed": seed, "steps": env.steps, "score": env.score,
        "reached_goal": env.reached_goal, "total_reward": total_reward,
    }


# ══════════════════════════════════════════════════════════════════════════════
# A* baseline runners
# ══════════════════════════════════════════════════════════════════════════════

def run_snake_astar(seed: int) -> Dict[str, Any]:
    rng = random.Random(seed)
    env = SnakeEnv(width=10, height=10, seed=seed)
    env.reset()
    total_reward = 0.0
    t = 0
    for t in range(300):
        action = snake_astar_action(env)
        if action is None or env.is_dead_move(action):
            # A* returned no path or path leads to death — fall back to safe random
            safe = env.safe_actions()
            action = rng.choice(safe) if safe else rng.choice(["U", "D", "L", "R"])
        res = env.step(action)
        total_reward += res.reward
        if not res.alive:
            break
    return {"seed": seed, "steps": t + 1, "score": env.score, "total_reward": total_reward}


def run_maze_astar(seed: int) -> Dict[str, Any]:
    rng = random.Random(seed)
    env = MazeEnv(seed=seed)
    env.reset()
    total_reward = 0.0
    while env.alive:
        action = maze_astar_action(env)
        if action is None:
            safe = env.safe_actions()
            action = rng.choice(safe) if safe else rng.choice(["U", "D", "L", "R"])
        res = env.step(action)
        total_reward += res.reward
        if not res.alive:
            break
    return {
        "seed": seed, "steps": env.steps, "score": env.score,
        "reached_goal": env.reached_goal, "total_reward": total_reward,
    }


def run_gw_astar(seed: int) -> Dict[str, Any]:
    rng = random.Random(seed)
    env = GridworldEnv(seed=seed)
    env.reset()
    total_reward = 0.0
    while env.alive:
        action = gridworld_astar_action(env)
        if action is None:
            safe = env.safe_actions()
            action = rng.choice(safe) if safe else rng.choice(["U", "D", "L", "R", "WAIT"])
        res = env.step(action)
        total_reward += res.reward
        if not res.alive:
            break
    return {
        "seed": seed, "steps": env.steps, "score": env.score,
        "reached_goal": env.reached_goal, "total_reward": total_reward,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Table printing
# ══════════════════════════════════════════════════════════════════════════════

def _delta(a: float, b: float, higher_is_better: bool = True) -> str:
    """Percentage change from b to a."""
    if b == 0:
        return "  n/a"
    diff = a - b
    pct  = diff / abs(b) * 100
    sign = "+" if diff >= 0 else ""
    better = (diff > 0) == higher_is_better
    marker = " ✓" if better else (" ✗" if diff != 0 else "  ")
    return f"{sign}{pct:.0f}%{marker}"


def _fmt(val: float, d: int = 1) -> str:
    return f"{val:.{d}f}"


def print_table(title: str, rows: List[tuple], width: int = 74) -> None:
    # rows: (metric, rnd_raw, astar_raw, pragma_raw, higher_is_better, fmt_fn)
    bar = "─" * width
    print(f"\n{'═' * width}")
    print(f"  {title}")
    print(f"{'═' * width}")
    print(f"  {'Metric':<26} {'Random':>8}  {'A*':>8}  {'Pragma':>8}  {'Δ(A*/Rnd)':>10}  {'Δ(Pragma/A*)':>13}")
    print(f"  {bar}")
    for metric, rnd, astar_v, pragma, hi, fmt in rows:
        d1 = _delta(astar_v,  rnd,     hi)
        d2 = _delta(pragma,   astar_v, hi)
        print(f"  {metric:<26} {fmt(rnd):>8}  {fmt(astar_v):>8}  {fmt(pragma):>8}  {d1:>10}  {d2:>13}")
    print(f"  {bar}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print(f"\nRunning {N} episodes × 3 agents × 3 benchmarks …")
    print("(Pragma uses rollouts=200; A* and Random are instant)\n")

    # ── SNAKE ─────────────────────────────────────────────────────────────────
    print("Snake random …",  end=" ", flush=True)
    snake_rnd  = [run_snake_random(i)  for i in range(N)]
    print("done.")
    print("Snake A*     …",  end=" ", flush=True)
    snake_ast  = [run_snake_astar(i)   for i in range(N)]
    print("done.")
    print("Snake Pragma …",  end=" ", flush=True)
    snake_prag = [snake_pragma(seed=i, priors=None) for i in range(N)]
    print("done.")

    def _s(results): return [r["score"]        for r in results]
    def _t(results): return [r["steps"]        for r in results]
    def _r(results): return [r["total_reward"] for r in results]

    sr, sa, sp = _s(snake_rnd), _s(snake_ast), _s(snake_prag)
    tr, ta, tp = _t(snake_rnd), _t(snake_ast), _t(snake_prag)
    rr, ra, rp = _r(snake_rnd), _r(snake_ast), _r(snake_prag)

    fp = lambda v: f"{v:.1f}"
    fi = lambda v: str(int(v))

    print_table(f"SNAKE — 10×10 grid ({N} episodes each)", [
        ("Avg score",  statistics.mean(sr), statistics.mean(sa), statistics.mean(sp), True,  fp),
        ("Max score",  max(sr),             max(sa),             max(sp),             True,  fi),
        ("Avg steps",  statistics.mean(tr), statistics.mean(ta), statistics.mean(tp), False, fp),
        ("Avg reward", statistics.mean(rr), statistics.mean(ra), statistics.mean(rp), True,  fp),
    ])

    # ── MAZE ──────────────────────────────────────────────────────────────────
    print("\nMaze random  …", end=" ", flush=True)
    maze_rnd  = [run_maze_random(i)  for i in range(N)]
    print("done.")
    print("Maze A*      …", end=" ", flush=True)
    maze_ast  = [run_maze_astar(i)   for i in range(N)]
    print("done.")
    print("Maze Pragma  …", end=" ", flush=True)
    maze_prag = [maze_pragma(seed=i, priors=None) for i in range(N)]
    print("done.")

    def _solved(results): return sum(1 for r in results if r["reached_goal"])
    def _steps_solved(results):
        s = [r["steps"] for r in results if r["reached_goal"]]
        return s or [0]

    mr = {"solved": _solved(maze_rnd),  "steps": statistics.mean(_t(maze_rnd)),
          "steps_s": statistics.mean(_steps_solved(maze_rnd)),  "reward": statistics.mean(_r(maze_rnd))}
    ma = {"solved": _solved(maze_ast),  "steps": statistics.mean(_t(maze_ast)),
          "steps_s": statistics.mean(_steps_solved(maze_ast)),  "reward": statistics.mean(_r(maze_ast))}
    mp = {"solved": _solved(maze_prag), "steps": statistics.mean(_t(maze_prag)),
          "steps_s": statistics.mean(_steps_solved(maze_prag)), "reward": statistics.mean(_r(maze_prag))}

    fof = lambda v: f"{int(v)}/{N}"

    print_table(f"MAZE — 15×15 recursive backtracker ({N} episodes each)", [
        ("Solved",             mr["solved"],   ma["solved"],   mp["solved"],   True,  fof),
        ("Avg steps (all)",    mr["steps"],    ma["steps"],    mp["steps"],    False, fp),
        ("Avg steps (solved)", mr["steps_s"],  ma["steps_s"],  mp["steps_s"],  False, fp),
        ("Avg reward",         mr["reward"],   ma["reward"],   mp["reward"],   True,  fp),
    ])

    # ── GRIDWORLD ─────────────────────────────────────────────────────────────
    print("\nGridworld random  …", end=" ", flush=True)
    gw_rnd  = [run_gw_random(i)  for i in range(N)]
    print("done.")
    print("Gridworld A*      …", end=" ", flush=True)
    gw_ast  = [run_gw_astar(i)   for i in range(N)]
    print("done.")
    print("Gridworld Pragma  …", end=" ", flush=True)
    gw_prag = [gw_pragma(seed=i, priors=None) for i in range(N)]
    print("done.")

    def _gw_stats(results):
        return {
            "solved":  sum(1 for r in results if r["reached_goal"]),
            "killed":  sum(1 for r in results if not r["reached_goal"] and r["steps"] < 300),
            "timeout": sum(1 for r in results if not r["reached_goal"] and r["steps"] >= 300),
            "steps":   statistics.mean(_t(results)),
            "reward":  statistics.mean(_r(results)),
        }

    gr, ga, gp = _gw_stats(gw_rnd), _gw_stats(gw_ast), _gw_stats(gw_prag)

    print_table(f"GRIDWORLD — 15×15, 5 wandering hazards ({N} episodes each)", [
        ("Solved",           gr["solved"],  ga["solved"],  gp["solved"],  True,  fof),
        ("Killed by hazard", gr["killed"],  ga["killed"],  gp["killed"],  False, fof),
        ("Timed out",        gr["timeout"], ga["timeout"], gp["timeout"], False, fof),
        ("Avg steps",        gr["steps"],   ga["steps"],   gp["steps"],   False, fp),
        ("Avg reward",       gr["reward"],  ga["reward"],  gp["reward"],  True,  fp),
    ])

    # ── Notes ─────────────────────────────────────────────────────────────────
    w = 74
    print(f"\n{'═' * w}")
    print("  Notes")
    print(f"{'═' * w}")
    print("  Random : uniform choice from safe_actions() — no pathfinding, no DIC.")
    print("  A*     : shortest path to goal, replanned every step; falls back to")
    print("           safe random when no clear path exists.  No DIC pipeline.")
    print("           Snake A* uses body-aware neighbour filtering (tail unblocked).")
    print("           Gridworld A* treats current hazard positions as blocked cells.")
    print("  Pragma : full 7-stage DIC — FMEA, Monte Carlo, circuit breaker.")
    print("  Δ(A*/Rnd)    = value of pathfinding above random action selection.")
    print("  Δ(Pragma/A*) = value of DIC pipeline above optimal path planning.")
    print(f"  ✓ = improvement   ✗ = regression")
    print(f"{'═' * w}\n")
