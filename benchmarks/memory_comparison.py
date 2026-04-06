"""
Memory comparison: Run all three benchmarks twice.
Pass 1: no prior memory (cold start — uniform Beta priors).
Pass 2: with memory saved from Pass 1 (warm start).
"""

import os
import statistics
from typing import Any, Dict, List

from core.episodic_memory import EpisodicMemory
from benchmarks.snake.run      import run_episode as snake_ep, MEMORY_PATH as SNAKE_MEM
from benchmarks.maze.run       import run_episode as maze_ep,  MEMORY_PATH as MAZE_MEM
from benchmarks.gridworld.run  import run_episode as gw_ep,    MEMORY_PATH as GW_MEM

N = 50


# ─── helpers ─────────────────────────────────────────────────────────────────

def run_snake(priors) -> List[Dict[str, Any]]:
    results, rp = [], priors
    mem = EpisodicMemory(SNAKE_MEM)
    for i in range(N):
        s  = snake_ep(seed=i, priors=rp)
        rp = mem.extract(s["bayes_state"])
        results.append(s)
    mem.save(rp)
    return results


def run_maze(priors) -> List[Dict[str, Any]]:
    results, rp = [], priors
    mem = EpisodicMemory(MAZE_MEM)
    for i in range(N):
        s  = maze_ep(seed=i, priors=rp)
        rp = mem.extract(s["bayes_state"])
        results.append(s)
    mem.save(rp)
    return results


def run_gw(priors) -> List[Dict[str, Any]]:
    results, rp = [], priors
    mem = EpisodicMemory(GW_MEM)
    for i in range(N):
        s  = gw_ep(seed=i, priors=rp)
        rp = mem.extract(s["bayes_state"])
        results.append(s)
    mem.save(rp)
    return results


def snake_stats(results):
    scores  = [r["score"]        for r in results]
    steps   = [r["steps"]        for r in results]
    rewards = [r["total_reward"] for r in results]
    return {
        "avg_score":  statistics.mean(scores),
        "max_score":  max(scores),
        "avg_steps":  statistics.mean(steps),
        "avg_reward": statistics.mean(rewards),
    }


def nav_stats(results):
    solved  = sum(1 for r in results if r["reached_goal"])
    killed  = sum(1 for r in results if not r["reached_goal"] and r["steps"] < 300)
    timeout = sum(1 for r in results if not r["reached_goal"] and r["steps"] >= 300)
    steps   = [r["steps"] for r in results]
    rewards = [r["total_reward"] for r in results]
    return {
        "solved":     solved,
        "killed":     killed,
        "timeout":    timeout,
        "avg_steps":  statistics.mean(steps),
        "avg_reward": statistics.mean(rewards),
    }


def pct(a, b):
    """Percentage change from b to a."""
    if b == 0:
        return "n/a"
    d = (a - b) / abs(b) * 100
    return f"{'+' if d >= 0 else ''}{d:.1f}%"


def print_compare(title, rows):
    # rows: (metric, cold_val, warm_val, higher_is_better, fmt_fn)
    w = 64
    print(f"\n{'═'*w}")
    print(f"  {title}")
    print(f"{'═'*w}")
    print(f"  {'Metric':<28} {'No Memory':>10}  {'With Memory':>11}  {'Δ':>8}")
    print(f"  {'─'*w}")
    for metric, cold, warm, hi, fmt in rows:
        delta = pct(warm, cold)
        marker = " ✓" if ((warm > cold) == hi) else (" ✗" if warm != cold else "  ")
        print(f"  {metric:<28} {fmt(cold):>10}  {fmt(warm):>11}  {delta+marker:>10}")
    print(f"  {'─'*w}")


# ─── main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── PASS 1: no memory ────────────────────────────────────────────────────
    print("\n── Pass 1: No Memory (cold start) ──────────────────────────")

    print("  Snake  …", end=" ", flush=True)
    snake1 = run_snake(None)
    print("done.")

    print("  Maze   …", end=" ", flush=True)
    maze1 = run_maze(None)
    print("done.")

    print("  Gridworld …", end=" ", flush=True)
    gw1 = run_gw(None)
    print("done.")

    ss1 = snake_stats(snake1)
    ms1 = nav_stats(maze1)
    gs1 = nav_stats(gw1)

    # ── PASS 2: with memory ──────────────────────────────────────────────────
    print("\n── Pass 2: With Memory (warm start from Pass 1) ────────────")

    snake_mem = EpisodicMemory(SNAKE_MEM)
    maze_mem  = EpisodicMemory(MAZE_MEM)
    gw_mem    = EpisodicMemory(GW_MEM)

    print("  Snake  …", end=" ", flush=True)
    snake2 = run_snake(snake_mem.load())
    print("done.")

    print("  Maze   …", end=" ", flush=True)
    maze2 = run_maze(maze_mem.load())
    print("done.")

    print("  Gridworld …", end=" ", flush=True)
    gw2 = run_gw(gw_mem.load())
    print("done.")

    ss2 = snake_stats(snake2)
    ms2 = nav_stats(maze2)
    gs2 = nav_stats(gw2)

    # ── RESULTS ──────────────────────────────────────────────────────────────

    fp  = lambda v: f"{v:.1f}"
    fi  = lambda v: str(int(v))
    fof = lambda v: f"{int(v)}/{N}"  # fraction of N

    print_compare("SNAKE — 10×10 grid (50 episodes)", [
        ("Avg score",  ss1["avg_score"],  ss2["avg_score"],  True,  fp),
        ("Max score",  ss1["max_score"],  ss2["max_score"],  True,  fi),
        ("Avg steps",  ss1["avg_steps"],  ss2["avg_steps"],  False, fp),
        ("Avg reward", ss1["avg_reward"], ss2["avg_reward"], True,  fp),
    ])

    print_compare("MAZE — 15×15 recursive backtracker (50 episodes)", [
        ("Solved",     ms1["solved"],     ms2["solved"],     True,  fof),
        ("Killed",     ms1["killed"],     ms2["killed"],     False, fof),
        ("Timed out",  ms1["timeout"],    ms2["timeout"],    False, fof),
        ("Avg steps",  ms1["avg_steps"],  ms2["avg_steps"],  False, fp),
        ("Avg reward", ms1["avg_reward"], ms2["avg_reward"], True,  fp),
    ])

    print_compare("GRIDWORLD — 15×15, 5 hazards (50 episodes)", [
        ("Solved",     gs1["solved"],     gs2["solved"],     True,  fof),
        ("Killed",     gs1["killed"],     gs2["killed"],     False, fof),
        ("Timed out",  gs1["timeout"],    gs2["timeout"],    False, fof),
        ("Avg steps",  gs1["avg_steps"],  gs2["avg_steps"],  False, fp),
        ("Avg reward", gs1["avg_reward"], gs2["avg_reward"], True,  fp),
    ])

    print(f"\n{'═'*64}")
    print("  Notes")
    print(f"{'═'*64}")
    print("  Pass 1: memory.json absent → uniform Beta(1,1) priors.")
    print("  Pass 2: priors loaded from memory.json saved by Pass 1.")
    print("  Within each pass, priors carry episode-to-episode.")
    print("  ✓ = memory helped   ✗ = memory hurt or no change")
    print(f"{'═'*64}\n")
