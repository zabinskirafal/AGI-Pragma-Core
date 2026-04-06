"""
A* pathfinding utilities for benchmark baselines.

Generic A* + three per-environment next_action helpers:
  - snake_astar_action(env)     → action string, or None if no path
  - maze_astar_action(env)      → action string, or None if no path
  - gridworld_astar_action(env) → action string, or None if no path

Each helper returns the first step of the shortest safe path to the goal.
A fallback to safe_actions() random choice is the caller's responsibility.
"""

import heapq
from typing import Callable, Dict, List, Optional, Set, Tuple

Point = Tuple[int, int]


# ── Generic A* ────────────────────────────────────────────────────────────────

def astar(
    start:        Point,
    goal:         Point,
    neighbors_fn: Callable[[Point], List[Point]],
    heuristic_fn: Callable[[Point, Point], float],
) -> Optional[List[Point]]:
    """
    Return a path [start, ..., goal] or None if unreachable.
    Uses Manhattan distance as the default heuristic when heuristic_fn
    is provided by the caller.
    """
    open_heap: List[Tuple[float, Point]] = [(0.0, start)]
    came_from: Dict[Point, Point] = {}
    g: Dict[Point, float] = {start: 0.0}

    while open_heap:
        _, current = heapq.heappop(open_heap)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return list(reversed(path))

        for nb in neighbors_fn(current):
            tentative_g = g[current] + 1.0
            if nb not in g or tentative_g < g[nb]:
                came_from[nb] = current
                g[nb] = tentative_g
                f = tentative_g + heuristic_fn(nb, goal)
                heapq.heappush(open_heap, (f, nb))

    return None


def _manhattan(a: Point, b: Point) -> float:
    return float(abs(a[0] - b[0]) + abs(a[1] - b[1]))


# ── Snake A* ──────────────────────────────────────────────────────────────────

def snake_astar_action(env) -> Optional[str]:
    """
    Return the first A* step from snake head toward food.

    Blocked cells: all body cells except the tail (tail will vacate next turn).
    Coordinate system: (x, y) — same as SnakeEnv.
    """
    from benchmarks.snake.snake_env import ACTIONS

    head: Point = env.snake[0]
    food: Point = env.food
    width: int  = env.width
    height: int = env.height

    # Tail will move on the next step unless we eat (conservative: block all
    # body cells; then fall back if no path found, unblock tail and retry)
    def _path(blocked: Set[Point]) -> Optional[List[Point]]:
        def neighbors(p: Point) -> List[Point]:
            x, y = p
            return [
                (x + dx, y + dy)
                for dx, dy in ACTIONS.values()
                if 0 <= x + dx < width
                and 0 <= y + dy < height
                and (x + dx, y + dy) not in blocked
            ]
        return astar(head, food, neighbors, _manhattan)

    # First try: block all body cells (conservative — correct when eating)
    body_set = set(env.snake)
    path = _path(body_set)

    # Second try: unblock tail (correct when not eating — tail vacates)
    if path is None and len(env.snake) > 1:
        body_no_tail = set(env.snake[:-1])
        path = _path(body_no_tail)

    if path is None or len(path) < 2:
        return None

    # Map (dx, dy) back to action name
    dx = path[1][0] - path[0][0]
    dy = path[1][1] - path[0][1]
    delta_to_action = {v: k for k, v in ACTIONS.items()}
    return delta_to_action.get((dx, dy))


# ── Maze A* ───────────────────────────────────────────────────────────────────

def maze_astar_action(env) -> Optional[str]:
    """
    Return the first A* step from agent position toward the goal.

    Blocked cells: walls (env._is_wall). Maze is static — A* always finds
    the shortest path if one exists.
    Coordinate system: (row, col) — same as MazeEnv.
    """
    from benchmarks.maze.maze_env import ACTIONS

    start: Point = env.agent_pos
    goal:  Point = env.goal

    def neighbors(p: Point) -> List[Point]:
        r, c = p
        return [
            (r + dr, c + dc)
            for dr, dc in ACTIONS.values()
            if not env._is_wall((r + dr, c + dc))
        ]

    path = astar(start, goal, neighbors, _manhattan)

    if path is None or len(path) < 2:
        return None

    dr = path[1][0] - path[0][0]
    dc = path[1][1] - path[0][1]
    delta_to_action = {v: k for k, v in ACTIONS.items()}
    return delta_to_action.get((dr, dc))


# ── Gridworld A* ──────────────────────────────────────────────────────────────

def gridworld_astar_action(env) -> Optional[str]:
    """
    Return the first A* step from agent toward goal, treating current hazard
    positions as impassable.  Replanned every step — hazards move so there is
    no value in caching.  WAIT is returned when all paths are hazard-blocked
    but waiting is safe (hazards may clear next turn).

    Coordinate system: (row, col) — same as GridworldEnv.
    """
    from benchmarks.gridworld.gridworld_env import ACTIONS, SIZE

    start:   Point     = env.agent_pos
    goal:    Point     = env.goal
    hazards: Set[Point] = env.hazard_positions()

    def neighbors(p: Point) -> List[Point]:
        r, c = p
        return [
            (r + dr, c + dc)
            for action, (dr, dc) in ACTIONS.items()
            if action != "WAIT"
            and 0 <= r + dr < SIZE
            and 0 <= c + dc < SIZE
            and (r + dr, c + dc) not in hazards
        ]

    path = astar(start, goal, neighbors, _manhattan)

    if path is None or len(path) < 2:
        # No clear path — prefer WAIT if it is safe, otherwise random safe action
        return "WAIT" if "WAIT" in env.safe_actions() else None

    dr = path[1][0] - path[0][0]
    dc = path[1][1] - path[0][1]
    delta_to_action = {
        v: k for k, v in ACTIONS.items() if k != "WAIT"
    }
    return delta_to_action.get((dr, dc))
