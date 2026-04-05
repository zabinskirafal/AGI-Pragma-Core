from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Tuple
import random

Point = Tuple[int, int]

ACTIONS: Dict[str, Tuple[int, int]] = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}

MAX_STEPS = 300


@dataclass
class StepResult:
    alive: bool
    reached_goal: bool
    reward: float


class MazeEnv:
    """
    15x15 grid maze environment.

    Grid values: 0 = open cell, 1 = wall.
    Maze is procedurally generated via recursive backtracker (DFS).
    Same seed always produces the same maze.

    - Hitting a wall is a no-op (agent stays in place, reward 0).
    - alive=False when step budget is exhausted (timeout = failure).
    - score = steps remaining when goal is reached; 0 if goal not reached.
    """

    SIZE = 15  # must be odd

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.rng = random.Random(seed)
        self.grid = self._generate()
        self.start: Point = (1, 1)
        self.goal: Point = (self.SIZE - 2, self.SIZE - 2)
        self._bfs_dist: Dict[Point, int] = self._precompute_bfs()
        self.reset()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self) -> dict:
        self.agent_pos: Point = self.start
        self.steps: int = 0
        self.alive: bool = True
        self.reached_goal: bool = False
        self.score: int = 0
        self.visit_counts: Dict[Point, int] = {self.start: 1}
        return self.obs()

    def clone(self, seed: int = 0) -> "MazeEnv":
        """Return a copy of this env with the same maze but independent RNG."""
        sim = MazeEnv.__new__(MazeEnv)
        sim.seed = seed
        sim.rng = random.Random(seed)
        sim.grid = [row[:] for row in self.grid]
        sim._bfs_dist = self._bfs_dist  # immutable after construction
        sim.start = self.start
        sim.goal = self.goal
        sim.agent_pos = self.agent_pos
        sim.steps = self.steps
        sim.alive = self.alive
        sim.reached_goal = self.reached_goal
        sim.score = self.score
        sim.visit_counts = dict(self.visit_counts)
        return sim

    def step(self, action: str) -> StepResult:
        if not self.alive:
            return StepResult(alive=False, reached_goal=self.reached_goal, reward=0.0)

        dr, dc = ACTIONS[action]
        r, c = self.agent_pos
        nr, nc = r + dr, c + dc

        if self._is_wall((nr, nc)):
            # wall no-op: position unchanged, no reward, no step cost
            return StepResult(alive=True, reached_goal=False, reward=0.0)

        self.agent_pos = (nr, nc)
        self.steps += 1
        self.visit_counts[self.agent_pos] = self.visit_counts.get(self.agent_pos, 0) + 1

        if self.agent_pos == self.goal:
            self.reached_goal = True
            self.alive = False
            self.score = MAX_STEPS - self.steps
            return StepResult(alive=False, reached_goal=True, reward=10.0)

        if self.steps >= MAX_STEPS:
            self.alive = False
            self.score = 0
            return StepResult(alive=False, reached_goal=False, reward=-10.0)

        return StepResult(alive=True, reached_goal=False, reward=-0.1)

    def is_dead_move(self, action: str) -> bool:
        """True if the action walks into a wall."""
        dr, dc = ACTIONS[action]
        r, c = self.agent_pos
        return self._is_wall((r + dr, c + dc))

    def is_dead_end(self, pos: Point) -> bool:
        """True if pos has exactly one open neighbour (forced backtrack)."""
        return len(self.open_neighbours(pos)) == 1

    def open_neighbours(self, pos: Point) -> List[Point]:
        """All adjacent cells that are not walls."""
        r, c = pos
        return [
            (r + dr, c + dc)
            for dr, dc in ACTIONS.values()
            if not self._is_wall((r + dr, c + dc))
        ]

    def safe_actions(self) -> List[str]:
        return [a for a in ACTIONS if not self.is_dead_move(a)]

    def bfs_to_goal(self, pos: Point) -> int:
        """Actual path length through open cells from pos to goal.
        Returns SIZE*SIZE as a large fallback if pos is unreachable (shouldn't occur)."""
        return self._bfs_dist.get(pos, self.SIZE * self.SIZE)

    def manhattan_to_goal(self, pos: Point) -> int:
        return abs(pos[0] - self.goal[0]) + abs(pos[1] - self.goal[1])

    def obs(self) -> dict:
        return {
            "agent_pos": self.agent_pos,
            "goal": self.goal,
            "steps": self.steps,
            "alive": self.alive,
            "reached_goal": self.reached_goal,
            "score": self.score,
            "manhattan": self.manhattan_to_goal(self.agent_pos),
        }

    # ------------------------------------------------------------------
    # BFS distance precomputation
    # ------------------------------------------------------------------

    def _precompute_bfs(self) -> Dict[Point, int]:
        """BFS from goal outward — gives exact path distance to goal for every open cell."""
        dist: Dict[Point, int] = {self.goal: 0}
        queue: deque = deque([self.goal])
        while queue:
            pos = queue.popleft()
            for nb in self.open_neighbours(pos):
                if nb not in dist:
                    dist[nb] = dist[pos] + 1
                    queue.append(nb)
        return dist

    # ------------------------------------------------------------------
    # Maze generation — recursive backtracker (DFS)
    # ------------------------------------------------------------------

    def _generate(self) -> List[List[int]]:
        size = self.SIZE
        grid = [[1] * size for _ in range(size)]

        def carve(r: int, c: int):
            grid[r][c] = 0
            directions = [(-2, 0), (2, 0), (0, -2), (0, 2)]
            self.rng.shuffle(directions)
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 < nr < size and 0 < nc < size and grid[nr][nc] == 1:
                    grid[r + dr // 2][c + dc // 2] = 0
                    carve(nr, nc)

        carve(1, 1)
        return grid

    def _is_wall(self, pos: Point) -> bool:
        r, c = pos
        if r < 0 or r >= self.SIZE or c < 0 or c >= self.SIZE:
            return True
        return self.grid[r][c] == 1
