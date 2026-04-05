# AGI Pragma
**Decision Intelligence Framework for Controlled Autonomy under Uncertainty**

> *Intelligence as iterative decision-making, not opaque optimization.*

---

## Overview

**AGI Pragma** is a research framework for **decision intelligence under dynamic and uncertain environments**.

It does **not** attempt to replicate human cognition, consciousness, or emotions.  
Instead, it formalizes **human-like adaptive decision mechanisms**:
filtering reality, assessing risk, and updating beliefs **before acting**.

> Intelligent behavior emerges from systematically **constraining decisions**
> under uncertainty — not from unconstrained optimization.

---

## What AGI Pragma Is / Is Not

### AGI Pragma IS
- a methodology-first framework for risk-aware autonomy
- a **Decision Intelligence Core (DIC)** built around explicit decision gates
- a research artifact with reproducible benchmarks and auditable traces
- a foundation for safety-oriented autonomous systems

### AGI Pragma IS NOT
- a human-like AGI
- a black-box learning system
- a reward-maximization benchmark
- a production-ready general intelligence

---

## Core Architecture — Decision Intelligence Core (DIC)

Each decision follows a fixed and auditable pipeline:

**1. Branching** — enumerate feasible actions, eliminate invalid ones.

**2. Critical Path Estimation** — Monte Carlo rollouts estimate:
- probability of catastrophic failure,
- probability of entering irreversible traps,
- expected steps until failure.

**3. Risk Assessment (FMEA)** — each action scored by:
- Severity (S) × Occurrence (O) × Detection difficulty (D) = **RPN**

**4. Decision Integrity Gate** — actions exceeding risk threshold are blocked before execution.

**5. Circuit Breaker** — autonomy dynamically constrained:
- OK → WARN → SLOW → STOP

**6. Decision Selection** — utility balances survival probability, goal progress, residual risk.

**7. Belief Update** — Bayesian trackers update internal hazard estimates.

---

## Safety Model

Safety in AGI Pragma is **preventive**, not reactive.

- self-harm equals failure,
- no action bypasses risk evaluation,
- all decisions are auditable,
- autonomy is conditional, not absolute.

See: [docs/safety.md](docs/safety.md)

---

## Benchmark Results — Snake

**Agent:** PragmaSnakeAgent  
**Environment:** SnakeEnv 10×10

### v1.0 — 50 episodes (2026-04-05)

| Metric                 | Value         |
|------------------------|---------------|
| Average score          | 22.8          |
| Min / Max score        | 9 / 33        |
| Average reward         | 102.4         |
| Average steps          | 201           |
| Survived to step limit | 2/50          |
| Scores ≥ 25            | 21/50 (42%)   |
| Scores < 15            | 4/50 (8%)     |

### v0.1 — 10 episodes (2026-04-04) — initial run

| Metric                 | Value         |
|------------------------|---------------|
| Average score          | 25.0          |
| Min / Max score        | 18 / 33       |
| Average reward         | 113.1         |
| Average steps          | 214           |
| Survived to step limit | 0/10          |

> Note: v0.1 used only 10 seeds — higher average reflects small sample size.
> v1.0 with 50 seeds gives a more reliable picture of agent behavior.

### Key finding — passive vs active agent

| Config                      | Avg score | Avg reward |
|-----------------------------|-----------|------------|
| dist weight = 0.2 (passive) | 0.4       | ~0         |
| dist weight = 1.5 (active)  | 22.8      | 102.4      |

**One parameter change produced a 57× improvement in score.**

### Interpretation

42% of episodes scored 25 or above.  
Only 8% of episodes scored below 15 — rare failures, not systemic.  
The agent accepts risk to pursue goals and dies actively, not passively.

This confirms the core AGI Pragma trade-off:  
**safety ≠ passivity. Controlled risk is required for goal achievement.**

To run the benchmark (50 episodes, results written to `artifacts/snake/`):
```bash
python3 -m benchmarks.snake.run
```

See: [docs/benchmarks/snake.md](docs/benchmarks/snake.md)

---

## Benchmark Results — Maze

**Agent:** PragmaMazeAgent  
**Environment:** MazeEnv 15×15 (recursive backtracker generation)

### v2.0 — 50 episodes (2026-04-05)

| Metric                              | Value          |
|-------------------------------------|----------------|
| Solved                              | 50 / 50 (100%) |
| Steps — avg / min / max             | 46.1 / 24 / 76 |
| Score (steps remaining) — avg / min / max | 253.9 / 224 / 276 |

### Key finding — BFS distance vs manhattan distance

| Utility signal | Solved | Avg steps |
|----------------|--------|-----------|
| Manhattan distance (v1.1) | 4/50 (8%) | 277.9 |
| BFS path distance (v2.0)  | 50/50 (100%) | 46.1 |

**One signal change produced a 12.5× reduction in steps and a 100% solve rate.**

### Interpretation

Manhattan distance is unreliable in mazes where walls force long detours.
Replacing it with exact BFS path distance — precomputed once per maze, O(1) per lookup —
gave the utility function accurate topological information and immediately solved all episodes.

The FMEA and circuit breaker operated correctly throughout; the failure in v1.x was
a utility signal problem, not a safety pipeline problem.

To run the benchmark (50 episodes, results written to `artifacts/maze/`):
```bash
python3 -m benchmarks.maze.run
```

See: [docs/benchmarks/maze.md](docs/benchmarks/maze.md)

---

## Methodology

See: [docs/Methodology.md](docs/Methodology.md)

---

## Reproducibility

Each benchmark run produces:
- decision-level logs (JSONL)
- episode summaries (JSON)
- reproducible configurations

---

## Related Projects

**ChaosGym / Reverse Reality Sandbox** — physics-breaking simulation environment
designed to stress-test AGI Pragma's decision integrity under non-stationary rules.

- [AGI-Development](https://github.com/zabinskirafal/AGI-Development) — iterative development history and experimental branches of the AGI Pragma framework
- [developmental-agi-sandbox](https://github.com/zabinskirafal/developmental-agi-sandbox) — Unity-based reverse-physics sandbox environment for testing AGI Pragma under non-stationary world rules

---

## Licensing & Commercial Use

**Author:** Rafał Żabiński

**Free use:** academic research, education, non-commercial projects, open-source experimentation.

**Commercial use:** requires a separate written agreement with the author.

zabinskirafal@outlook.com  
https://www.linkedin.com/in/zabinskirafal

---

## Project Status

Current version: **v3.0.0**

AGI Pragma is an active research program, not a finished product.

Future work includes additional benchmarks, stronger baselines, and formal evaluation protocols.

---

## Citation

If you use this work in research, please cite via: [CITATION.cff](CITATION.cff)

**Rafał Żabiński** — Founder and original author (January 2026)