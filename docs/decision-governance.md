# Decision Governance for Agentic AI

## The Problem

Modern AI agents — LLMs with tool access, autonomous coding assistants, multi-agent
pipelines — share a structural weakness: they act before they evaluate.

A language model generates a candidate action (a shell command, an API call, a file
write) and executes it. The model's internal confidence is not a risk score.
It does not estimate the probability of catastrophic failure. It does not check
whether the action is reversible. It does not constrain itself when uncertainty is high.

Post-hoc monitoring catches failures after they occur. Prompt engineering nudges
behavior but provides no enforcement. Neither approach answers the question that
matters in an agentic context: **is this action safe to execute right now, given
what we know about the current state?**

---

## The DIC as a Governance Layer

AGI Pragma's Decision Intelligence Core (DIC) is a structured filter that sits
between action generation and action execution.

Its seven stages run before any action is committed:

**1. Branching** — enumerate candidate actions and eliminate physically or logically
impossible ones. The action space is constrained before evaluation begins.

**2. Critical Path Estimation** — Monte Carlo rollouts project each candidate forward
over a finite horizon, estimating:
- probability of catastrophic failure within that horizon,
- probability of entering a trap or irreversible state,
- expected steps until failure.

This gives each action a forward-looking risk profile, not just a point estimate.

**3. FMEA Risk Scoring** — each candidate is scored by:
> **RPN = Severity × Occurrence × Detection difficulty**

Severity captures consequence magnitude. Occurrence is drawn from the Monte Carlo
estimates. Detection captures how silently a failure mode manifests — silent failures
score higher and are penalised more heavily.

**4. Decision Integrity Gate** — any action whose RPN exceeds a configurable threshold
is blocked before execution. The gate is a hard constraint, not a soft preference.

**5. Circuit Breaker** — autonomy is dynamically constrained based on the current
risk profile: OK → WARN → SLOW → STOP. When cumulative risk is elevated, the system
reduces its own decision depth before the operator intervenes.

**6. Decision Selection** — among actions that cleared the gate, utility balances
survival probability, goal progress, and residual risk.

**7. Belief Update** — Bayesian trackers update hazard estimates after each action,
so the system's risk model sharpens over time rather than remaining static.

The result is a decision that is **auditable at every stage**: which actions were
blocked, why, what the RPN was, and what the circuit breaker state was.

---

## Why This Matters for the Agentic AI Era

As AI agents take more real-world actions — executing code, calling external APIs,
modifying files, spawning sub-agents — the cost of a wrong action increases
qualitatively. A hallucinated sentence is correctable. A dropped database table,
a sent email, or a committed security misconfiguration may not be.

The DIC addresses this with three properties that prompt-level interventions cannot provide:

**Pre-execution enforcement.** The gate blocks actions before they run. There is no
path from candidate to execution that bypasses risk evaluation. In both the Snake and
maze benchmarks, the circuit breaker engaged correctly throughout — it was never
circumvented, regardless of how high the agent's utility estimate was for a given action.

**Calibrated autonomy.** The circuit breaker's four states (OK / WARN / SLOW / STOP)
allow the system to reduce its own scope of action as risk increases, without requiring
operator intervention at every step. Autonomy is conditional, not binary.

**Domain-agnostic safety.** The DIC pipeline makes no assumptions about the underlying
action generator. It evaluates outputs, not internals. An LLM, a rule-based planner,
or a neural policy can sit upstream — the governance layer is indifferent to the source.
The maze benchmark demonstrated this directly: when the utility signal was miscalibrated
(manhattan distance), the safety pipeline continued to operate correctly. The governance
layer did not fail because the generator was giving poor suggestions.

---

## Position in a System

```
┌─────────────────────────────┐
│   Action Generator          │  LLM, planner, neural policy
│   (upstream component)      │
└────────────┬────────────────┘
             │ candidate actions
             ▼
┌─────────────────────────────┐
│   Decision Intelligence     │  Branching → Critical Path → FMEA
│   Core (DIC)                │  → Gate → Circuit Breaker → Selection
│                             │  → Belief Update
└────────────┬────────────────┘
             │ approved action + audit trace
             ▼
┌─────────────────────────────┐
│   Environment / Execution   │  real world, simulation, API
└─────────────────────────────┘
```

The DIC does not replace the action generator. It governs what the generator is
allowed to do — and produces an auditable record of every decision it makes.

This is the distinction between **intelligence that acts** and
**intelligence that acts within bounds it can justify**.

---

## Empirical Limits: What the Benchmarks Revealed

Three benchmarks validated the governance layer across qualitatively
different risk environments. They also revealed a precise boundary on
where structured decision governance provides measurable value versus
where the constraints are upstream of it.

**Where DIC governance is unambiguously effective:**

- In all three benchmarks, the safety pipeline (FMEA gate + circuit breaker)
  operated without failure across 150 total benchmark episodes. No pipeline
  bypass occurred regardless of utility estimates.
- In the Gridworld benchmark, the circuit breaker operated in WARN/SLOW range
  throughout — the system correctly self-constrained without ever reaching STOP.
- Baseline comparison (random policy from the same safe_actions filter) showed
  +789% score improvement in Snake, +1567% solve rate improvement in Maze,
  and +3800% in Gridworld versus the random policy — isolating the DIC pipeline
  as the source of improvement above the shared branching filter.

**Where governance alone cannot help:**

The episodic memory experiment demonstrated that **belief accuracy is not
sufficient if the utility function cannot act on it**. Carrying calibrated
Bayesian priors across sessions showed zero measurable effect across all
three benchmarks. The structural reasons:

1. Deterministic environments mean MC estimates are identical regardless
   of prior state — the blended signal converges to the same value.
2. In-session Bayesian accumulation overwhelms inter-session priors within
   the first 3–4 episodes of a 50-episode run.
3. Benchmarks operating at performance ceilings have no headroom
   for improved priors to express.

The implication for governance system design: **episodic memory and prior
calibration provide value in proportion to the stochasticity and variability
of the operating environment**. In stable, repeatable environments, the
governance layer's in-session belief update is sufficient. Cross-session
memory becomes critical when environments drift, sessions are short, or
cold-start performance is safety-critical.

This is an empirically grounded boundary: the DIC architecture is correct,
but the conditions under which each of its seven stages provides measurable
lift depend on properties of the environment, not just properties of the agent.
