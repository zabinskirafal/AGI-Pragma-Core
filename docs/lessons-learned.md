# Lessons Learned — AGI Pragma

Practical lessons from building the Decision Intelligence Core across three
benchmark environments, two real-world demo domains, a REST API, and three
framework integrations.

---

## Part 1 — The DIC Pipeline (from the arXiv paper)

### L1 — The safety pipeline is not the performance bottleneck

Across all three benchmarks and every calibration iteration, the FMEA gate
and circuit breaker worked correctly on the first attempt and never needed
modification.  Every performance failure was traceable to the utility function
or the critical path estimate — not to the safety stages.

**What this means in practice:** Don't tune the pipeline when the agent
underperforms.  Check Stage 6 (utility signal) and Stage 2 (rollout depth)
first.  The safety constraint is stable and domain-agnostic; the progress
signal is domain-specific and almost always the real problem.

---

### L2 — The utility function is the primary design variable

Every benchmark improvement came from fixing the goal-progress signal, not
the safety constraints:

- **Snake:** increasing the food-distance weight from 0.2 to 1.5 produced a
  57× score improvement.
- **Maze:** replacing Manhattan distance with BFS path distance lifted the
  solve rate from 8% to 100%.

The correct architectural separation is **invariant safety** (Stages 1–5,
fixed) and **domain-specific progress** (Stage 6, tuned per domain).  Once
you get that boundary right, safety and performance don't trade off — they
compose.

---

### L3 — Critical path calibration must respect domain solution length

In Maze v1.0 the rollout depth (25 steps) was shorter than the minimum maze
solution length under the rollout policy.  `p_death` was never triggered —
it silently produced zero for every candidate action, giving the utility
function nothing to work with.

**Rule:** calibrate rollout depth to the *expected time-to-failure under the
rollout policy*, not to the episode step budget.  A depth that never reaches
failure is not a conservative safety measure — it is a broken signal.

---

### L4 — Signal accuracy without variance is uninformative

After the Maze v1.0 depth fix, `p_death ≈ 1.0` for all candidate actions:
accurate, but providing no basis for choosing between them.  Useful risk
estimation requires *variance across the action set*, not merely correctness
in aggregate.

The Gridworld is the first environment in the suite where `p_death` is
genuinely differentiating: moving toward a hazard cluster scores measurably
higher than WAIT or evasion.  That variance is what makes Monte Carlo forward
simulation worth the compute.

---

## Part 2 — DIC v2.0 Additions

### L5 — Reversibility is a first-class risk dimension, not a derived score

The original RPN formula was `S × O × D`.  Reversibility was implicit —
baked into Severity estimates rather than stated directly.  The problem: two
actions with identical S, O, D scores can have radically different consequences
depending on whether the damage can be undone.

Adding R as an explicit fourth dimension (`RPN = S × O × D × R`) has two
effects:

1. It forces the FMEA author to state reversibility as a deliberate design
   decision, not a side-effect of severity scoring.
2. It creates a clean threshold: `DELETE` scores `R=10`, `WRITE new` scores
   `R=3`, `READ` scores `R=1`.  The difference is not a matter of degree —
   it is a categorical distinction that the RPN now reflects directly.

With `R` in place, `DROP TABLE` reaches RPN 3780 and `DELETE` reaches 3150 —
both well above the 2400 threshold — purely from irreversibility, even if
every other dimension is moderate.

---

### L6 — Use p95, not mean, as the FMEA Occurrence input

Early versions fed `mean(p_death)` from Monte Carlo rollouts directly into
the FMEA Occurrence score.  This works on average but is systematically
optimistic: the mean smooths over tail outcomes where the agent is already
in a dangerous cluster of states.

Switching Occurrence to a **blended p95 estimate** (`0.7 × p95_death + 0.3 ×
tracker.mean`) makes the gate conservative at the right moment.  The p95
fires when the tail of the rollout distribution is dangerous, even if most
rollouts survive.  The tracker mean anchors it against one-off noise spikes.

The difference shows up in dynamic environments: the Gridworld circuit breaker
operated in WARN/SLOW range (RPN 180–200) precisely because p95 variance
across candidate actions was meaningful, not saturated.

---

### L7 — Bayesian belief tracking catches drift that Monte Carlo misses

Monte Carlo rollouts measure *current-state* risk: they simulate forward from
the exact current configuration.  They cannot see LLM-level behavioral drift —
a model that is becoming incrementally riskier across a session.

The Beta tracker (`BetaTracker(a, b)`) captures this.  Each DIC evaluation
updates the posterior: a risky signal (RPN above half-threshold) increments
`a`; a safe signal increments `b`.  The posterior mean feeds back into the
FMEA Occurrence calculation, so a session with repeated risky proposals
steadily raises the baseline Occurrence score for subsequent calls.

**Practical consequence:** a single bad proposal doesn't lock out the agent.
A pattern of bad proposals does.  This mirrors how a human supervisor would
respond to a colleague who repeatedly proposes dangerous actions.

---

## Part 3 — Real-World Demos

### L8 — Detection scores matter as much as Severity

The most surprising result from the file-operations demo was the blocking
reason for `DELETE temp.txt`.  The dominant failure mode was not
`permanent_data_loss` (which scored `D=2` — highly detectable, easy to catch
before it's too late) but `wrong_file_deleted` (`D=7` — an LLM may
hallucinate the target filename and the mistake is hard to catch before
execution).

The full FMEA captures this.  A naive risk model that only asks "how bad is
the outcome?" misses "how likely is it that no one notices until it's too
late?"  High Detection score (meaning *hard to detect*) is often the
irreversibility multiplier for LLM-generated actions, where the input itself
is unverified.

---

### L9 — Stage 1 scope enforcement must be in the pipeline, not only in the executor

The Executor has its own sandbox guard (`_safe_resolve` raises if the path
escapes the sandbox).  That guard should never fire in normal operation —
Stage 1 (Branching) blocks the action before it reaches the executor.

Having both layers is correct: defence in depth means the executor does not
trust that DIC was called.  But the policy decision belongs in Stage 1.  If
Stage 1 is skipped or misconfigured, a path traversal attack
(`../../etc/passwd`) should fail at the executor — but the audit trail should
make clear it was a pipeline misconfiguration, not a legitimate block.

---

### L10 — Cascade failure scores higher than direct schema destruction

In the database demo, `DROP TABLE` has two failure modes: `schema_destruction`
(RPN 2700) and `cascade_failure` (RPN 3780).  The cascade failure scores
*higher* because downstream application references break silently (`D=6`)
rather than immediately (`D=1`).

Direct, loud failures are easier to catch and recover from.  Silent downstream
failures — foreign key violations, missing tables discovered hours later by an
unrelated process — are the actual high-risk outcome.  The Detection dimension
captures this distinction; Severity alone does not.

---

### L11 — 100% irreversible block rate with 0% task completion impact is the key number

The controlled experiment (50 sessions DIC vs. no-DIC on the file-operations
agent) produced one result that matters above all others:

- Without DIC: 50/50 irreversible operations executed, 50/50 tasks completed.
- With DIC: 0/50 irreversible operations executed, 50/50 tasks completed.

The pipeline eliminated every irreversible operation without blocking a single
productive one.  This is the number to lead with in any investor or customer
conversation.  It is not about making the agent "safer" in a vague sense —
it is about a hard enforcement layer that draws a clear line between reversible
and irreversible and never crosses it.

---

## Part 4 — Framework Integrations

### L12 — Find the single lowest-level execution hook in each framework

Each framework has one place where tool execution is unavoidable:

| Framework | Hook | Called by |
|-----------|------|-----------|
| LangGraph | `DICGuardNode.__call__(state)` | `graph.invoke()` edge routing |
| AutoGen | `BaseTool.run_json(args, token, call_id)` | `StaticWorkbench.call_tool()` |
| LlamaIndex | `AsyncBaseTool.acall(**kwargs)` | `BaseWorkflowAgent._run_tool()` |

Intercepting at the right level means the wrapper is invisible to everything
above it (the agent, the graph, the orchestrator) and everything below it
(the actual tool function).  Intercepting too high means some code paths
bypass the guard.  Intercepting too low means you cannot return a structured
block message to the agent.

Researching the right hook took more lines than the implementation in every
case.

---

### L13 — The block message must go back to the LLM as a tool result, not as an exception

The naive implementation raises an exception on block.  The correct
implementation returns a descriptive string as if the tool executed normally.

If the block is an exception, the agent framework either crashes, retries
silently, or treats it as a transient error and repeats the same action.  If
the block is a tool result, the LLM sees `[DIC BLOCKED] RPN 3150 ≥ threshold
2400` in its context and can reason about why the action was refused and
propose an alternative.

All three integrations return strings or `ToolOutput` objects rather than
raising.  The LLM is the recovery mechanism; the block message is the signal
it needs to engage that mechanism.

---

### L14 — Pass isinstance checks or the framework silently discards your wrapper

LangGraph routes on node type.  AutoGen's `AssistantAgent` type-checks tools:

```python
if isinstance(tool, BaseTool):
    self._tools.append(tool)
else:
    self._tools.append(FunctionTool(tool, ...))   # re-wraps, discarding DIC
```

If the wrapper does not pass `isinstance(tool, BaseTool)`, the framework
silently discards it and re-wraps the original — undoing the entire
integration without any error or warning.  All three wrappers subclass the
relevant base class (`BaseTool`, `AsyncBaseTool`) rather than duck-typing,
precisely to prevent this silent failure mode.

---

### L15 — Share one governor across all tools for session-level escalation

Each integration provides `dic_wrap_tool(tool)` (single tool, own governor)
and `dic_wrap_tools(tools)` (list, shared governor).  The shared governor
matters because the circuit breaker is a *session-level* escalation mechanism:

```
OK → WARN → SLOW → STOP
```

If each tool has its own governor, the circuit breaker resets between tool
calls.  A pattern of `write → delete → write → delete` never triggers SLOW or
STOP because each call sees a fresh state.  With a shared governor, the same
pattern escalates correctly regardless of which tool proposed each action.

---

### L16 — Tool name normalization is necessary and fragile

Every framework has different naming conventions for the same logical operation:

| Logical operation | Names seen in the wild |
|-------------------|------------------------|
| Write a file | `write_file`, `write`, `create_file`, `save_file` |
| Delete a file | `delete_file`, `delete`, `remove_file`, `remove` |
| Read a file | `read_file`, `read`, `open_file`, `load_file` |

The name-to-`FileOp` map in each integration covers the common cases.  Any
tool not in the map passes through without DIC evaluation — this is the right
default.  It is better to miss an unusual tool name than to block a legitimate
non-file operation because it matched a pattern by accident.

**Practical consequence:** if you add custom tools to your agent, check that
their names are in the map.  The integration cannot protect operations it does
not recognise.

---

## Summary table

| # | Lesson | Source |
|---|--------|--------|
| L1 | Safety pipeline is not the bottleneck | All three benchmarks |
| L2 | Utility function is the primary design variable | Snake, Maze |
| L3 | Rollout depth must reflect time-to-failure under policy | Maze v1.0 |
| L4 | Signal variance across actions matters more than accuracy | Maze, Gridworld |
| L5 | Reversibility is a first-class FMEA dimension | DIC v2.0 |
| L6 | Use p95, not mean, for FMEA Occurrence | DIC v2.0 |
| L7 | Beta tracking catches session-level behavioral drift | DIC v2.0 |
| L8 | Detection scores matter as much as Severity for LLM agents | File ops demo |
| L9 | Scope enforcement belongs in Stage 1, not only in the executor | File ops demo |
| L10 | Cascade failures score higher than direct failures | Database demo |
| L11 | 100% irreversible block, 0% task completion impact | Controlled experiment |
| L12 | Find the single lowest-level execution hook per framework | All integrations |
| L13 | Block messages must be tool results, not exceptions | All integrations |
| L14 | Pass isinstance checks or the framework discards your wrapper | AutoGen, LlamaIndex |
| L15 | Share one governor across all tools for session-level escalation | All integrations |
| L16 | Tool name normalization is necessary and fragile | All integrations |
