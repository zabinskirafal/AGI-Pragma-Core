# Failure Modes — AGI Pragma

Honest accounting of what DIC gets wrong, what it misses, and where
it cannot be trusted without additional safeguards.

> Known issues are tracked here. Contributions and fixes welcome via [GitHub issues](https://github.com/zabinskirafal/AGI-Pragma-Core/issues).

---

## 1 — False Positives (blocks that shouldn't happen)

These are cases where DIC returns `approved=False` for an action that is
actually safe.

### FP-1: DELETE on a file that doesn't exist

`critical_path.py` sets `p_irreversible=0.05` when the target file is absent
(the delete is a no-op), but the FMEA table does not distinguish this case.
`wrong_file_deleted` is still scored with full `S=9, D=7, R=10`, producing
RPN 3150+ regardless of whether the file exists.

A DELETE of a nonexistent temp file will be blocked.  The action is harmless.

```python
# file "scratch.tmp" does not exist — delete is a no-op
gov.evaluate(FileAction(op=FileOp.DELETE, path="scratch.tmp", reason="cleanup"))
# approved=False, RPN 3150  ← false positive
```

---

### FP-2: Repeated safe WRITEs inflate the Beta tracker

The Beta tracker treats any proposal with `RPN ≥ rpn_threshold / 2` (≥ 1200)
as a "risky signal", which raises the posterior mean and increases Occurrence
for all subsequent actions.

A WRITE to an existing file scores `overwrite_data_loss` at `S=8, R=8`.
With `o_base=5` the RPN is 960 — below the risky-signal threshold.  But if
a session involves several overwrites in sequence, each one slightly raises
the tracker mean, which in turn raises the Occurrence for the *next* action.
A later legitimate WRITE may be blocked not because it is risky but because
earlier legitimate WRITEs accumulated a false session risk signal.

---

### FP-3: Large legitimate WRITEs are penalised

`critical_path.py` adds a side-effect warning for writes over 10 000 bytes:

```python
if content_size > 10_000:
    side_effects.append(f"Large write ({content_size} bytes) increases corruption risk")
```

This does not affect the RPN directly, but it increases `p_irreversible` to
0.55 (overwrite path), which feeds into the utility penalty.  Writing a large
legitimate file — a data export, a generated report — produces the same
`p_irreversible` as overwriting a small config file.  The utility score will
be lower, making it more likely to fall below the approval threshold in future
utility-gated extensions.

---

### FP-4: Any WRITE to a file with "config", "env", ".key", or similar in its name

The FMEA table does not inspect filename patterns.  It only uses `file_exists`
and `action.op`.  There is no elevated severity for writing to
`config.yaml`, `.env`, or `id_rsa.pub`.  However, the *absence* of elevated
scoring for sensitive filenames is itself a false-negative risk (see §2).

The false positive here is the inverse: a project that legitimately names its
output files `config_output.json` or `env_summary.txt` gets no penalty, so
DIC does not over-block these.  This is correct behaviour — but it means DIC
provides no filename-based signal at all.

---

### FP-5: First DELETE in a new session is blocked more aggressively than it deserves

`BetaTracker` is initialised at `Beta(1.0, 1.0)`, mean=0.50.  This prior
assumes 50% of proposals are risky, which is conservative for a fresh session
with a well-calibrated agent.  The first DELETE in any session faces an
Occurrence score derived from a 0.50 prior regardless of the agent's actual
track record.

Sessions with strong agents are penalised by the uninformed prior until enough
safe proposals accumulate to pull the mean down.

---

## 2 — False Negatives (bypasses that DIC misses)

These are cases where a dangerous action gets through.

### FN-1: Tool name not in the integration map

All three framework integrations (LangGraph, AutoGen, LlamaIndex) operate on
a hardcoded name-to-`FileOp` map:

```python
_TOOL_NAME_MAP = {
    "read_file": FileOp.READ,
    "write_file": FileOp.WRITE,
    "delete_file": FileOp.DELETE,
    ...
}
```

Any tool whose name is not in this map passes through without evaluation.
A tool called `save_document`, `overwrite_config`, `purge_records`, or
`execute_shell` is invisible to DIC.  The agent can call it freely.

This is the most significant bypass in the integration layer.

---

### FN-2: Multi-step operations that individually look safe

DIC evaluates one action at a time.  A sequence of individually low-RPN
operations can produce an irreversible outcome that no single step would
have triggered:

1. READ `config.yaml` → approved (RPN 125)
2. WRITE `config.yaml` with modified content → approved (RPN 960, below 2400)
3. WRITE `config.yaml` again, truncating it → approved (RPN 960)

Each step is approved.  The net effect is a destroyed config file.  DIC has
no cross-action state beyond the Beta tracker and the circuit breaker.  It
does not model "what has the agent already done this session" in terms of
filesystem state.

---

### FN-3: READ followed by exfiltration

DIC assigns `R=1` to READ operations and scores them at RPN ~125 (always
approved).  The reasoning is that reading is non-destructive.  This is true
for filesystem integrity — but it ignores the downstream use of the read data.

An agent that READs a file containing secrets and then sends the contents
to an external API is never evaluated on the exfiltration step, because the
exfiltration is not a file operation.  DIC does not cover network calls,
subprocess execution, or clipboard operations.

---

### FN-4: Sandbox escape via symlink

Stage 1 (Branching) resolves the proposed path and checks
`resolved.relative_to(self.sandbox_root)`.  Path traversal via `..` is also
blocked.  However, a symlink inside the sandbox that points outside it will
pass both checks: the symlink itself resolves to a path inside the sandbox,
and following it reaches an external location.

```
sandbox/link_to_etc -> /etc/   # symlink created before DIC session starts
```

Writing to `sandbox/link_to_etc/passwd` passes Stage 1 (`resolved` is inside
sandbox before symlink is followed) and would be executed by the Executor.
The Executor's `_safe_resolve` has the same limitation.

This requires the symlink to be pre-placed — it is not something the LLM can
do unilaterally within the current demo, but it is a real bypass in any
deployment where the sandbox directory is shared or pre-populated.

---

### FN-5: DONE signal is never evaluated

`FileOp.DONE` bypasses all seven DIC stages:

```python
# dic_governor.py — Stage 1
if action.op == FileOp.DONE:
    return True, "DONE signal — no path to check"
```

An agent that encodes a malicious payload in the `reason` field of a DONE
action, or that uses DONE as a sentinel to trigger a side effect in a custom
executor, is not evaluated.  This is by design (DONE has no filesystem
effect), but it means any logic that misuses DONE bypasses the pipeline.

---

### FN-6: Beta tracker resets between sessions

`DICGovernor` creates a fresh `BetaTracker(1.0, 1.0)` on instantiation.
There is no cross-session persistence.  An agent that has proposed 20 high-risk
actions across 20 prior sessions starts each new session at the uninformed
prior.

The episodic memory / prior-transfer mechanism described in the paper is not
implemented in `demos/dic_llm`.  Session-level risk history is lost on every
`DICGovernor()` instantiation.

---

## 3 — Known Limitations

### L-1: FMEA table is hardcoded and manually calibrated

The severity, detection, and reversibility values in `risk_fmea.py` were set
by hand:

```python
# DELETE — permanent data loss
s, o, d, r = 10, min(10, o_base + 2), 2, 10
```

These values reflect one author's judgement about file operation risk.  They
have not been validated against real incident data, user studies, or
red-teaming.  A deployment in a domain where DELETE is routine (log rotation,
cache eviction) would require recalibration — the defaults will produce
constant false positives.

The FMEA table covers exactly three operations (READ, WRITE, DELETE) with two
sub-cases for WRITE.  Any domain extension requires writing a new table from
scratch.

---

### L-2: One domain at a time — no cross-domain evaluation

Each demo (`dic_llm`, `dic_db`) has its own governor, its own FMEA table, and
its own circuit breaker.  An agent that interleaves file operations with
database operations — writing a query result to a file, then deleting the
source table — is not evaluated as a combined action.  Each domain sees only
its own slice of the agent's behaviour.

There is no unified risk model across action types.

---

### L-3: Occurrence is derived from a single session-local tracker

The Beta tracker observes only whether the current session's proposals exceed
half the RPN threshold.  It does not distinguish between:
- An agent that consistently proposes risky operations (genuine bad actor)
- An agent that had one unusual proposal early in the session (statistical noise)
- An agent that is operating in a genuinely high-risk domain (appropriate risk)

All three raise the tracker mean identically.  A session involving many
legitimate high-RPN proposals (e.g. a backup tool that genuinely needs to
overwrite files repeatedly) will gradually inflate Occurrence scores for all
subsequent proposals, including safe ones.

---

### L-4: RPN threshold is a single global constant

`RPN_THRESHOLD = 2400` is applied uniformly to every action, every domain,
every session.  There is no mechanism to:
- lower the threshold for high-stakes environments (production databases)
- raise the threshold for low-stakes environments (scratch directories)
- adapt the threshold based on time-of-day, operator identity, or prior audit

The threshold was chosen to block DELETE (RPN ~3150+) while allowing WRITE-new
(RPN ~420) and WRITE-overwrite (RPN ~960).  It works for the demos.  Any
deployment with a different risk/utility balance requires manual recalibration.

---

### L-5: Critical path analysis is static, not Monte Carlo

Despite the name, `reversibility_profile()` performs no simulation.  It is a
lookup table with fixed `p_irreversible` values:

```python
FileOp.DELETE  → p_irreversible = 0.95  (or 0.05 if file doesn't exist)
FileOp.WRITE   → p_irreversible = 0.55  (overwrite) or 0.10 (new file)
FileOp.READ    → p_irreversible = 0.02
```

These values are constants, not estimates derived from any model of the file
system's state.  The "critical path" analysis in the paper's benchmarks
(which does use Monte Carlo rollouts) is a different, environment-specific
implementation.  The file-operations domain uses a deterministic approximation
because file system dynamics do not lend themselves to rollout-based
simulation.

This means `p_irreversible` cannot adapt to context: a DELETE in a directory
with hourly backups is identical to a DELETE in a directory with no backups.

---

### L-6: Circuit breaker ESCALATE resets immediately

When ESCALATE fires, `_reset_counters()` is called and the consecutive-stop
counter returns to zero.  The next proposal starts from a clean STOP-tier
counter.

This means an agent that repeatedly cycles through (STOP × 3 → ESCALATE →
STOP × 3 → ESCALATE) is never permanently suspended — it re-escalates every
three STOPs.  There is no cumulative escalation count that results in a harder
block.  `escalation_count` on the governor tracks how many times ESCALATE has
fired, but nothing in the pipeline acts on it beyond that.

---

### L-7: No content inspection

DIC evaluates the *operation* and the *path* but not the *content* of a WRITE.
Writing `rm -rf /` to a shell script, writing a malicious payload to a
`.bashrc`, or writing SQL injection strings to a config file all produce the
same RPN as writing "hello world" to the same path.

Content-aware risk scoring would require integrating a content classifier,
which is deliberately out of scope for the current implementation.

---

### L-8: No audit log persistence

The `stage_log` attached to each `DICDecision` exists only in memory for the
duration of the session.  There is no durable write to disk, no structured
log file, and no replay capability.  The REST API returns the audit trace in
the HTTP response, but does not write it anywhere.

A deployment that requires post-incident analysis, compliance auditing, or
detection of pattern-of-behaviour attacks has no log to inspect after the
session ends.

---

## Summary

| ID | Type | Severity | What goes wrong |
|----|------|----------|-----------------|
| FP-1 | False positive | Medium | DELETE on nonexistent file blocked as if real |
| FP-2 | False positive | Low | Repeated safe WRITEs inflate tracker, raise Occurrence |
| FP-3 | False positive | Low | Large WRITEs penalised equally regardless of legitimacy |
| FP-5 | False positive | Low | Uninformed Beta prior penalises first actions in every session |
| FN-1 | False negative | **High** | Tool names outside the map bypass DIC entirely |
| FN-2 | False negative | **High** | Multi-step sequences can destroy state one approved step at a time |
| FN-3 | False negative | **High** | READ + exfiltration is never evaluated |
| FN-4 | False negative | Medium | Symlinks inside sandbox can point outside it |
| FN-5 | False negative | Low | DONE signal is never evaluated |
| FN-6 | False negative | Medium | No cross-session risk memory |
| L-1 | Limitation | **High** | FMEA values are manually set, unvalidated |
| L-2 | Limitation | **High** | No cross-domain evaluation |
| L-3 | Limitation | Medium | Single tracker conflates bad actor, noise, and legitimate high-risk domain |
| L-4 | Limitation | Medium | Global RPN threshold, not context-sensitive |
| L-5 | Limitation | Medium | Critical path is a static lookup, not simulation |
| L-6 | Limitation | Low | ESCALATE resets immediately, no cumulative hard block |
| L-7 | Limitation | **High** | No content inspection |
| L-8 | Limitation | Medium | No durable audit log |

---

## False Positive Rate Benchmark

**Setup:** 50 sessions. Each session uses a fresh `DICGovernor` (Beta tracker
reset to `Beta(1.0, 1.0)`). Each session proposes the same safe action
sequence: `READ readme.txt` → `WRITE new_file_1.md` → `WRITE new_file_2.md`
→ `DONE`. Approved WRITEs are executed so files persist across sessions,
making sessions 2–50 test the overwrite path rather than the new-file path.

**Result: 0 false positives across 50 sessions (150 actions).**

### Per-operation RPN distribution

| Operation | Variant | Count | RPN min | RPN max | RPN mean | Blocked |
|-----------|---------|------:|--------:|--------:|--------:|--------:|
| READ | (all) | 50 | 125 | 125 | 125 | 0 |
| WRITE | new file (session 1 only) | 2 | 336 | 420 | 378 | 0 |
| WRITE | overwrite (sessions 2–50) | 98 | 1344 | 1536 | 1440 | 0 |
| **Total** | | **150** | | | | **0** |

**Safety margin:** highest RPN observed was **1536** against a threshold of
**2400** — a margin of 864 points (36% below threshold).

### Why WRITE RPNs vary (336 → 1536)

The spread reflects two compounding effects:

**1. New file vs overwrite.** Session 1 creates both files fresh (new-file
path: `R=3`, dominant mode `unintended_write` → RPN 336–420). Sessions 2–50
find both files already present (overwrite path: `R=8`, dominant mode
`overwrite_data_loss` → RPN 1344–1536).

**2. Beta tracker inflation within a session.** The overwrite RPN (1344–1536)
exceeds the risky-signal threshold (`rpn_threshold / 2 = 1200`), so it
registers as a risky signal and increments the Beta tracker's `a` counter.
This raises the Occurrence estimate for the second WRITE in the same session:

| Action | Beta state before | mean | O (from mean) | max RPN |
|--------|------------------|------|---------------|---------|
| READ | Beta(1.0, 1.0) | 0.500 | — | 125 |
| WRITE new_file_1.md | Beta(1.0, 2.0) | 0.333 | 7 | 1344 |
| WRITE new_file_2.md | Beta(2.0, 2.0) | 0.500 | 8 | 1536 |

98 of 150 actions (65%) were flagged as risky signals to the Beta tracker —
all from legitimate overwrites. This confirms **L-3**: the tracker cannot
distinguish a legitimate overwrite from a genuinely risky proposal.
The overwrites inflate Occurrence for subsequent actions every session, but
the resulting RPN (max 1536) stays well below the 2400 threshold, so no
false positive occurs.

### What this benchmark does and does not cover

**Covered:**
- Fresh-governor sessions with only safe file operations
- Both new-file and overwrite WRITE paths
- Beta tracker inflation effect within a session
- Circuit breaker behaviour under low-RPN load (remained at `OK` throughout)

**Not covered by this benchmark:**
- Sessions that mix safe and risky actions (Beta tracker starts inflated)
- Tool name bypass (FN-1) — DIC is never invoked for unrecognised tool names
- Multi-step filesystem destruction (FN-2) — no cross-action state check
- Content-based risks (L-7) — content is not inspected regardless of RPN
