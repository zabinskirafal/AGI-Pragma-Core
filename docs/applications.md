# Applications of AGI Pragma

The Decision Intelligence Core (DIC) is domain-agnostic.
Its seven-stage pipeline operates on abstract actions, risk scores, and belief states —
none of which are specific to any environment.

This document illustrates how to apply the DIC to a physical sensor domain:
**accelerometer-based anomaly detection**, as found in industrial equipment monitoring,
wearables, and autonomous vehicles.

---

## Example: Accelerometer-Based Anomaly Detection

**Scenario:** an embedded controller reads tri-axial accelerometer data `[ax, ay, az]`
at regular intervals and must decide between three actions:

| Action       | Description                                      |
|--------------|--------------------------------------------------|
| `CONTINUE`   | maintain current operation                       |
| `SLOW_DOWN`  | reduce operating speed or load                   |
| `EMERGENCY_STOP` | halt immediately and raise an alert          |

The controller must make this decision at each timestep under uncertainty —
it cannot be certain whether a spike in readings represents genuine danger
or sensor noise.

---

## Mapping the DIC Pipeline to Accelerometer Data

### 1. Branching

Enumerate feasible actions and remove physically impossible ones.
At any timestep, all three actions (`CONTINUE`, `SLOW_DOWN`, `EMERGENCY_STOP`) are
available unless a hardware interlock has already forced a stop.

### 2. Critical Path Estimation

Monte Carlo rollouts project sensor trajectories forward over a short horizon
(e.g. 10–25 timesteps) using a simple physics model or learned transition distribution.
Each rollout estimates:

- probability of exceeding a critical vibration threshold within the horizon,
- probability of entering a regime with no safe recovery (structural resonance trap),
- expected timesteps until threshold breach.

### 3. Risk Assessment (FMEA)

Each candidate action is scored using Severity × Occurrence × Detection = **RPN**.

**Mapping sensor readings to FMEA inputs:**

```python
import math
from benchmarks.snake.risk_fmea import FMEAItem, clamp10, occ_from_prob

def sensor_to_fmea(
    ax: float, ay: float, az: float,
    p_threshold_breach: float,
    p_resonance_trap: float,
) -> dict:
    """
    Convert accelerometer readings and Monte Carlo estimates into an FMEA table.

    ax, ay, az          : acceleration in m/s² on each axis
    p_threshold_breach  : P(exceeding critical vibration level within horizon)
    p_resonance_trap    : P(entering unrecoverable resonance regime)
    """
    magnitude = math.sqrt(ax**2 + ay**2 + az**2)
    immediate_danger = magnitude > 50.0  # m/s² — hardware-defined critical limit

    if immediate_danger:
        # Immediate breach: catastrophic, certain, fully detectable
        return {
            "immediate_breach": FMEAItem(
                failure_mode="Immediate vibration limit exceeded",
                severity=10, occurrence=10, detection=1,
                rpn=100,
            )
        }

    # Probabilistic breach within rollout horizon
    s_breach = 10                           # structural damage is catastrophic
    o_breach = occ_from_prob(p_threshold_breach)
    d_breach = 3                            # rollouts make this estimable
    rpn_breach = s_breach * o_breach * d_breach

    # Resonance trap risk (harder to detect, slightly lower severity)
    s_trap = 8
    o_trap = occ_from_prob(p_resonance_trap)
    d_trap = 6                              # resonance onset is subtle
    rpn_trap = s_trap * o_trap * d_trap

    return {
        "prob_breach": FMEAItem("Vibration breach within horizon",
                                s_breach, o_breach, d_breach, rpn_breach),
        "resonance_trap": FMEAItem("Resonance trap / unrecoverable regime",
                                   s_trap, o_trap, d_trap, rpn_trap),
    }
```

`occ_from_prob` converts a probability (0–1) from the Monte Carlo rollouts into
an Occurrence score (1–10) using the same heuristic mapping used in the Snake benchmark.

### 4. Decision Integrity Gate

Actions whose maximum RPN exceeds the configured threshold are blocked before execution.
For `CONTINUE`, a high RPN from `prob_breach` or `resonance_trap` causes the gate to
block the action outright — the system cannot choose to continue operating unsafely.

### 5. Circuit Breaker

The circuit breaker maps the peak RPN of the chosen action to an autonomy state:

| RPN range   | State   | Effect                                        |
|-------------|---------|-----------------------------------------------|
| < 180       | OK      | normal autonomous operation                   |
| 180 – 219   | WARN    | action allowed, anomaly flagged to operator   |
| 220 – 259   | SLOW    | operating speed reduced, decision depth cut   |
| >= 260      | STOP    | action blocked, `EMERGENCY_STOP` forced       |

### 6. Decision Selection

Among actions that passed the gate, utility is computed as:

```
U(action) = w_survival × P(safe horizon) − w_risk × RPN_norm + w_goal × progress
```

`SLOW_DOWN` will typically dominate when RPN is elevated but below the STOP threshold —
it reduces future breach probability without the cost of a full halt.

### 7. Belief Update

After each timestep, Bayesian trackers update estimates of:

- `p_threshold_breach` — using the Beta distribution, updated by whether the
  vibration level actually rose or fell after the chosen action,
- `p_resonance_trap` — updated by whether oscillation amplitude increased.

These updated beliefs feed directly into the next timestep's Monte Carlo rollouts,
making the system's risk estimates sharpen over time.

---

## Example Decision Trace

Given a single timestep with readings `ax=12.3, ay=−4.1, az=31.7` (magnitude ≈ 33.9 m/s²),
rollout estimates `p_threshold_breach=0.28`, `p_resonance_trap=0.09`:

| Stage               | Output                                                  |
|---------------------|---------------------------------------------------------|
| Branching           | `{CONTINUE, SLOW_DOWN, EMERGENCY_STOP}`                 |
| Critical Path       | p_breach=0.28, p_trap=0.09, expected breach in ~18 steps|
| FMEA (CONTINUE)     | RPN_breach=180, RPN_trap=48, max RPN=180                |
| Decision Gate       | RPN 180 — at threshold boundary, passes gate            |
| Circuit Breaker     | WARN state — action flagged                             |
| Selection           | `SLOW_DOWN` preferred (lower future RPN projection)     |
| Belief Update       | Beta trackers updated; p_breach posterior shifts down   |

---

## Extensibility

The same pipeline applies to any sensor domain by substituting domain-specific
failure modes into the FMEA table:

| Sensor domain     | Failure modes for FMEA                            |
|-------------------|---------------------------------------------------|
| Gyroscope         | attitude instability, gimbal lock approach        |
| Pressure sensor   | overpressure breach, sensor drift / silent failure|
| GPS / odometry    | position divergence, geofence violation           |
| Temperature       | thermal runaway, cooling system saturation        |

The DIC stages — and the RPN thresholds that govern the circuit breaker — remain
identical across domains. Only the failure mode definitions and the Monte Carlo
transition model need to be adapted.
