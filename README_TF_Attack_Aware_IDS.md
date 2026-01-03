
# Attack-Aware TF Integrity IDS with CUSUM

## Overview
This module implements an **attack-aware integrity monitoring mechanism for ROS 2 TF streams**, targeting
*stealth TF spoofing attacks* that introduce slow but persistent drift in robot pose estimation.

Unlike classical rule-based detectors that rely on instantaneous physical bounds, the proposed approach
leverages **sequential change detection (CUSUM)** to identify anomalies that are individually subtle but
statistically significant over time.

The system is designed to be **lightweight, modular, and directly applicable** to real ROS 2 navigation stacks.

---

## Threat Model
We consider an adversary that:
- Publishes a forged transform (`odom → base_link_spoofed`)
- Injects **low-magnitude translational and/or rotational drift**
- Respects physical motion limits to evade threshold-based checks

Such attacks are realistic in ROS-based systems, where TF is often assumed trustworthy by planners and costmaps.

---

## Method

### Per-Step Normalized Score
At each TF update, the monitor estimates linear and angular motion:
- Linear velocity: \( v_t \)
- Angular velocity: \( \omega_t \)

These are normalized by expected bounds:
```
v_ratio = v_t / v_max
w_ratio = |ω_t| / w_max
```

The anomaly score is defined as:
```
score_t = max(v_ratio, w_ratio)
```

This produces a **unitless, interpretable signal**:
- `score < 1.0` → physically plausible motion
- elevated but sub-threshold values → potential stealth behavior

---

### Sequential Detection via CUSUM
To detect persistent deviations, we apply a one-sided CUSUM test:
```
g_t = max(0, g_{t-1} + (score_t - k))
```

An alarm is raised when:
```
g_t ≥ h
```

Where:
- `k` controls sensitivity to small deviations
- `h` controls detection delay vs false positives

---

## Implementation
**Node:** `tf_integrity_monitor_cusum.py`

**Inputs**
- TF transform (`parent_frame → child_frame`)

**Outputs**
- `/ids/tf_score` : instantaneous normalized anomaly score
- `/ids/tf_cusum` : accumulated CUSUM statistic
- `/ids/tf_alarm` : boolean alarm (latched)

The detector operates purely on TF dynamics and requires **no map, no ground truth, and no sensor-specific assumptions**.

---

## Experimental Results

### Scenario
A TF spoofing injector introduces **slow drift**:
- Translation: ~0.02 m/s
- Rotation: ~0.01 rad/s
- Motion remains within physical bounds

Baseline velocity-threshold detectors **do not trigger**.

---

### Detection Performance
With parameters:
```
v_max = 0.25 m/s
w_max = 0.35 rad/s
k     = 0.08
h     = 2.5
```

Observed behavior:
- Instantaneous score remains below 1.0
- CUSUM statistic increases monotonically
- Alarm triggered after ~4–5 seconds

---

### Qualitative Outcome
| Method                    | Detects Stealth Drift | Detection Delay | False Alarms |
|---------------------------|----------------------|-----------------|--------------|
| Instantaneous Thresholds  | ❌ No               | N/A             | Low          |
| CUSUM-based TF IDS (ours) | ✅ Yes              | Low (seconds)   | Low (tunable)|

This confirms that **sequential analysis is necessary** to detect stealth TF attacks.

---

## Discussion
Key observations:
- TF streams constitute a **critical attack surface** in ROS
- Physical plausibility checks are insufficient for security
- Sequential detectors provide strong guarantees with minimal overhead

The proposed method is:
- Computationally lightweight
- Interpretable
- Easily extensible to costmaps, planners, or sensor fusion pipelines

---

## Limitations & Future Work
- Parameter calibration currently requires offline tuning
- No semantic validation of TF graph consistency
- Future extensions:
  - Costmap spoofing detection
  - TF graph topology integrity
  - Multi-sensor trust fusion

---

## Disclaimer
This code and documentation are intended for **research and educational use only**.
They are not certified for safety-critical deployment.

**Author:** Panagiota Grosdouli  
