# Innovation-Based Intrusion Detection for UKF Sensor Fusion

## Abstract
This presents an **innovation-based intrusion detection mechanism** embedded directly into a Unscented Kalman Filter (UKF) used for robot state estimation.
The core idea is to treat **state estimation consistency** not only as a filtering property, but also as a **security signal**.

Without modifying code execution or system logic, an attacker can still compromise navigation by **manipulating sensor measurements** (e.g., spoofing or replaying data).
This work shows that such attacks can be detected *purely statistically*, using the innovation process of the filter itself, and mitigated through **adaptive trust weighting**.

---

## Threat Model
We consider an attacker who:
- cannot alter the source code or filter logic,
- cannot access internal filter states,
- **can manipulate sensor measurements** (e.g., VO, odometry, pose estimates).

Covered attack types:
- **Bias / spoofing attacks** (systematic offset injection),
- **Replay attacks** (re-transmission of previously valid measurements).

The system is required to remain operational and stable even under partial sensor compromise.

---

## Detection Principle

At each time step, the UKF computes the **innovation**:

νₖ = zₖ − ẑₖ

where:
- zₖ is the incoming measurement,
- ẑₖ is the predicted measurement.

The innovation covariance is given by:

Sₖ = H Pₖ Hᵀ + R_nominal

The statistical consistency of the measurement is evaluated using the **Mahalanobis distance**:

dₖ² = νₖᵀ Sₖ⁻¹ νₖ

Under nominal conditions, dₖ² follows a chi-square distribution:

dₖ² ~ χ²(DOF)

An anomaly is flagged whenever:

dₖ² > χ²(α, DOF)

To reduce spurious detections, a trigger is raised only after **N consecutive violations**, following an initial warm-up phase.

---

## Detection vs Mitigation (Key Design Insight)

A central design choice of this system is the **strict separation between detection and mitigation**.

### Detection
- Always uses the **nominal measurement covariance**
- Ensures that adaptive mitigation **cannot mask anomalies**
- Produces reliable and repeatable triggers

### Mitigation: Adaptive Trust Weighting
Once an anomaly is detected, the affected sensor is **down-weighted smoothly**, rather than rejected outright.

A scaling factor is computed as:

scaleₖ = 1 + k · max(0, dₖ² / thr − 1)

The effective covariance used for the update becomes:

R_eff = scaleₖ · R_nominal

This design yields graceful degradation under attack, filter stability, and continued navigation capability.

---

## Experimental Setup
- **State**: [x, y, yaw]
- **Sensors**:
  - Visual Odometry (VO)
  - IMU yaw
- **Attack scenario**:
  - Replay of previously recorded VO measurements
- **Evaluation metrics**:
  - Detection delay
  - False alarm rate
  - Trust scaling factor
  - Trigger frequency

All experiments are executed **offline**, without ROS dependencies.

---

## Results (Replay Attack)

| Metric | Value |
|------|------|
| Attack start | t = 100 |
| Detection time | t = 104 |
| Detection delay | 4 steps |
| False alarms (pre-attack) | 0% |
| Mean trust scale (post-attack) | 3.92 |
| Maximum trust scale | 23.36 |

Generated plots:
- Mahalanobis distance vs threshold
- Adaptive trust scaling over time

---

## Files

| File | Description |
|----|----|
| security_monitor.py | Innovation-based IDS logic |
| ukf_fusion.py | UKF with integrated detection and mitigation |
| eval_ids_replay.py | Replay attack experiment |
| log_ids_replay_csv.py | CSV logging |
| plot_ids_from_csv.py | Plot and metric generation |
| ids_replay_log.csv | Experiment data |

---

## Contributions
- Sensor fusion used as a **formal security monitor**
- Detection independent from mitigation
- Continuous trust adaptation instead of binary rejection
- Zero false alarms prior to attack
- Minimal assumptions on attacker capabilities

---

## Future Work
- Integration with ROS 2 / Nav2
- Multi-sensor consistency checks
- Automatic transition to safe-mode planners
- STRIDE-aligned threat modeling

---


