# Human–Robot Trust Dynamics + Human Preference–Aware Risk Navigation

This module extends the dynamic navigation system by coupling:

1️⃣ **Robot Self-Trust**  
2️⃣ **Human Trust in the Robot**  
3️⃣ **Human Preference Awareness**

into a unified **risk-aware navigation policy controller**.  
The system allows the robot to update its behavior based on experience (success, failure, human override, approval) *and* adaptively follow human comfort preferences.

---

## ✨ Concept Overview

### 🔹 Baseline Navigation
The system already supports risk–aware navigation using:

- self-trust
- OOD awareness
- drift awareness
- calibrated uncertainty

with a classical cost model:

\[
\text{Cost} = L(\pi) + \lambda \cdot R(\pi)
\]

where λ controls how aggressively or conservatively the robot navigates.

---

## 🔥 Human Preference Layer
Humans express preferences like:

- “Prefer safer route even if slower”
- “Reach fast, I accept risk”
- “Avoid dark / low-feature regions”
- “Balanced”

These are parsed into:

- continuous risk preference h ∈ [0,1]
- semantic constraints:
  - avoid_dark_areas
  - avoid_low_feature_areas
  - prefer_well_mapped

The preference modifies λ using a human influence factor:

\[
\lambda_{\text{effective}} =
f(\lambda_\text{robot}, h, \alpha_\text{human})
\]

---

## 🤝 Human–Robot Trust Dynamics

We introduce **Trust Dynamics**, meaning both sides “learn each other”:

### ✅ Robot Self-Trust
Increases when:
- navigation succeeds
Decreases when:
- near-miss
- failure
- human override

### ✅ Human Trust in Robot
Estimated internally by the robot.
Increases with:
- success
- human approval
Decreases with:
- failures
- overrides
- unsafe behavior

Trust is normalized in \[0,1\].

---

## 🧠 Trust → Policy Mapping

### λ Robot (Risk Weight)
- Low self_trust → **increase λ** (safer)
- High self_trust → **reduce λ** (more aggressive)

### Human Influence Scale
- High human_trust → **preferences weigh more**
- Low human_trust → **preferences weigh less**

### Safe Mode
Enabled when:
- robot self-trust is too low
- estimated human trust is too low

---

## 🧪 Demo Scripts

### 1️⃣ Trust Dynamics Only

```bash
python3 run_trust_dynamics_demo.py


# Human–Robot Trust Dynamics + Human Preference–Aware Risk Navigation

This module extends the dynamic navigation system by coupling:

1️⃣ **Robot Self-Trust**  
2️⃣ **Human Trust in the Robot**  
3️⃣ **Human Preference Awareness**

into a unified **risk-aware navigation policy controller**.  
The system allows the robot to update its behavior based on experience (success, failure, human override, approval) *and* adaptively follow human comfort preferences.

---

## ✨ Concept Overview

### 🔹 Baseline Navigation
The system already supports risk–aware navigation using:

- self-trust
- OOD awareness
- drift awareness
- calibrated uncertainty

with a classical cost model:

\[
\text{Cost} = L(\pi) + \lambda \cdot R(\pi)
\]

where λ controls how aggressively or conservatively the robot navigates.

---

## 🔥 Human Preference Layer
Humans express preferences like:

- “Prefer safer route even if slower”
- “Reach fast, I accept risk”
- “Avoid dark / low-feature regions”
- “Balanced”

These are parsed into:

- continuous risk preference h ∈ [0,1]
- semantic constraints:
  - avoid_dark_areas
  - avoid_low_feature_areas
  - prefer_well_mapped

The preference modifies λ using a human influence factor:

\[
\lambda_{\text{effective}} =
f(\lambda_\text{robot}, h, \alpha_\text{human})
\]

---

## 🤝 Human–Robot Trust Dynamics

We introduce **Trust Dynamics**, meaning both sides “learn each other”:

### ✅ Robot Self-Trust
Increases when:
- navigation succeeds
Decreases when:
- near-miss
- failure
- human override

### ✅ Human Trust in Robot
Estimated internally by the robot.
Increases with:
- success
- human approval
Decreases with:
- failures
- overrides
- unsafe behavior

Trust is normalized in \[0,1\].

---

## 🧠 Trust → Policy Mapping

### λ Robot (Risk Weight)
- Low self_trust → **increase λ** (safer)
- High self_trust → **reduce λ** (more aggressive)

### Human Influence Scale
- High human_trust → **preferences weigh more**
- Low human_trust → **preferences weigh less**

### Safe Mode
Enabled when:
- robot self-trust is too low
- estimated human trust is too low

---

## 🧪 Demo Scripts

### 1️⃣ Trust Dynamics Only

```bash
python3 run_trust_dynamics_demo.py


-----
Shows:

self_trust_robot

human_trust_in_robot

λ_robot

human influence

safe mode flag


2️⃣ Trust + Human Preferences Integration

Shows step-by-step:

event (SUCCESS, FAILURE, HUMAN_OVERRIDE, etc.)

human preference text

trust evolution

λ_robot

human influence scale

human risk preference h

final λ_effective sent to the planner

This demonstrates how trust and preference co-evolve.

3️⃣ Export Results for Plots
python3 save_trust_preference_results.py


Generates:

trust_preference_results.csv


Containing:

step

event

human preference text

robot self-trust

human trust

λ_robot

λ_effective

safe mode

Ready for:

Excel

pandas analysis

matplotlib / seaborn plots

research figures

🎯 Why This Is Important

This framework connects:

Human-centered robotics

Trust modeling

Risk-aware navigation

Explainable robotic decision-making

It enables experiments on:

how humans influence robot risk decisions

how trust changes robot behavior

how failures / successes reshape risk policies

This is currently hot research in:

Human–Robot Interaction

Trust-Aware Autonomy

Human Preference Integration

🚀 Ready for Extension

Next steps may include:

ROS Integration

Real navigation-based trust adaptation

Trust learning via Bayesian / RL approaches

UI for real user input preferences

Full paper-ready experimental evaluation

✔️ Summary

This module delivers:

Human-aware navigation

Trust-evolving autonomy

Adaptive λ-risk control

Semantic risk constraints

Real experiment logging

It transforms the navigator from a static planner to an adaptive, self-aware, human-aligned system.
python3 run_trust_and_preferences_demo.py

