# Human Preference–Aware Risk Navigation

This module extends a standard risk-aware navigation framework with a **human preference–aware risk model**, combining:

- robot-side risk reasoning (self-trust, OOD, drift, uncertainty), and  
- explicit human comfort & behavior preferences,

to enable navigation that is **risk-aware and human-centered**.

---

## 1️⃣ Core Idea

We start from a baseline risk-aware planner with:

\[
cost(path) = L(path) + \lambda \cdot R(path)
\]

Where:

- **L(path)** = length / time / energy
- **R(path)** = risk metric (obstacle proximity, collision probability, uncertainty)
- **λ** = risk sensitivity weight  
  - large λ → conservative / safer
  - small λ → more aggressive / performance-oriented

---

## 1.1 Human Risk Attitude

We introduce a continuous human preference parameter:

\[
h \in [0,1]
\]

- h = 0 → highly risk-averse  
- h = 0.5 → balanced  
- h = 1 → risk tolerant

The robot provides:

\[
\lambda_\text{robot}
\]

Human preference modifies it into:

\[
\lambda_\text{eff} = \lambda_\text{robot} \cdot f(h,\alpha)
\]

Where α controls how strongly human preference influences behavior.

Intuition:

- Safe human → λ_eff > λ_robot  
- Risk-tolerant human → λ_eff < λ_robot  

Planner uses:

cost = L + λ_eff * R


---

## 1.2 Semantic Preferences

Humans may also specify **qualitative comfort constraints**, e.g.:

- “Avoid dark areas”
- “Avoid low-feature areas”
- “Prefer well-mapped regions”

Mapped into flags:

void_dark_areas
avoid_low_feature_areas
prefer_well_mapped_areas

These introduce penalties on affected edges/cells.

---

## 2️⃣ Modules

---

### ✔ user_preferences.py

Parses natural language (English + Greek) into:

