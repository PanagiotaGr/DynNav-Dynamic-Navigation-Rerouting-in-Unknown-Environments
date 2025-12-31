# Human Preference–Aware Risk Navigation

This module extends the existing navigation framework with a **human preference–aware risk model**.  
The goal is to combine:

- the robot’s internal risk reasoning (self-trust, OOD detection, drift, uncertainty), and  
- explicit human comfort and behavior preferences,

so that navigation is **both risk-aware and human-centered**.

---

## 1. Core Idea

We start from a standard risk-aware planner with a cost function:

```
cost(path) = L(path) + λ * R(path)
```

Where:

- **L(path)** = path length / time / energy  
- **R(path)** = risk metric (e.g., obstacle proximity, collision probability, uncertainty)  
- **λ** = risk weight  

Meaning:

- large λ → conservative, safe behavior  
- small λ → aggressive, performance-oriented behavior  

---

### 1.1 Human risk attitude

We introduce a human risk preference:

```
h ∈ [0, 1]
h = 0   → very risk-averse
h = 0.5 → balanced
h = 1   → accepts high risk
```

The robot provides a baseline:

```
λ_robot
```

Human preference modifies it into an effective λ:

```
λ_eff = λ_robot * f(h, α)
```

Where α controls how strongly the human influences λ.

Intuition:

- safe human (low h) → λ_eff > λ_robot  
- aggressive human (high h) → λ_eff < λ_robot  

Then planner uses:

```
cost = L + λ_eff * R
```

---

### 1.2 Semantic preferences

Humans may also express qualitative constraints such as:

- “Avoid dark areas”
- “Avoid low-feature regions”
- “Prefer well-mapped areas”

Mapped into flags:

```
avoid_dark_areas
avoid_low_feature_areas
prefer_well_mapped_areas
```

These are applied as penalties on edges or cells that violate the preference.

---

## 2. Modules

---

### 2.1 Core human preference logic

#### `user_preferences.py`

Parses human preference text (English and Greek).

Produces:

```
HumanPreference
    risk_preference: float (h ∈ [0, 1])
    avoid_dark_areas: bool
    avoid_low_feature_areas: bool
    prefer_well_mapped_areas: bool
```

This is the bridge between natural language and planner parameters.

---

#### `human_risk_policy.py`

Defines:

**HumanRiskConfig**
- human influence scale
- min/max λ scaling
- penalties for dark / low-feature areas

**HumanRiskPolicy**
Main policy with:

```python
lambda_eff = policy.compute_lambda_effective(lambda_robot, human_pref)

cost = policy.edge_cost(
    base_length,
    edge_risk,
    lambda_effective=lambda_eff,
    human_pref=human_pref,
    is_dark=...,
    is_low_feature=...,
)
```

The planner only needs to call `edge_cost(...)`.

---

#### `risk_cost_utils.py`

Reusable helper:

```python
human_aware_edge_cost(
    base_length,
    edge_risk,
    lambda_effective,
    human_pref,
    is_dark=False,
    is_low_feature=False,
    low_feature_penalty=5.0,
    dark_area_penalty=5.0,
)
```

Planner-agnostic → usable in PRM, RRT, NBV, coverage planners, etc.

---

## 3. Toy Planners & Experiments

---

### `simple_risk_planner.py`

Baseline risk-aware planner with 3 synthetic paths A, B, C.

Cost:

```
cost = length + lambda_robot * risk
```

Selects lowest cost path.

---

### `simple_risk_planner_human.py`

Same A, B, C but:

- parses human preference → HumanPreference
- computes λ_eff
- adds penalties (dark / low-feature)

Shows differences in:

- λ_eff
- selected path

---

### `save_simple_planner_results.py`

Runs:

- baseline planner
- human-aware planner with multiple preference texts

Saves results to:

```
simple_planner_results.csv
```

Each row includes:

- mode (baseline / human)
- human_pref_text
- path_name (A/B/C)
- length, risk
- is_dark, is_low_feature
- lambda_robot, lambda_effective
- cost
- is_best (1 if selected)

---

### `analyze_simple_planner_results.py`

Reads `simple_planner_results.csv`

Computes per (mode, human_pref_text):

- path win frequency
- average cost
- average λ_eff

Useful for tables and report figures.

---

## 4. Real Planner Integration

---

### `human_aware_real_planner.py`

Provides:

```python
class HumanAwarePlannerWrapper:
    ...
```

Wrapper on top of any planner with `plan(...)`.

#### Usage

```python
from modules.graph_planning.prm_planner import PRMPlanner
from human_aware_real_planner import HumanAwarePlannerWrapper

base_planner = PRMPlanner(...)

human_planner = HumanAwarePlannerWrapper(
    underlying_planner=base_planner,
    human_pref_text="Prefer safer route even if slower",
    human_influence_scale=1.0,
)

lambda_robot = 1.0

result, lambda_eff = human_planner.plan_with_human_lambda(
    lambda_robot=lambda_robot,
    start=start,
    goal=goal,
)
```

The wrapper:

- parses preferences
- computes λ_eff
- tries to pass λ as:
  - `lambda_weight=`
  - `risk_weight=`
  - `lambda_risk=`
- else calls `plan()` normally.

---

## 5. How to Run

From project root:

---

### 5.1 Toy examples

```
python3 simple_risk_planner.py
python3 simple_risk_planner_human.py
```

---

### 5.2 Save & analyze

```
python3 save_simple_planner_results.py
python3 analyze_simple_planner_results.py
```

---

### 5.3 Real planner demo

```
python3 run_human_preference_exp.py
python3 run_real_human_preference_demo.py
```

(Depends on your planner setup)

---

## 6. Integrating With a Custom Planner

Expose λ in your cost:

```python
cost = length + lambda_weight * risk
```

Optional semantics:

```python
if human_pref.avoid_dark_areas and is_dark:
    cost += dark_penalty
if human_pref.avoid_low_feature_areas and is_low_feature:
    cost += low_feature_penalty
```

Use wrapper:

```
plan_with_human_lambda(...)
```

---

## 7. Summary

This module provides a human-centered risk adaptation layer:

- natural language → structured preference
- risk preference h ∈ [0,1]
- semantic comfort preferences
- fusion with robot trust / OOD / drift
- modular
- works with existing planners
- tested with toy + CSV experiments
- ready for real robot integration

Result:

**Navigation becomes risk-aware and aligned with human comfort.**
