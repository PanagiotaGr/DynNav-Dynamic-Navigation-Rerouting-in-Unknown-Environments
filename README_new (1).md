# Self-Healing Navigation & Language-Driven Safety

This repository contains a minimal, research-style Python implementation of two modules:

1. **Self-Healing Navigation**  
2. **Language-Driven Safety (English + Greek)**  

The code is intentionally lightweight and self-contained so you can:

- plug it into an existing navigation or planning stack, or  
- use it as a prototype for experiments in human-centered, risk-aware autonomy.

---

## 1. Project Structure

```text
.
├── self_healing_policy.py       # Self-healing navigation core policy
├── language_safety_policy.py    # Language-driven safety policy (EN + EL)
├── self_healing_demo.py         # Demo for self-healing behaviour
├── language_safety_demo.py      # Demo for language-driven risk modulation
└── README.md
```

No external dependencies are required beyond the Python standard library.

---

## 2. Self-Healing Navigation

**File:** `self_healing_policy.py`

The self-healing policy monitors normalized reliability metrics in `[0, 1]`:

- `drift` – localization drift / visual odometry confidence  
- `calibration_error` – miscalibration of uncertainty estimates  
- `heuristic_regret` – deviation from (approximate) optimal cost  
- `failure_rate` – recent failure / near-miss probability  

When these metrics exceed configurable thresholds, the policy:

- proposes **corrective actions** (e.g., adjust state estimator, retrain heuristic),  
- increases a **risk-weight parameter** λ to bias planning toward safer behaviour,  
- optionally activates **Safe Mode** under severe conditions,  
- logs **human-readable reasons** explaining _why_ self-healing is triggered.

### 2.1 Basic Usage

```python
from self_healing_policy import SelfHealingPolicy, SelfHealingConfig

policy = SelfHealingPolicy(SelfHealingConfig())

metrics = {
    "drift": 0.7,
    "calibration_error": 0.3,
    "heuristic_regret": 0.8,
    "failure_rate": 0.5,
}

decision = policy.evaluate(step=7, metrics=metrics)
print(decision.to_dict())
```

Example output:

```text
SELF_HEALING_TRIGGER = True
reasons:
  - drift 0.70 ≥ threshold 0.60
  - heuristic_regret 0.80 ≥ threshold 0.60
  - failure_rate 0.50 ≥ threshold 0.40
recommended_actions:
  - adjust_state_estimator
  - retrain_heuristic_small_batch
  - review_recent_failures
  - activate_safe_mode
  - increase_monitoring_frequency
  - increase_risk_weight
```

`decision.new_lambda` reflects the updated risk-weight to be fed into your planner.

---

## 3. Language-Driven Safety

**File:** `language_safety_policy.py`

This module maps natural language (English + Greek) into **risk** and **uncertainty** scaling factors:

- crowding / many people → ↑ risk  
- slippery or wet floor → ↑↑ risk and ↑ uncertainty  
- stairs / elevation changes → ↑ physical risk  
- children and elderly present → ↑ ethical and collision risk  
- unseen or hidden hazard → ↑ uncertainty  

The implementation is:

- **rule-based** (transparent keyword rules),  
- **bilingual** (EN + EL),  
- **bounded** (upper caps on scaling to remain conservative),  
- **explainable** (textual explanation of active factors).

### 3.1 Basic Usage

```python
from language_safety_policy import LanguageSafetyPolicy

policy = LanguageSafetyPolicy()
message = "There are stairs ahead, and many elderly people."
decision = policy.evaluate(message)
print(decision.to_dict())
```

Example output:

```text
risk_scale        = 2.55
uncertainty_scale = 1.00
explanation       =
Language-driven factors:
  - stairs / elevation changes
  - elderly people present
```

A Greek example:

```python
policy.evaluate("Πρόσεχε, ο διάδρομος είναι γλιστερός.")
```

This will increase both risk and uncertainty, encouraging the planner to choose safer trajectories.

Neutral example:

```python
policy.evaluate("Balanced corridor, nothing special here.")
```

→ no adjustment (risk and uncertainty remain at 1.0).

---

## 4. Demos

### 4.1 Self-Healing Demo

**File:** `self_healing_demo.py`

Simulates a sequence of planning steps with synthetic metrics:

- Step 2: drift rises → estimator adjustment  
- Step 7: high regret + high failure → Safe Mode + retraining  
- Step 10: repeated drift → new self-healing trigger  

Run:

```bash
python3 self_healing_demo.py
```

You will see step-by-step logs of metrics, triggers, reasons, and suggested actions.

---

### 4.2 Language-Driven Safety Demo

**File:** `language_safety_demo.py`

Evaluates several example messages (English + Greek) and prints the resulting risk/uncertainty scales and explanations.

Run:

```bash
python3 language_safety_demo.py
```

---

## 5. Installation & Quick Start

1. (Optional) Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate   # Linux / macOS
# .venv\\Scripts\\activate  # Windows PowerShell
```

2. Run the demos:

```bash
python3 self_healing_demo.py
python3 language_safety_demo.py
```

---

## 6. Integration Ideas

You can integrate these modules with:

- a ROS-based navigation stack (e.g., feeding `new_lambda` into a cost-based planner),  
- a custom motion planner that accepts:
  - a risk weight λ, and/or  
  - per-region cost/uncertainty multipliers,  
- simulation environments for experiments in:
  - **Human–Robot Interaction**
  - **Trust-Aware Autonomy**
  - **Uncertainty-Aware Planning**
  - **Ethical and Socially Aware Robotics**

Typical integration loop:

1. Read robot metrics (drift, failures, etc.) → `SelfHealingPolicy.evaluate(...)`.  
2. Adjust planning parameters based on `decision.new_lambda` and `decision.recommended_actions`.  
3. Parse human language messages → `LanguageSafetyPolicy.evaluate(message)`.  
4. Use `risk_scale` and `uncertainty_scale` to modulate local or global cost maps, or to choose safer behaviours.

---

## 7. License

Add your preferred license here (e.g., MIT, BSD-3-Clause, Apache-2.0).
