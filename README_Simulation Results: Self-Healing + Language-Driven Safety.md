## Simulation Results: Self-Healing + Language-Driven Safety

This module currently runs **as a software-level simulation**, not directly on the physical robot.  
The behavior is still *scientifically meaningful*: it shows how self-healing autonomy and language-driven
safety would modulate a navigation planner when integrated into a real system.

### What is Being Simulated?

At each step, the system receives:

- **Robot reliability metrics** (simulated):
  - `drift`
  - `calibration_error`
  - `heuristic_regret`
  - `failure_rate`
- **A human language message** (English or Greek)
- It then:
  1. Evaluates **Self-Healing Navigation Policy**
  2. Evaluates **Language-Driven Safety Policy**
  3. Updates an `AbstractPlanner` with new:
     - max velocities
     - obstacle inflation radius
     - goal tolerance
     - risk weight λ (`lambda_risk`)
     - safe mode on/off

The planner in this demo is an **abstract, software-only planner** (no real robot actuation yet), but its
parameters are exactly the kind a real navigation stack would use.

### Example Output (from the integrated demo)

For example, the demo produces sequences like:

```text
===== STEP 1 =====
Message: Balanced corridor, nothing special here.

Self-Healing Decision: {
  'SELF_HEALING_TRIGGER': False,
  'reasons': [],
  'recommended_actions': [],
  'new_lambda': 0.75,
  'safe_mode': False
}

Language Safety: {
  'risk_scale': 1.0,
  'uncertainty_scale': 1.0,
  'factors': [],
  'explanation': 'No language-driven adjustment (neutral message).'
}

Planner Parameters: {
  'max_linear_velocity': 0.6,
  'max_angular_velocity': 1.0,
  'obstacle_inflation_radius': 0.4,
  'goal_tolerance': 0.15,
  'lambda_risk': 0.75
}


----
Interpretation:

The corridor is described as “nothing special” → no language-induced risk.

The robot metrics are healthy → no self-healing trigger.

The planner remains in a nominal configuration.

===== STEP 2 =====
Message: There are many people and children ahead.

Self-Healing Decision: {
  'SELF_HEALING_TRIGGER': False,
  'reasons': [],
  'recommended_actions': [],
  'new_lambda': 0.5,
  'safe_mode': False
}

Language Safety: {
  'risk_scale': 2.4,
  'uncertainty_scale': 1.21,
  'factors': ['crowding / many people', 'children present'],
  'explanation': 'Language-driven factors:\n  - crowding / many people\n  - children present'
}

Planner Parameters: {
  'max_linear_velocity': 0.25,
  'max_angular_velocity': 0.42,
  'obstacle_inflation_radius': 0.96,
  'goal_tolerance': 0.18,
  'lambda_risk': 0.5
}

Interpretation:

Language mentions crowding and children → the system increases risk_scale.

As a consequence, the planner:

reduces max speed (safer, more cautious motion),

inflates obstacles more (keeping a larger safety buffer),

slightly increases goal tolerance.

This is a realistic, human-centered adjustment: the robot behaves more conservatively when informed about
vulnerable humans and crowded spaces.

An example with self-healing and hidden hazard:

===== STEP 3 =====
Message: Be careful, hidden danger around the corner.

Self-Healing Decision: {
  'SELF_HEALING_TRIGGER': True,
  'reasons': ['drift 0.70 ≥ threshold 0.60'],
  'recommended_actions': ['adjust_state_estimator', 'increase_risk_weight'],
  'new_lambda': 0.75,
  'safe_mode': False
}

Language Safety: {
  'risk_scale': 1.4,
  'uncertainty_scale': 1.4,
  'factors': ['unseen / hidden hazard'],
  'explanation': 'Language-driven factors:\n  - unseen / hidden hazard'
}

Planner Parameters: {
  'max_linear_velocity': 0.43,
  'max_angular_velocity': 0.71,
  'obstacle_inflation_radius': 0.56,
  'goal_tolerance': 0.21,
  'lambda_risk': 0.75
}


Interpretation:

The metrics indicate high drift → the self-healing policy triggers and recommends:

adjusting the state estimator,

increasing risk weight λ.

The language mentions a hidden hazard → uncertainty and risk are increased.

Together, they push the planner into a more conservative configuration:

slower motion,

larger safety margins,

higher risk penalty in the cost function.

Finally, an ethically critical case in Greek:

===== STEP 4 =====
Message: Ο διάδρομος είναι γλιστερός και επικίνδυνος.

Self-Healing Decision: {
  'SELF_HEALING_TRIGGER': False,
  'reasons': [
    'heuristic_regret 0.90 ≥ threshold 0.60',
    'failure_rate 0.70 ≥ threshold 0.40'
  ],
  'recommended_actions': [
    'retrain_heuristic_small_batch',
    'review_recent_failures',
    'activate_safe_mode',
    'increase_monitoring_frequency',
    'increase_risk_weight'
  ],
  'new_lambda': 0.75,
  'safe_mode': True
}

Language Safety: {
  'risk_scale': 1.8,
  'uncertainty_scale': 1.5,
  'factors': ['slippery / wet floor'],
  'explanation': 'Language-driven factors:\n  - slippery / wet floor'
}

Planner Parameters: {
  'max_linear_velocity': 0.2,
  'max_angular_velocity': 0.5,
  'obstacle_inflation_radius': 0.72,
  'goal_tolerance': 0.225,
  'lambda_risk': 0.75
}


Interpretation:

The robot shows high heuristic regret and failure rate → self-healing recommends retraining and activates Safe Mode.

The Greek message indicates a slippery corridor → risk and uncertainty increase.

Combined effect:

very low speed (0.2 m/s),

increased obstacle inflation,

Safe Mode engaged.

This is a plausible and desirable behavior in hazardous conditions.
