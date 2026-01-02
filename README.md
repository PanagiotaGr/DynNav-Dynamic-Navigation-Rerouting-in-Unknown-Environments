# Dynamic Navigation and Uncertainty-Aware Replanning in Unknown Environments

This repository implements a full research-oriented pipeline for **autonomous robotic navigation in unknown environments under sensing and localization uncertainty**.  
The system integrates:

- **ROS 2** navigation stack  
- **Visual Odometry (VO)** and SLAM-style mapping  
- **Uncertainty-aware coverage planning & exploration**  
- **Learned A\*** heuristics with online adaptation  
- **Belief–risk planning, self-trust and OOD-aware behavior**  
- **Multi-objective planners** for safety, coverage and efficiency  

with **extensive quantitative evaluation, ablation studies and statistical validation**.

The work was developed as an **individual research project** at the  
School of Electrical and Computer Engineering, Democritus University of Thrace (D.U.Th.).

---

## 1. Research Problem

Autonomous navigation in realistic, unknown environments is fundamentally constrained by:

- **Sensor uncertainty**, especially visual odometry drift and feature sparsity  
- **Incomplete, evolving maps** during exploration  
- **Dynamic obstacles and online replanning** requirements  
- **Trade-offs** between path optimality, coverage, safety and computation time  

Classical global planners (e.g. A\*, RRT\*) typically assume:

- reliable state estimation  
- static cost maps  
- and no explicit notion of risk/uncertainty in the objective.

In practice, **pose drift, tracking failures and miscalibrated uncertainty estimates** directly affect navigation quality and safety.  
This project studies **how navigation performance can be improved by explicitly modeling uncertainty**, learning data-driven heuristics, and adding **self-trust and out-of-distribution (OOD)-aware behavior** to the planning loop.

---

## 2. Main Contributions

### 2.1 Core Navigation & Coverage

- **Uncertainty-aware coverage planning** using VO-derived feature density and pose uncertainty.  
- **Dynamic navigation and replanning** based on an uncertainty-weighted priority field.  
- **Information Gain (IG) & Next-Best-View (NBV) exploration** in partially known maps.  
- **Multi-objective navigation** combining entropy reduction, coverage, path length and safety.

### 2.2 Learned Heuristics for A\*

- **Neural learned heuristic for A\*** that preserves optimality while reducing node expansions.  
  - Up to **~70% reduction in expansions** (e.g. *1099 → 324* expansions)  
    with **no degradation** in the optimal path cost.  
- **Online self-improving loop**:
  - the heuristic is refined iteratively as more planning data is collected,  
  - leading to **steady improvements** across successive iterations.

### 2.3 Uncertainty-Aware Exploration and Replanning

- **VO-based coverage replanning**:
  - coverage improves from **~25.49% → 100%** with uncertainty-aware missing-cell replanning.  
- **Uncertainty-aware exploration**:
  - IG-based NBV selection augmented with **VO-derived uncertainty**  
  - avoids over-exploring regions with unreliable pose estimates.  
- **Multi-objective planner**:
  - combines **entropy**, **uncertainty** and **path cost**,  
  - achieving **mean planning time ~0.8 ms per query**.

### 2.4 Drift Prediction, Calibration and Safe Behavior

- **Drift-aware exploration**:
  - VO drift is predicted via a learned model (e.g. ridge regression / neural net),  
  - drift is **penalized directly in the exploration objective**,  
  - discouraging trajectories through regions with high expected drift.  
- **UKF fusion of VO & wheel odometry**:
  - reduces mean drift (e.g. **mean ≈ 0.045** in normalized units),  
  - provides **pose uncertainty estimates** that are propagated into the planning grid.  
- **Uncertainty calibration**:
  - learns and calibrates predictive uncertainty grids,  
  - enabling **risk-sensitive planning** that is robust to over/under-confident models.

---

## 3. System Architecture

The overall pipeline is structured as:

1. **SLAM + Visual Odometry**
   - Monocular ORB-based VO
   - Essential matrix pose recovery
   - Drift and inlier statistics
2. **Coverage Grid Construction**
   - Project robot trajectory into a coverage grid
   - Estimate feature/texture density and VO-based uncertainty
3. **Uncertainty & Priority Field Mapping**
   - Build uncertainty and information maps  
   - Construct a **priority field** combining coverage deficit and uncertainty
4. **Dynamic Replanning**
   - Weighted dynamic replanning on the priority field  
   - Risk-adjusted global path planning with A\*
5. **Information Gain & NBV Planning**
   - Entropy-based IG evaluation for candidate viewpoints  
   - Drift-aware NBV selection
6. **Learned A\* Heuristic Integration**
   - Data collection from classical A\* runs  
   - Neural heuristic training and evaluation  
   - Online updates and ablation studies
7. **Belief–Risk Planning & Self-Trust**
   - Trade-off between geometric cost and integrated risk  
   - Self-trust–adaptive risk weighting and OOD-aware safe modes
8. **Benchmarking & Statistical Validation**
   - Multi-seed experiments, ablations, t-tests and confidence intervals

---

## 4. Key Modules

### 4.1 Photogrammetry-Inspired Coverage Planning

Located in `photogrammetry_module/`:

- Coverage path planning for **rectangular and polygonal areas of interest (AOIs)**  
- Detection and replanning for **missing cells**  
- **Uncertainty-weighted** priority fields for coverage  
- **Adaptive online replanning** to improve coverage under drift  
- Coverage improvement evaluation and visualization

### 4.2 Visual Odometry and Uncertainty Modeling

- VO and SLAM-related components under `dynamic_nav/`, `modules/`, and `visual_odometry/`-related scripts:
  - Monocular ORB-based VO
  - Essential matrix pose recovery
  - VO drift estimation and statistics
- **Uncertainty modeling**:
  - EKF / UKF-based sensor fusion (`ekf_fusion.py`, `ukf_fusion.py`, `ukf_fusion_vo.py`)  
  - Pose uncertainty propagation into the grid  
  - Entropy and uncertainty contour modeling

### 4.3 Learned A\* Heuristic

Core scripts include:

- `train_heuristic.py`, `train_heuristic_curriculum.py`  
- `eval_astar_learned.py`, `astar_learned_heuristic.py`  

Features:

- Neural regression heuristic for A\* on navigation graphs  
- Curriculum training and dataset sweeps (`planner_dataset*.npz`)  
- Heuristic **uncertainty-aware variants** (`train_heuristic_uncertainty.py`)  
- Benchmarking vs classical A\*:
  - up to **~70% fewer node expansions**  
  - same optimal path cost  
- Ablation studies and logging (`ablation_study.py`, `heuristic_logger.py`).

### 4.4 Information Gain, NBV and Multi-Objective Planners

Main scripts:

- `info_gain_planner.py`  
- `ig_planners_demo.py`  
- `multiobj_planner.py`  
- `multiobj_viz.py`, `multiobj_entropy_viz.png`, `multiobj_pareto_front.csv`  

Capabilities:

- IG-based **Next-Best-View** (NBV) selection  
- Pareto-front **multi-objective navigation**:
  - entropy reduction
  - coverage
  - path length
  - risk / uncertainty  
- Weighted replanning under uncertainty and path smoothing.

---

## 5. Quantitative Results

### 5.1 Learned Heuristic for A\*

- **Node expansions** reduced by up to **~70%** (e.g. 1099 → 324)  
  while preserving **optimal path cost**.  
- **Online self-improving loop**:
  - as new trajectories are added to the dataset,
  - the heuristic improves over successive training iterations.

### 5.2 Coverage & Exploration Performance

- **VO-based coverage replanning**:
  - initial coverage ~**25.49%**  
  - improved to **100%** after uncertainty-aware replanning.  
- **Uncertainty-aware exploration**:
  - avoids low-texture / high-drift regions  
  - maintains coverage while limiting exposure to unreliable pose estimates.

### 5.3 Risk-Sensitive Planning, Self-Trust and OOD-Aware Behavior

We designed a synthetic benchmark with **multiple navigation scenarios**.  
Each trial contains three candidate paths with different:

- geometric path length  
- drift exposure  
- integrated uncertainty  

We compare four policies:

1. **baseline_shortest**  
   - Ignores risk entirely; always chooses the shortest path.  
   - Mean path length: **≈ 10.18**  
   - Mean drift exposure: **≈ 2.58**  
   - Success rate: **~70%**

2. **fixed_risk**  
   - Uses a fixed weight λ on drift/uncertainty.  
   - Mean path length: **≈ 10.8**  
   - Mean drift ≈ **1.6**  
   - Success rate: **> 90%**

3. **adaptive_risk**  
   - Dynamically adapts λ based on a **self-trust index S**  
   - Similar path length to fixed_risk (**≈ 10.8**)  
   - Mean drift ≈ **1.6**  
   - Success rate: **> 90%**

4. **ood_aware**  
   - Activates a **safe mode** in “difficult” environments,  
   - chooses paths with **minimum drift/uncertainty**.  
   - Mean path length: **≈ 11.16**  
   - Mean drift: **≈ 1.47**  
   - Success rate: **~94%**  
   - Safe mode triggered in **~31.5%** of trials.

**Conclusion:**  
Explicit modeling of uncertainty, self-trust and OOD-awareness yields **substantially safer navigation** (lower drift, higher success rate) at **minimal cost in path length**.

### 5.4 Uncertainty Calibration & Drift Modeling

- **Uncertainty calibration**:
  - learned vs calibrated uncertainty grids are compared on fixed paths  
  - calibration reshapes risk distribution, correcting over/under-confidence while preserving trajectory structure.  
- **Drift-aware NBV**:
  - comparison of classical vs drift-aware NBV:
    - drift-aware policy systematically reduces exposure to high-drift regions  
    - maintains comparable information gain over multi-step exploration.  

Corresponding scripts and plots:

- `analyze_learned_vs_calib_paths.py`, `plot_learned_vs_calib_paths.py`  
- `drift_aware_nbv_experiment.py`, `drift_aware_nbv_multistep.py`  
- `nbv_*` CSV files and figures in the repository.

---

## 6. Repository Structure (High-Level)

- `dynamic_nav/` – core navigation and ROS 2 integration  
- `modules/` – planning, coverage, photogrammetry-inspired components  
- `photogrammetry_module/` – coverage planning and AOI handling  
- `neural_uncertainty/` – learned uncertainty models and calibration  
- `research_experiments/` – experiment configurations and scripts  
- `research_results/` – aggregated research logs and summary outputs  
- `results/`, `figures/`, `data/plots/` – plots, tables and evaluation artifacts  
- `cpp_extension/` – C++ extensions and RL/ROS2 integration  
- `install/`, `launch/` – setup and launch files for ROS 2 / Gazebo

---

## 7. Installation

1. **Create and activate a Python environment** (recommended):

```bash
python -m venv venv
source venv/bin/activate      # Linux / macOS
# or
venv\Scripts\activate         # Windows
```

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

3. (Optional) **Install as a package**:

```bash
pip install -e .
```

Additional dependencies may be required for full **ROS 2 + Gazebo** integration  
(ROS 2 distribution, TurtleBot3 packages, simulation worlds, etc.).

---

## 8. Running Experiments

Below are typical entry points (see code comments for detailed options):

### 8.1 Learned Heuristic Experiments

```bash
# Train a learned heuristic for A*
python train_heuristic.py

# Evaluate learned vs classical A*
python eval_astar_learned.py
```

### 8.2 Uncertainty-Aware & Multi-Objective Planning

```bash
# Run information-gain planners and demos
python ig_planners_demo.py

# Run multi-objective planner experiments
python multiobj_planner.py
```

### 8.3 Belief–Risk Planning and λ-Sweep

```bash
# Sweep λ for belief–risk planning
python sweep_lambda_fused_risk.py

# Plot λ-sweep results
python plot_belief_risk_lambda_sweep.py
```

### 8.4 Drift-Aware NBV and OOD Experiments

```bash
# Drift-aware single-step NBV comparison
python drift_aware_nbv_experiment.py

# Multi-step drift-aware exploration
python drift_aware_nbv_multistep.py
```

Additional scripts in `research_experiments/`, `logs_*` and `research_results/`  
provide **reproducible pipelines** for all reported experiments.

---

## 9. Simulation Environment

The navigation framework is designed to run with:

- **ROS 2** middleware  
- **Gazebo** simulator  
- **TurtleBot3** mobile robot platform  

Features include:

- LiDAR-based obstacle avoidance  
- Dynamic map updates and replanning  
- Integration of VO, odometry and uncertainty into the navigation stack.

---

## 10. Applications

This work is relevant to:

- Autonomous mobile robots (ground and aerial)  
- UAV exploration and mapping  
- Search and rescue operations in unknown environments  
- Inspection and infrastructure monitoring  
- Active perception systems and safety-critical autonomous systems.

---

## 11. Author

**Panagiota Grosdouli**  
Electrical & Computer Engineering, D.U.Th.

---

## 12. Disclaimer

This repository is intended for **research and educational use only**.  
Trained neural models and experimental datasets are provided for **reproducibility**,  
and should not be used as-is in safety-critical real-world systems  
without additional validation, testing and certification.


© 2026 Panagiota Grosdouli

This repository is released under the Creative Commons
Attribution-NonCommercial-ShareAlike 4.0 International License.

Use of the code, results, figures or ideas in publications
requires explicit citation and/or permission.

