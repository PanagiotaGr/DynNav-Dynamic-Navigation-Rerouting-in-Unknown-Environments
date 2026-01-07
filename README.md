# Dynamic Navigation and Uncertainty-Aware Replanning in Unknown Environments

**TL;DR**  
A research-oriented navigation framework for autonomous robots operating in **unknown, evolving environments under uncertainty**, combining **risk-aware replanning**, **uncertainty-aware exploration**, and **learning-augmented planning** with extensive quantitative evaluation.

This repository presents a research-oriented framework for autonomous robotic navigation in unknown and evolving environments under uncertainty.  
The work focuses on dynamic replanning, uncertainty-aware exploration, and risk-sensitive decision-making, combining classical planning, probabilistic estimation, and learning-based heuristics.

Developed as an **individual research project** at the  
**School of Electrical and Computer Engineering, Democritus University of Thrace (D.U.Th.)**.

> 📘 For a detailed technical description, experimental analysis, and quantitative results, see **`README-large info.md`**.

---

## Abstract

Autonomous navigation in unknown environments is fundamentally constrained by:

- Partial observability and incrementally built maps  
- Visual odometry drift and feature sparsity  
- Dynamic obstacles requiring frequent replanning  
- Trade-offs between optimality, safety, coverage, and runtime  

This project introduces a **unified navigation pipeline** that explicitly models uncertainty and risk, integrates **learned heuristics** into classical planners, and supports **adaptive behavior under distributional shift (OOD)**.  
The framework is accompanied by **extensive quantitative evaluation**, ablations, and reproducible experiments.


## How to Run This Repository (Step-by-step)

This repo includes both **offline experiments** (pure Python, no ROS) and **ROS 2 / Gazebo** integration.
If you are new here, start with the offline runs first.

---

### 0) Recommended setup (clean Python environment)

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
# or
venv\Scripts\activate           # Windows

pip install -r requirements.txt
If something fails, check:

Python version (3.9+ recommended)

pip install -r requirements.txt completed successfully

Part A — Offline experiments (no ROS required)
These scripts run on a laptop and reproduce core research claims.

A1) Learned A* vs Classical A* (primary entry point)
bash
Αντιγραφή κώδικα
python eval_astar_learned.py
Expected outcome

Console output comparing:

node expansions (learned vs classical)

path cost (should remain optimal / unchanged)

Optional plots saved under figures/ or results/ (depending on script config)

If you want to retrain:

bash
Αντιγραφή κώδικα
python train_heuristic.py
Related files:

train_heuristic.py, train_heuristic_curriculum.py

eval_astar_learned.py, astar_learned_heuristic.py

datasets: planner_dataset*.npz

A2) Irreversibility-aware navigation (hard feasibility vs soft risk weighting)
Run a feasibility sweep in a bottleneck environment:

bash
Αντιγραφή κώδικα
python run_irreversibility_bottleneck_sweep.py
Expected outcome

A feasibility phase transition vs threshold τ

CSV logs and plots saved under results/ or figures/

To compare hard vs soft planning overlays (if enabled in repo scripts):

bash
Αντιγραφή κώδικα
python plot_path_overlay_hard_vs_soft.py
Related docs:

Irreversibility_Aware_Navigation_New_Contribution.md

Proposition_Irreversibility_vs_Risk_Weighting.md

A3) Risk-weighted planning (λ-sweep)
bash
Αντιγραφή κώδικα
python run_risk_weighted_lambda_sweep.py
python plot_belief_risk_lambda_sweep.py
Expected outcome

Trade-off curves: path length vs risk exposure across λ

Outputs saved under results/, figures/, and CSV logs

A4) IDS (Innovation-based intrusion detection inside UKF)
Replay-attack experiment:

bash
Αντιγραφή κώδικα
python eval_ids_replay.py
python plot_ids_from_csv.py --csv ids_replay_log.csv
Expected outcome

Mahalanobis distance vs χ² threshold

Adaptive trust scaling over time

Detection delay / false alarm metrics (printed or saved)

Related docs:

README_Innovation-Based_IDS_for_UKF_Sensor_Fusion.md

A5) TF integrity IDS (CUSUM) — stealth drift detection
bash
Αντιγραφή κώδικα
python plot_ids_to_planner_hook.py
# or (depending on your entry scripts)
python eval_ids_sweep.py
Expected outcome

A score signal that stays sub-threshold

A CUSUM statistic that accumulates and triggers an alarm under stealth drift

Related docs:

README_TF_Attack_Aware_IDS.md

Part B — ROS 2 + Gazebo integration (TurtleBot3)
This part requires a working ROS 2 installation and TurtleBot3 packages.
If you only want the research experiments, Part A is sufficient.

B1) ROS 2 prerequisites (high level)
You will need:

ROS 2 distribution installed

Gazebo simulator

TurtleBot3 packages / simulation worlds

Related notes:

README_LiDAR_SLAM_TurtleBot3_ROS2.md

B2) Typical usage
Once ROS 2 is configured, you can:

run the navigation stack

stream VO / SLAM / LiDAR

enable replanning and uncertainty-aware planners

(Exact commands depend on your local ROS 2 setup; see the ROS-specific README above.)

Troubleshooting
Common issues
Missing packages → re-run pip install -r requirements.txt

Script cannot find data → check data/, results/, research_results/

Plotting fails → ensure matplotlib installed and try running scripts from repo root

If you are stuck, open an Issue with:

script name

error traceback

OS + Python version

---

---

## Quickstart 

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
# or
venv\Scripts\activate         # Windows

pip install -r requirements.txt
python eval_astar_learned.py
```

---

##  Documentation & Research Notes

This repository includes extensive research documentation for readers interested in
specific subsystems, experimental results, and theoretical contributions:

- **`README-large info.md`** – Extended technical description of the full navigation pipeline.
- **`Abstract_and_Contributions.md`** – High-level research abstract and summary.
- **`Irreversibility_Aware_Navigation_New_Contribution.md`** – Irreversibility-aware planning formulation and evaluation.
- **`Proposition_Irreversibility_vs_Risk_Weighting.md`** – Theoretical comparison of irreversibility vs risk weighting.
- **`Frontier-Restricted NBV Benchmark.md`** – Frontier-restricted NBV benchmark documentation.
- **`Returnability- & Irreversibility-Aware Frontier NBV.md`** – Returnability/irreversibility-aware NBV.
- **`README_safe_mode_experiments.md`** – OOD-aware safe modes and threshold ablations.
- **`README_trust_navigation.md`** – Self-trust modeling and adaptive navigation.
- **`Human Preference–Aware Risk Navigation.md`** – Human-centered risk-aware navigation.
- **`Multi-Robot Safe Mode Navigation under Uncertainty.md`** – Multi-robot extensions.
- **`README_Innovation-Based_IDS_for_UKF_Sensor_Fusion.md`** – Innovation-based IDS for UKF.
- **`README_TF_Attack_Aware_IDS.md`** – TF integrity monitoring and attack-aware IDS.
- **`README_LiDAR_SLAM_TurtleBot3_ROS2.md`** – LiDAR SLAM and TurtleBot3 ROS 2 notes.
- **`README Self-Healing Navigation & Language-Driven Safety.md`** – Language-driven safety and self-healing navigation.
- **`READMEResults: Self-Healing + Language Safety + Trust + Ethical Navigation.md`** – Aggregated experimental results.

---

## Key Research Contributions

### 1. Uncertainty-Aware Dynamic Navigation (ROS 2)
- Risk-adjusted global planning on evolving occupancy maps  
- Online replanning with uncertainty-weighted A*  
- ROS 2 / Gazebo integration with TurtleBot3  

### 2. Coverage Planning & Exploration Under Uncertainty
- AOI coverage planning inspired by photogrammetry  
- Missing-cell detection and uncertainty-aware replanning  
- Information Gain (IG) and Next-Best-View (NBV) strategies under VO drift  

### 3. Learned Admissible A* Heuristics
- Neural heuristic for A* with preserved optimality  
- Curriculum training and online self-improvement  
- Significant node-expansion reduction with formal benchmarking  

### 4. Belief–Risk Planning & Self-Trust Adaptation
- Planning objective combining geometric cost and uncertainty exposure  
- Adaptive risk weighting via self-trust indices  
- OOD-aware safe modes with threshold ablations  

### 5. Irreversibility & Returnability-Aware Navigation
- Planning with bottleneck exposure and irreversibility penalties  
- Frontier/NBV selection constrained by returnability feasibility  
- Synthetic benchmarks and parameter sweeps  

### 6. Security-Aware Estimation & Planning Hooks
- Innovation-based IDS for UKF sensor fusion  
- TF integrity monitoring with CUSUM / EWMA variants  
- Attack-aware adaptive trust weighting  

### 7. Human, Language & Ethics-Aware Extensions
- Human preference–aware risk navigation  
- Language-driven safety logic and self-healing behavior  
- Ethical constraint layers for decision-making  

---

## Reproducibility & Evaluation

This repository includes:
- Multi-seed experiment pipelines  
- Ablation studies  
- Statistical validation artifacts (tables, plots, t-tests)  

Results are stored under `research_results/`, `results/`, and `figures/`.

---

## Citation

If you use this work, please cite:

See `CITATION.cff`.

---

## License

Creative Commons Attribution–NonCommercial–ShareAlike 4.0 International  
See `LICENSE`.

---

## Disclaimer

This repository is intended for **research and educational use only**.  
The provided models and policies must not be deployed in safety-critical systems without additional validation.

---

## Author

**Panagiota Grosdouli**  
Electrical & Computer Engineering, D.U.Th.

