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

---

## Quickstart (No ROS)

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
# or
venv\Scripts\activate         # Windows

pip install -r requirements.txt
python eval_astar_learned.py
```

---

## 📚 Documentation & Research Notes

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

© 2026 Panagiota Grosdouli
