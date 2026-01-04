# Abstract & Main Contributions

## Abstract

Autonomous navigation in unknown environments is fundamentally limited by sensing uncertainty, incomplete maps, and the risk of entering regions that are difficult or impossible to safely exit. Classical motion planners typically optimize geometric cost under assumptions of reliable state estimation and static environments, leading to brittle behavior in the presence of drift, miscalibrated uncertainty, or structural bottlenecks.

This work presents a unified framework for uncertainty-aware and safety-oriented autonomous navigation that integrates irreversibility-aware planning, returnability-aware exploration, learned heuristics, and adaptive risk management. We introduce irreversibility as a planning primitive that captures the difficulty of escaping a region under uncertainty, and show how hard feasibility constraints and soft risk penalties induce fundamentally different behaviors. To support safe exploration, we further propose a frontier-restricted Next-Best-View (NBV) formulation that jointly optimizes information gain, irreversibility, and returnability to a trusted base.

The framework is implemented in a fully reproducible research pipeline with extensive ablation studies, statistical validation, and multi-scenario benchmarks. Experimental results demonstrate substantial reductions in planning brittleness, improved safety and recoverability during exploration, and significant computational gains through learned A* heuristics, all achieved with minimal sacrifice in path optimality or information gain.

---

## Main Contributions

1. **Irreversibility-Aware Navigation Planning**  
   We introduce irreversibility as a quantitative measure of how risky it is to enter a region due to escape difficulty under uncertainty. We study two complementary mechanisms:
   - hard irreversibility constraints that enforce feasibility thresholds, and  
   - soft irreversibility penalties that enable smooth risk–effort trade-offs.  
   We empirically demonstrate the brittleness of hard constraints and the shaping behavior of soft penalties across bottleneck and cul-de-sac environments.

2. **Returnability- & Irreversibility-Aware Frontier NBV**  
   We propose a safety-oriented frontier-based NBV formulation that scores candidate viewpoints using information gain, irreversibility, and returnability. Frontier restriction ensures reachability, while returnability explicitly accounts for the feasibility of returning to a trusted base under risk constraints. The resulting exploration strategy avoids curiosity traps while preserving high exploratory value.

3. **Learned A* Heuristics with Online Improvement**  
   We develop a neural heuristic for A* that significantly reduces node expansions while preserving optimal path cost. An online self-improving loop allows the heuristic to refine itself as new planning data is collected, yielding consistent performance improvements over time.

4. **Uncertainty Calibration and Drift-Aware Planning**  
   We integrate learned drift and uncertainty models with calibration mechanisms that correct over- and under-confidence in risk estimates. Calibrated uncertainty grids reshape the effective risk landscape, enabling more robust risk-sensitive planning without altering the underlying geometry of the environment.

5. **Self-Trust, OOD Awareness, and Safe-Mode Policies**  
   We introduce self-trust and out-of-distribution (OOD) signals to adapt risk sensitivity online. When operating conditions degrade, the system activates a safe mode that prioritizes minimum-risk solutions, improving success rates in challenging environments at modest additional cost.

6. **Attack-Aware State Estimation Monitoring**  
   We provide an innovation-based intrusion detection system (IDS) for UKF-based state estimation pipelines, enabling detection of integrity violations and abnormal estimation behavior. The IDS is fully integrated with the navigation stack and evaluated through controlled attack injection experiments.

---

## Reproducibility

All experiments, figures, and tables reported in this work are reproducible via scripts included in the repository. Each major result is accompanied by corresponding execution commands, logged outputs, and plotting utilities.

