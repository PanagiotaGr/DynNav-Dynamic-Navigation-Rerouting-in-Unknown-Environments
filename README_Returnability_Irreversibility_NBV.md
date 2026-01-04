Returnability- & Irreversibility-Aware Frontier NBV

This module introduces a safety-oriented exploration mechanism that avoids “curiosity traps” in uncertain environments. We combine irreversibility (how risky it is to enter a state) with returnability (how feasible it is to return to a trusted base), and apply them to frontier-restricted Next-Best-View (NBV) goal selection.

Core Idea

We score candidate frontier goals using:

score(g)=IG(g)−αI(g)−β(1−R(g))

IG(g): information gain surrogate (uncertainty-driven exploration)

I(g): irreversibility penalty (risk of entering hard-to-exit regions)

R(g): returnability (recoverability to base under a risk constraint)

Frontier restriction ensures candidates are both informative and reachable, producing exploration targets that remain safe and recoverable.

Key Result (Frontier Top-10)

In the bottleneck template benchmark:

mean IG ≈ 0.800

mean I ≈ 0.351

mean R ≈ 0.911

This preserves high exploratory value while strongly improving safety and recoverability compared to unconstrained IG-only NBV.

How to Run
# Score frontier candidates and export top-10
python run_nbv_frontier_demo.py

# Plot overlays of top-10 goals
python plot_nbv_frontier_top10_overlay.py

Outputs

nbv_frontier_scores.csv

nbv_frontier_top10.csv

nbv_frontier_top10_on_R.png

nbv_frontier_top10_on_I.png
