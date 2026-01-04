# Returnability- & Irreversibility-Aware Frontier NBV

This module introduces a safety-oriented exploration mechanism that avoids “curiosity traps” in uncertain environments. It combines **irreversibility** (how risky it is to enter a state/region) with **returnability** (how feasible it is to return to a trusted base), and applies them to **frontier-restricted Next-Best-View (NBV)** goal selection.

---

## Core idea

Candidate **frontier** goals are scored with a composite objective:

**score(g) = IG(g) − α·I(g) − β·(1 − R(g))**

Where:
- **IG(g)**: information gain surrogate (uncertainty-driven exploration)
- **I(g)**: irreversibility penalty (risk of entering hard-to-exit / high-uncertainty regions)
- **R(g)**: returnability (recoverability to base under a risk constraint)
- **α, β**: trade-off weights for safety vs exploration utility

**Frontier restriction** ensures candidates are informative *and* reachable, producing exploration targets that remain safer and more recoverable than IG-only goal selection.

---

## Key result (Frontier top-10)

In the **bottleneck template** benchmark, the selected top-10 frontier goals achieve:

- **mean IG ≈ 0.800**
- **mean I ≈ 0.351**
- **mean R ≈ 0.911**

This preserves high exploratory value while strongly improving safety and recoverability compared to unconstrained IG-only NBV.

---

## How to run

```bash
# Score frontier candidates and export top-10
python run_nbv_frontier_demo.py

# Plot overlays of top-10 goals
python plot_nbv_frontier_top10_overlay.py
```

---

## Outputs

- `nbv_frontier_scores.csv`
- `nbv_frontier_top10.csv`
- `nbv_frontier_top10_on_R.png`
- `nbv_frontier_top10_on_I.png`
