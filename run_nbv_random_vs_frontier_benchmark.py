# run_nbv_random_vs_frontier_benchmark.py
import numpy as np
import pandas as pd

from build_world_context import build_bottleneck_context
from nbv_irreversibility_scoring import score_candidates, topk, sample_candidates
from nbv_frontier_sampling import (
    find_frontier_cells,
    filter_reachable_frontiers,
    filter_frontiers_by_I,
    sample_from_list,
)


def summarize_topk(df_scores: pd.DataFrame, score_col: str, k: int = 10):
    t = topk(df_scores, score_col, k)
    return {
        "mean_IG": float(t["IG"].mean()),
        "mean_I": float(t["I_local"].mean()),
        "mean_R": float(t["R_local"].mean()),
    }


def summarize_random_pick(df_scores: pd.DataFrame, k: int = 10, rng=None):
    """Pick k candidates uniformly at random from scored table (not top-k)."""
    if rng is None:
        rng = np.random.default_rng(0)
    if len(df_scores) == 0:
        return {"mean_IG": np.nan, "mean_I": np.nan, "mean_R": np.nan}
    idx = rng.choice(len(df_scores), size=min(k, len(df_scores)), replace=False)
    sub = df_scores.iloc[idx]
    return {
        "mean_IG": float(sub["IG"].mean()),
        "mean_I": float(sub["I_local"].mean()),
        "mean_R": float(sub["R_local"].mean()),
    }


def main():
    n_trials = 30
    n_candidates = 500
    topK = 10

    alpha = 0.8
    beta = 0.5
    score_col = "score_IG_I_R"

    unc_thresh = 0.6
    known_thresh = 0.5
    R_min = 0.05
    I_cap = 0.6

    ctx = build_bottleneck_context(tauR=0.85)
    print(f"Base/start={ctx.start} tauR={ctx.tauR}")

    # frontier pool (reachable boundary + I-cap)
    frontier = find_frontier_cells(ctx.free_mask, ctx.unc_grid, unc_thresh=unc_thresh, known_thresh=known_thresh)
    frontier = filter_reachable_frontiers(frontier, ctx.R, R_min=R_min)
    frontier = filter_frontiers_by_I(frontier, ctx.I_world, I_cap=I_cap)

    print(f"Frontier pool size = {len(frontier)} (reachable, I<= {I_cap})")
    print(f"Trials={n_trials}, candidates/method={n_candidates}, topK={topK}")
    print(f"alpha={alpha}, beta={beta}, unc_thresh={unc_thresh}, known_thresh={known_thresh}, R_min={R_min}")

    rows = []

    for t in range(n_trials):
        rng = np.random.default_rng(100 + t)

        # -------------------------
        # 1) RANDOM_GLOBAL baseline
        # -------------------------
        rand_global = sample_candidates(rng, ctx.free_mask, n=n_candidates)
        df_rg = score_candidates(rand_global, ctx.unc_grid, ctx.I_world, ctx.R, alpha=alpha, beta=beta, r_local=2)
        s_rg = summarize_topk(df_rg, score_col=score_col, k=topK)
        rows.append({"trial": t, "method": "random_global", **s_rg})

        # -------------------------------------
        # 2) RANDOM_FRONTIER baseline (FAIR)
        # -------------------------------------
        rand_frontier = sample_from_list(rng, frontier, n=n_candidates)
        df_rf = score_candidates(rand_frontier, ctx.unc_grid, ctx.I_world, ctx.R, alpha=alpha, beta=beta, r_local=2)

        # pick random topK from frontier candidates (NOT sorted)
        s_rf = summarize_random_pick(df_rf, k=topK, rng=rng)
        rows.append({"trial": t, "method": "random_frontier", **s_rf})

        # -------------------------------------
        # 3) FRONTIER_SCORED (YOUR METHOD)
        # -------------------------------------
        s_fs = summarize_topk(df_rf, score_col=score_col, k=topK)
        rows.append({"trial": t, "method": "frontier_scored", **s_fs})

        if (t + 1) % 5 == 0:
            print(f"[{t+1:02d}/{n_trials}] done")

    out = "nbv_random_vs_frontier_benchmark.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("Saved:", out)


if __name__ == "__main__":
    main()
