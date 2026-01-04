# run_nbv_frontier_demo.py
from build_world_context import build_bottleneck_context
from nbv_frontier_sampling import find_frontier_cells, filter_reachable_frontiers
from nbv_irreversibility_scoring import score_candidates, topk


def main():
    ctx = build_bottleneck_context()

    frontier = find_frontier_cells(ctx.free_mask, ctx.unc_grid, unc_thresh=0.6, known_thresh=0.5)
    frontier = filter_reachable_frontiers(frontier, ctx.R, R_min=0.05)

    print(f"Base/start={ctx.start} tauR={ctx.tauR}")
    print(f"Frontier candidates (reachable): {len(frontier)}")

    df = score_candidates(frontier, ctx.unc_grid, ctx.I_world, ctx.R, alpha=0.8, beta=0.5, r_local=2)
    df.to_csv("nbv_frontier_scores.csv", index=False)

    top = topk(df, "score_IG_I_R", 10)
    top.to_csv("nbv_frontier_top10.csv", index=False)

    print("Top-10 Frontier NBV summary:")
    print(
        f"mean IG={top['IG'].mean():.3f}  "
        f"mean I={top['I_local'].mean():.3f}  "
        f"mean R={top['R_local'].mean():.3f}"
    )


if __name__ == "__main__":
    main()
