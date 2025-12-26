from dataclasses import dataclass
from typing import Tuple, List, Dict
import numpy as np
import pandas as pd
import heapq


# ==========================
# ΒΑΣΙΚΕΣ ΔΟΜΕΣ & ΠΥΡΗΝΑΣ
# ==========================

@dataclass(frozen=True)
class GridCell:
    i: int  # row index
    j: int  # column index


def neighbors(cell: GridCell, shape: Tuple[int, int]) -> List[GridCell]:
    H, W = shape
    res: List[GridCell] = []
    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # 4-connected
        ni, nj = cell.i + di, cell.j + dj
        if 0 <= ni < H and 0 <= nj < W:
            res.append(GridCell(ni, nj))
    return res


def astar_risk_aware(
    grid_unc: np.ndarray,
    start: GridCell,
    goal: GridCell,
    lambda_risk: float = 1.0,
):
    """
    A* σε grid όπου κάθε βήμα έχει κόστος:
        cost_step = 1 + λ * uncertainty(cell)
    """
    H, W = grid_unc.shape

    def heuristic(c: GridCell) -> float:
        # Manhattan distance
        return abs(c.i - goal.i) + abs(c.j - goal.j)

    # (priority, counter, cell) ώστε να μην συγκρίνει GridCell objects
    open_set: List[Tuple[float, int, GridCell]] = []
    counter = 0
    heapq.heappush(open_set, (0.0, counter, start))

    came_from: Dict[GridCell, GridCell] = {}
    g_cost: Dict[GridCell, float] = {start: 0.0}

    while open_set:
        _, _, current = heapq.heappop(open_set)

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path, g_cost[path[-1]]

        for nb in neighbors(current, (H, W)):
            step_cost = 1.0
            unc = grid_unc[nb.i, nb.j]
            if np.isnan(unc):
                unc = 0.0

            cost = step_cost + lambda_risk * float(unc)
            tentative_g = g_cost[current] + cost

            if nb not in g_cost or tentative_g < g_cost[nb]:
                g_cost[nb] = tentative_g
                priority = tentative_g + heuristic(nb)
                counter += 1
                heapq.heappush(open_set, (priority, counter, nb))
                came_from[nb] = current

    return None, np.inf


def load_grid_with_uncertainty(
    csv_path: str,
    col_i: str,
    col_j: str,
    col_unc: str,
) -> np.ndarray:
    """
    Generic loader: του δίνεις:
      - όνομα αρχείου
      - όνομα στήλης για i
      - όνομα στήλης για j
      - όνομα στήλης για uncertainty
    και επιστρέφει grid [H, W] με uncertainty.
    """
    df = pd.read_csv(csv_path)
    print(f"[INFO] Loading: {csv_path}")
    print("[INFO] Columns in CSV:", df.columns.tolist())

    for name, col in [("i", col_i), ("j", col_j), ("uncertainty", col_unc)]:
        if col not in df.columns:
            raise ValueError(
                f"Στο {csv_path} περιμένω στήλη '{col}' για {name}, "
                f"αλλά δεν υπάρχει. Δες τα ονόματα και άλλαξέ το."
            )

    max_i = int(df[col_i].max())
    max_j = int(df[col_j].max())
    grid_unc = np.full((max_i + 1, max_j + 1), np.nan, dtype=float)

    for _, row in df.iterrows():
        i = int(row[col_i])
        j = int(row[col_j])
        grid_unc[i, j] = float(row[col_unc])

    return grid_unc


def run_planner_on_grid(grid_unc: np.ndarray, lambda_risk: float, tag: str):
    """
    Κοινό helper: τρέχει A* σε ένα grid και σώζει το path.
    """
    H, W = grid_unc.shape
    print(f"[{tag}] Grid shape: {H} x {W}")

    start = GridCell(0, 0)
    goal = GridCell(H - 1, W - 1)

    path, total_cost = astar_risk_aware(grid_unc, start, goal, lambda_risk=lambda_risk)
    print(f"[{tag}] λ = {lambda_risk}")
    if path is None:
        print(f"[{tag}] Δεν βρέθηκε διαδρομή.")
        return

    print(f"[{tag}] Path length (cells): {len(path)}")
    print(f"[{tag}] Total cost: {total_cost}")

    out = [(c.i, c.j) for c in path]
    out_arr = np.array(out, dtype=int)
    out_file = f"belief_risk_path_{tag}.csv"
    np.savetxt(out_file, out_arr, delimiter=",", fmt="%d")
    print(f"[{tag}] Έσωσα το path στο {out_file}")


# ==========================
# DEMOS ΓΙΑ ΔΙΑΦΟΡΕΤΙΚΕΣ ΠΕΡΙΠΤΩΣΕΙΣ
# ==========================

def demo_small_grid():
    """
    Μικρό synthetic παράδειγμα (5x5) για sanity check.
    """
    grid = np.zeros((5, 5), dtype=float)
    grid[2, 2] = 10.0  # πολύ risky κελί στο κέντρο

    start = GridCell(0, 0)
    goal = GridCell(4, 4)

    path, total_cost = astar_risk_aware(grid, start, goal, lambda_risk=1.0)
    print("=== DEMO SMALL GRID ===")
    print("Path:", path)
    print("Total cost:", total_cost)


def demo_case1_coverage_basic():
    """
    Case 1:
      coverage_grid_with_uncertainty.csv
      στήλες: i, j, uncertainty
    """
    csv_path = "coverage_grid_with_uncertainty.csv"
    grid_unc = load_grid_with_uncertainty(
        csv_path=csv_path,
        col_i="i",
        col_j="j",
        col_unc="uncertainty",
    )
    run_planner_on_grid(grid_unc, lambda_risk=0.5, tag="case1_basic")


def demo_case2_pose_unc():
    """
    Case 2:
      coverage_grid_with_uncertainty_pose.csv
      στήλες: row, col, uncertainty_fused (ή άλλη αν προτιμάς)
    """
    csv_path = "coverage_grid_with_uncertainty_pose.csv"
    grid_unc = load_grid_with_uncertainty(
        csv_path=csv_path,
        col_i="row",                # από το CSV
        col_j="col",                # από το CSV
        col_unc="uncertainty_fused" # ΕΔΩ διαλέγουμε τι σημαίνει "risk"
    )
    run_planner_on_grid(grid_unc, lambda_risk=0.7, tag="case2_pose")

def demo_case2_pose_only():
    """
    Case 2b:
      Ίδιο grid, αλλά risk = pose_uncertainty_norm
    """
    csv_path = "coverage_grid_with_uncertainty_pose.csv"
    grid_unc = load_grid_with_uncertainty(
        csv_path=csv_path,
        col_i="row",
        col_j="col",
        col_unc="pose_uncertainty_norm",
    )
    run_planner_on_grid(grid_unc, lambda_risk=0.7, tag="case2_pose_only")


def demo_case3_vla_unc():
    """
    Case 3:
      coverage_grid_with_uncertainty_vla.csv
      π.χ. στήλες: grid_i, grid_j, vla_uncertainty
    """
    csv_path = "coverage_grid_with_uncertainty_vla.csv"
    grid_unc = load_grid_with_uncertainty(
        csv_path=csv_path,
        col_i="grid_i",          # άλλαξέ το αν χρειάζεται
        col_j="grid_j",
        col_unc="vla_uncertainty",  # ή ό,τι έχεις στο CSV
    )
    run_planner_on_grid(grid_unc, lambda_risk=0.7, tag="case3_vla")


def demo_case4_feature_unc():
    """
    Case 4:
      feature_uncertainty_grid.csv
      π.χ. στήλες: grid_i, grid_j, feature_uncertainty
    """
    csv_path = "feature_uncertainty_grid.csv"
    grid_unc = load_grid_with_uncertainty(
        csv_path=csv_path,
        col_i="grid_i",                # άλλαξέ το αν χρειάζεται
        col_j="grid_j",
        col_unc="feature_uncertainty",  # ή άλλο όνομα στο CSV
    )
    run_planner_on_grid(grid_unc, lambda_risk=0.5, tag="case4_feature")


# ==========================
# MAIN
# ==========================

if __name__ == "__main__":
    demo_small_grid()
    demo_case2_pose_unc()      # fused
    demo_case2_pose_only()   # βγάζεις σχόλιο αν θες να το τρέξεις κι αυτό

    # ΔΙΑΛΕΓΕΙΣ ΠΕΡΙΠΤΩΣΗ ΑΝΑΛΟΓΑ ΜΕ ΤΟ CSV ΠΟΥ ΘΕΣ
    # Βγάλε το σχόλιο από όποιο demo θέλεις να τρέξεις.

    # demo_case1_coverage_basic()
    # demo_case2_pose_unc()
    # demo_case3_vla_unc()
    # demo_case4_feature_unc()
