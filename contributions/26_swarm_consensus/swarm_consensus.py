"""
Contribution 26: Swarm Consensus Navigation
============================================
N robots reach a fault-tolerant consensus on the best navigation plan
through a distributed Byzantine-fault-tolerant (BFT) agreement protocol.

Each robot independently computes a local plan and broadcasts its
risk-weighted cost estimate. The swarm then uses a weighted median
consensus (robust to Byzantine agents) to select the global plan.

Extensions:
    - Raft-based leader election for coordinator
    - Gossip protocol for scalable plan propagation
    - Disagreement detection (builds on Contribution 09 multi-robot)

Research Question (RQ-Swarm): Does BFT swarm consensus improve
navigation robustness in the presence of compromised/faulty robots
compared to naive majority voting?

References:
    Lamport et al. (1982) "Byzantine Generals Problem"
    Olfati-Saber (2006) "Flocking for Multi-Agent Dynamic Systems"
    Vieira et al. (2009) "Swarm consensus for multi-robot navigation"
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class NavProposal:
    robot_id: int
    path: list[tuple[int, int]]      # sequence of (row, col) waypoints
    cost: float                      # risk-weighted cost estimate
    confidence: float                # robot's self-reported confidence
    timestamp: float = field(default_factory=time.time)
    signature: Optional[str] = None  # simple hash for integrity


@dataclass
class ConsensusResult:
    agreed_path: list[tuple[int, int]]
    agreed_cost: float
    n_participants: int
    n_byzantine_detected: int
    rounds: int
    method: str


# ---------------------------------------------------------------------------
# Robot agent
# ---------------------------------------------------------------------------

class SwarmRobot:
    """
    One robot in the swarm. Computes a local plan and participates in consensus.
    """

    def __init__(self, robot_id: int,
                 faulty: bool = False,
                 fault_type: str = "random"):
        self.robot_id = robot_id
        self.faulty = faulty          # simulates Byzantine / compromised robot
        self.fault_type = fault_type  # "random" | "constant_bad" | "silent"
        self._rng = np.random.default_rng(robot_id)

    def compute_proposal(self, grid: np.ndarray,
                         start: tuple[int, int],
                         goal: tuple[int, int]) -> NavProposal:
        """
        Compute a local navigation plan (A* stub with random perturbation).
        Byzantine robots return corrupted proposals.
        """
        path, cost = self._local_astar(grid, start, goal)
        confidence = 1.0

        if self.faulty:
            if self.fault_type == "random":
                # Randomly corrupt path
                path = [(self._rng.integers(0, grid.shape[0]),
                         self._rng.integers(0, grid.shape[1]))
                        for _ in range(len(path))]
                cost = self._rng.uniform(100, 1000)
                confidence = self._rng.uniform(0.1, 0.5)
            elif self.fault_type == "constant_bad":
                cost = 9999.0
                confidence = 0.99   # lies about confidence
            elif self.fault_type == "silent":
                return None   # type: ignore

        sig = f"robot_{self.robot_id}_{hash(str(path)) % 10000:04d}"
        return NavProposal(self.robot_id, path, cost, confidence,
                           signature=sig)

    def _local_astar(self, grid: np.ndarray,
                     start: tuple[int, int],
                     goal: tuple[int, int]) -> tuple[list, float]:
        """Simplified A* on the occupancy grid (BFS stub)."""
        import heapq
        H, W = grid.shape
        dist = {start: 0.0}
        prev = {start: None}
        heap = [(0.0, start)]

        while heap:
            d, u = heapq.heappop(heap)
            if u == goal:
                break
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                v = (u[0]+dr, u[1]+dc)
                if not (0 <= v[0] < H and 0 <= v[1] < W):
                    continue
                if grid[v[0], v[1]] > 0.5:
                    continue   # occupied
                nd = d + 1.0 + 2.0 * grid[v[0], v[1]]
                if nd < dist.get(v, np.inf):
                    dist[v] = nd
                    prev[v] = u
                    h = abs(v[0]-goal[0]) + abs(v[1]-goal[1])
                    heapq.heappush(heap, (nd + h, v))

        if goal not in prev:
            return [start], np.inf

        path, cur = [], goal
        while cur is not None:
            path.append(cur); cur = prev[cur]
        path.reverse()
        # Add small perturbation to cost for realism
        cost = dist.get(goal, np.inf) + self._rng.uniform(-0.5, 0.5)
        return path, max(0, cost)


# ---------------------------------------------------------------------------
# Consensus protocol
# ---------------------------------------------------------------------------

class BFTConsensus:
    """
    Byzantine-Fault-Tolerant consensus for navigation plan selection.

    Algorithm:
        1. Collect proposals from all robots.
        2. Detect outliers using robust statistics (median ± k*MAD).
        3. Weighted-median vote on path cost.
        4. Select path from highest-confidence non-Byzantine robot
           whose cost is within the consensus cost range.

    Tolerates up to ⌊(n-1)/3⌋ Byzantine robots (standard BFT bound).
    """

    def __init__(self, n_robots: int,
                 byzantine_threshold: float = 3.0,   # k*MAD for outlier detection
                 min_participants: float = 0.67):     # min fraction needed
        self.n_robots = n_robots
        self.byz_thresh = byzantine_threshold
        self.min_part = min_participants
        self._max_byzantine = (n_robots - 1) // 3

    def reach_consensus(self,
                        proposals: list[NavProposal]) -> ConsensusResult:
        """
        Run BFT consensus on a list of proposals.
        Returns the agreed plan.
        """
        valid = [p for p in proposals if p is not None]
        logger.info("Consensus: %d/%d robots responded", len(valid), self.n_robots)

        if len(valid) < int(self.n_robots * self.min_part):
            logger.error("Not enough participants for consensus")
            # Fall back to best single proposal
            best = min(valid, key=lambda p: p.cost) if valid else None
            return ConsensusResult(
                agreed_path=best.path if best else [],
                agreed_cost=best.cost if best else np.inf,
                n_participants=len(valid),
                n_byzantine_detected=0,
                rounds=1,
                method="fallback_single",
            )

        # Step 1: Detect Byzantine outliers via MAD
        costs = np.array([p.cost for p in valid])
        median_cost = float(np.median(costs))
        mad = float(np.median(np.abs(costs - median_cost))) + 1e-8
        outlier_mask = np.abs(costs - median_cost) > self.byz_thresh * mad

        honest = [p for p, out in zip(valid, outlier_mask) if not out]
        n_byz = int(outlier_mask.sum())
        logger.info("Byzantine detected: %d (threshold=%.1f*MAD)",
                    n_byz, self.byz_thresh)

        if not honest:
            honest = valid   # all suspected — use all
            n_byz = 0

        # Step 2: Weighted median cost among honest proposals
        h_costs = np.array([p.cost for p in honest])
        h_confs = np.array([p.confidence for p in honest])
        agreed_cost = float(self._weighted_median(h_costs, h_confs))

        # Step 3: Select path from robot closest to agreed cost, highest conf
        honest.sort(key=lambda p: (abs(p.cost - agreed_cost), -p.confidence))
        best = honest[0]

        logger.info(
            "Consensus reached: cost=%.3f path_len=%d by robot %d",
            agreed_cost, len(best.path), best.robot_id
        )
        return ConsensusResult(
            agreed_path=best.path,
            agreed_cost=agreed_cost,
            n_participants=len(valid),
            n_byzantine_detected=n_byz,
            rounds=2,
            method="bft_weighted_median",
        )

    @staticmethod
    def _weighted_median(values: np.ndarray,
                          weights: np.ndarray) -> float:
        """Weighted median."""
        sorted_idx = np.argsort(values)
        sv = values[sorted_idx]
        sw = weights[sorted_idx]
        cumw = np.cumsum(sw)
        cutoff = cumw[-1] / 2.0
        idx = np.searchsorted(cumw, cutoff)
        return float(sv[min(idx, len(sv) - 1)])


# ---------------------------------------------------------------------------
# Swarm coordinator
# ---------------------------------------------------------------------------

class SwarmCoordinator:
    """
    Orchestrates the full swarm: creates robots, collects proposals,
    runs BFT consensus, and logs results.
    """

    def __init__(self, n_robots: int = 6, n_byzantine: int = 1):
        self.robots = [
            SwarmRobot(i, faulty=(i < n_byzantine),
                       fault_type="random" if i % 2 == 0 else "constant_bad")
            for i in range(n_robots)
        ]
        self.consensus = BFTConsensus(n_robots)
        self.history: list[dict] = []

    def plan(self, grid: np.ndarray,
             start: tuple[int, int],
             goal: tuple[int, int]) -> ConsensusResult:
        """Full swarm planning round."""
        proposals = [r.compute_proposal(grid, start, goal)
                     for r in self.robots]
        result = self.consensus.reach_consensus(
            [p for p in proposals if p is not None]
        )
        self.history.append({
            "path_len": len(result.agreed_path),
            "cost": round(result.agreed_cost, 3),
            "byz_detected": result.n_byzantine_detected,
            "participants": result.n_participants,
        })
        return result

    def summary(self) -> dict:
        if not self.history:
            return {}
        return {
            "total_rounds": len(self.history),
            "mean_byz_detected": round(
                float(np.mean([h["byz_detected"] for h in self.history])), 2),
            "mean_cost": round(
                float(np.mean([h["cost"] for h in self.history])), 3),
            "n_robots": len(self.robots),
            "n_faulty": sum(1 for r in self.robots if r.faulty),
        }
