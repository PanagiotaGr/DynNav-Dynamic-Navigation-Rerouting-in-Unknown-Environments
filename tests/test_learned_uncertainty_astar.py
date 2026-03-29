"""
Tests for contributions/learned_uncertainty_astar/

Covers:
  - UncertaintyHeuristicNet architecture and output
  - gaussian_nll_loss
  - extract_features
  - EuclideanHeuristic
  - LearnedUncertaintyHeuristic
  - astar() with both heuristics
  - AStarResult
  - train pipeline (smoke test on tiny dataset)
  - evaluate pipeline (smoke test)
"""

import sys
import os

CODE_DIR = os.path.join(
    os.path.dirname(__file__), "..",
    "contributions", "learned_uncertainty_astar", "code",
)
sys.path.insert(0, CODE_DIR)

import math
import pytest
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_grid():
    """5x5 grid, all free."""
    return np.zeros((5, 5), dtype=np.int32)


def _grid_with_wall():
    """10x10 grid with a vertical wall at x=5."""
    g = np.zeros((10, 10), dtype=np.int32)
    g[1:9, 5] = 1
    return g


# ===========================================================================
# uncertainty_heuristic_net
# ===========================================================================

@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestUncertaintyHeuristicNet:
    def _make_model(self, input_dim=11, hidden_dim=16, n_hidden=2):
        from uncertainty_heuristic_net import UncertaintyHeuristicNet
        return UncertaintyHeuristicNet(input_dim=input_dim, hidden_dim=hidden_dim, n_hidden=n_hidden)

    def test_output_shapes_batch(self):
        model = self._make_model()
        x = torch.randn(8, 11)
        mean, std = model(x)
        assert mean.shape == (8, 1)
        assert std.shape == (8, 1)

    def test_output_shapes_single(self):
        model = self._make_model()
        x = torch.randn(11)
        mean, std = model(x)
        assert mean.shape == (1, 1)
        assert std.shape == (1, 1)

    def test_mean_non_negative(self):
        model = self._make_model()
        x = torch.randn(50, 11)
        mean, _ = model(x)
        assert (mean >= 0).all()

    def test_std_positive(self):
        model = self._make_model()
        x = torch.randn(50, 11)
        _, std = model(x)
        assert (std > 0).all()

    def test_std_above_min_floor(self):
        model = self._make_model()
        x = torch.randn(20, 11)
        _, std = model(x)
        assert (std >= model.min_std).all()

    def test_predict_returns_floats(self):
        model = self._make_model()
        x = np.random.randn(11).astype(np.float32)
        mean, std = model.predict(x)
        assert isinstance(mean, float)
        assert isinstance(std, float)

    def test_numpy_input_accepted(self):
        model = self._make_model()
        x = np.ones(11, dtype=np.float32)
        mean, std = model(x)
        assert mean.shape == (1, 1)

    def test_different_inputs_different_outputs(self):
        model = self._make_model()
        x1 = torch.zeros(1, 11)
        x2 = torch.ones(1, 11) * 10
        m1, s1 = model(x1)
        m2, s2 = model(x2)
        assert not torch.allclose(m1, m2)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestGaussianNLLLoss:
    def test_loss_is_scalar(self):
        from uncertainty_heuristic_net import gaussian_nll_loss
        import torch
        mu = torch.tensor([[1.0], [2.0]])
        sigma = torch.tensor([[0.5], [0.5]])
        y = torch.tensor([1.0, 2.0])
        loss = gaussian_nll_loss(mu, sigma, y)
        assert loss.shape == ()

    def test_perfect_prediction_lower_than_bad(self):
        from uncertainty_heuristic_net import gaussian_nll_loss
        import torch
        mu_good = torch.tensor([[5.0]])
        mu_bad = torch.tensor([[0.0]])
        sigma = torch.tensor([[1.0]])
        y = torch.tensor([5.0])
        loss_good = gaussian_nll_loss(mu_good, sigma, y)
        loss_bad = gaussian_nll_loss(mu_bad, sigma, y)
        assert loss_good < loss_bad

    def test_lower_std_better_when_accurate(self):
        from uncertainty_heuristic_net import gaussian_nll_loss
        import torch
        mu = torch.tensor([[3.0]])
        y = torch.tensor([3.0])
        sigma_tight = torch.tensor([[0.1]])
        sigma_loose = torch.tensor([[2.0]])
        loss_tight = gaussian_nll_loss(mu, sigma_tight, y)
        loss_loose = gaussian_nll_loss(mu, sigma_loose, y)
        assert loss_tight < loss_loose


# ===========================================================================
# uncertainty_astar — extract_features
# ===========================================================================

class TestExtractFeatures:
    def test_output_length(self):
        from uncertainty_astar import extract_features
        grid = _simple_grid()
        feat = extract_features((2, 2), (4, 4), grid)
        assert feat.shape == (11,)

    def test_dtype_float32(self):
        from uncertainty_astar import extract_features
        grid = _simple_grid()
        feat = extract_features((0, 0), (4, 4), grid)
        assert feat.dtype == np.float32

    def test_goal_node_zero_distances(self):
        from uncertainty_astar import extract_features
        grid = _simple_grid()
        feat = extract_features((3, 3), (3, 3), grid)
        assert feat[2] == pytest.approx(0.0)   # euclid
        assert feat[3] == pytest.approx(0.0)   # manhattan
        assert feat[4] == pytest.approx(0.0)   # chebyshev

    def test_euclidean_distance_correct(self):
        from uncertainty_astar import extract_features
        grid = _simple_grid()
        feat = extract_features((0, 0), (3, 4), grid)
        assert feat[2] == pytest.approx(5.0)   # 3-4-5 triangle

    def test_norm_coords_in_range(self):
        from uncertainty_astar import extract_features
        grid = _simple_grid()
        feat = extract_features((0, 0), (4, 4), grid)
        assert 0.0 <= feat[9] <= 1.0   # norm_x
        assert 0.0 <= feat[10] <= 1.0  # norm_y

    def test_obstacle_density_nonzero_near_wall(self):
        from uncertainty_astar import extract_features
        grid = np.zeros((10, 10), dtype=np.int32)
        grid[5, 5] = 1
        feat = extract_features((5, 5), (9, 9), grid)
        assert feat[7] > 0.0  # obstacle_density


# ===========================================================================
# EuclideanHeuristic
# ===========================================================================

class TestEuclideanHeuristic:
    def test_returns_tuple(self):
        from uncertainty_astar import EuclideanHeuristic
        h = EuclideanHeuristic()
        result = h.h((0, 0), (3, 4), _simple_grid())
        assert len(result) == 2

    def test_correct_distance(self):
        from uncertainty_astar import EuclideanHeuristic
        h = EuclideanHeuristic()
        mean, std = h.h((0, 0), (3, 4), _simple_grid())
        assert mean == pytest.approx(5.0)
        assert std == pytest.approx(0.0)

    def test_zero_at_goal(self):
        from uncertainty_astar import EuclideanHeuristic
        h = EuclideanHeuristic()
        mean, std = h.h((2, 2), (2, 2), _simple_grid())
        assert mean == pytest.approx(0.0)

    def test_admissible_vs_grid_path(self):
        """Euclidean dist <= actual grid path length (admissibility check)."""
        from uncertainty_astar import EuclideanHeuristic, astar
        grid = _simple_grid()
        h = EuclideanHeuristic()
        result = astar(grid, (0, 0), (4, 4), h)
        assert result.found
        mean, _ = h.h((0, 0), (4, 4), grid)
        assert mean <= result.path_length + 1e-9


# ===========================================================================
# A* search
# ===========================================================================

class TestAStarClassic:
    def test_finds_path_open_grid(self):
        from uncertainty_astar import EuclideanHeuristic, astar
        result = astar(_simple_grid(), (0, 0), (4, 4), EuclideanHeuristic())
        assert result.found
        assert result.path[0] == (0, 0)
        assert result.path[-1] == (4, 4)

    def test_path_is_connected(self):
        from uncertainty_astar import EuclideanHeuristic, astar
        result = astar(_simple_grid(), (0, 0), (4, 4), EuclideanHeuristic())
        path = result.path
        for i in range(len(path) - 1):
            x0, y0 = path[i]
            x1, y1 = path[i + 1]
            assert abs(x1 - x0) + abs(y1 - y0) == 1

    def test_no_path_through_full_wall(self):
        from uncertainty_astar import EuclideanHeuristic, astar
        grid = np.ones((5, 5), dtype=np.int32)
        grid[0, 0] = 0
        grid[4, 4] = 0
        result = astar(grid, (0, 0), (4, 4), EuclideanHeuristic())
        assert not result.found

    def test_same_start_goal(self):
        from uncertainty_astar import EuclideanHeuristic, astar
        result = astar(_simple_grid(), (2, 2), (2, 2), EuclideanHeuristic())
        assert result.found
        assert result.path_length == pytest.approx(0.0)

    def test_expansions_positive(self):
        from uncertainty_astar import EuclideanHeuristic, astar
        result = astar(_simple_grid(), (0, 0), (4, 4), EuclideanHeuristic())
        assert result.expansions > 0

    def test_path_length_manhattan_lower_bound(self):
        from uncertainty_astar import EuclideanHeuristic, astar
        grid = _simple_grid()
        start, goal = (0, 0), (4, 4)
        result = astar(grid, start, goal, EuclideanHeuristic())
        manhattan = abs(goal[0] - start[0]) + abs(goal[1] - start[1])
        assert result.path_length >= manhattan - 1e-9


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestAStarLearned:
    def _make_heuristic(self, beta=0.0):
        from uncertainty_heuristic_net import UncertaintyHeuristicNet
        from uncertainty_astar import make_uncertainty_astar
        model = UncertaintyHeuristicNet(input_dim=11, hidden_dim=16, n_hidden=2)
        return make_uncertainty_astar(model, beta=beta)

    def test_finds_path(self):
        from uncertainty_astar import astar
        h = self._make_heuristic(beta=0.0)
        result = astar(_simple_grid(), (0, 0), (4, 4), h, beta=0.0)
        assert result.found

    def test_risk_averse_finds_path(self):
        from uncertainty_astar import astar
        h = self._make_heuristic(beta=1.0)
        result = astar(_simple_grid(), (0, 0), (4, 4), h, beta=1.0)
        assert result.found

    def test_risk_seeking_finds_path(self):
        from uncertainty_astar import astar
        h = self._make_heuristic(beta=-1.0)
        result = astar(_simple_grid(), (0, 0), (4, 4), h, beta=-1.0)
        assert result.found

    def test_h_stds_logged(self):
        from uncertainty_astar import astar
        h = self._make_heuristic(beta=0.0)
        result = astar(_simple_grid(), (0, 0), (4, 4), h, beta=0.0)
        assert len(result.h_stds) > 0
        assert all(s > 0 for s in result.h_stds)

    def test_h_means_non_negative(self):
        from uncertainty_astar import astar
        h = self._make_heuristic(beta=0.0)
        result = astar(_simple_grid(), (0, 0), (4, 4), h, beta=0.0)
        assert all(m >= 0 for m in result.h_means)

    def test_f_score_formula(self):
        from uncertainty_astar import LearnedUncertaintyHeuristic
        from uncertainty_heuristic_net import UncertaintyHeuristicNet
        model = UncertaintyHeuristicNet(input_dim=11, hidden_dim=8, n_hidden=1)
        h = LearnedUncertaintyHeuristic(model=model, beta=2.0)
        f = h.f_score(g=5.0, h_mean=3.0, h_std=1.0)
        assert f == pytest.approx(10.0)  # 5 + 3 + 2*1


# ===========================================================================
# Training smoke test
# ===========================================================================

@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestTrainingPipeline:
    def test_build_dataset_returns_correct_shapes(self):
        from train import build_dataset
        X, y = build_dataset(n_grids=5, seed=0)
        assert X.ndim == 2
        assert X.shape[1] == 11
        assert y.ndim == 1
        assert len(X) == len(y)
        assert len(X) > 0

    def test_targets_non_negative(self):
        from train import build_dataset
        X, y = build_dataset(n_grids=5, seed=1)
        assert (y >= 0).all()

    def test_train_model_returns_model(self):
        from train import build_dataset, train_model
        from uncertainty_heuristic_net import UncertaintyHeuristicNet
        X, y = build_dataset(n_grids=10, seed=0)
        model = train_model(X, y, hidden_dim=16, n_hidden=2, epochs=3, batch_size=64)
        assert isinstance(model, UncertaintyHeuristicNet)

    def test_save_load_roundtrip(self):
        import tempfile
        from train import build_dataset, train_model, save_model, load_model
        X, y = build_dataset(n_grids=5, seed=0)
        model = train_model(X, y, hidden_dim=16, n_hidden=2, epochs=2, batch_size=64)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.pt")
            save_model(model, path)
            loaded = load_model(path)
        x = torch.randn(1, 11)
        m1, s1 = model(x)
        m2, s2 = loaded(x)
        assert torch.allclose(m1, m2)
        assert torch.allclose(s1, s2)


# ===========================================================================
# Evaluation smoke test
# ===========================================================================

@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestEvaluatePipeline:
    def test_evaluate_classic_only(self):
        from evaluate import evaluate
        summaries = evaluate(n_grids=10, model_path=None, seed=0)
        assert len(summaries) == 1
        assert summaries[0].method == "classic_astar"

    def test_evaluate_with_model(self):
        import tempfile
        from train import build_dataset, train_model, save_model
        from evaluate import evaluate
        X, y = build_dataset(n_grids=10, seed=0)
        model = train_model(X, y, hidden_dim=16, n_hidden=2, epochs=2, batch_size=64)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.pt")
            save_model(model, path)
            summaries = evaluate(n_grids=10, model_path=path, betas=[0.0, 1.0], seed=1)
        methods = {s.method for s in summaries}
        assert "classic_astar" in methods
        assert any("beta=+0.0" in m for m in methods)
        assert any("beta=+1.0" in m for m in methods)

    def test_found_rate_in_range(self):
        from evaluate import evaluate
        summaries = evaluate(n_grids=20, model_path=None, seed=0)
        for s in summaries:
            assert 0.0 <= s.found_rate <= 1.0

    def test_suboptimality_classic_is_one(self):
        from evaluate import evaluate
        summaries = evaluate(n_grids=20, model_path=None, seed=0)
        classic = next(s for s in summaries if s.method == "classic_astar")
        assert classic.mean_suboptimality == pytest.approx(1.0)
