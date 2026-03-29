import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'contributions', '01_learned_astar', 'code'))

import pytest
import numpy as np
import tempfile
try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestHeuristicNet:
    def test_forward_single_input(self):
        from learned_heuristic import HeuristicNet
        model = HeuristicNet(input_dim=4, hidden_dim=16)
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        out = model(x)
        assert out.shape == (1, 1)

    def test_forward_batch(self):
        from learned_heuristic import HeuristicNet
        model = HeuristicNet(input_dim=4, hidden_dim=16)
        x = torch.randn(8, 4)
        out = model(x)
        assert out.shape == (8, 1)

    def test_forward_numpy_input(self):
        from learned_heuristic import HeuristicNet
        model = HeuristicNet(input_dim=3, hidden_dim=8)
        x = np.array([1.0, 0.5, 0.2])
        out = model(x)
        assert out.shape == (1, 1)

    def test_compat_in_dim_alias(self):
        from learned_heuristic import HeuristicNet
        model = HeuristicNet(in_dim=5, hidden=32)
        x = torch.randn(2, 5)
        out = model(x)
        assert out.shape == (2, 1)

    def test_missing_input_dim_raises(self):
        from learned_heuristic import HeuristicNet
        with pytest.raises(ValueError):
            HeuristicNet()

    def test_output_is_scalar_per_sample(self):
        from learned_heuristic import HeuristicNet
        model = HeuristicNet(input_dim=6, hidden_dim=16)
        x = torch.randn(5, 6)
        out = model(x)
        assert out.shape[0] == 5
        assert out.shape[1] == 1


class TestHeuristicLogger:
    def test_record_saves_sample(self):
        from heuristic_logger import HeuristicLogger
        cost_map = {(1, 2): 5.0, (3, 4): 3.0}
        logger = HeuristicLogger(true_cost_map=cost_map)
        features = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        logger.record(features, (1, 2))
        assert len(logger.X_list) == 1
        assert len(logger.y_list) == 1
        assert logger.y_list[0] == pytest.approx(5.0)

    def test_record_ignores_unknown_node(self):
        from heuristic_logger import HeuristicLogger
        logger = HeuristicLogger(true_cost_map={(0, 0): 1.0})
        logger.record(np.array([1.0, 2.0]), (9, 9))
        assert len(logger.X_list) == 0

    def test_record_ignores_without_cost_map(self):
        from heuristic_logger import HeuristicLogger
        logger = HeuristicLogger(true_cost_map=None)
        logger.record(np.array([1.0, 2.0]), (0, 0))
        assert len(logger.X_list) == 0

    def test_save_creates_file(self):
        from heuristic_logger import HeuristicLogger
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_log.npz")
            logger = HeuristicLogger(out_path=path, true_cost_map={(0, 0): 2.0})
            logger.record(np.array([1.0, 2.0, 3.0], dtype=np.float32), (0, 0))
            logger.save()
            assert os.path.exists(path)
            data = np.load(path)
            assert data["X"].shape == (1, 3)
            assert data["y"][0] == pytest.approx(2.0)

    def test_save_appends_existing_file(self):
        from heuristic_logger import HeuristicLogger
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "log.npz")
            cost_map = {(0, 0): 1.0, (1, 1): 2.0}
            logger = HeuristicLogger(out_path=path, true_cost_map=cost_map)
            logger.record(np.array([1.0, 2.0], dtype=np.float32), (0, 0))
            logger.save()
            logger.record(np.array([3.0, 4.0], dtype=np.float32), (1, 1))
            logger.save()
            data = np.load(path)
            assert data["X"].shape[0] == 2
