import os

import pytest

from mt_tsp.model import MTSPMICP, Target
from mt_tsp.visualization import Visualizer


class TestVisualizer:
    @pytest.fixture
    def sample_model(self, tmp_path):
        targets = [
            Target("A", (0, 10), (0, 0), (0, 1000)),
            Target("B", (10, 0), (0, 0), (0, 1000)),
        ]
        model = MTSPMICP(
            targets=targets, depot=(0, 0), T=1000.0, vmax=2.0, square_side=20.0
        )
        model.tour = [0, 1, 2, 3]
        model.agent_time_points = [0.0, 5.0, 8.0, 10.0]
        return model

    def test_animation_generation(self, sample_model, tmp_path):
        output_path = tmp_path / "output.gif"
        viz = Visualizer(model=sample_model, gif_path=str(output_path), frames=10)
        viz.animate()

        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 1024

    def test_agent_position_calculation(self, sample_model):
        viz = Visualizer(sample_model)

        assert viz._get_agent_pos(0.0) == (0.0, 0.0)

        x, y = viz._get_agent_pos(2.5)
        assert 0.0 < x < 5.0
        assert 0.0 < y < 10.0

        assert viz._get_agent_pos(15.0) == (10.0, 0.0)
