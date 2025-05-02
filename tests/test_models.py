import numpy as np
import pytest

from mt_tsp.model import MTSPMICP, Target


class TestMTSPMICP:
    @pytest.fixture
    def static_targets(self):
        return [
            Target("A", (0, 0), (0, 0), (0, 10)),
            Target("B", (5, 5), (0, 0), (0, 10)),
        ]

    @pytest.fixture
    def moving_targets(self):
        return [
            Target("C", (0, 0), (1, 0), (0, 10)),
            Target("D", (10, 10), (0, -1), (0, 10)),
        ]

    def test_model_init(self, static_targets):
        model = MTSPMICP(
            targets=static_targets, depot=(0, 0), T=10.0, vmax=2.0, square_side=10.0
        )

        assert model.n_targets == 2
        assert model.depot.tolist() == [0, 0]
        assert model.vmax == 2.0

    def test_solve_static_case(self, static_targets):
        model = MTSPMICP(
            targets=static_targets, depot=(0, 0), T=100.0, vmax=2.0, square_side=10.0
        )
        tour = model.solve()
        assert tour == [0, 1, 2, 3]

        assert model.t[1].Xn >= 0
        assert model.t[2].Xn >= model.t[1].Xn

    @pytest.mark.skip(reason="Need Gurobi license.")
    def test_moving_targets(self, moving_targets):
        model = MTSPMICP(
            targets=moving_targets, depot=(5, 5), T=20.0, vmax=3.0, square_side=20.0
        )
        tour = model.solve()

        assert len(tour) == 4  # 0 -> 1 -> 2 -> 3
        assert model.t[3].Xn <= 20.0
