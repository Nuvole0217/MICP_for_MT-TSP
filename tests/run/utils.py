from pathlib import Path
from typing import List, Tuple, Any
import pytest
import gurobipy as gp
import math
from mt_tsp.model import load_config, MTSPMICP


def load_cases(case_name: str) ->MTSPMICP:
    root = Path(__file__).resolve().parent
    target_file = root / case_name / "target.json"
    agent_file = root / case_name / "agent.toml"
    model = load_config(target_file, agent_file)
    return model

def model_testing(model: MTSPMICP) -> None:
    tour: List[int] = model.solve()
    @pytest.fixture
    def node_num() -> None:
        # test solution counts
        assert len(tour) == len(model.targets) + 2, "fail tour count solution."
        assert len(model.agent_time_points) == len(model.targets) + 2, "fail time count solution."
    
    @pytest.fixture
    def test_tsp_basic() -> None:
        y_e: gp.tupledict[Tuple[Any, ...], gp.Var] = model.y_e
        s: int = 0
        s_end: int = model.N - 1
        assert gp.quicksum(y_e[(s, j)] for j in range(1, s_end)) == 1
        assert gp.quicksum(y_e[(i, s_end)] for i in range(1, s_end)) == 1
        for k in range(1, s_end):
            assert (
                gp.quicksum(y_e[(i, k)] for i in model.nodes if i != k) == 1
            )
            assert (
                gp.quicksum(y_e[(k, j)] for j in model.nodes if j != k) == 1
            )
        return
    
    @pytest.fixture
    def test_time_feasibility() -> None:
        agent_time_points: List[float] = model.agent_time_points
        for idx in range(1, model.N - 1):
            t_min, t_max = model.targets[idx - 1].t_window
            for idx_, obj in enumerate(tour):
                if obj == idx:
                    assert agent_time_points[idx_] >= t_min, "fail time feasibility."
                    assert agent_time_points[idx_] <= t_max, "fail time feasibility."
        assert model.t[0] == 0
        assert model.t[model.N - 1] <= model.T and model.t[model.N - 1] >=0
        return
    
    @pytest.fixture
    def test_position_relationship() -> None:
        for i, j in model.E:
            assert (
                model.lt[(i, j)]
                <= model.vmax * (model.t[j] - model.t[i] + model.T * (1 - model.y_e[(i, j)]))
            )
            # boundary set
            assert(
                model.l[(i, j)] == model.lt[(i, j)] + math.sqrt(model.square_side**2 * 2) * (1 - model.y_e[(i, j)])
            )
            # second conic constraints
            assert(
                model.lx[(i, j)] ** 2 + model.ly[(i, j)] ** 2 <= model.l[(i, j)] ** 2
            )
