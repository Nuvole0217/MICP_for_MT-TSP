import math
from pathlib import Path
from typing import Any, List, Tuple

import gurobipy as gp
import pytest

from mt_tsp.model import MTSPMICP, load_config


def load_case(case_name: str) -> MTSPMICP:
    root = Path(__file__).resolve().parent
    target_file = root / case_name / "targets.json"
    agent_file = root / case_name / "agent.toml"
    return load_config(target_file, agent_file)


@pytest.mark.parametrize("case_name", ["simple", "complex"])
def test_mtsp_solution(case_name: str) -> None:
    model = load_case(case_name)
    tour: List[int] = model.solve()

    # check solution counts
    print("Check solution counts: \n")
    assert (
        len(tour) == len(model.targets) + 2
    ), "Tour must include depot, all targets once, and return to depot."
    assert (
        len(model.agent_time_points) == len(model.targets) + 2
    ), "Agent time points must include depot, all targets once, and return to depot."

    s, s_end = 0, model.N - 1
    start_time = model.t[0].Xn
    end_time = model.t[s_end].Xn
    assert pytest.approx(0.0) == start_time, "Start time must be zero."
    assert 0.0 <= end_time <= model.T, "End time must be within the horizon."

    # check tsp-basics
    print("Check TSP basics: \n")
    y_e: gp.tupledict[Tuple[Any, ...], gp.Var] = model.y_e
    out_deg = sum(y_e[(s, j)].Xn for j in range(1, s_end))
    assert out_deg == 1, "Exactly one edge must leave the depot."
    in_deg = sum(y_e[(i, s_end)].Xn for i in range(1, s_end))
    assert in_deg == 1, "Exactly one edge must enter the end depot."
    for k in range(1, s_end):
        indeg = sum(y_e[(i, k)].Xn for i in model.nodes if i != k)
        outdeg = sum(y_e[(k, j)].Xn for j in model.nodes if j != k)
        assert indeg == 1, f"Node {k} must have exactly one incoming edge."
        assert outdeg == 1, f"Node {k} must have exactly one outgoing edge."

    # check time feasibility
    print("Check time feasibility: \n")
    for node_idx in range(1, s_end):
        tmin, tmax = model.targets[node_idx - 1].t_window
        visit_time = model.t[node_idx].Xn
        assert (
            tmin <= visit_time <= tmax
        ), f"Visit time for node {node_idx} must lie within its window."

    for i, j in model.E:
        chosen = y_e[(i, j)].Xn > 0.5
        if chosen:
            t_i = model.t[i].Xn
            t_j = model.t[j].Xn
            travel_time = model.lt[(i, j)].Xn
            assert (
                t_j + 1e-6 >= t_i + travel_time
            ), f"Time progression violated on edge {(i, j)}."

    # check auxiliary bindings and second conic constraints
    print("Check auxiliary bindings and second conic constraints: \n")
    R = math.sqrt(2 * (model.square_side**2))
    for i, j in model.E:
        yi = y_e[(i, j)].Xn
        lt_val = model.lt[(i, j)].Xn
        assert (
            lt_val
            <= model.vmax * (model.t[j].Xn - model.t[i].Xn + model.T * (1 - yi)) + 1e-6
        ), f"Time feasibility violated on edge {(i, j)}."
        l_val = model.l[(i, j)].Xn
        expected_l = lt_val + R * (1 - yi)
        assert (
            pytest.approx(expected_l, rel=1e-6) == l_val
        ), f"Auxiliary binding violated on edge {(i, j)}."
        # Second-order conic constraints
        lx = model.lx[(i, j)].Xn
        ly = model.ly[(i, j)].Xn
        assert (
            lx * lx + ly * ly <= l_val * l_val + 1e-6
        ), f"SOC constraint violated on edge {(i, j)}."
