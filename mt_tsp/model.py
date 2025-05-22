import json
import math
from pathlib import Path
from typing import Any, List, Tuple

import gurobipy as gp
import numpy as np
import tomli
from gurobipy import GRB
from numpy.typing import NDArray


class Target:
    def __init__(
        self,
        name: str,
        p0: Tuple[float, float],
        v: Tuple[float, float],
        t_window: Tuple[float, float],
        radius: float = 0.0,
    ) -> None:
        self.name: str = name
        self.p0: NDArray[np.float64] = np.array(p0, dtype=float)
        self.v: NDArray[np.float64] = np.array(v, dtype=float)
        self.t_window: Tuple[float, float] = t_window
        self.radius: float = radius

    def position(self, t: float) -> NDArray[np.float64]:
        return self.p0 + self.v * t


class MTSPMICP:
    def __init__(
        self,
        targets: List[Target],
        depot: Tuple[float, float] = (0.0, 0.0),
        T: float = 10.0,
        vmax: float = 2.0,
        square_side: float = 10.0,
    ) -> None:
        self.targets: List[Target] = targets
        self.depot: NDArray[np.float64] = np.array(depot, dtype=float)
        self.T: float = T
        self.vmax: float = vmax
        self.square_side: float = square_side
        self.tour: List[int] = []
        self.agent_time_points: List[float] = []
        self.delta_x_list: List[float] = []
        self.delta_y_list: List[float] = []
        self._build_graph()
        self._build_model()

    def _build_graph(self) -> None:
        self.n_targets: int = len(self.targets)
        self.N: int = self.n_targets + 2  # 0=s, 1..n targets, n+1=s'
        self.nodes: List[int] = list(range(self.N))
        self.E: List[Tuple[int, int]] = [
            (i, j) for i in self.nodes for j in self.nodes if i != j
        ]

    def _build_model(self) -> None:
        m: gp.Model = gp.Model("MT-TSP-MICP")

        # decision variables
        self.t: gp.tupledict[int, gp.Var] = m.addVars(
            self.N, lb=0.0, ub=self.T, name="t"
        )
        # TODO: we can set a tighter upperbound for delta_x and delta_y here
        self.delta_x: gp.tupledict[int, gp.Var] = m.addVars(
            self.N, lb=-self.square_side, ub=self.square_side, name="delta_x"
        )
        self.delta_y: gp.tupledict[int, gp.Var] = m.addVars(
            self.N, lb=-self.square_side, ub=self.square_side, name="delta_y"
        )

        self.y_e: gp.tupledict[Tuple[Any, ...], gp.Var] = m.addVars(
            self.E, vtype=GRB.BINARY, name="y_e"
        )
        self.lt: gp.tupledict[Tuple[Any, ...], gp.Var] = m.addVars(
            self.E, lb=0.0, name="l_tilde"
        )
        self.lx: gp.tupledict[Tuple[Any, ...], gp.Var] = m.addVars(
            self.E, lb=-GRB.INFINITY, name="l_x"
        )
        self.ly: gp.tupledict[Tuple[Any, ...], gp.Var] = m.addVars(
            self.E, lb=-GRB.INFINITY, name="l_y"
        )
        self.l: gp.tupledict[Tuple[Any, ...], gp.Var] = m.addVars(
            self.E, lb=0.0, name="l"
        )

        # objectives (1)
        m.setObjective(gp.quicksum(self.lt[e] for e in self.E), GRB.MINIMIZE)

        # flow controls for basic tsp problems (2)-(5)
        s: int = 0
        s_end: int = self.N - 1
        m.addConstr(gp.quicksum(self.y_e[(s, j)] for j in range(1, s_end)) == 1)
        m.addConstr(gp.quicksum(self.y_e[(i, s_end)] for i in range(1, s_end)) == 1)
        for k in range(1, s_end):
            m.addConstr(
                gp.quicksum(self.y_e[(i, k)] for i in self.nodes if i != k) == 1
            )
            m.addConstr(
                gp.quicksum(self.y_e[(k, j)] for j in self.nodes if j != k) == 1
            )

        # time windows restrictions (6)
        for idx in range(1, self.N - 1):
            tmin, tmax = self.targets[idx - 1].t_window
            m.addConstr(self.t[idx] >= tmin)
            m.addConstr(self.t[idx] <= tmax)

        # square area uppperbound
        R: float = math.sqrt(self.square_side**2 * 2)

        # time feasibility
        m.addConstr(self.t[0] == 0.0)  # initial time t_s = 0
        m.addConstr(self.t[self.N - 1] >= 0)
        m.addConstr(self.t[self.N - 1] <= self.T)  # end time t_s' <= T

        # add constraints for almost tsp
        for i in range(self.N):
            r_i = 0.0 if i in (s, s_end) else self.targets[i - 1].radius
            m.addQConstr(self.delta_x[i] ** 2 + self.delta_y[i] ** 2 <= r_i**2)

        # node relationship
        for i, j in self.E:
            if i in (s, s_end):
                p_i, t_i0 = self.depot, 0.0
                v_i = np.array([0.0, 0.0], dtype=float)
            else:
                tgt_i = self.targets[i - 1]
                p_i, t_i0 = tgt_i.p0, tgt_i.t_window[0]
                v_i = tgt_i.v

            if j in (s, s_end):
                p_j, t_j0 = self.depot, 0.0
                v_j = np.array([0.0, 0.0], dtype=float)
            else:
                tgt_j = self.targets[j - 1]
                p_j, t_j0 = tgt_j.p0, tgt_j.t_window[0]
                v_j = tgt_j.v

            # position definition (7)-(8)
            m.addConstr(
                self.lx[(i, j)]
                == (p_j[0] + self.delta_x[j] + v_j[0] * self.t[j] - v_j[0] * t_j0)
                - (p_i[0] + self.delta_x[i] + v_i[0] * self.t[i] - v_i[0] * t_i0)
            )

            m.addConstr(
                self.ly[(i, j)]
                == (p_j[1] + self.delta_y[j] + v_j[1] * self.t[j] - v_j[1] * t_j0)
                - (p_i[1] + self.delta_y[i] + v_i[1] * self.t[i] - v_i[1] * t_i0)
            )

            # time feasibility
            m.addConstr(
                self.lt[(i, j)]
                <= self.vmax * (self.t[j] - self.t[i] + self.T * (1 - self.y_e[(i, j)]))
            )
            m.addConstr(
                self.t[j]
                >= self.t[i] + self.lt[(i, j)] - self.T * (1 - self.y_e[(i, j)])
            )

            m.addConstr(self.l[(i, j)] == self.lt[(i, j)] + R * (1 - self.y_e[(i, j)]))

            # second conic constraints
            m.addQConstr(
                self.lx[(i, j)] ** 2 + self.ly[(i, j)] ** 2 <= self.l[(i, j)] ** 2
            )

        self.model = m

    def solve(self) -> List[int]:
        # resove model
        self.model.optimize()
        self.model.write("model.lp")
        if self.model.status == GRB.INFEASIBLE:
            self.model.computeIIS()
            self.model.write("infeasible.ilp")
        print("Status:", self.model.status, " SolCount:", self.model.SolCount)
        assert self.model.SolCount > 0, "There is no feasible solutions!"

        # return the idx list of the solutions
        tour: List[int] = [0]
        current: int = 0
        while current != self.N - 1:
            for i, j in self.E:
                if i == current and self.y_e[(i, j)].Xn > 0.5:
                    tour.append(j)
                    current = j
                    break
        self.tour = tour
        self.agent_time_points = [self.t[i].Xn for i in tour]
        self.delta_x_list = [self.delta_x[i].Xn for i in tour]
        self.delta_y_list = [self.delta_y[i].Xn for i in tour]
        assert len(tour) > 0, "There is no feasible solutions, no output for tour!"
        assert len(tour) == len(self.agent_time_points), "Time point mismatch!"
        assert len(tour) == len(self.targets), "Not all targets are in the solution!"
        print(f"Agent time points: {self.agent_time_points}")
        print(f"delta_x: {self.delta_x_list}")
        print(f"delta_y: {self.delta_y_list}")
        return tour


def load_config(target_path: Path, agent_path: Path) -> MTSPMICP:
    with open(target_path, encoding="utf-8") as f:
        target_data = json.load(f)

    with open(agent_path, "rb") as g:
        agent_data = tomli.load(g)

    targets: List[Target] = []
    for key, data in target_data.items():
        targets.append(
            Target(
                name=key,
                p0=(data["px"], data["py"]),
                v=(data["vx"], data["vy"]),
                t_window=(data["tmin"], data["tmax"]),
                radius=data["radius"],
            )
        )
    return MTSPMICP(
        targets=targets,
        depot=(agent_data["depot"]["px"], agent_data["depot"]["py"]),
        T=agent_data["param"]["T"],
        vmax=agent_data["param"]["vmax"],
        square_side=agent_data["param"]["R"],
    )
