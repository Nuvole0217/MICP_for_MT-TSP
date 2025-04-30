import math
from typing import Any, List, Tuple

import gurobipy as gp
import numpy as np
from gurobipy import GRB
from numpy.typing import NDArray


class Target:
    def __init__(
        self,
        name: str,
        p0: Tuple[float, float],
        v: Tuple[float, float],
        t_window: Tuple[float, float],
    ) -> None:
        self.name: str = name
        self.p0: NDArray[np.float64] = np.array(p0, dtype=float)
        self.v: NDArray[np.float64] = np.array(v, dtype=float)
        self.t_window: Tuple[float, float] = t_window

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
        self.y_e: gp.tupledict[Tuple[Any, ...], gp.Var] = m.addVars(
            self.E, vtype=GRB.BINARY, name="y_e"
        )
        self.lt: gp.tupledict[Tuple[Any, ...], gp.Var] = m.addVars(
            self.E, lb=0.0, name="l_tilde"
        )
        self.lx: gp.tupledict[Tuple[Any, ...], gp.Var] = m.addVars(self.E, name="l_x")
        self.ly: gp.tupledict[Tuple[Any, ...], gp.Var] = m.addVars(self.E, name="l_y")
        self.l: gp.tupledict[Tuple[Any, ...], gp.Var] = m.addVars(
            self.E, lb=0.0, name="l"
        )

        # objectives (1)
        m.setObjective(gp.quicksum(self.lt[e] for e in self.E), GRB.MINIMIZE)

        # flow controls for basic tsp problems (2)-(5)
        s: int = 0
        s_end: int = self.N - 1
        m.addConstr(gp.quicksum(self.y_e[(s, j)] for j in range(1, self.N)) == 1)
        m.addConstr(gp.quicksum(self.y_e[(i, s_end)] for i in range(self.N - 1)) == 1)
        for k in range(1, self.N - 1):
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
        R: float = math.sqrt(self.square_side * self.square_side * 2)

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
                == (p_j[0] + v_j[0] * self.t[j] - v_j[0] * t_j0)
                - (p_i[0] + v_i[0] * self.t[i] - v_i[0] * t_i0)
            )

            m.addConstr(
                self.ly[(i, j)]
                == (p_j[1] + v_j[1] * self.t[j] - v_j[1] * t_j0)
                - (p_i[1] + v_i[1] * self.t[i] - v_i[1] * t_i0)
            )

            # time feasibility
            m.addConstr(
                self.lt[(i, j)]
                <= self.vmax * (self.t[j] - self.t[i] + self.T * (1 - self.y_e[(i, j)]))
            )

            m.addConstr(self.t[j] - self.t[i] + self.T * (1 - self.y_e[(i, j)]) >= 0)
            m.addConstr(self.t[0] == 0.0)  # initial time t_s = 0
            m.addConstr(self.t[self.N - 1] <= self.T)  # end time t_s' <= T
            m.addConstr(self.t[j] - self.t[i] + self.T * (1 - self.y_e[(i, j)]) >= 0)

            m.addConstr(self.l[(i, j)] == self.lt[(i, j)] + R * (1 - self.y_e[(i, j)]))

            # second conic constraints
            m.addQConstr(
                self.lx[(i, j)] * self.lx[(i, j)] + self.ly[(i, j)] * self.ly[(i, j)]
                <= self.l[(i, j)] * self.l[(i, j)]
            )

            # update time
            m.addConstr(
                self.t[j]
                >= self.t[i]
                + self.lt[(i, j)]  # travel time = l_tilde
                - self.T * (1 - self.y_e[(i, j)])
            )

        self.model = m

    def solve(self) -> List[int]:
        # resove model
        self.model.optimize()
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
        return tour
