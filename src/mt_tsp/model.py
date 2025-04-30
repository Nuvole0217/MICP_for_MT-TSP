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
    ) -> None:
        self.targets: List[Target] = targets
        self.depot: NDArray[np.float64] = np.array(depot, dtype=float)
        self.T: float = T
        self.vmax: float = vmax
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
        self.y_e: gp.tupledict[Any, gp.Var] = m.addVars(
            self.E, vtype=GRB.BINARY, name="y"
        )
        self.lt: gp.tupledict[Any, gp.Var] = m.addVars(self.E, lb=0.0, name="l_tilde")
        self.lx: gp.tupledict[Any, gp.Var] = m.addVars(self.E, name="l_x")
        self.ly: gp.tupledict[Any, gp.Var] = m.addVars(self.E, name="l_y")
        self.l: gp.tupledict[Any, gp.Var] = m.addVars(self.E, lb=0.0, name="l")

        # objectives
        m.setObjective(gp.quicksum(self.lt[e] for e in self.E), GRB.MINIMIZE)

        # flow controls
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

        # time windows restrictions
        for idx in range(1, self.N - 1):
            tmin, tmax = self.targets[idx - 1].t_window
            m.addConstr(self.t[idx] >= tmin)
            m.addConstr(self.t[idx] <= tmax)

        # tsp definition
        R: np.floating[Any] | np.float64 = (
            np.linalg.norm(self.depot - np.array([10.0, 10.0])) * 2.0
        )

        # node relationship
        for i, j in self.E:
            if i in (s, s_end):
                p_i, v_i, t_i0 = self.depot, np.zeros(2), 0.0
            else:
                tgt_i = self.targets[i - 1]
                p_i, v_i, t_i0 = tgt_i.p0, tgt_i.v, tgt_i.t_window[0]
            if j in (s, s_end):
                p_j, v_j, t_j0 = self.depot, np.zeros(2), 0.0
            else:
                tgt_j = self.targets[j - 1]
                p_j, v_j, t_j0 = tgt_j.p0, tgt_j.v, tgt_j.t_window[0]

            # position definition
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

            m.addConstr(self.l[(i, j)] == self.lt[(i, j)] + R * (1 - self.y_e[(i, j)]))

            # second conic constraints
            m.addQConstr(
                self.lx[(i, j)] * self.lx[(i, j)] + self.ly[(i, j)] * self.ly[(i, j)]
                <= self.l[(i, j)] * self.l[(i, j)]
            )

        self.model = m

    def solve(self) -> List[int]:
        # resove model, return the idx of the nodes
        self.model.optimize()
        tour: List[int] = [0]
        current: int = 0
        while current != self.N - 1:
            for i, j in self.E:
                if i == current and self.y_e[(i, j)].X > 0.5:
                    tour.append(j)
                    current = j
                    break
        return tour
