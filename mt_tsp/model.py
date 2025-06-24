import math
from typing import Any, Dict, List, Tuple

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
        max_time: float = 10.0,
        vmax: float = 2.0,
        square_side: float = 10.0,
    ) -> None:
        self.targets: List[Target] = targets
        self.depot: NDArray[np.float64] = np.array(depot, dtype=float)
        self.max_time: float = max_time
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
        self.nodes: int = self.n_targets + 2  # 0=s, 1..n targets, n+1=s'
        self.nodes: List[int] = list(range(self.nodes))
        self.edges: List[Tuple[int, int]] = [
            (i, j) for i in self.nodes for j in self.nodes if i != j
        ]

    def _build_model(self) -> None:
        m: gp.Model = gp.Model("MT-TSP-MICP")

        # decision variables
        self.t: gp.tupledict[int, gp.Var] = m.addVars(
            self.nodes, lb=0.0, ub=self.max_time, name="t"
        )
        # TODO: we can set a tighter upperbound for delta_x and delta_y here
        self.delta_x: gp.tupledict[int, gp.Var] = m.addVars(
            self.nodes, lb=-self.square_side, ub=self.square_side, name="delta_x"
        )
        self.delta_y: gp.tupledict[int, gp.Var] = m.addVars(
            self.nodes, lb=-self.square_side, ub=self.square_side, name="delta_y"
        )

        self.y_e: gp.tupledict[Tuple[Any, ...], gp.Var] = m.addVars(
            self.edges, vtype=GRB.BINARY, name="y_e"
        )
        self.lt: gp.tupledict[Tuple[Any, ...], gp.Var] = m.addVars(
            self.edges, lb=0.0, name="l_tilde"
        )
        self.lx: gp.tupledict[Tuple[Any, ...], gp.Var] = m.addVars(
            self.edges, lb=-GRB.INFINITY, name="l_x"
        )
        self.ly: gp.tupledict[Tuple[Any, ...], gp.Var] = m.addVars(
            self.edges, lb=-GRB.INFINITY, name="l_y"
        )
        self.l: gp.tupledict[Tuple[Any, ...], gp.Var] = m.addVars(
            self.edges, lb=0.0, name="l"
        )

        # objectives (1)
        m.setObjective(gp.quicksum(self.lt[e] for e in self.edges), GRB.MINIMIZE)

        # flow controls for basic tsp problems (2)-(5)
        s: int = 0
        s_end: int = self.nodes - 1
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
        for idx in range(1, self.nodes - 1):
            tmin, tmax = self.targets[idx - 1].t_window
            m.addConstr(self.t[idx] >= tmin)
            m.addConstr(self.t[idx] <= tmax)

        # square area uppperbound
        R: float = math.sqrt(self.square_side**2 * 2)

        # time feasibility
        m.addConstr(self.t[0] == 0.0)  # initial time t_s = 0
        m.addConstr(self.t[self.nodes - 1] >= 0)
        m.addConstr(self.t[self.nodes - 1] <= self.max_time)  # end time t_s' <= T

        # add constraints for almost tsp
        for i in range(self.nodes):
            r_i = 0.0 if i in (s, s_end) else self.targets[i - 1].radius
            m.addQConstr(self.delta_x[i] ** 2 + self.delta_y[i] ** 2 <= r_i**2)

        # node relationship
        for i, j in self.edges:
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
                <= self.vmax
                * (self.t[j] - self.t[i] + self.max_time * (1 - self.y_e[(i, j)]))
            )
            m.addConstr(
                self.t[j]
                >= self.t[i] + self.lt[(i, j)] - self.max_time * (1 - self.y_e[(i, j)])
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
        while current != self.nodes - 1:
            for i, j in self.edges:
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
        assert (
            len(tour) == len(self.targets) + 2
        ), "Not all targets are in the solution!"
        print(f"Agent time points: {self.agent_time_points}")
        print(f"delta_x: {self.delta_x_list}")
        print(f"delta_y: {self.delta_y_list}")
        runtime = self.model.Runtime
        print(f"Runtime: {runtime:.2f} seconds")
        return tour


class MTSPMICPGCS:
    def __init__(
        self,
        targets: List[Target],
        depot: Tuple[float, float] = (0.0, 0.0),
        max_time: float = 150.0,
        vmax: float = 8.0,
    ) -> None:
        self.targets: List[Target] = targets
        self.depot: Tuple[float, float] = depot
        self.max_time: float = max_time
        self.vmax: float = vmax

        # save the works
        self.tour: List[int] = []
        self.agent_time_points: List[float] = []
        self.agent_positions: List[Tuple[float, float]] = []

        # inner mapping, easier for gorubi solver
        self.name_to_idx: Dict[str, int] = {}
        self.idx_to_name: Dict[int, str] = {}

        self._build_graph()
        self._build_model()

    def _build_graph(self) -> None:
        """
        build the graph (edge, nodes) and preprocessing the node data.
        """
        self.n_targets: int = len(self.targets)
        self.nodes: int = self.n_targets + 2

        self.s_node_idx: int = 0
        self.s_prime_node_idx: int = self.nodes - 1

        # build index to name
        self.idx_to_name = {self.s_node_idx: "Depot", self.s_prime_node_idx: "Depot"}
        self.V_tar_indices: List[int] = []
        for i, tgt in enumerate(self.targets):
            node_idx = i + 1
            self.name_to_idx[tgt.name] = node_idx
            self.idx_to_name[node_idx] = tgt.name
            self.V_tar_indices.append(node_idx)

        self.edges: List[Tuple[int, int]] = []
        self.edges.extend([(self.s_node_idx, j) for j in self.V_tar_indices])
        self.edges.extend(
            [(i, j) for i in self.V_tar_indices for j in self.V_tar_indices if i != j]
        )
        self.edges.extend([(i, self.s_prime_node_idx) for i in self.V_tar_indices])

        # preprocess node data
        self.node_data: Dict[int, Dict[str, Any]] = {}
        for tgt in self.targets:
            node_idx = self.name_to_idx[tgt.name]
            t_underline = tgt.t_window[0]
            # calculate the time at the begining of the time windows p_underline
            # GCS model takes p(t) = p_underline + v * (t - t_underline)
            p_underline = tgt.p0 + tgt.v * t_underline

            self.node_data[node_idx] = {
                "p_underline": p_underline,
                "v": tgt.v,
                "t_underline": t_underline,
                "t_bar": tgt.t_window[1],
                "radius": tgt.radius,
            }

        self.node_data[self.s_node_idx] = {
            "p_underline": self.depot,
            "v": np.array([0.0, 0.0]),
            "t_underline": 0.0,
            "t_bar": 0.0,
            "radius": 0.0,
        }
        self.node_data[self.s_prime_node_idx] = {
            "p_underline": self.depot,
            "v": np.array([0.0, 0.0]),
            "t_underline": 0.0,
            "t_bar": self.max_time,
            "radius": 0.0,
        }

    def _build_model(self) -> None:
        m = gp.Model("MT-TSP-MICP-GCS")

        self.y_e = m.addVars(self.edges, vtype=GRB.BINARY, name="y_e")
        self.z_x = m.addVars(
            self.edges, vtype=GRB.CONTINUOUS, name="zx", lb=-GRB.INFINITY
        )
        self.z_y = m.addVars(
            self.edges, vtype=GRB.CONTINUOUS, name="zy", lb=-GRB.INFINITY
        )
        self.z_t = m.addVars(self.edges, vtype=GRB.CONTINUOUS, name="zt")
        self.z_prime_x = m.addVars(
            self.edges, vtype=GRB.CONTINUOUS, name="z_prime_x", lb=-GRB.INFINITY
        )
        self.z_prime_y = m.addVars(
            self.edges, vtype=GRB.CONTINUOUS, name="z_prime_y", lb=-GRB.INFINITY
        )
        self.z_prime_t = m.addVars(self.edges, vtype=GRB.CONTINUOUS, name="z_prime_t")
        self.l = m.addVars(self.edges, vtype=GRB.CONTINUOUS, lb=0.0, name="l")
        self.l_x = m.addVars(
            self.edges, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="lx"
        )
        self.l_y = m.addVars(
            self.edges, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="ly"
        )
        self.delta_x = m.addVars(
            self.nodes,
            vtype=GRB.CONTINUOUS,
            lb=-GRB.INFINITY,
            ub=GRB.INFINITY,
            name="delta_x",
        )
        self.delta_y = m.addVars(
            self.nodes,
            vtype=GRB.CONTINUOUS,
            lb=-GRB.INFINITY,
            ub=GRB.INFINITY,
            name="delta_y",
        )
        self.delta_x_pos = m.addVars(
            self.nodes, vtype=GRB.CONTINUOUS, lb=0.0, ub=GRB.INFINITY, name="x_pos"
        )
        self.delta_x_neg = m.addVars(
            self.nodes, vtype=GRB.CONTINUOUS, lb=0.0, ub=GRB.INFINITY, name="x_neg"
        )
        self.delta_y_pos = m.addVars(
            self.nodes, vtype=GRB.CONTINUOUS, lb=0.0, ub=GRB.INFINITY, name="y_pos"
        )
        self.delta_y_neg = m.addVars(
            self.nodes, vtype=GRB.CONTINUOUS, lb=0.0, ub=GRB.INFINITY, name="y_neg"
        )
        self.w_x_pos = m.addVars(
            self.edges,
            vtype=GRB.CONTINUOUS,
            lb=0.0,
            ub=GRB.INFINITY,
            name="w_x_pos",
        )
        self.w_y_pos = m.addVars(
            self.edges,
            vtype=GRB.CONTINUOUS,
            lb=0.0,
            ub=GRB.INFINITY,
            name="w_y_pos",
        )
        self.w_x_neg = m.addVars(
            self.edges,
            vtype=GRB.CONTINUOUS,
            lb=0.0,
            ub=GRB.INFINITY,
            name="w_x_neg",
        )
        self.w_y_neg = m.addVars(
            self.edges, vtype=GRB.CONTINUOUS, lb=0.0, ub=GRB.INFINITY, name="w_y_neg"
        )
        self.w_prime_x_pos = m.addVars(
            self.edges,
            vtype=GRB.CONTINUOUS,
            lb=0.0,
            ub=GRB.INFINITY,
            name="w_prime_x",
        )
        self.w_prime_x_neg = m.addVars(
            self.edges,
            vtype=GRB.CONTINUOUS,
            lb=0.0,
            ub=GRB.INFINITY,
            name="w_prime_x_neg",
        )
        self.w_prime_y_pos = m.addVars(
            self.edges,
            vtype=GRB.CONTINUOUS,
            lb=0.0,
            ub=GRB.INFINITY,
            name="w_prime_y_pos",
        )
        self.w_prime_y_neg = m.addVars(
            self.edges,
            vtype=GRB.CONTINUOUS,
            lb=0.0,
            ub=GRB.INFINITY,
            name="w_prime_y_neg",
        )

        # take objective functions
        m.setObjective(gp.quicksum(self.l[i, j] for i, j in self.edges), GRB.MINIMIZE)

        # take constraints
        m.addConstr(self.y_e.sum(self.s_node_idx, "*") == 1)
        m.addConstr(self.y_e.sum("*", self.s_prime_node_idx) == 1)
        for i in self.V_tar_indices:
            m.addConstr(self.y_e.sum("*", i) == 1)
            m.addConstr(self.y_e.sum("*", i) == self.y_e.sum(i, "*"))
            m.addConstr(
                gp.quicksum(self.z_prime_x[k, i] for k, j in self.edges if j == i)
                == gp.quicksum(self.z_x[i, j] for k, j in self.edges if k == i)
            )
            m.addConstr(
                gp.quicksum(self.z_prime_y[k, i] for k, j in self.edges if j == i)
                == gp.quicksum(self.z_y[i, j] for k, j in self.edges if k == i)
            )
            m.addConstr(
                gp.quicksum(self.z_prime_t[k, i] for k, j in self.edges if j == i)
                == gp.quicksum(self.z_t[i, j] for k, j in self.edges if k == i)
            )
            m.addConstr(
                self.delta_x[i] ** 2 + self.delta_y[i] ** 2
                <= self.node_data[i]["radius"] ** 2
            )
            m.addConstr(self.delta_x_pos[i] <= self.node_data[i]["radius"])
            m.addConstr(self.delta_x_neg[i] <= self.node_data[i]["radius"])
            m.addConstr(self.delta_y_pos[i] <= self.node_data[i]["radius"])
            m.addConstr(self.delta_y_neg[i] <= self.node_data[i]["radius"])

        for i, j in self.edges:
            m.addConstr(self.l_x[i, j] == self.z_prime_x[i, j] - self.z_x[i, j])
            m.addConstr(self.l_y[i, j] == self.z_prime_y[i, j] - self.z_y[i, j])
            m.addConstr(
                self.l[i, j] <= self.vmax * (self.z_prime_t[i, j] - self.z_t[i, j])
            )
            m.addConstr(self.l_x[i, j] ** 2 + self.l_y[i, j] ** 2 <= self.l[i, j] ** 2)

            data_i = self.node_data[i]
            data_j = self.node_data[j]
            m.addConstr(self.z_t[i, j] >= data_i["t_underline"] * self.y_e[i, j])
            m.addConstr(self.z_t[i, j] <= data_i["t_bar"] * self.y_e[i, j])
            m.addConstr(self.z_prime_t[i, j] >= data_j["t_underline"] * self.y_e[i, j])
            m.addConstr(self.z_prime_t[i, j] <= data_j["t_bar"] * self.y_e[i, j])

            m.addConstr(self.delta_x_pos[i] <= data_i["radius"] * self.y_e[i, j])
            m.addConstr(self.delta_x_neg[i] <= data_i["radius"] * self.y_e[i, j])
            m.addConstr(self.delta_y_pos[i] <= data_i["radius"] * self.y_e[i, j])
            m.addConstr(self.delta_y_neg[j] <= data_j["radius"] * self.y_e[i, j])

            const_ix = data_i["p_underline"][0] - data_i["t_underline"] * data_i["v"][0]
            const_iy = data_i["p_underline"][1] - data_i["t_underline"] * data_i["v"][1]
            m.addConstr(
                self.z_x[i, j]
                - data_i["v"][0] * self.z_t[i, j]
                - (self.w_x_pos[i, j] - self.w_x_neg[i, j])
                == const_ix * self.y_e[i, j]
            )
            m.addConstr(
                self.z_y[i, j]
                - data_i["v"][1] * self.z_t[i, j]
                - (self.w_y_pos[i, j] - self.w_y_neg[i, j])
                == const_iy * self.y_e[i, j]
            )

            const_jx = data_j["p_underline"][0] - data_j["t_underline"] * data_j["v"][0]
            const_jy = data_j["p_underline"][1] - data_j["t_underline"] * data_j["v"][1]
            m.addConstr(
                self.z_prime_x[i, j]
                - data_j["v"][0] * self.z_prime_t[i, j]
                - (self.w_prime_x_pos[i, j] - self.w_prime_x_neg[i, j])
                == const_jx * self.y_e[i, j]
            )
            m.addConstr(
                self.z_prime_y[i, j]
                - data_j["v"][1] * self.z_prime_t[i, j]
                - (self.w_prime_y_pos[i, j] - self.w_prime_y_neg[i, j])
                == const_jy * self.y_e[i, j]
            )

            # add constraints for delta_x and delta_y perspective
            radius_i = data_i["radius"]
            m.addConstr(self.w_x_pos[i, j] <= radius_i * self.y_e[i, j])
            m.addConstr(self.w_x_neg[i, j] <= radius_i * self.y_e[i, j])
            m.addConstr(self.w_y_pos[i, j] <= radius_i * self.y_e[i, j])
            m.addConstr(self.w_y_neg[i, j] <= radius_i * self.y_e[i, j])
            m.addConstr(self.w_x_pos[i, j] <= self.delta_x_pos[i])
            m.addConstr(self.w_x_neg[i, j] <= self.delta_x_neg[i])
            m.addConstr(self.w_y_pos[i, j] <= self.delta_y_pos[i])
            m.addConstr(self.w_y_neg[i, j] <= self.delta_y_neg[j])
            m.addConstr(
                self.w_x_pos[i, j]
                >= self.delta_x_pos[i] - radius_i * (1 - self.y_e[i, j])
            )
            m.addConstr(
                self.w_x_neg[i, j]
                >= self.delta_x_neg[i] - radius_i * (1 - self.y_e[i, j])
            )
            m.addConstr(
                self.w_y_pos[i, j]
                >= self.delta_y_pos[i] - radius_i * (1 - self.y_e[i, j])
            )
            m.addConstr(
                self.w_y_neg[i, j]
                >= self.delta_y_neg[j] - radius_i * (1 - self.y_e[i, j])
            )

            radius_j = data_j["radius"]
            m.addConstr(self.w_prime_x_pos[i, j] <= radius_j * self.y_e[i, j])
            m.addConstr(self.w_prime_x_neg[i, j] <= radius_j * self.y_e[i, j])
            m.addConstr(self.w_prime_y_pos[i, j] <= radius_j * self.y_e[i, j])
            m.addConstr(self.w_prime_y_neg[i, j] <= radius_j * self.y_e[i, j])
            m.addConstr(self.w_prime_x_pos[i, j] <= self.delta_x_pos[j])
            m.addConstr(self.w_prime_x_neg[i, j] <= self.delta_x_neg[j])
            m.addConstr(self.w_prime_y_pos[i, j] <= self.delta_y_pos[j])
            m.addConstr(self.w_prime_y_neg[i, j] <= self.delta_y_neg[j])
            m.addConstr(
                self.w_prime_x_pos[i, j]
                >= self.delta_x_pos[j] - radius_j * (1 - self.y_e[i, j])
            )
            m.addConstr(
                self.w_prime_x_neg[i, j]
                >= self.delta_x_neg[j] - radius_j * (1 - self.y_e[i, j])
            )
            m.addConstr(
                self.w_prime_y_pos[i, j]
                >= self.delta_y_pos[j] - radius_j * (1 - self.y_e[i, j])
            )
            m.addConstr(
                self.w_prime_y_neg[i, j]
                >= self.delta_y_neg[j] - radius_j * (1 - self.y_e[i, j])
            )
        self.model = m

    def solve(self) -> List[int]:
        self.model.setParam("MIPGap", 0.1)
        self.model.optimize()

        if self.model.status == GRB.OPTIMAL or self.model.status == GRB.SUBOPTIMAL:
            print(f"{self.model.status}, total distance: {self.model.ObjVal:.2f}")

            tour_indices = [self.s_node_idx]
            current_idx = self.s_node_idx
            while current_idx != self.s_prime_node_idx:
                for i, j in self.edges:
                    if i == current_idx and self.y_e[i, j].X > 0.5:
                        tour_indices.append(j)
                        current_idx = j
                        break

            tour_name = [self.idx_to_name[i] for i in tour_indices]
            print("Tour names: ", tour_name)

            visit_points = {}
            p_s_x = sum(
                self.z_x[self.s_node_idx, j].X
                for _, j in self.edges
                if _ == self.s_node_idx
            )
            p_s_y = sum(
                self.z_y[self.s_node_idx, j].X
                for _, j in self.edges
                if _ == self.s_node_idx
            )
            t_s = sum(
                self.z_t[self.s_node_idx, j].X
                for _, j in self.edges
                if _ == self.s_node_idx
            )
            visit_points[self.s_node_idx] = {"pos": (p_s_x, p_s_y), "time": t_s}

            for i in self.V_tar_indices + [self.s_prime_node_idx]:
                p_i_x = sum(self.z_prime_x[k, i].X for k, _ in self.edges if _ == i)
                p_i_y = sum(self.z_prime_y[k, i].X for k, _ in self.edges if _ == i)
                t_i = sum(self.z_prime_t[k, i].X for k, _ in self.edges if _ == i)
                visit_points[i] = {"pos": (p_i_x, p_i_y), "time": t_i}

            self.agent_positions = [visit_points[i]["pos"] for i in tour_indices]
            self.agent_time_points = [visit_points[i]["time"] for i in tour_indices]
            radius_set = [self.node_data[i]["radius"] for i in tour_indices]

            delta_x = [self.delta_x[i].Xn for i in tour_indices]
            delta_y = [self.delta_y[i].Xn for i in tour_indices]
            print(f"delta_x: {delta_x}")
            print(f"delta_y: {delta_y}")
            print(f"radius: {radius_set}")

            self.tour = tour_indices
            assert (
                len(self.tour) == self.n_targets + 2
            ), "Wrong number of targets in the tour!"

            print(f"Best tour: {' -> '.join(tour_name)}")
            print(f"Access time points: {[f'{t:.2f}' for t in self.agent_time_points]}")
            print(
                f"Access positions: {[f'({p[0]:.2f}, {p[1]:.2f})' for p in self.agent_positions]}"
            )
            runtime = self.model.Runtime
            print(f"Runtime: {runtime:.2f} seconds")
            return self.tour

        elif self.model.status == GRB.INFEASIBLE:
            print(" There is no feasible solution.")
            return []
        else:
            print(
                f" Optimization finished, but there is no best solution, with exit code: {self.model.status}"
            )
            return []
