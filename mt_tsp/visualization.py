from typing import Any, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Arrow
from numpy.typing import NDArray

from mt_tsp.model import MTSPMICP


class Visualizer:
    def __init__(
        self,
        model: MTSPMICP,
        gif_path: str = "mt_tsp.gif",
        fps: int = 10,
        frames: int = 1000,
        speed_arrow_scale: float = 3,
    ) -> None:
        self.model = model
        self.gif_path = gif_path
        self.fps = fps
        self.speed_arrow_scale = speed_arrow_scale
        self.times: NDArray[np.float64] = np.linspace(0.0, model.T, frames)

        self.tour: List[int] = model.tour
        self.agent_time_points: List[float] = [model.t[i].Xn for i in self.tour]

        # visualization parameters
        self.target_colors = plt.cm.tab10(np.linspace(0, 1, len(model.targets)))
        self.agent_color = "#E64A19"  # Material Orange 700
        self.path_color = "#616161"  # Material Grey 700
        self.arrow_color = "#37474F"  # Material Blue Grey 800

    def _get_path_segments(
        self,
    ) -> List[Tuple[float, float, NDArray[np.float64], NDArray[np.float64]]]:
        segments = []
        for i in range(len(self.tour) - 1):
            from_node = self.tour[i]
            to_node = self.tour[i + 1]
            t_start = self.agent_time_points[i]
            t_end = self.agent_time_points[i + 1]

            if from_node == 0:
                start_pos = self.model.depot
            else:
                target = self.model.targets[from_node - 1]
                start_pos = target.position(t_start)

            if to_node == self.model.N - 1:
                end_pos = self.model.depot
            else:
                target = self.model.targets[to_node - 1]
                end_pos = target.position(t_end)

            segments.append((t_start, t_end, np.array(start_pos), np.array(end_pos)))
        return segments

    def _get_agent_pos(self, t: float) -> Tuple[float, float]:
        for t_start, t_end, start_pos, end_pos in self._get_path_segments():
            if t_start <= t <= t_end:
                ratio = (t - t_start) / (t_end - t_start) if t_end > t_start else 0.0
                x = start_pos[0] + (end_pos[0] - start_pos[0]) * ratio
                y = start_pos[1] + (end_pos[1] - start_pos[1]) * ratio
                return (x, y)
        return (self.model.depot[0], self.model.depot[1])

    def animate(self) -> None:
        fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
        ax.set(
            xlim=(-self.model.square_side, self.model.square_side ),
            ylim=(-self.model.square_side, self.model.square_side ),
            title=f"MT-TSP Solution (vmax={self.model.vmax})",
            xlabel="X Coordinate",
            ylabel="Y Coordinate",
        )
        ax.grid(True, alpha=0.3)

        agent_dot = ax.plot(
            [],
            [],
            "o",
            color=self.agent_color,
            markersize=14,
            markeredgewidth=2,
            markeredgecolor="white",
            label="Agent",
        )[0]

        target_dots: List[Any] = []
        speed_arrows: List[Arrow] = []

        for i, tgt in enumerate(self.model.targets):
            # target dots
            dot = ax.plot(
                [],
                [],
                "o",
                color=self.target_colors[i],
                markersize=12,
                alpha=0.9,
                label=f"Target {tgt.name}",
            )[0]
            target_dots.append(dot)

            # speed arrows
            arrow = Arrow(0, 0, 0, 0, width=0.3, color=self.arrow_color, alpha=0.7)
            ax.add_patch(arrow)
            speed_arrows.append(arrow)

        # path elements
        path_line = ax.plot(
            [],
            [],
            "--",
            color=self.path_color,
            linewidth=2,
            alpha=0.7,
            label="Agent Path",
        )[0]

        ax.legend(loc="upper right", frameon=True, shadow=True)

        # define the animation
        def init() -> List[Any]:
            agent_dot.set_data([], [])
            for dot in target_dots:
                dot.set_data([], [])
            for arrow in speed_arrows:
                arrow.set_data(x=0, y=0, width=0)
            path_line.set_data([], [])
            return [agent_dot] + target_dots + speed_arrows + [path_line]

        def update(frame: int) -> List[Any]:
            t = self.times[frame]

            # udapte target objects
            updates: List[Any] = []
            for i, tgt in enumerate(self.model.targets):
                # currentt posistion
                pos = tgt.position(t)
                target_dots[i].set_data([pos[0]], [pos[1]])

                # show arrows
                if np.linalg.norm(tgt.v) > 1e-3:
                    dx = tgt.v[0] * self.speed_arrow_scale
                    dy = tgt.v[1] * self.speed_arrow_scale
                    speed_arrows[i].set_data(pos[0], pos[1], dx, dy, width=0.3)
                else:
                    speed_arrows[i].set_data(0, 0, 0, 0, 0)

                updates.extend([target_dots[i], speed_arrows[i]])

            # udapte position
            agent_pos = self._get_agent_pos(t)
            agent_dot.set_data([agent_pos[0]], [agent_pos[1]])
            updates.append(agent_dot)

            path_x, path_y = [], []
            last_seg: Tuple[float, float, NDArray[np.float64], NDArray[np.float64]] = (
                0,
                0,
                np.array(agent_pos),
                np.array(agent_pos),
            )
            first_seg: Tuple[float, float, NDArray[np.float64], NDArray[np.float64]] = (
                0,
                0,
                np.array(agent_pos),
                np.array(agent_pos),
            )
            for seg in self._get_path_segments():
                if seg[1] <= t:  # already taken time
                    path_x.extend([seg[2][0], seg[3][0]])
                    path_y.extend([seg[2][1], seg[3][1]])
                    if seg[0] >= last_seg[0]:
                        last_seg = seg
                if seg[0] == 0:
                    first_seg = seg

            if first_seg[1] >= t:
                path_x.extend([first_seg[2][0], agent_pos[0]])
                path_y.extend([first_seg[2][1], agent_pos[1]])

            path_x.extend([last_seg[3][0], agent_pos[0]])
            path_y.extend([last_seg[3][1], agent_pos[1]])
            path_line.set_data(path_x, path_y)
            updates.append(path_line)

            return updates

        # generate the image
        anim = FuncAnimation(
            fig,
            update,
            frames=len(self.times),
            init_func=init,
            blit=True,
            interval=50,
            repeat=False,
        )

        print(f"Generating {self.gif_path}...")
        anim.save(
            self.gif_path,
            writer=PillowWriter(fps=self.fps),
            progress_callback=lambda i, n: print(
                f"Rendering frame {i+1}/{n}", end="\r"
            ),
        )
        plt.close()
        print("\nAnimation saved successfully!")
