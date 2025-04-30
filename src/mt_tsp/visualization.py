from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from numpy.typing import NDArray

from mt_tsp.model import MTSPMICP


class Visualizer:
    def __init__(
        self,
        model: MTSPMICP,
        gif_path: str = "mt_tsp.gif",
        fps: int = 10,
        frames: int = 100,
    ) -> None:
        self.model: MTSPMICP = model
        self.gif_path: str = gif_path
        self.fps: int = fps
        self.times: NDArray[np.floating] = np.linspace(0.0, model.T, frames)

    def animate(self) -> None:
        fig, ax = plt.subplots()
        ax.set(xlim=(-5, 15), ylim=(-5, 15))
        (agent_dot,) = ax.plot([], [], "ro", label="Agent")
        target_dots: List[Any] = [
            ax.plot([], [], "bo", label=f"Target {t.name}")[0]
            for t in self.model.targets
        ]

        def init() -> List[Any]:
            agent_dot.set_data([], [])
            for dot in target_dots:
                dot.set_data([], [])
            return [agent_dot] + target_dots

        def update(frame: int) -> List[Any]:
            t = self.times[frame]
            # update accordinbg to timeframe
            for dot, tgt in zip(target_dots, self.model.targets):
                pos = tgt.position(t)
                dot.set_data(pos[0], pos[1])
            # update agent position
            agent_dot.set_data(self.model.depot[0], self.model.depot[1])
            return [agent_dot] + target_dots

        anim = FuncAnimation(
            fig, update, frames=len(self.times), init_func=init, blit=True
        )
        writer = PillowWriter(fps=self.fps)
        anim.save(self.gif_path, writer=writer)
        plt.close(fig)
