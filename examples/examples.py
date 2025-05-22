from mt_tsp.model import MTSPMICP, Target, load_config
from mt_tsp.visualization import Visualizer


def run_examples() -> None:
    # give input data
    targets = [
        Target("A", (0.0, 10.0), (0.01, 0.0), (0.0, 40.0), 1.0),
        Target("B", (8.0, 0.0), (0.0, 0.02), (0.0, 100.0), 1.0),
        Target("C", (5.0, 5.0), (-0.01, -0.015), (0.0, 50.0), 1.0),
        Target("D", (1.0, 6.0), (-0.01, 0.02), (0.0, 50.0), 1.0),
        Target("E", (2.0, 2.0), (0.01, -0.015), (0.0, 100.0), 1.0),
        Target("F", (0.0, 0.0), (0.0, -0.02), (0.0, 1000.0), 1.0),
    ]
    model = MTSPMICP(targets, depot=(0.0, 4.0), T=1000.0, vmax=0.35, square_side=10.0)
    tour = model.solve()
    print("Tour:", tour)
    viz = Visualizer(model, gif_path="mt_tsp.gif", fps=10, frames=1000)
    viz.animate()
    print("Saved animation to mt_tsp.gif")


run_examples()
