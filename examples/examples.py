from mt_tsp.model import MTSPMICP, MTSPMICPGCS, Target
from mt_tsp.utils.loader import load_config
from mt_tsp.visualization import Visualizer


def run_examples() -> None:
    # give input data
    targets = [
        Target("A", (0.0, 10.0), (0.01, 0.0), (0.0, 40.0), 1.5),
        Target("B", (8.0, 0.0), (0.0, 0.02), (0.0, 100.0), 1.5),
        Target("C", (5.0, 5.0), (-0.01, -0.015), (0.0, 50.0), 1.0),
        Target("D", (1.0, 6.0), (-0.01, 0.02), (0.0, 50.0), 1.5),
        Target("E", (2.0, 2.0), (0.01, -0.015), (0.0, 100.0), 1.5),
        Target("F", (0.0, 0.0), (0.0, -0.02), (0.0, 1000.0), 1.5),
        Target("G", (4.0, 8.0), (0.01, 0.0), (0.0, 100.0), 1.0),
        Target("H", (6.0, 2.0), (0.0, 0.01), (0.0, 100.0), 1.0),
        Target("I", (9.0, 9.0), (0.0, 0.0), (0.0, 100.0), 1.0),
        Target("J", (3.0, 3.0), (0.0, 0.0), (0.0, 100.0), 1.0),
        Target("K", (7.0, 7.0), (0.0, 0.0), (0.0, 100.0), 1.0),
        Target("L", (10.0, 10.0), (0.0, 0.0), (0.0, 100.0), 1.0),
        Target("M", (0.0, 5.0), (0.01, 0.0), (0.0, 100.0), 1.5),
        # Target("N", (5.0, 0.0), (0.0, 0.02), (0.0, 100.0), 1.0),
        # Target("O", (3.0, 7.0), (-0.01, 0.02), (0.0, 50.0), 1.0),
        # Target("P", (8.0, 3.0), (-0.01, -0.015), (0.0, 50.0), 1.0),
        # Target("Q", (2.0, 8.0), (0.01, -0.015), (0.0, 100.0), 1.0),
    ]
    # model = MTSPMICP(targets, depot=(0.0, 4.0), max_time=1000.0, vmax=0.35, square_side=10.0)
    model = MTSPMICPGCS(targets, depot=(0.0, 4.0), max_time=1000.0, vmax=0.35)
    tour = model.solve()
    print("Tour:", tour)
    viz = Visualizer(model, gif_path="mt_cetsp_13_targets_example.gif", fps=10, frames=500)
    viz.animate()
    print("Saved animation to mt_tsp.gif")


run_examples()
