from mt_tsp.model import MTSPMICP, Target
from mt_tsp.visualization import Visualizer


def main() -> None:
    # give input data
    # TODO: reading targets from a config file
    targets = [
        Target("A", (0.0, 10.0), (0.1, 0.0), (0.0, 40.0)),
        Target("B", (8.0, 0.0), (0.0, 0.3), (0.0, 100.0)),
        Target("C", (5.0, 5.0), (-0.1, 0.3), (0.0, 30.0)),
        Target("D", (1.0, 6.0), (0.1, 0.2), (0.0, 30.0)),
        Target("E", (2.0, 2.0), (0.1, 0.1), (0.0, 100.0)),
        Target("F", (3.0, 3.0), (0.2, 0.3), (0.0, 1000.0)),
    ]
    model = MTSPMICP(targets, depot=(0.0, 0.0), T=1000.0, vmax=0.5, square_side=10.0)
    tour = model.solve()
    print("Tour:", tour)
    viz = Visualizer(model, gif_path="mt_tsp.gif", fps=10, frames=1000)
    viz.animate()
    print("Saved animation to mt_tsp.gif")


if __name__ == "__main__":
    main()
