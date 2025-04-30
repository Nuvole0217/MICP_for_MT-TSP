from mt_tsp.model import Target, MTSPMICP
from mt_tsp.visualization import Visualizer

def main() -> None:
    # give input data
    # TODO: reading it from a config file
    targets = [
        Target("A", (0.0, 10.0), (0.3, -0.1), (1.0, 8.0)),
        Target("B", (8.0,  0.0), (-0.2, 0.3), (2.0, 9.0)),
        Target("C", (5.0,  5.0), (0.1,  0.1), (0.0, 10.0)),
    ]
    model = MTSPMICP(targets, depot=(0.0, 0.0), T=10.0, vmax=2.0)
    tour = model.solve()
    print("Tour:", tour)
    viz = Visualizer(model, gif_path="mt_tsp.gif", fps=10, frames=100)
    viz.animate()
    print("Saved animation to mt_tsp.gif")

if __name__ == "__main__":
    main()
