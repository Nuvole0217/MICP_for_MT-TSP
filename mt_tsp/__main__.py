from pathlib import Path

import typer
from typing_extensions import Annotated

from mt_tsp.utils.loader import load_config
from mt_tsp.visualization import Visualizer

app = typer.Typer(add_completion=False)


@app.command()
def run(
    target_path: Annotated[
        Path,
        typer.Argument(help="Path to the targets file in JSON format."),
    ] = Path("."),
    agent_path: Annotated[
        Path,
        typer.Argument(help="Path to the agent configuration file in toml format."),
    ] = Path("."),
    choice: Annotated[
        int,
        typer.Argument(
            help="Choice of model you want to run, 1 for MTSPMICP, 2 for MTSPMICPGCS."
        ),
    ] = 1,
) -> None:
    model = load_config(target_path, agent_path, choice)
    tour = model.solve()
    print("Tour: ", tour)
    viz = Visualizer(model, gif_path="mt_tsp.gif", fps=10, frames=500)
    viz.animate()
    print("Saved animation to mt_tsp.gif")
    return
