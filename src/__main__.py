import typer
from pathlib import Path
from typing_extensions import Annotated
from src.mt_tsp.model import MTSPMICP, Target
from src.mt_tsp.visualization import Visualizer
from typing import List
import json
import tomllib

app = typer.Typer(add_completion=False)

@app.command()
def run(
    target_path: Annotated[
        Path,
        typer.Argument (
            help="Path to the targets file in JSON format."
        ),
    ] = Path("."),
    agent_path: Annotated[
        Path,
        typer.Argument (
            help="Path to the agent configuration file in toml format."
        ),
    ] = Path("."),
) -> None:
    with open(target_path, "r", encoding='utf-8') as f:
        target_data = json.load(f)
        
    with open(agent_path, "rb") as g:
        agent_data = tomllib.load(g)
    
    targets: List[Target] = []
    for key, data in target_data.items():
        targets.append(
            Target(
                name=key,
                p0=(data["px"], data['py']),
                v=(data["vx"], data["vy"]),
                t_window=(data["tmin"], data["tmax"]),
            )
        )
    model=MTSPMICP(
        targets=targets,
        depot=(agent_data["depot"]["px"], agent_data["depot"]["py"]),
        T=agent_data["param"]["T"],
        vmax=agent_data["param"]["vmax"],
        square_side=agent_data["param"]["R"],
    )
    tour = model.solve()
    print("Tour: ", tour)
    viz = Visualizer(model, gif_path="mt_tsp.gif", fps=10, frames=1000)
    viz.animate()
    print("Saved animation to mt_tsp.gif")
    return