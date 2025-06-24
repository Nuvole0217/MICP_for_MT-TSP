import json
from pathlib import Path
from typing import List

import tomli

from mt_tsp.model import MTSPMICP, MTSPMICPGCS, Target


def load_config(
    target_path: Path, agent_path: Path, choice: int
) -> MTSPMICP | MTSPMICPGCS:
    if choice not in [1, 2]:
        raise ValueError("Choice must be 1 for MTSPMICP or 2 for MTSPMICPGCS.")

    with open(target_path, encoding="utf-8") as f:
        target_data = json.load(f)

    with open(agent_path, "rb") as g:
        agent_data = tomli.load(g)

    targets: List[Target] = []
    for key, data in target_data.items():
        targets.append(
            Target(
                name=key,
                p0=(data["px"], data["py"]),
                v=(data["vx"], data["vy"]),
                t_window=(data["tmin"], data["tmax"]),
                radius=data["radius"],
            )
        )
    if choice == 1:
        return MTSPMICP(
            targets=targets,
            depot=(agent_data["depot"]["px"], agent_data["depot"]["py"]),
            max_time=agent_data["param"]["T"],
            vmax=agent_data["param"]["vmax"],
            square_side=agent_data["param"]["R"],
        )
    else:
        return MTSPMICPGCS(
            targets=targets,
            depot=(agent_data["depot"]["px"], agent_data["depot"]["py"]),
            max_time=agent_data["param"]["T"],
            vmax=agent_data["param"]["vmax"],
        )
