from pathlib import Path

import numpy as np
import pytest

from mt_tsp.model import load_config


def load_cases(case_name: Path) -> None:
    root = Path(__file__).resolve().parent
    target_file = root / case_name / "target.json"
    agent_file = root / case_name / "agent.toml"
    model = load_config(target_file, agent_file)
