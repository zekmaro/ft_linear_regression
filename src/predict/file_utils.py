import json
from header import THETAS_PATH


def load_thetas(filename=THETAS_PATH) -> tuple[float, float]:
    """Load thetas from a JSON file."""
    with open(filename, "r") as f:
        data = json.load(f)
        return data["theta0"], data["theta1"]
