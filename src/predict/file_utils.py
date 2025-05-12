import json


def load_thetas(filename="../data/thetas.json"):
    """Load thetas from a JSON file."""
    with open(filename, "r") as f:
        data = json.load(f)
        return data["theta0"], data["theta1"]
