import json

def save_thetas(theta0, theta1, filename="thetas.json"):
    """Save thetas to a JSON file."""
    with open(filename, "w") as f:
        json.dump({"theta0": theta0, "theta1": theta1}, f)