def predict_price(mileage: float, theta0: float, theta1: float) -> float:
    """Predict the price of a car based on
    its mileage using linear regression parameters.
    """
    return theta0 + theta1 * mileage
