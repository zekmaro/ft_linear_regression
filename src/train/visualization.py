import matplotlib.pyplot as plt
import numpy as np


def plot_regression_line(x: np.ndarray, y: np.ndarray, teta0: float, teta1: float) -> None:
	"""Plot the regression line based on the given parameters."""
	plt.scatter(x, y, color='blue', label='Data points')
	plt.plot(x, teta0 + teta1 * x, color='red', label='Regression line')
	plt.xlabel('Mileage (km)')
	plt.ylabel('Price')
	plt.title('Linear Regression Fit')
	plt.legend()
	plt.show()


def plot_error_history(error_history: list) -> None:
	"""Plot the error history over iterations."""
	plt.plot(error_history)
	plt.xlabel('Iterations')
	plt.ylabel('Mean Squared Error')
	plt.title('Error History')
	plt.show()
