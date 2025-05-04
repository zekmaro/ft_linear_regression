from typing import Tuple
import pandas as pd
import numpy as np


def gradient_descent(
		dataset_output: np.ndarray,
		input_feature: np.ndarray, 
		teta0: float,
		teta1: float,
		learning_rate: float = 0.01,
		iterations: int = 1000
	) -> Tuple[float, float]:
	"""Perform gradient descent to find the optimal parameters."""
	for _ in range(iterations):
		predictions = teta0 + teta1 * input_feature
		error = dataset_output - predictions

		d_teta0 = -2 * np.mean(error)
		d_teta1 = -2 * np.mean(error * input_feature)

		teta0 -= learning_rate * d_teta0
		teta1 -= learning_rate * d_teta1

	return teta0, teta1


def apply_linear_regression(df: pd.DataFrame) -> Tuple[float, float]:
	"""Apply linear regression to the given dataframe."""
	
	if not isinstance(df, pd.DataFrame):
		raise TypeError("df must be a pandas DataFrame")

	if df.empty:
		raise ValueError("DataFrame is empty")

	if "mileage" not in df.columns or "price" not in df.columns:
		raise ValueError("DataFrame must contain 'mileage' and 'price' columns")

	dataset_output = df["price"].values
	input_feature = df["mileage"].values

	teta0 = 0.0
	teta1 = 0.0

	teta0, teta1 = gradient_descent(dataset_output, input_feature, teta0, teta1)

	return teta0, teta1
