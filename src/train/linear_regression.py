from typing import Tuple
import pandas as pd
import numpy as np
from visualization import plot_regression_line, plot_error_history


def get_mape(kms: np.ndarray, prices: np.ndarray, teta0: float, teta1: float) -> float:
	predicted_prices = teta0 + teta1 * kms
	error = prices - predicted_prices
	mape = np.mean(np.abs(error / prices)) * 100
	return mape


def gradient_descent(
		dataset_output: np.ndarray,
		input_feature: np.ndarray, 
		teta0: float,
		teta1: float,
		learning_rate: float = 0.1,
		iterations: int = 1000,
		epsilon: float = 1e-6
	) -> Tuple[float, float]:
	"""Perform gradient descent to find the optimal parameters."""

	prev_mse = float('inf')

	error_history = []
	for _ in range(iterations):
		predictions = teta0 + teta1 * input_feature

		error = dataset_output - predictions
		mse = np.mean(error ** 2)  # ← Mean Squared Error
		error_history.append(mse)
		if abs(prev_mse - mse) < epsilon:
			break
		prev_mse = mse

		d_teta0 = -2 * np.mean(error)
		d_teta1 = -2 * np.mean(error * input_feature)

		teta0 -= learning_rate * d_teta0
		teta1 -= learning_rate * d_teta1

	plot_error_history(error_history)
	return teta0, teta1


def normalize_data(raw_data: np.ndarray) -> np.ndarray:
    """Normalize the data"""

    mean = raw_data.mean()
    std_dev = raw_data.std()
    normalized_data = (raw_data - mean) / std_dev
    return normalized_data


def reverse_normalize_tetas(
		teta0_norm: float,
		teta1_norm: float,
		mean_input: float,
		std_dev_input: float,
		mean_output: float,
		std_dev_output: float
	) -> Tuple[float, float]:
	"""Reverse normalize the teta values"""

	teta0 = (teta0_norm - teta1_norm * mean_input / std_dev_input) * std_dev_output + mean_output
	teta1 = teta1_norm * std_dev_output / std_dev_input
	return teta0, teta1


def apply_linear_regression(df: pd.DataFrame) -> Tuple[float, float]:
	"""Apply linear regression to the given dataframe."""
	
	if not isinstance(df, pd.DataFrame):
		raise TypeError("df must be a pandas DataFrame")

	if df.empty:
		raise ValueError("DataFrame is empty")

	if "km" not in df.columns or "price" not in df.columns:
		raise ValueError("DataFrame must contain 'mileage' and 'price' columns")

	dataset_output = df["price"].values
	input_feature = df["km"].values

	normalized_dataset_output = normalize_data(dataset_output)
	normalized_input_feature = normalize_data(input_feature)

	teta0_norm = 0.0
	teta1_norm = 0.0

	teta0_norm, teta1_norm = gradient_descent(
		normalized_dataset_output,
		normalized_input_feature,
		teta0_norm,
		teta1_norm
	)

	mean_out = dataset_output.mean()
	std_dev_out = dataset_output.std()
	mean_in = input_feature.mean()
	std_dev_in = input_feature.std()

	teta0, teta1 = reverse_normalize_tetas(teta0_norm, teta1_norm, mean_in, std_dev_in, mean_out, std_dev_out)
	mape = get_mape(input_feature, dataset_output, teta0, teta1)

	plot_regression_line(input_feature, dataset_output, teta0, teta1)

	return teta0, teta1, mape
