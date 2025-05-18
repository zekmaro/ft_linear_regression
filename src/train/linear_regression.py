from typing import Tuple
import pandas as pd
import numpy as np
from visualization import plot_regression_line, plot_error_history
from header import (
    INIT_TETA0,
    INIT_TETA1,
    LEARNING_RATE,
    MAX_ITERATIONS,
    EPSILON
)
from header import (
    DF_TYPE_ERROR,
    DF_EMPTY,
    DF_STRUCTURE_ERROR,
    DF_NANS_ERROR
)


def get_mape(
    kms: np.ndarray,
    prices: np.ndarray,
    teta0: float,
    teta1: float
) -> float:
    """Calculate the Mean Absolute Percentage Error (MAPE)"""

    predicted_prices = teta0 + teta1 * kms
    error = prices - predicted_prices
    mape = np.mean(np.abs(error / prices)) * 100
    return mape  # type: ignore


def gradient_descent(
    dataset_output: np.ndarray,
    input_feature: np.ndarray,
    teta0: float,
    teta1: float,
    learning_rate: float = LEARNING_RATE,
    iterations: int = MAX_ITERATIONS,
    epsilon: float = EPSILON
) -> Tuple[float, float]:
    """Perform gradient descent to find the optimal parameters."""

    prev_mse = float('inf')

    error_history = []
    for _ in range(iterations):
        predictions = teta0 + teta1 * input_feature

        error = dataset_output - predictions
        mse = np.mean(error ** 2)
        error_history.append(mse)
        if abs(prev_mse - mse) < epsilon:
            break
        prev_mse = mse

        d_teta0 = -2 * np.mean(error)
        d_teta1 = -2 * np.mean(error * input_feature)

        teta0 -= learning_rate * d_teta0  # type: ignore
        teta1 -= learning_rate * d_teta1  # type: ignore

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

    scaled_bias = -teta1_norm * mean_input / std_dev_input
    teta0 = (teta0_norm + scaled_bias) * std_dev_output + mean_output
    teta1 = teta1_norm * std_dev_output / std_dev_input
    return teta0, teta1


def apply_linear_regression(df: pd.DataFrame) -> Tuple[float, float, float]:
    """Apply linear regression to the given dataframe."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError(DF_TYPE_ERROR)

    if df.empty:
        raise ValueError(DF_EMPTY)

    if "km" not in df.columns or "price" not in df.columns:
        raise ValueError(DF_STRUCTURE_ERROR)

    if df.isnull().values.any():
        raise ValueError(DF_NANS_ERROR)

    dataset_output = df["price"].values
    input_feature = df["km"].values

    normalized_dataset_output = normalize_data(dataset_output)  # type: ignore
    normalized_input_feature = normalize_data(input_feature)  # type: ignore

    teta0_norm = INIT_TETA0
    teta1_norm = INIT_TETA1

    teta0_norm, teta1_norm = gradient_descent(
        normalized_dataset_output,
        normalized_input_feature,
        teta0_norm,
        teta1_norm
    )

    mean_out = dataset_output.mean()  # type: ignore
    std_dev_out = dataset_output.std()  # type: ignore
    mean_in = input_feature.mean()  # type: ignore
    std_dev_in = input_feature.std()  # type: ignore

    teta0, teta1 = reverse_normalize_tetas(
        teta0_norm,
        teta1_norm,
        mean_in,
        std_dev_in,
        mean_out,
        std_dev_out
    )

    mape = get_mape(
        input_feature,  # type: ignore
        dataset_output,  # type: ignore
        teta0,
        teta1
    )
    plot_regression_line(
        input_feature,  # type: ignore
        dataset_output,  # type: ignore
        teta0,
        teta1
    )

    return teta0, teta1, mape
