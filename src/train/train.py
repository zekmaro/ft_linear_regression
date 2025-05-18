from load_csv import load
from file_utils import save_thetas
from header import DATA_PATH
from linear_regression import apply_linear_regression
import sys


def main():
    """Main function to train the model"""
    df = load(DATA_PATH)
    teta0, teta1, mape = apply_linear_regression(df)
    save_thetas(teta0, teta1)
    print(f"Model trained with Mean Absolute Percentage Error: {mape:.2f}")


if __name__ == "__main__":
    """Entry point for the script"""
    try:
        main()
    except Exception as e:
        print(f"AssertionError: {e}")
        sys.exit(1)
