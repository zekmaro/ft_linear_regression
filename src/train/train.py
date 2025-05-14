from load_csv import load
from file_utils import save_thetas
from linear_regression import apply_linear_regression


def main():
	"""Main function to train the model"""
	df = load("../data/data.csv")
	teta0, teta1, mape = apply_linear_regression(df)
	save_thetas(teta0, teta1)
	print(f"Model trained with Mean Absolute Percentage Error: {mape:.2f}")


if __name__ == "__main__":
	main()