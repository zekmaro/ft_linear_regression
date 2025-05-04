from src.train.load_csv import load
from src.train.file_utils import save_thetas
from src.train.linear_regression import apply_linear_regression

def main():
	"""Main function to train the model"""
	df = load("data/train.csv")
	teta0, teta1 = apply_linear_regression(df)
	save_thetas(teta0, teta1)

if __name__ == "__main__":
	main()