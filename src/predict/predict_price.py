from estimate_price import predict_price
from file_utils import load_thetas


def main():
	"""Main function to predict the price of a car based on user input."""
	mileage_input = input("Enter the mileage of the car (in km): ")
	teta0, teta1 = load_thetas()
	predicted_price = predict_price(float(mileage_input), teta0, teta1)
	print(f"The predicted price of the car is: {predicted_price:.2f}")
	

if __name__ == "__main__":
	main()
