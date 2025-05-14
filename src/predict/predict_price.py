from estimate_price import predict_price
from file_utils import load_thetas
import sys


def main():
    """Main function to predict the price of a car based on user input."""
    mileage_input = input("Enter the mileage of the car (in km): ")

    if not mileage_input.isdigit():
        print("Please enter a valid number for mileage.")
        return

    mileage_input = int(mileage_input)
    if mileage_input < 0:
        print("Mileage cannot be negative.")
        return
    if mileage_input > 400000:
        print("Mileage seems too high. Please check the input.")
        return

    teta0, teta1 = load_thetas()
    predicted_price = predict_price(float(mileage_input), teta0, teta1)
    print(f"The predicted price of the car is: {predicted_price:.2f}")
    

if __name__ == "__main__":
    """Entry point for the script"""
    try:
        main()
    except Exception as e:
        print(f"AssertionError: {e}")
        sys.exit(1)
