from predict_price import predict_price
from file_utils import load_thetas
import sys
from header import (
    ENTER_MILEAGE,
    MILEAGE_TYPE_ERROR,
    MILEAGE_VALUE_ERROR,
    FILE_NOT_FOUND,
    FILE_EMPTY
)


def main() -> None:
    """Main function to predict the price of a car based on user input."""
    mileage_input = input(ENTER_MILEAGE)

    if not mileage_input.isdigit():
        print(MILEAGE_TYPE_ERROR)
        return

    mileage_input = int(mileage_input)
    if mileage_input < 0:
        print(MILEAGE_VALUE_ERROR)
        return
    mileage_input = int(mileage_input)

    try:
        teta0, teta1 = load_thetas()
    except FileNotFoundError:
        print(FILE_NOT_FOUND)
        return
    except ValueError:
        print(FILE_EMPTY)
        return

    predicted_price = max(0, predict_price(float(mileage_input), teta0, teta1))
    print(f"The predicted price of the car is: {predicted_price:.2f}")


if __name__ == "__main__":
    """Entry point for the script"""
    try:
        main()
    except Exception as e:
        print(f"AssertionError: {e}")
        sys.exit(1)
