# ft\_linear\_regression - Predicting Car Prices with Linear Regression

## 📈 Project Overview

`ft_linear_regression` is a simple machine learning project from the 42 curriculum that implements a basic linear regression algorithm in Python to predict car prices based on mileage. The goal is to understand the mathematical foundation of linear regression and implement training, evaluation, and prediction from scratch without using ML libraries.

## 🚀 Features

* **Gradient Descent Implementation** - Trains the model using gradient descent.
* **Manual Data Handling** - Loads and parses CSV input data.
* **Model Training & Evaluation** - Trains the model and evaluates performance visually.
* **Prediction Mode** - Takes user input for mileage and outputs predicted price.
* **Data Visualization** - Plots the regression line vs actual data using matplotlib.
* **Persistent Model Saving** - Stores `theta0` and `theta1` in a file for reuse.

## 🧠 Core Concepts

* Hypothesis function: `price = theta0 + theta1 * mileage`
* Cost function (MSE): `J = (1/2m) * sum((h(x) - y)^2)`
* Optimization via gradient descent:

  * `theta0 -= learning_rate * dJ/dtheta0`
  * `theta1 -= learning_rate * dJ/dtheta1`

## 📦 Installation

### Prerequisites

* Python 3.x
* `matplotlib` (for visualization)

Install the dependencies:

```sh
pip install -r requirements.txt
```

## 🛠️ Usage

### 1. Training the Model

```sh
python train.py
```

* Reads `data.csv`
* Trains using gradient descent
* Saves parameters to `thetas.json`

### 2. Predicting a Price

```sh
python predict.py
```

* Prompts for mileage
* Predicts price using saved thetas

### 3. Plotting the Regression

```sh
python plot.py
```

* Plots training data and regression line

## 📁 File Structure

```
📂 ft_linear_regression/
├── data.csv         # Input data (mileage, price)
├── train.py         # Training script
├── predict.py       # User prediction interface
├── plot.py          # Data visualization
├── thetas.json      # Stored model parameters
├── requirements.txt # Python dependencies
```

## 📊 Example

#### Input CSV (`data.csv`):

```
km,price
10000,20000
25000,18000
...
```

#### Prediction:

```
Enter mileage: 30000
Estimated price: 17042.58$
```

## 🏗️ Future Enhancements

* Add polynomial regression for better fitting.
* Implement normalization for better convergence.
* Add unit tests for robustness.

## 🏆 Credits

* **Developer:** [zekmaro](https://github.com/zekmaro)
* **Project:** Part of the 42 School curriculum
* **Inspiration:** Andrew Ng’s ML course & scikit-learn's simplicity

---

🤖 A foundational step into machine learning. Explore, tweak, and enjoy learning!
