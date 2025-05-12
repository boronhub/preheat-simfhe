#!/usr/bin/env python3.11
import pandas as pd
from pysr import PySRRegressor
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv("experiment.csv")  # Replace with your actual file name

# Ensure required columns exist
required_columns = ['funits', 'logq', 'dnum', 'cache_size', 'total cycles (slow, worst case)']
if not all(col in df.columns for col in required_columns):
    raise ValueError(f"CSV must contain columns: {required_columns}")

# Define features and target
X = df[['funits', 'logq', 'dnum', 'cache_size']]
y = df['total cycles (slow, worst case)']

# Optional: split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train PySR model
model = PySRRegressor(
    niterations=100,           # You can increase this for better accuracy
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["square", "cube", "sqrt", "log"],
    model_selection="best",    # You can change to 'accuracy' for best R^2
    verbosity=1,
    progress=True,
    random_state=0
)

model.fit(X_train, y_train)

# Print the best equation
print("Best equation:")
print(model.get_best())

# Optional: evaluate on test data
from sklearn.metrics import mean_squared_error
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"\nTest MSE: {mse:.2f}")
