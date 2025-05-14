#!/usr/bin/env python3.11
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pysr import PySRRegressor

# Load data
df = pd.read_csv("data_full.csv")

# Check required columns
total_cycles = 'total cycles (slow, worst case)'
required_columns = ['funits', 'logq', 'dnum', 'cache_size', total_cycles]
if not all(col in df.columns for col in required_columns):
    raise ValueError(f"CSV must contain columns: {required_columns}")

# Features and target
X = df[['funits', 'logq', 'dnum', 'cache_size']]
y = df[total_cycles]
feature_names = X.columns.tolist()

# Train PySR
model = PySRRegressor(
    niterations=100,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["square", "cube", "sqrt", "log"],
    model_selection="best",
    verbosity=1,
    progress=True,
    random_state=0
)
model.fit(X, y)

# Get symbolic expression
equation = model.get_best()
print("Best-fit equation:")
print(equation)

# also train models for total_energy and total_area
ys = ['total_energy', 'total_area']
for y_col in ys:
    if y_col not in df.columns:
        raise ValueError(f"Column '{y_col}' not found in the DataFrame.")

    y = df[y_col]
    model.fit(X, y)
    equation = model.get_best()
    print(f"Best-fit equation for {y_col}:")
    print(equation)


# Create subplots
fig, axes = plt.subplots(4, 2, figsize=(12, 16))
axes = axes.flatten()

# For prediction: hold non-plotted features constant at median values
medians = X.median()

# Plot each feature with the best-fit line
for i, feature in enumerate(feature_names):
    ax = axes[i]

    # Scatter plot
    ax.scatter(df[feature], y, alpha=0.6, edgecolor='k', label='Data')

    # Generate line for best-fit using PySR model
    x_vals = np.linspace(df[feature].min(), df[feature].max(), 300)
    X_line = pd.DataFrame({f: [medians[f]] * len(x_vals) for f in feature_names})
    X_line[feature] = x_vals
    y_pred_line = model.predict(X_line)

    ax.plot(x_vals, y_pred_line, color='red', label='PySR fit')
    ax.set_xlabel(feature)
    ax.set_ylabel("total_cycles")
    ax.set_title(f"{feature} vs total_cycles")
    ax.grid(True)
    ax.legend()

# also plot cache_size vs funits, logq, dnum
for i, feature in enumerate(['funits', 'logq', 'dnum']):
    ax = axes[i + 4]

    # Scatter plot
    ax.scatter(df[feature], df['cache_size'], alpha=0.6, edgecolor='k', label='Data')

    ax.set_xlabel(feature)
    ax.set_ylabel("cache_size")
    ax.set_title(f"{feature} vs cache_size")
    ax.grid(True)
    ax.legend()

# Adjust layout and save
plt.tight_layout()
output_file = "feature_vs_total_cycles_with_fit.png"
plt.savefig(output_file)
plt.close()
print(f"Saved plot with PySR fit to: {output_file}")
