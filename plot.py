import pandas as pd
df = pd.read_csv('data.csv')

import numpy as np
X = df['funits'].values.reshape(-1, 1)
y = df['total cycles (slow best case)'].values

from pysr import PySRRegressor
model = PySRRegressor(
    niterations=100,
    populations=20,
    model_selection="accuracy",
    loss="loss(prediction, target) = (prediction - target)^2"
)

model.fit(X, y)

print(model)
predictions = model.predict(X)