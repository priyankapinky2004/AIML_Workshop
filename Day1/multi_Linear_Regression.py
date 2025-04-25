import numpy as np
from sklearn.linear_model import LinearRegression

# Independent variables (Area, Bedrooms, Age)
X = [
    [1500, 3, 20],
    [1600, 3, 15],
    [1700, 4, 18],
    [1800, 4, 12],
    [1900, 3, 10],
    [2000, 4, 5],
    [2100, 5, 8],
    [2200, 5, 3]
]

# Dependent variable (Price)
y = [300000, 320000, 340000, 360000, 380000, 400000, 420000, 440000]

# Create the model
model = LinearRegression()

# Fit the model
model.fit(X, y)

# Predict price for a new house
# Area = 2400, Bedrooms = 4, Age = 4
predicted_price = model.predict([[2400, 4, 4]])

print(f"Predicted price: ${predicted_price[0]:,.2f}")
print("Model Coefficients:", model.coef_)
print("Model Intercept:", model.intercept_)
print("R² Score:", model.score(X, y))
print("Model Coefficients:", model.coef_)