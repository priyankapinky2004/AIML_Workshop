import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Dataset
square_feet = [1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400]
prices = [300000, 320000, 340000, 360000, 380000, 390000, 400000, 410000, 430000, 450000]

# Preparing data for the model
x = [[i] for i in square_feet]
y = prices

# Creating and training the model
model = LinearRegression()
model.fit(x, y)

# Prediction for 4000 sq. ft
predicted_price = model.predict([[3200]])[0]
print(f"Predicted price for 3200 square feet: {predicted_price}")
print(f"Model Coefficient: {model.coef_[0]}")
print(f"Model Intercept: {model.intercept_}")
print(f"Model Score (R^2): {model.score(x, y)}")

# Predicting values for plotting the regression line
predicted_prices = model.predict(x)

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(square_feet, prices, color='blue', label='Actual Prices')
plt.plot(square_feet, predicted_prices, color='red', label='Regression Line')
plt.scatter([4000], [predicted_price], color='green', marker='x', s=100, label='Prediction for 4000 sqft')
plt.title('Linear Regression: House Prices vs Square Feet')
plt.xlabel('Square Feet')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()