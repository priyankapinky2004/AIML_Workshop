import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

# Dataset
square_feet = [1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400]
prices = [300000, 320000, 340000, 360000, 380000, 390000, 400000, 410000, 430000, 450000]

# Prepare data
x = [[i] for i in square_feet]
y = prices

# Create and train the model
model = DecisionTreeRegressor()
model.fit(x, y)

# Predict price for 4000 sq. ft
predicted_price = model.predict([[4000]])[0]
print(f"Predicted price for 4000 square feet: {predicted_price}")
print(f"Model Score (R^2): {model.score(x, y)}")

# Prepare a smooth curve for plotting
x_plot = [[i] for i in range(1400, 4100, 10)]
y_plot = model.predict(x_plot)

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(square_feet, prices, color='blue', label='Actual Prices')
plt.plot([i[0] for i in x_plot], y_plot, color='orange', label='Decision Tree Prediction Line')
plt.scatter([4000], [predicted_price], color='green', marker='x', s=100, label='Prediction for 4000 sqft')
plt.title('Decision Tree Regression: House Prices vs Square Feet')
plt.xlabel('Square Feet')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
