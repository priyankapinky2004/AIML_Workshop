import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree

# Simple dataset: Experience (in years) and Salary
experience = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
salary = [30000, 35000, 45000, 50000, 60000, 65000, 70000, 75000, 85000, 90000]

# Prepare input
X = [[i] for i in experience]
y = salary

# Create and train the Decision Tree Regressor
model = DecisionTreeRegressor()
model.fit(X, y)

# Predict salary for 6.5 years of experience
prediction = model.predict([[6.5]])[0]
print(f"Predicted salary for 6.5 years experience: ₹{prediction:.2f}")
print(f"Model Score (R^2): {model.score(X, y):.2f}")

# Create a smooth line for visualization
x_test = [[i / 10] for i in range(10, 105)]  # from 1.0 to 10.4
y_pred = model.predict(x_test)

# Plotting the result
plt.figure(figsize=(10, 6))
plt.scatter(experience, salary, color='blue', label='Actual Data')
plt.plot([i[0] for i in x_test], y_pred, color='orange', label='Decision Tree Prediction')
plt.scatter([6.5], [prediction], color='green', marker='x', s=100, label='Prediction for 6.5 yrs')
plt.xlabel("Years of Experience")
plt.ylabel("Salary (₹)")
plt.title("Decision Tree Regression: Salary vs Experience")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
