import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

# Example Medical Dataset
age = [25, 30, 35, 40, 45, 50, 55, 60, 65, 70]
blood_pressure = [120, 122, 125, 130, 135, 140, 145, 150, 155, 160]

# Preparing data
x = [[i] for i in age]
y = blood_pressure

# Create and train the model
model = DecisionTreeRegressor()
model.fit(x, y)

# Predict blood pressure for age 58
predicted_bp = model.predict([[58]])[0]
print(f"Predicted blood pressure for 58 years: {predicted_bp}")
print(f"Model Score (R^2): {model.score(x, y)}")

# Prepare a smooth curve for plotting
x_plot = [[i] for i in range(20, 75, 1)]
y_plot = model.predict(x_plot)

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(age, blood_pressure, color='blue', label='Actual Blood Pressure')
plt.plot([i[0] for i in x_plot], y_plot, color='orange', label='Decision Tree Prediction Line')
plt.scatter([58], [predicted_bp], color='green', marker='x', s=100, label='Prediction for 58 years')
plt.title('Decision Tree Regression: Blood Pressure vs Age')
plt.xlabel('Age (years)')
plt.ylabel('Blood Pressure (mm Hg)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
