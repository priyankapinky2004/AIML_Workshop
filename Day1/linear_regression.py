from sklearn.linear_model import LinearRegression

dataset = {
    "SqureFeet": [1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400],
    "Price": [300000, 320000, 340000, 360000, 380000, 390000, 400000, 410000, 430000, 450000]
}

x = [[i] for i in dataset["SqureFeet"]]
y = dataset["Price"]

model = LinearRegression()

model.fit(x,y)

predicted_price = model.predict([[4000]])

print(f"Predicted price for 4000 square feet: {predicted_price}")

print(f"Model Coefficient: {model.coef_[0]}")
print(f"Model Intercept: {model.intercept_}")
print(f"Model Score: {model.score(x, y)}")
