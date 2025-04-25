# random_forest.py

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# 2. Create a DataFrame for visualization
df = pd.DataFrame(X, columns=feature_names)
df['target'] = [target_names[i] for i in y]

print("🌸 Iris Dataset Preview:")
print(df.head())

# 3. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Train the Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 5. Predict on test set
y_pred = rf_model.predict(X_test)

# 6. Evaluate the model
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))
print(f"✅ Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# 7. Feature Importances
feature_imp = pd.Series(rf_model.feature_importances_, index=feature_names).sort_values(ascending=False)

# Plotting feature importances
plt.figure(figsize=(8, 5))
sns.barplot(x=feature_imp, y=feature_imp.index)
plt.title("🔍 Feature Importance (Iris Dataset)")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.show()
