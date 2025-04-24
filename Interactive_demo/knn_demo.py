

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap

# Create a student study habits dataset
def create_student_dataset():
    # Study hours and social media hours
    X = np.array([
        [1, 5],    # Study hours, Social media hours
        [2, 4],
        [1.5, 5],
        [3, 4],
        [2, 2.5],
        [4, 1],
        [5, 1],
        [3.5, 2],
        [2, 4.5],
        [5, 3],
        [1, 6],
        [4, 2]
    ])

    # Outcome: 0 = Fail, 1 = Pass
    y = np.array([0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1])

    return X, y

# Create and visualize dataset
X, y = create_student_dataset()

# Print dataset
print("Student Study Dataset:")
print("ID | Study Hrs | Social Media Hrs | Result")
print("-" * 45)
for i in range(len(X)):
    result = "Pass" if y[i] == 1 else "Fail"
    print(f"{i+1:2d} | {X[i,0]:9.1f} | {X[i,1]:15.1f} | {result}")

# Create and train KNN model
k = 3
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X, y)

# Test on a new student
new_student = np.array([[3, 3]])  # 3 hours study, 3 hours social media
prediction = knn.predict(new_student)
result = "Pass" if prediction[0] == 1 else "Fail"
print(f"\nPrediction for new student (study={new_student[0,0]}h, social={new_student[0,1]}h): {result}")

# Function to plot decision boundary
def plot_decision_boundary(X, y, model, k_value):
    h = 0.1  # step size in the mesh

    # Create color maps for the plot
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00'])

    # Plot the decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.3)

    # Plot the training points
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
                edgecolor='k', s=100, label=["Fail", "Pass"])

    # Add ID labels to points
    for i in range(len(X)):
        plt.text(X[i,0]+0.1, X[i,1], f"{i+1}", fontsize=10)

    # Mark the test point
    plt.scatter(new_student[0,0], new_student[0,1],
                marker='*', s=200, edgecolor='k',
                color='blue', label='New Student')

    # Find K nearest neighbors
    distances, indices = knn.kneighbors(new_student)

    # Draw lines to k nearest neighbors
    for i in range(len(indices[0])):
        idx = indices[0][i]
        plt.plot([new_student[0,0], X[idx,0]], [new_student[0,1], X[idx,1]], 'b--')
        # Add distance label
        mid_x = (new_student[0,0] + X[idx,0]) / 2
        mid_y = (new_student[0,1] + X[idx,1]) / 2
        plt.text(mid_x-0.2, mid_y, f"{distances[0,i]:.1f}",
                bbox=dict(facecolor='white', alpha=0.7))

    # Set plot labels and title
    plt.xlabel('Study Hours', fontsize=12)
    plt.ylabel('Social Media Hours', fontsize=12)
    plt.title(f'KNN Classifier (k={k_value}) - Student Pass/Fail Prediction', fontsize=14)
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Visualize the KNN classification
plot_decision_boundary(X, y, knn, k)

# Show effect of different k values
print("\nEffect of different k values:")
for k_value in [1, 3, 5, 7]:
    model = KNeighborsClassifier(n_neighbors=k_value)
    model.fit(X, y)
    prediction = model.predict(new_student)
    result = "Pass" if prediction[0] == 1 else "Fail"
    print(f"With k={k_value}, prediction: {result}")

    # Plot decision boundary for this k
    plot_decision_boundary(X, y, model, k_value)
     