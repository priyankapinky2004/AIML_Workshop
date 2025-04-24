import numpy as np
from sklearn.neighbors import KNeighborsClassifier

x = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([0, 1, 1])

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x, y)

new_student = np.array([[4, 5]])
prediction = knn.predict(new_student)
print(f"The predicted class for the new student is: {prediction}")
