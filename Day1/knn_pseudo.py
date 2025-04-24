from collections import Counter
import math 
data = [
    ((160, 2), "Apple"),
    ((170, 3), "Apple"),
    ((150, 1), "Orange"),
    ((180, 4), "Apple"),
    ((155, 2), "Orange"),
    ((165, 3), "Apple"),
    ((145, 1), "Orange"),
    ((175, 4), "Apple"),
]

def euclidean_distance(point1, point2):
    """Calculate the Euclidean distance between two points."""
    #formula for euclidean distance
    # Euclidean distance formula: sqrt((x2 - x1)^2 + (y2 - y1)^2 + ...)
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))


def knn(new_point,k):

    distances = []
    # Calculate the distance from the new point to all other points
    for features, point in data:
        distance = euclidean_distance(features, new_point)
        print(f"distance from {new_point} to {features} is: {distance}")
        distances.append((distance, point))
    print(f"the distances are: {distances}")

    distances.sort()

    neighbors = [label for _, label in distances[:k]]
    print(f"the {k} nearest neighbors are: {neighbors}")
    vote = Counter(neighbors)
    print(f"the votes are: {vote}")
    most_common = vote.most_common(1)[0][0]
    print(f"the most common label is: {most_common}")
    print(f"the most common label among the {k} nearest neighbors is: {most_common}")

    return most_common

new_point = (160, 5)
k = 4
neighbor = knn(new_point, k)