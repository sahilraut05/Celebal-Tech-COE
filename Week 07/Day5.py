# Day 5: k-Nearest Neighbors and Decision Trees

# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# k-Nearest Neighbors
def knn(X, y, n_neighbors):
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X, y)
    return model

# Decision Trees
def decision_tree(X, y):
    model = DecisionTreeClassifier()
    model.fit(X, y)
    return model

# Example usage
X_knn = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
y_knn = np.array([0, 0, 1, 1, 1])

X_tree = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
y_tree = np.array([0, 0, 1, 1, 1])

knn_model = knn(X_knn, y_knn, n_neighbors=3)
tree_model = decision_tree(X_tree, y_tree)

print("k-NN Class Predictions:", knn_model.predict(X_knn))
print("Decision Tree Class Predictions:", tree_model.predict(X_tree))
