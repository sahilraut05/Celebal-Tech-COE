# Day 6: Random Forest and Support Vector Machines

# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Random Forest
def random_forest(X, y, n_estimators):
    model = RandomForestClassifier(n_estimators=n_estimators)
    model.fit(X, y)
    return model

# Support Vector Machines
def support_vector_machine(X, y, kernel):
    model = SVC(kernel=kernel)
    model.fit(X, y)
    return model

# Example usage
X_forest = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
y_forest = np.array([0, 0, 1, 1, 1])

X_svm = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
y_svm = np.array([0, 0, 1, 1, 1])

forest_model = random_forest(X_forest, y_forest, n_estimators=10)
svm_model = support_vector_machine(X_svm, y_svm, kernel='linear')

print("Random Forest Class Predictions:", forest_model.predict(X_forest))
print("SVM Class Predictions:", svm_model.predict(X_svm))
