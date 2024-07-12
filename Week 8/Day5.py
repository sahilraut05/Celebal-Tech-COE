# Day 5: Bagging, Boosting, Stacking

# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

# Bagging
def bagging(X, y):
    model = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=10)
    model.fit(X, y)
    return model

# Boosting
def boosting(X, y):
    model = AdaBoostClassifier(n_estimators=50)
    model.fit(X, y)
    return model

# Stacking
def stacking(X, y):
    estimators = [
        ('dt', DecisionTreeClassifier()),
        ('lr', LogisticRegression())
    ]
    model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
    model.fit(X, y)
    return model

# Example usage
X = np.array([[1, 2], [1, 4], [1, 0],
              [4, 2], [4, 4], [4, 0]])
y = np.array([0, 0, 0, 1, 1, 1])

bagging_model = bagging(X, y)
boosting_model = boosting(X, y)
stacking_model = stacking(X, y)

print("Bagging Class Predictions:", bagging_model.predict(X))
print("Boosting Class Predictions:", boosting_model.predict(X))
print("Stacking Class Predictions:", stacking_model.predict(X))
