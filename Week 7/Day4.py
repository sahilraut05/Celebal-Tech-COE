# Day 4: Logistic Regression and Naive Bayes

# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

# Logistic Regression
def logistic_regression(X, y):
    model = LogisticRegression()
    model.fit(X, y)
    return model

# Naive Bayes
def naive_bayes(X, y):
    model = GaussianNB()
    model.fit(X, y)
    return model

# Example usage
X_logistic = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
y_logistic = np.array([0, 0, 1, 1, 1])

X_naive = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
y_naive = np.array([0, 0, 1, 1, 1])

logistic_model = logistic_regression(X_logistic, y_logistic)
naive_bayes_model = naive_bayes(X_naive, y_naive)

print("Logistic Regression Coefficients:", logistic_model.coef_)
print("Naive Bayes Class Prior:", naive_bayes_model.class_prior_)
