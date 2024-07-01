# Day 1: Simple Linear Regression and Multiple Linear Regression

# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Simple Linear Regression
def simple_linear_regression(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

# Multiple Linear Regression
def multiple_linear_regression(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

# Example usage
X_simple = np.array([[1], [2], [3], [4], [5]])
y_simple = np.array([1, 2, 3, 4, 5])

X_multiple = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
y_multiple = np.array([2, 3, 4, 5, 6])

simple_model = simple_linear_regression(X_simple, y_simple)
multiple_model = multiple_linear_regression(X_multiple, y_multiple)

print("Simple Linear Regression Coefficients:", simple_model.coef_)
print("Multiple Linear Regression Coefficients:", multiple_model.coef_)
