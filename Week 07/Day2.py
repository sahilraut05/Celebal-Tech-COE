# Day 2: Polynomial Regression and Ridge Regression

# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

# Polynomial Regression
def polynomial_regression(X, y, degree):
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)
    return model

# Ridge Regression
def ridge_regression(X, y, alpha):
    model = Ridge(alpha=alpha)
    model.fit(X, y)
    return model

# Example usage
X_poly = np.array([[1], [2], [3], [4], [5]])
y_poly = np.array([1, 4, 9, 16, 25])

X_ridge = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
y_ridge = np.array([2, 3, 4, 5, 6])

poly_model = polynomial_regression(X_poly, y_poly, degree=2)
ridge_model = ridge_regression(X_ridge, y_ridge, alpha=1.0)

print("Polynomial Regression Coefficients:", poly_model.coef_)
print("Ridge Regression Coefficients:", ridge_model.coef_)
