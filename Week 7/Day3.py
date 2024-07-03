# Day 3: Lasso Regression and ElasticNet Regression

# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, ElasticNet

# Lasso Regression
def lasso_regression(X, y, alpha):
    model = Lasso(alpha=alpha)
    model.fit(X, y)
    return model

# ElasticNet Regression
def elasticnet_regression(X, y, alpha, l1_ratio):
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
    model.fit(X, y)
    return model

# Example usage
X_lasso = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
y_lasso = np.array([2, 3, 4, 5, 6])

X_elasticnet = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
y_elasticnet = np.array([2, 3, 4, 5, 6])

lasso_model = lasso_regression(X_lasso, y_lasso, alpha=1.0)
elasticnet_model = elasticnet_regression(X_elasticnet, y_elasticnet, alpha=1.0, l1_ratio=0.5)

print("Lasso Regression Coefficients:", lasso_model.coef_)
print("ElasticNet Regression Coefficients:", elasticnet_model.coef_)
