import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import Lasso, Ridge, ElasticNet

X_classification = np.array([[1, 2], [1, 4], [1, 0],
                             [4, 2], [4, 4], [4, 0]])
y_classification = np.array([0, 0, 0, 1, 1, 1])

X_regression = np.array([[1, 2], [1, 4], [1, 0],
                         [4, 2], [4, 4], [4, 0]])
y_regression = np.array([1, 2, 3, 4, 5, 6])

random_forest_model = RandomForestClassifier(n_estimators=10)
random_forest_model.fit(X_classification, y_classification)

gradient_boosting_model = GradientBoostingClassifier(n_estimators=50)
gradient_boosting_model.fit(X_classification, y_classification)

lasso_model = Lasso(alpha=1.0)
lasso_model.fit(X_regression, y_regression)

ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_regression, y_regression)

elasticnet_model = ElasticNet(alpha=1.0, l1_ratio=0.5)
elasticnet_model.fit(X_regression, y_regression)

print("Random Forest Class Predictions:", random_forest_model.predict(X_classification))
print("Gradient Boosting Class Predictions:", gradient_boosting_model.predict(X_classification))
print("Lasso Regression Coefficients:", lasso_model.coef_)
print("Ridge Regression Coefficients:", ridge_model.coef_)
print("ElasticNet Regression Coefficients:", elasticnet_model.coef_)
