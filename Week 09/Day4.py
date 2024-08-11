# Ensemble Methods
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

random_forest = RandomForestClassifier()
random_forest.fit(X_train, y_train)
print("Random Forest accuracy:", random_forest.score(X_test, y_test))

gradient_boosting = GradientBoostingClassifier()
gradient_boosting.fit(X_train, y_train)
print("Gradient Boosting accuracy:", gradient_boosting.score(X_test, y_test))
