# Bagging and Variations
from sklearn.ensemble import BaggingClassifier

bagging = BaggingClassifier(base_estimator=decision_tree, n_estimators=50, random_state=42)
bagging.fit(X_train, y_train)
print("Bagging accuracy:", bagging.score(X_test, y_test))

# Random Subspace Method and Random Patches Method can be achieved by tweaking the parameters of BaggingClassifier
subspace_method = BaggingClassifier(base_estimator=decision_tree, n_estimators=50, max_features=0.5, random_state=42)
subspace_method.fit(X_train, y_train)
print("Random Subspace Method accuracy:", subspace_method.score(X_test, y_test))

random_patches_method = BaggingClassifier(base_estimator=decision_tree, n_estimators=50, max_samples=0.5, max_features=0.5, random_state=42)
random_patches_method.fit(X_train, y_train)
print("Random Patches Method accuracy:", random_patches_method.score(X_test, y_test))
