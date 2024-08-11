# Day 4: Gaussian Mixture Models

# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# Gaussian Mixture Models
def gaussian_mixture_models(X, n_components):
    model = GaussianMixture(n_components=n_components)
    model.fit(X)
    return model

# Example usage
X = np.array([[1, 2], [1, 4], [1, 0],
              [4, 2], [4, 4], [4, 0]])

gmm_model = gaussian_mixture_models(X, n_components=2)

print("Means:", gmm_model.means_)
print("Covariances:", gmm_model.covariances_)

# Plotting the results
labels = gmm_model.predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.title('Gaussian Mixture Models Clustering')
plt.show()
