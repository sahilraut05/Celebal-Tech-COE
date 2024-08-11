# Day 3: DBSCAN

# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# DBSCAN Clustering
def dbscan_clustering(X, eps, min_samples):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    model.fit(X)
    return model

# Example usage
X = np.array([[1, 2], [1, 4], [1, 0],
              [4, 2], [4, 4], [4, 0]])

dbscan_model = dbscan_clustering(X, eps=1.5, min_samples=2)

print("Labels:", dbscan_model.labels_)

# Plotting the results
plt.scatter(X[:, 0], X[:, 1], c=dbscan_model.labels_, cmap='viridis')
plt.title('DBSCAN Clustering')
plt.show()
