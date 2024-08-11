# Day 1: K-Means Clustering

# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# K-Means Clustering
def kmeans_clustering(X, n_clusters):
    model = KMeans(n_clusters=n_clusters)
    model.fit(X)
    return model

# Example usage
X = np.array([[1, 2], [1, 4], [1, 0],
              [4, 2], [4, 4], [4, 0]])

kmeans_model = kmeans_clustering(X, n_clusters=2)

print("Cluster Centers:", kmeans_model.cluster_centers_)
print("Labels:", kmeans_model.labels_)

# Plotting the results
plt.scatter(X[:, 0], X[:, 1], c=kmeans_model.labels_, cmap='viridis')
plt.scatter(kmeans_model.cluster_centers_[:, 0], kmeans_model.cluster_centers_[:, 1], s=300, c='red')
plt.title('K-Means Clustering')
plt.show()
