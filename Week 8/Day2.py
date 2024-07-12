# Day 2: Hierarchical Clustering

# Import necessary libraries
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt

# Hierarchical Clustering
def hierarchical_clustering(X):
    linked = linkage(X, 'single')
    return linked

# Example usage
X = np.array([[1, 2], [1, 4], [1, 0],
              [4, 2], [4, 4], [4, 0]])

linked = hierarchical_clustering(X)

# Plotting the dendrogram
plt.figure(figsize=(10, 7))
dendrogram(linked,
           orientation='top',
           distance_sort='descending',
           show_leaf_counts=True)
plt.title('Hierarchical Clustering Dendrogram')
plt.show()
