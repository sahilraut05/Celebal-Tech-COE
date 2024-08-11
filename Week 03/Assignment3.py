import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from scipy import stats

# Step 1: Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
data = pd.read_csv(url, header=None, names=column_names)

# Step 2: Data Cleaning

data.drop_duplicates(inplace=True)

data = data[(np.abs(stats.zscore(data.select_dtypes(include=[np.number]))) < 3).all(axis=1)]

# Step 3: Feature Scaling

scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(data.select_dtypes(include=[np.number]))

scaler = StandardScaler()
standardized_data = scaler.fit_transform(data.select_dtypes(include=[np.number]))

# Step 4: Encoding Categorical Variables

label_encoder = LabelEncoder()
data['species'] = label_encoder.fit_transform(data['species'])

# Step 5: Feature Engineering


data['sepal_area'] = data['sepal_length'] * data['sepal_width']
data['petal_area'] = data['petal_length'] * data['petal_width']

X = data.drop('species', axis=1)
y = data['species']

selector = SelectKBest(score_func=f_classif, k=3)
X_new = selector.fit_transform(X, y)

# Step 6: Visualization

# Histogram
plt.figure(figsize=(10, 6))
sns.histplot(data['sepal_length'], kde=True)
plt.title('Distribution of Sepal Length')
plt.xlabel('Sepal Length')
plt.ylabel('Frequency')
plt.show()

# Scatter Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='sepal_length', y='sepal_width', data=data, hue='species')
plt.title('Scatter Plot of Sepal Length vs Sepal Width')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.show()

# Advanced Visualization: Pair Plot
plt.figure(figsize=(10, 6))
sns.pairplot(data, hue='species')
plt.suptitle('Pair Plot of Iris Dataset', y=1.02)
plt.show()

# Advanced Visualization: Correlation Heatmap
plt.figure(figsize=(10, 6))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()
