import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
import warnings

warnings.filterwarnings("ignore")

# Load the dataset
data = pd.read_csv("https://www.kaggle.com/competitions/anomaly-detection/download/data")

# Display the first few rows of the dataset
print(data.head())

# Data Analysis
print(data.shape)
print(data.describe())
print(data.info())
print(data.isnull().sum())

# Data Preprocessing
data.fillna(value=0, inplace=True)
print(data.isnull().sum())

# Exploratory Data Analysis (EDA)
# Boxplot to check for outliers
fig, ax = plt.subplots(ncols=4, nrows=2, figsize=(15, 10))
index = 0
ax = ax.flatten()
for col, value in data.items():
    sns.boxplot(y=col, data=data, ax=ax[index], color="#A259FF")
    index += 1
plt.tight_layout(pad=0.5, w_pad=0.7, h_pad=5.0)
plt.show()

# Distplot to check data distribution
fig, ax = plt.subplots(ncols=4, nrows=2, figsize=(15, 10))
index = 0
ax = ax.flatten()
for col, value in data.items():
    sns.histplot(value, kde=True, ax=ax[index], color="#A259FF")
    index += 1
plt.tight_layout(pad=0.5, w_pad=0.7, h_pad=5.0)
plt.show()

# Min-Max Normalization
cols = data.columns
scaler = MinMaxScaler()
data[cols] = scaler.fit_transform(data[cols])

# Standardization
scaler = StandardScaler()
data[cols] = scaler.fit_transform(data[cols])

# Correlation Matrix
corr = data.corr()
plt.figure(figsize=(20, 10))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()

# Feature and Label Split (Assuming 'label' is the target column)
X = data.drop(columns=['label'], axis=1)
y = data['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training: Isolation Forest
model = IsolationForest(contamination=0.1)
model.fit(X_train)

# Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Convert predictions from {-1, 1} to {0, 1}
y_train_pred = [1 if x == -1 else 0 for x in y_train_pred]
y_test_pred = [1 if x == -1 else 0 for x in y_test_pred]

# Evaluation
print("Training Classification Report")
print(classification_report(y_train, y_train_pred))
print("Testing Classification Report")
print(classification_report(y_test, y_test_pred))

# Confusion Matrix
print("Training Confusion Matrix")
print(confusion_matrix(y_train, y_train_pred))
print("Testing Confusion Matrix")
print(confusion_matrix(y_test, y_test_pred))

# Visualization
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Training data visualization
sns.scatterplot(x=X_train.iloc[:, 0], y=X_train.iloc[:, 1], hue=y_train_pred, ax=ax[0], palette='viridis')
ax[0].set_title("Training Data - Anomalies vs Normal")

# Testing data visualization
sns.scatterplot(x=X_test.iloc[:, 0], y=X_test.iloc[:, 1], hue=y_test_pred, ax=ax[1], palette='viridis')
ax[1].set_title("Testing Data - Anomalies vs Normal")

plt.tight_layout()
plt.show()
