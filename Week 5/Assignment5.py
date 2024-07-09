import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
data = pd.read_csv(url)

print(data.head())

data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)

data['Age'].fillna(data['Age'].median(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
data['Fare'].fillna(data['Fare'].median(), inplace=True)

label_encoder = LabelEncoder()
data['Sex'] = label_encoder.fit_transform(data['Sex'])

one_hot_encoder = OneHotEncoder(drop='first')
embarked_encoded = one_hot_encoder.fit_transform(data[['Embarked']]).toarray()
embarked_df = pd.DataFrame(embarked_encoded, columns=['Embarked_'+col for col in one_hot_encoder.categories_[0][1:]])
data = pd.concat([data, embarked_df], axis=1)
data.drop(columns=['Embarked'], inplace=True)

data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

data['Fare'] = np.log1p(data['Fare'])

scaler = StandardScaler()
numerical_features = ['Age', 'Fare', 'FamilySize']
data[numerical_features] = scaler.fit_transform(data[numerical_features])

print(data.head())

plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

X = data.drop('Survived', axis=1)
y = data['Survived']

print(f"Feature set shape: {X.shape}")
print(f"Target variable shape: {y.shape}")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
