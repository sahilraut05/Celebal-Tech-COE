import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

# Configuration
warnings.filterwarnings("ignore")

# Load Data
data = pd.read_csv("./HousingData.csv")

# Initial Data Exploration
print(data.shape)
print(data.describe())
print(data.info())
print(data.isnull().sum())

# Fill missing values with 0
data.fillna(value=0, inplace=True)
print(data.isnull().sum())


# Plotting helper function
def plot_distribution(data, plot_type='boxplot'):
    fig, ax = plt.subplots(ncols=7, nrows=2, figsize=(15, 10))
    ax = ax.flatten()
    for index, (col, value) in enumerate(data.items()):
        if plot_type == 'boxplot':
            sns.boxplot(y=col, data=data, ax=ax[index], color="#A259FF")
        elif plot_type == 'distplot':
            sns.histplot(value, ax=ax[index], color="#A259FF", kde=True)
    plt.tight_layout(pad=0.5, w_pad=0.7, h_pad=5.0)
    plt.show()


# Plot initial data distributions
plot_distribution(data, plot_type='boxplot')
plot_distribution(data, plot_type='distplot')

# Normalize specific columns
cols_to_normalize = ["CRIM", "ZN", "TAX", "B"]
data[cols_to_normalize] = preprocessing.MinMaxScaler().fit_transform(data[cols_to_normalize])

# Standardize specific columns
scaler = preprocessing.StandardScaler()
data[cols_to_normalize] = scaler.fit_transform(data[cols_to_normalize])

# Plot normalized data distributions
plot_distribution(data, plot_type='distplot')

# Correlation heatmap
plt.figure(figsize=(20, 10))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.show()

# Regression plots
fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(20, 10))
sns.regplot(y=data['MEDV'], x=data['LSTAT'], color="#A259FF", ax=ax[0])
sns.regplot(y=data['MEDV'], x=data['RM'], color="#A259FF", ax=ax[1])
plt.show()

# Prepare data for modeling
X = data.drop(columns=['MEDV', 'RAD'])
y = data['MEDV']

# Insert columns for model predictions
models = ["LinearRegression", "DecisionTreeRegressor", "RandomForestRegressor", "ExtraTreesRegressor"]
for model_name in models:
    data[model_name] = 0


# Training and evaluating models
def train_and_evaluate(model, X, y, colnum):
    x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=42)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    X_features = X.columns  # Ensure we use the correct features for prediction

    for i in range(len(data)):
        data.iloc[i, colnum] = model.predict([data.loc[i, X_features]])[0]

    cv_score = np.abs(np.mean(cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=5)))

    print(f"Model Report for {model.__class__.__name__}")
    print("MSE:", mean_squared_error(y_test, predictions))
    print('CV Score:', cv_score)

    # Plot feature importance or coefficients
    if hasattr(model, 'coef_'):
        pd.Series(model.coef_, X.columns).sort_values().plot(kind='bar', title='Model Coefficients')
    elif hasattr(model, 'feature_importances_'):
        pd.Series(model.feature_importances_, X.columns).sort_values(ascending=False).plot(kind='bar',
                                                                                           title='Feature Importance')
    plt.show()


# Linear Regression
train_and_evaluate(LinearRegression(), X, y, data.columns.get_loc("LinearRegression"))

# Decision Tree Regressor
train_and_evaluate(DecisionTreeRegressor(), X, y, data.columns.get_loc("DecisionTreeRegressor"))

# Random Forest Regressor
train_and_evaluate(RandomForestRegressor(), X, y, data.columns.get_loc("RandomForestRegressor"))

# Extra Trees Regressor
train_and_evaluate(ExtraTreesRegressor(), X, y, data.columns.get_loc("ExtraTreesRegressor"))
