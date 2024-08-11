import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings

# Configuration
warnings.filterwarnings('ignore')

# Load the datasets
train_df = pd.read_csv('./Training Dataset.csv')
test_df = pd.read_csv('./Test Dataset.csv')
sample_submission_df = pd.read_csv('./Sample_Submission.csv')

# Data preprocessing function
def preprocess_data(df):
    # Fill missing values for categorical columns with the mode
    categorical_cols = ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Credit_History']
    for col in categorical_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # Fill missing values for numerical columns with the mean
    numerical_cols = ['LoanAmount', 'Loan_Amount_Term', 'ApplicantIncome', 'CoapplicantIncome']
    for col in numerical_cols:
        df[col].fillna(df[col].mean(), inplace=True)

    # Encode categorical variables
    label_encoder = LabelEncoder()
    categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
    for col in categorical_cols:
        df[col] = label_encoder.fit_transform(df[col])

    return df

# Preprocess the data
train_df = preprocess_data(train_df)
test_df = preprocess_data(test_df)

# Prepare training and validation datasets
X = train_df.drop(columns=['Loan_ID', 'Loan_Status'])
y = train_df['Loan_Status'].apply(lambda x: 1 if x == 'Y' else 0)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(test_df.drop(columns=['Loan_ID']))

# Train a RandomForestClassifier model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {accuracy:.2f}")

# Predict on the test data
test_predictions = model.predict(X_test)
test_predictions = ['Y' if pred == 1 else 'N' for pred in test_predictions]

# Prepare the submission file
submission_df = pd.DataFrame({'Loan_ID': test_df['Loan_ID'], 'Loan_Status': test_predictions})
submission_df.to_csv('submission.csv', index=False)
print("Submission file has been created.")

# Display the submission file
submission_data = pd.read_csv('submission.csv')
print(submission_data.head())
