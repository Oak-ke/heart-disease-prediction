import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('heart.csv')
# Display the first few rows of the dataset
print(df.head())
print(df.shape)
print(df.info())
print(df.isnull().sum())
# Identify the categorical columns to encode
categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

# Perform one-hot encoding
# The 'get_dummies' function creates new columns for each category
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Display the first few rows of the new, encoded DataFrame
print(df_encoded.head())

# Print the shape of the new DataFrame to see how many new columns were created
print(f"Original shape: {df.shape}")
print(f"Encoded shape: {df_encoded.shape}")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
X = df_encoded.drop('target', axis=1) # Features
y = df_encoded['target'] # Target variable
# Assuming 'X' is your features and 'y' is your target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Data has been successfully split!")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Initialize the model with an increased number of iterations
# For example, let's try 1000 iterations
model = LogisticRegression(max_iter=1000)

# Now, fit the model as usual
model.fit(X_train, y_train)


from sklearn.metrics import accuracy_score
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

