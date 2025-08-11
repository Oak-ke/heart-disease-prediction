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
