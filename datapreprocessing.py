# 01_data_preprocessing.py

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np

df = pd.read_excel('dataset.xlsx')

# Display the first few rows of the dataframe
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Normalize specified columns
df[['Temperature', 'Traffic Volume']] = scaler.fit_transform(df[['Temperature', 'Traffic Volume']])

# One-hot encode categorical variables
df = pd.get_dummies(df, columns=['Location', 'Day of Week'])

# Extract new time-based features
df['Hour'] = df['Time'].apply(lambda x: x.hour)
df['Day'] = df['Date'].apply(lambda x: x.day)
df['Month'] = df['Date'].apply(lambda x: x.month)
df['Year'] = df['Date'].apply(lambda x: x.year)

# Create lagged features for traffic volume
df['Traffic_Lag_1'] = df['Traffic Volume'].shift(1)
df['Traffic_Lag_2'] = df['Traffic Volume'].shift(2)
df['Traffic_Lag_3'] = df['Traffic Volume'].shift(3)

# Define the size for the training set
train_size = int(len(df) * 0.8)

# Split the data chronologically for time series analysis
train = df[:train_size]
test = df[train_size:]

# Optionally, create a validation set from the training data
val_size = int(len(train) * 0.2)
train, val = train[:train_size], train[train_size:]

# Display the shapes of the splits
print("Training set shape:", train.shape)
print("Validation set shape:", val.shape)
print("Testing set shape:", test.shape)

# Save preprocessed data to new files (optional)
train.to_csv('train_preprocessed.csv', index=False)
val.to_csv('val_preprocessed.csv', index=False)
test.to_csv('test_preprocessed.csv', index=False)

