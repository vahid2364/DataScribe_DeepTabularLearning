#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 18:47:44 2024

@author: attari.v
"""

import pandas as pd
import numpy as np

# Load the DataFrame from a .pkl file
file_path = 'NbCrVWZr_data_stoic_creep_equil_filtered_v2.csv'  
df = pd.read_csv(file_path)

# Ensure that the loaded object is a DataFrame
if isinstance(df, pd.DataFrame):
    print("The loaded object is a DataFrame.")
    print(f"DataFrame shape: {df.shape}")
    print("DataFrame head:")
    print(df.head())
    print(f"The type of the loaded data is: {type(df)}")
    print(df.dtypes)
else:
    print("The loaded object is not a DataFrame.")


# Assuming df is your DataFrame
df = df.drop(df.columns[:2], axis=1)


# %% Remove object columns from the DataFrame

#print(df['2000 Creep Lim El'].unique())
#non_numeric_values = df[~df['2000 Creep Lim El'].apply(lambda x: isinstance(x, (int, float)))]

# Identify object columns
object_columns = df.select_dtypes(include=['object']).columns
print(f"Object columns to be removed: {object_columns}")

# Remove object columns from the DataFrame
df_cleaned = df.drop(columns=object_columns)

# Identify object columns
bool_columns = df.select_dtypes(include=['bool']).columns
print(f"Object columns to be removed: {bool_columns}")

# Remove object columns from the DataFrame
df_cleaned = df.drop(columns=bool_columns)

# Check the resulting DataFrame
print(f"DataFrame shape after removing object columns: {df_cleaned.shape}")
print(df_cleaned.head())

print(df_cleaned.dtypes)

# %%

# Optionally, print out just the object data type columns if there are any left
object_columns_remaining = df_cleaned.select_dtypes(include=['object']).columns
print(f"Object columns remaining: {object_columns_remaining}")

# Summary of the cleaned DataFrame
print(df_cleaned.info())
print(df_cleaned.describe())

# %% Find out if df has any NaN and their columns

# Assuming df_cleaned is your DataFrame
has_nan = df_cleaned.isnull().values.any()

print(f"Does the DataFrame have any NaN values? {has_nan}")

total_nan = df_cleaned.isnull().sum().sum()

print(f"Total number of NaN values in the DataFrame: {total_nan}")

nan_counts_per_column = df_cleaned.isnull().sum()

print("Number of NaN values per column:")
print(nan_counts_per_column)


# %% Replace Infs with NaNs

## Replace infinite values with NaN
df_cleaned.replace([np.inf, -np.inf], np.nan, inplace=True)

# %% Find out if df has any NaN and their columns

# Assuming df_cleaned is your DataFrame
has_nan = df_cleaned.isnull().values.any()

print(f"Does the DataFrame have any NaN values? {has_nan}")

total_nan = df_cleaned.isnull().sum().sum()

print(f"Total number of NaN values in the DataFrame: {total_nan}")

# %%

# Identify columns with sum of zero
cols_to_remove = df_cleaned.columns[df_cleaned.sum() == 0]

# Drop those columns
df_cleaned = df_cleaned.drop(columns=cols_to_remove)

# %%

# Function to remove outliers using a conservative IQR approach
def remove_outliers_conservative_iqr(df, factor):
    df_cleaned = df.copy()
    for col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
    return df_cleaned

# Remove outliers from the DataFrame
#df_cleaned2 = remove_outliers(df_cleaned)
df_cleaned2 = remove_outliers_conservative_iqr(df_cleaned, factor=10000000000000)

# Display the original and cleaned DataFrame shape
print("Original DataFrame shape:", df_cleaned.shape)
print("Cleaned DataFrame shape:", df_cleaned2.shape)

has_nan = df_cleaned2.isnull().values.any()

print(f"Does the DataFrame have any NaN values? {has_nan}")

total_nan = df_cleaned2.isnull().sum().sum()

print(f"Total number of NaN values in the DataFrame: {total_nan}")

if total_nan ==0:
    
    # Save the DataFrame to a CSV file
    file_path = 'IQR_dataframe.csv'  # Replace with your desired file path
    df_cleaned2.to_csv(file_path, index=False)
    
    print(f"IQR DataFrame saved to {file_path}")
    
    df_cleaned2.describe()
    
    
else:

    # %% Linear interpolation
    
    ## Linear interpolation
    df_cleaned_filled = df_cleaned2.interpolate(method='linear', axis=0)
    
    #print(df_cleaned.iloc[7671,7])
    #print(df_cleaned_filled.iloc[7671,7])
    
    #print(df_cleaned.iloc[:,7].idxmin())
    #print(df_cleaned_filled.iloc[:,7].idxmin())
    
    # Assuming df_cleaned is your DataFrame
    has_nan = df_cleaned_filled.isnull().values.any()
    
    print(f"Does the DataFrame have any NaN values? {has_nan}")
    
    total_nan = df_cleaned_filled.isnull().sum().sum()
    
    print(f"Total number of NaN values in the DataFrame: {total_nan}")
    
    # %%
    
    # Save the interpolated DataFrame to a CSV file
    file_path = 'interpolated_dataframe.csv'  # Replace with your desired file path
    df_cleaned_filled.to_csv(file_path, index=False)
    
    print(f"Interpolated DataFrame saved to {file_path}")
    
    df_cleaned_filled.describe()

