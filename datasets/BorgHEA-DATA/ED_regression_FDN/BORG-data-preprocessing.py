#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 13:11:51 2024

@author: attari.v
"""

import pandas as pd
import re

csv_file_path = '../MPEA_dataset.csv'
df = pd.read_csv(csv_file_path, usecols=lambda column: column not in ['Unnamed: 0.3'])

# Calculate the percentage of missing data for each column
missing_percentage = (df.isnull().sum() / len(df)) * 100

# Print the result, showing columns with missing data percentages
print("Percentage of Missing Data in Each Column:")
print(missing_percentage)


# %%

# Step 1: Extract elements and compositions using regex
def parse_formula(formula):
    pattern = r'([A-Z][a-z]*)([0-9]*\.?[0-9]*)'  # Match element symbols and compositions
    matches = re.findall(pattern, formula)
    return {element: float(composition) if composition else 1.0 for element, composition in matches}

# Step 2: Apply parsing function to the column
parsed_data = df['FORMULA'].apply(parse_formula)

composition_df = pd.DataFrame(parsed_data.tolist()).fillna(0)  # Fill missing compositions with 0

# Concatenate the composition_df and original df along columns
merged_df = pd.concat([composition_df, df], axis=1)

# Display the merged DataFrame
print(merged_df)

merged_df.describe()

# Ensure your dataset only includes numerical columns
numerical_data = merged_df.select_dtypes(include=['float', 'int'])

# Save the DataFrame to a CSV file
numerical_data.to_csv('../Borg_df_updated.csv', index=False)

# Optional: Confirm the save
print("DataFrame saved to 'numerical_data.csv'")

# %%

# Count NaNs
for idx in numerical_data.columns:
    nan_count = numerical_data[idx].isna().sum()
    print(f"Number of NaNs: {nan_count} in column {idx} ")

pause
# %%

import seaborn as sns
import matplotlib.pyplot as plt

for idx in merged_df.columns:
    print(idx)
    
    # Check if the column is categorical
    if merged_df[idx].dtype == 'object' or merged_df[idx].nunique() < 10:
        print(f"Skipping {idx} as it is categorical.")
        continue  # Skip categorical columns
    
    # Plot KDE for numerical columns
    sns.kdeplot(data=merged_df, x=idx, log_scale=True)
    plt.title(f"KDE Plot for {idx}")
    plt.show()    
    
# %%



# %%

import seaborn as sns
import matplotlib.pyplot as plt

# Ensure your dataset only includes numerical columns
numerical_data = merged_df.select_dtypes(include=['float', 'int'])

# Create the pair plot
sns.pairplot(numerical_data, diag_kind="kde", corner=True)

# Show the plot
plt.show()    

# %%

for idx in numerical_data.columns:
    print(idx)
    
    # Check if the column is categorical
    if numerical_data[idx].dtype == 'object' or numerical_data[idx].nunique() < 10:
        print(f"Skipping {idx} as it is categorical.")
        continue  # Skip categorical columns
    
    # Plot KDE for numerical columns
    sns.kdeplot(data=numerical_data, x=idx, log_scale=False)
    plt.title(f"KDE Plot for {idx}")
    plt.show() 


pause

# Define the remaining features
columns_to_keep = [
#    'Nb', 'Cr', 'V', 'W', 'Zr', 'Creep Merit', '500 Min Creep CB [1/s]', '1300 Min Creep CB [1/s]'   # EQ 1273K Density (g/cc), EQ 1273K THCD (W/mK), YS 1000C PRIOR, 1000 Min Creep NH [1/s], Kou Criteria
#    'Nb', 'Cr', 'V', 'W', 'Zr', 'YS 1000C PRIOR', 'EQ 1273K Density (g/cc)', 'EQ 1273K THCD (W/mK)', 'Kou Criteria', '1300 Min Creep CB [1/s]',   # EQ 1273K Density (g/cc), EQ 1273K THCD (W/mK), YS 1000C PRIOR, 1000 Min Creep NH [1/s], Kou Criteria
#    'Nb', 'Cr', 'V', 'W', 'Zr', 'YS 1000C PRIOR', 'YS 1500C PRIOR', 'Pugh_Ratio_PRIOR', '1000 Min Creep CB [1/s]', '1500 Min Creep CB [1/s]'   # EQ 1273K Density (g/cc), EQ 1273K THCD (W/mK), YS 1000C PRIOR, 1000 Min Creep NH [1/s], Kou Criteria
##    'Nb', 'Cr', 'V', 'W', 'Zr', 'YS 1000C PRIOR', 'Pugh_Ratio_PRIOR', '1000 Min Creep CB [1/s]', 'Kou Criteria', 'Creep Merit'   # EQ 1273K Density (g/cc), EQ 1273K THCD (W/mK), YS 1000C PRIOR, 1000 Min Creep NH [1/s], Kou Criteria
     'band gap (eV)', 'density (g/cm³)', 'energy above hull (eV/atom)', 'stable',
     'volume (Å³)', 'energy per atom (eV/atom)', 'formation energy per atom (eV/atom)',
     'crystal system', 'oxide type', 'elements', 'enthalpy per atom (eV/atom)', 
     'scintillation attenuation length (cm)'
]

df = df[columns_to_keep]