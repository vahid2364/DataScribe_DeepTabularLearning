#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 13:11:51 2024

@author: attari.v
"""

import os
import pandas as pd
import re

os.makedirs('processed_data',exist_ok=True)

# %%

csv_file_path = 'MPEA_dataset.csv'
df = pd.read_csv(csv_file_path, usecols=lambda column: column not in ['Unnamed: 0.3'])

# Calculate the percentage of missing data for each column
missing_percentage = (df.isnull().sum() / len(df)) * 100

# Print the result, showing columns with missing data percentages
print("")
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
print(" ")
print("Display the updated data")
print(merged_df)

merged_df.describe()

# Ensure your dataset only includes numerical columns
numerical_data = merged_df.select_dtypes(include=['float', 'int'])

# Save the DataFrame to a CSV file
numerical_data.to_csv('processed_data/MPEA_data_processed.csv', index=False)

# Optional: Confirm the save
print("DataFrame saved to 'MPEA_data_processed.csv'")

# %%

# Count NaNs
for idx in numerical_data.columns:
    nan_count = numerical_data[idx].isna().sum()
    print(f"Number of NaNs: {nan_count} in column {idx} ")
    
# %% Latex Table for paper

# Select only the required columns
selected_columns = [
'Al', 'Co', 'Fe', 'Ni', 'Si', 'Cr', 'Mn', 'Nb', 'Mo', 'Ti', 'Cu', 'C',
       'V', 'Zr', 'B', 'Nd', 'Y', 'Sn', 'Li', 'Mg', 'Zn', 'Sc', 'Ta', 'Hf',
       'W', 'Re', 'Ca', 'Pd', 'Ga', 'Ag',
       #'PROPERTY: Microstructure', 
       #'PROPERTY: BCC/FCC/other', 
       'PROPERTY: grain size ($\mu$m)',
       'PROPERTY: Exp. Density (g/cm$^3$)',
       'PROPERTY: Calculated Density (g/cm$^3$)', 
       'PROPERTY: HV',
       #'PROPERTY: Type of test', 
       'PROPERTY: Test temperature ($^\circ$C)',
       'PROPERTY: YS (MPa)', 'PROPERTY: UTS (MPa)', 
       'PROPERTY: Elongation (%)',
       'PROPERTY: Elongation plastic (%)',
       'PROPERTY: Exp. Young modulus (GPa)',
       'PROPERTY: Calculated Young modulus (GPa)',
       'PROPERTY: O content (wppm)', 
       'PROPERTY: N content (wppm)',
       'PROPERTY: C content (wppm)',
]

# Calculate extended statistical summary with skewness and kurtosis
summary_df = merged_df[selected_columns].agg(['mean', 'std', 'min', 'max', 'median', 'skew', 'kurtosis']).T

# Optionally, include 25%, 50%, and 75% percentiles
percentiles = merged_df[selected_columns].describe(percentiles=[0.25, 0.5, 0.75]).T
summary_df['25%'] = percentiles['25%']
summary_df['75%'] = percentiles['75%']

# Generate LaTeX table with skewness and kurtosis, using .2g format
latex_table = r"""
\begin{table*}[!h]
    \centering
    \caption{MPEA dataset: Extended Statistical Summary of the Dataset including Skewness and Kurtosis}
    \scriptsize
    \label{tab:extended_data_stats_skew_kurt}
    \vspace{-3mm}
    \begin{tabular}{lccccccccc}
        \toprule
        \textbf{Feature} & \textbf{Mean} & \textbf{Std. Dev.} & \textbf{Min} & \textbf{Max} & \textbf{Median} & \textbf{25\%} & \textbf{75\%} & \textbf{Skewness} & \textbf{Kurtosis} \\
        \midrule
"""

# Populate the LaTeX table with the extended statistical summary using .2g format
for feature in summary_df.index:
    latex_table += f"        {feature} & {summary_df.loc[feature, 'mean']:.1f} & {summary_df.loc[feature, 'std']:.1f} & {summary_df.loc[feature, 'min']:.1f} & {summary_df.loc[feature, 'max']:.1f} & {summary_df.loc[feature, 'median']:.1f} & {summary_df.loc[feature, '25%']:.1f} & {summary_df.loc[feature, '75%']:.1f} & {summary_df.loc[feature, 'skew']:.1f} & {summary_df.loc[feature, 'kurtosis']:.1f} \\\\\n"
    #latex_table += f"        {feature} & {summary_df.loc[feature, 'mean']:.2f} & {summary_df.loc[feature, 'std']:.2f} & {summary_df.loc[feature, 'min']:.2f} & {summary_df.loc[feature, 'max']:.2f} & {summary_df.loc[feature, 'median']:.2f} & {summary_df.loc[feature, '25%']:.2f} & {summary_df.loc[feature, '75%']:.2f} & {summary_df.loc[feature, 'skew']:.2f} & {summary_df.loc[feature, 'kurtosis']:.2f} \\\\\n"
# End of the table
latex_table += r"""
        \bottomrule
    \end{tabular}
\end{table*}
"""

# Print LaTeX table
print(latex_table)

# %%

import os

os.makedirs('tSNE',exist_ok=True)

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import re
from matplotlib.colors import LogNorm

# Remove unacceptable characters from the filename

input_columns = [
    'Al', 'Co', 'Fe', 'Ni', 'Si', 'Cr', 'Mn', 'Nb', 'Mo', 'Ti', 'Cu', 'C',
       'V', 'Zr', 'B', 'Nd', 'Y', 'Sn', 'Li', 'Mg', 'Zn', 'Sc', 'Ta', 'Hf',
       'W', 'Re', 'Ca', 'Pd', 'Ga', 'Ag',
]

X = merged_df[input_columns]

property_columns = [
       #'PROPERTY: Microstructure', 
       #'PROPERTY: BCC/FCC/other', 
       'PROPERTY: grain size ($\mu$m)',
       'PROPERTY: Exp. Density (g/cm$^3$)',
       'PROPERTY: Calculated Density (g/cm$^3$)', 
       'PROPERTY: HV',
       #'PROPERTY: Type of test', 
       'PROPERTY: Test temperature ($^\circ$C)',
       'PROPERTY: YS (MPa)', 'PROPERTY: UTS (MPa)', 
       'PROPERTY: Elongation (%)',
       'PROPERTY: Elongation plastic (%)',
       'PROPERTY: Exp. Young modulus (GPa)',
       'PROPERTY: Calculated Young modulus (GPa)',
       'PROPERTY: O content (wppm)', 
       'PROPERTY: N content (wppm)',
       'PROPERTY: C content (wppm)',
    ]

properties = merged_df[property_columns]

idx=1
for prop in properties:
    print(prop)
    sanitized_filename = re.sub(r'[<>:"/\\|?*]', '_', str(prop)) + '.jpg'

    y = df[prop]

    # Standardize the input features (important for t-SNE)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply t-SNE to reduce dimensions to 2D
    tsne = TSNE(n_components=2, perplexity=60, random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)

    # Remove unacceptable characters from the filename
    sanitized_filename = re.sub(r'[<>:"/\\|?*]', '_', str(prop)) + '.jpg'

    # Determine if log scale is needed
    if y.min() > 0 and (y.max() / y.min() > 10):  # Log scale for large dynamic range
        print('y.max() / y.min()',y.max() / y.min())
        norm = LogNorm(vmin=y.min(), vmax=y.max())
        colorbar_label = f'Log Scale: {prop}'
    else:
        norm = None
        colorbar_label = prop

    # Plot the results
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', s=20, alpha=0.95, norm=norm)
    cbar = plt.colorbar(scatter, label=colorbar_label)

    # Remove frame and axes
    plt.axis('off')  # Turn off the axes
    plt.gca().set_frame_on(False)  # Remove the frame

    # Save the plot
    plt.tight_layout()
    plt.savefig('tSNE/tSNE_'+str(idx)+'.jpg', dpi=300, bbox_inches='tight', pad_inches=0)
    
    idx=idx+1

    plt.show()

pause
# %%

import seaborn as sns
import matplotlib.pyplot as plt

for idx in merged_df.columns:
    print(idx)
    
    # Check if the column is categorical
    if merged_df[idx].dtype == 'object' or merged_df[idx].nunique() < 2:
        print(f"Skipping {idx} as it is categorical.")
        continue  # Skip categorical columns
    
    # Plot KDE for numerical columns
    sns.kdeplot(data=merged_df, x=idx, log_scale=True)
    plt.title(f"KDE Plot for {idx}")
    plt.show()  

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