#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 18:47:44 2024
@author: attari.v
"""

import pandas as pd
import numpy as np
import os

os.makedirs('processed_data',exist_ok=True)

# Load the DataFrame from a .pkl file
#file_path = 'NbCrVWZr_data_stoic_creep_equil_filtered_v2.csv'  
file_path = 'NbCrVWZr_data_stoic_creep_equil_v3.csv'  
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


# %% Remove object columns from the DataFrame

# Identify object columns
object_columns = df.select_dtypes(include=['object']).columns
print(f"Object columns to be removed: {object_columns}")

# Remove object columns from the DataFrame
df_cleaned = df.drop(columns=object_columns)

# Identify object columns
bool_columns = df_cleaned.select_dtypes(include=['bool']).columns
print(f"Object columns to be removed: {bool_columns}")

# Remove object columns from the DataFrame
df_cleaned = df_cleaned.drop(columns=bool_columns)

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
df_cleaned2 = remove_outliers_conservative_iqr(df_cleaned, factor=10000000000000)

# Display the original and cleaned DataFrame shape
print("Original DataFrame shape:", df_cleaned.shape)
print("Cleaned DataFrame shape:", df_cleaned2.shape)

has_nan = df_cleaned2.isnull().values.any()

print(f"Does the DataFrame have any NaN values? {has_nan}")

total_nan = df_cleaned2.isnull().sum().sum()

print(f"Total number of NaN values in the DataFrame: {total_nan}")

if total_nan ==0:
    
    print('--- case 1 ---')
    
    # Save the DataFrame to a CSV file
    ##file_path = 'processed_data/IQR_dataframe-NbCrVWZr_data_stoic_creep_equil_v3.csv'  # Replace with your desired file path
    ##df_cleaned2.to_csv(file_path, index=False)
    
    print(f"IQR DataFrame saved to {file_path}")
    
    df_cleaned2.describe()
    
    # Convert DataFrame to LaTeX table
    latex_table = df_cleaned2.to_latex(index=False, float_format="%.1f", column_format="l" * len(df_cleaned2.columns))
    
else:

    # %% Linear interpolation

    print('--- case 2 ---')
    
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
    ##file_path = 'interpolated_dataframe-NbCrVWZr_data_stoic_creep_equil_v3.csv'  # Replace with your desired file path
    ##df_cleaned_filled.to_csv(file_path, index=False)
    
    print(f"Interpolated DataFrame saved to {file_path}")
    
    df_cleaned_filled.describe()

    # Convert interpolated DataFrame to LaTeX table
    latex_table = df_cleaned_filled.to_latex(index=False, float_format="%.1f", column_format="l" * len(df_cleaned_filled.columns))

# %% Latex Table for paper

from scipy.stats import skew, kurtosis

# Select only the required columns
selected_columns = [
    'Nb', 'Cr', 'V', 'W', 'Zr',
    'YS 1000C PRIOR',
    'EQ 1273K THCD (W/mK)',
    'EQ 1273K Density (g/cc)',
    '1300 Min Creep CB [1/s]',
    'PROP 1500C CTE (1/K)',
    'YS 1500C PRIOR',
    'Pugh_Ratio_PRIOR',
    'SCHEIL LT',
    'Kou Criteria',
    'Creep Merit'
]

# Calculate extended summary + feature complexity using consistent kurtosis (Pearson)
summary_data = []
for col in selected_columns:
    data = df_cleaned2[col].dropna()
    mean = data.mean()
    std = data.std()
    min_val = data.min()
    max_val = data.max()
    median = data.median()
    q25 = data.quantile(0.25)
    q75 = data.quantile(0.75)
    sk = skew(data)
    kurt = kurtosis(data, fisher=False)  # Pearson
    complexity = abs(sk) + abs(kurt - 3)
    
    summary_data.append([mean, std, min_val, max_val, median, q25, q75, sk, kurt, complexity])

# Build DataFrame
summary_df = pd.DataFrame(summary_data, 
    index=selected_columns, 
    columns=['Mean', 'Std. Dev.', 'Min', 'Max', 'Median', '25%', '75%', 'Skewness', 'Kurtosis', 'Feature Complexity']
)

# %% Generate LaTeX table
latex_table = r"""
\begin{table*}[!h]
    \centering
    \caption{Extended Statistical Summary of the Dataset including Skewness and Kurtosis}
    \scriptsize
    \label{tab:extended_data_stats_skew_kurt}
    \begin{tabular}{lcccccccccc}
        \toprule
        \textbf{Feature} & \textbf{Mean} & \textbf{Std. Dev.} & \textbf{Min} & \textbf{Max} & \textbf{Median} & \textbf{25\%} & \textbf{75\%} & \textbf{Skewness} & \textbf{Kurtosis} & \textbf{Feature Complexity}  \\
        \midrule
"""

for feature in summary_df.index:
    row = summary_df.loc[feature]
    latex_table += (f"        {feature} & "
        f"{row['Mean']:.2f} & {row['Std. Dev.']:.2f} & {row['Min']:.2f} & {row['Max']:.2f} & "
        f"{row['Median']:.2f} & {row['25%']:.2f} & {row['75%']:.2f} & "
        f"{row['Skewness']:.3f} & {row['Kurtosis']:.3f} & {row['Feature Complexity']:.2f} \\\\\n"
    )

latex_table += r"""
        \bottomrule
    \end{tabular}
\end{table*}
"""

print(latex_table)

pause
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

X = df[['Nb', 'Cr', 'V', 'W', 'Zr']]

properties = df_cleaned2.columns[2:]

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
    if y.min() > 0 and (y.max() / y.min() > 2.1):  # Log scale for large dynamic range
        norm = LogNorm(vmin=y.min(), vmax=y.max())
        colorbar_label = f'Log Scale: {prop}'
    else:
        norm = None
        colorbar_label = prop

    # Plot the results
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', s=10, alpha=0.5, norm=norm)
    cbar = plt.colorbar(scatter, label=colorbar_label)

    # Remove frame and axes
    plt.axis('off')  # Turn off the axes
    plt.gca().set_frame_on(False)  # Remove the frame

    # Save the plot
    plt.tight_layout()
    plt.savefig('tSNE/tSNE_'+str(idx)+'.jpg', dpi=300, bbox_inches='tight', pad_inches=0)
    
    idx=idx+1

    plt.show()

# %%

elements = ['Nb', 'Cr', 'V', 'W', 'Zr']

# Loop through the elements and plot them
for el in elements:
  plt.figure(figsize=(5,4))
  df = df.sort_values(by=el)
  plt.scatter(df['umap0'],df['umap1'],c=df[el])
  plt.colorbar(label=('at. % '+el))
  plt.axis('off')
  plt.show()

# %%

def generate_latex_table(df, columns_per_row=5, skip_columns=2):
    # Remove the first `skip_columns` columns
    column_names = df.columns.tolist()[skip_columns:]
    num_columns = len(column_names)
    rows = [column_names[i:i + columns_per_row] for i in range(0, num_columns, columns_per_row)]
    
    # LaTeX table header
    latex_table = "\\begin{table*}[ht]\n\\centering\n\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{" + "c " * columns_per_row + "}\n\\toprule\n"
    
    # Add rows of column names
    for i, row in enumerate(rows):
        row += [""] * (columns_per_row - len(row))  # Fill incomplete rows with empty cells
        latex_table += " & ".join(row) + " \\\\\n"
        if i == 0:  # Add midrule after the first row
            latex_table += "\\midrule\n"
    
    # LaTeX table footer
    latex_table += "\\bottomrule\n\\end{tabular}%\n}\n\\caption{List of DataFrame Columns (excluding the first two columns)}\n\\label{tab:columns_table}\n\\end{table*}"
    
    return latex_table

# Customize the columns per row to fit the table to the page width
columns_per_row = 5  # Adjust based on how many columns you want per row
latex_code = generate_latex_table(df, columns_per_row=columns_per_row)

print(latex_code)

# Optionally, save to a file
with open("columns_table.tex", "w") as f:
    f.write(latex_code)
