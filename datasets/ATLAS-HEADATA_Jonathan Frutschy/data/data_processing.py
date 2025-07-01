#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 18:47:44 2024
@author: attari.v
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns


# %%

def remove_outliers_iqr(df, column, lower_quantile=0.10, upper_quantile=0.90, factor=1.5):
    """
    Removes outliers based on a custom IQR range.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        column (str): The column to check for outliers.
        lower_quantile (float): Lower percentile threshold (default: 10%).
        upper_quantile (float): Upper percentile threshold (default: 90%).
        factor (float): Multiplier for IQR range (default: 1.5).

    Returns:
        pd.DataFrame: Filtered DataFrame without outliers.
    """
    Q1 = df[column].quantile(lower_quantile)  # Custom lower quantile
    Q3 = df[column].quantile(upper_quantile)  # Custom upper quantile
    IQR = Q3 - Q1

    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR

    df_filtered = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df_filtered

# %%


os.makedirs('processed_data',exist_ok=True)

# Load the DataFrame from a .pkl file

file_path = 'data_LIQUID_variable_temprange9.xlsx'  

# Check file extension and read accordingly
if file_path.endswith('.csv'):
    df = pd.read_csv(file_path, encoding='utf-8')
elif file_path.endswith('.xlsx'):
    df = pd.read_excel(file_path, engine='openpyxl')
else:
    raise ValueError("Unsupported file format. Please provide a CSV or XLSX file.")
    
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


# Print exact column names to identify issues
print(df.columns.tolist())

# Standardize column names by stripping spaces and replacing hidden characters
df.columns = df.columns.str.strip().str.replace(r'\s+', ' ', regex=True)

# # List of EDS columns
# eds_columns = [
#     'EDS Measured Atomic % Al', 'EDS Measured Atomic % Co',
#     'EDS Measured Atomic % Cr', 'EDS Measured Atomic % Cu',
#     'EDS Measured Atomic % Fe', 'EDS Measured Atomic % Mn',
#     'EDS Measured Atomic % Ni', 'EDS Measured Atomic % V'
# ]

# # Clean and convert EDS columns to numeric
# for col in eds_columns:
#     df[col] = df[col].astype(str).str.strip()  # Remove extra spaces and characters
#     df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to numeric, set errors to NaN
    
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

# %% remove outliers in a column (optional)

df_cleaned = remove_outliers_iqr(df_cleaned, 'phi_deltaT', lower_quantile=0.05, upper_quantile=0.95, factor=1.5)

# %% Linear interpolation

print('--- case 2 ---')

## Linear interpolation
df_cleaned_filled = df_cleaned.interpolate(method='linear', axis=0)

# Assuming df_cleaned is your DataFrame
has_nan = df_cleaned_filled.isnull().values.any()

print(f"Does the DataFrame have any NaN values? {has_nan}")

total_nan = df_cleaned_filled.isnull().sum().sum()

print(f"Total number of NaN values in the DataFrame: {total_nan}")

# %%

os.makedirs('KDEs', exist_ok=True)


plt.figure(figsize=(8, 5))

for idx in ['$/kg',    'phi_deltaT',    'TSC'                   ]:
    sns.kdeplot(df_cleaned_filled[idx], fill=True, color='blue', label='KDE')
    safe_idx = idx.replace('$','').replace('/','_').replace(' ','_')
    plt.savefig('KDEs/kde_plot_'+str(safe_idx)+'.jpg', dpi=300)
    plt.show() 

# %%

# Save the DataFrame to a CSV file
filepath_processed = os.path.splitext(file_path)[0] + '_processed.csv'
df_cleaned_filled.to_csv(filepath_processed, index=False)

print(f"Interpolated DataFrame saved to {filepath_processed}")

df_cleaned_filled.describe()

# Convert interpolated DataFrame to LaTeX table
latex_table = df_cleaned_filled.to_latex(index=False, float_format="%.1f", column_format="l" * len(df_cleaned_filled.columns))



# %% Latex Table for paper

# Select only the required columns
selected_columns = [
    'Al','Cu','Cr','Nb','Ni','Fe','Mo',
    'PROP LT (K)',
    'PROP ST (K)',
    'EQ RT THCD (W/mK)',                          
    'EQ 1200C LIQUID',                  
    'EQ 800C LIQUID',
    '$/kg',
    'phi_deltaT',
    'TSC',                   
]

# Calculate extended statistical summary with skewness and kurtosis
summary_df = df_cleaned_filled[selected_columns].agg(['mean', 'std', 'min', 'max', 'median', 'skew', 'kurtosis']).T

# Optionally, include 25%, 50%, and 75% percentiles
percentiles = df_cleaned_filled[selected_columns].describe(percentiles=[0.25, 0.5, 0.75]).T
summary_df['25%'] = percentiles['25%']
summary_df['75%'] = percentiles['75%']

# Generate LaTeX table with skewness and kurtosis, using .2g format
latex_table = r"""
\begin{table*}[!h]
    \centering
    \caption{Extended Statistical Summary of the Dataset including Skewness and Kurtosis}
    \scriptsize
    \label{tab:extended_data_stats_skew_kurt}
    \begin{tabular}{lccccccccc}
        \toprule
        \textbf{Feature} & \textbf{Mean} & \textbf{Std. Dev.} & \textbf{Min} & \textbf{Max} & \textbf{Median} & \textbf{25\%} & \textbf{75\%} & \textbf{Skewness} & \textbf{Kurtosis} \\
        \midrule
"""

# Populate the LaTeX table with the extended statistical summary using .2g format
for feature in summary_df.index:
    latex_table += f"        {feature} & {summary_df.loc[feature, 'mean']:.2g} & {summary_df.loc[feature, 'std']:.2g} & {summary_df.loc[feature, 'min']:.2g} & {summary_df.loc[feature, 'max']:.2g} & {summary_df.loc[feature, 'median']:.2g} & {summary_df.loc[feature, '25%']:.2g} & {summary_df.loc[feature, '75%']:.2g} & {summary_df.loc[feature, 'skew']:.2g} & {summary_df.loc[feature, 'kurtosis']:.2g} \\\\\n"

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
    'Al','Cu','Cr','Nb','Ni','Fe','Mo',
]

X = df_cleaned_filled[input_columns]

property_columns = [
    '$/kg',
    'phi_deltaT',
    'TSC',                   
    ]

properties = df_cleaned_filled[property_columns]

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
