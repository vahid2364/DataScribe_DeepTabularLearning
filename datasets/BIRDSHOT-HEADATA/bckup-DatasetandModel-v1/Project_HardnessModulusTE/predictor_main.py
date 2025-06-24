#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 13:40:43 2024

@author: attari.v
"""

#import os
import pandas as pd
from modules.predictor import predict_and_compare

# %%

csv_file_path = '../data/HTMDEC_MasterTable_Interpolated_Orange_Iterations_BBC_with_SFEcalc.csv'  # Replace with your CSV file path

df = pd.read_csv(csv_file_path)
# Drop columns with all zeros
df = df.loc[:, ~(df == 0).all()]

# Define input and output columns
input_columns = df.columns[3:11]
output_columns = df.columns[15:21] # Remaining columns
output_columns = output_columns.drop(['UTS/YS','Hardness (GPa)', 'Modulus (GPa)', 'Tension Elongation (%)'])

columns_to_keep = input_columns.tolist() + output_columns.tolist()
    
df = df[columns_to_keep]
df = df.dropna()
    
print("\nDataFrame after dropping all-zero columns:")
print(df)
print("\nInput Columns:")
print(input_columns)
print("\nOutput Columns:")
print(output_columns)

# %%

# UTS and YS Prediction
comparison_df = predict_and_compare(
    df=df.iloc[0:10,:],
    input_columns=input_columns,
    output_columns=output_columns,
    weight_folder='results/weights/',
    scale_folder='scales/',
    save_path='results/comparison_df_UTS_YS.csv',
)
print("UTS and YS Predictions:\n", comparison_df)
