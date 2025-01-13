#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 09:17:33 2024

@author: attari.v
"""

import pandas as pd

# Read the two CSV files into DataFrames
df1 = pd.read_csv('HTMDEC_MasterTable_Interpolated_Orange_Iterations_BBC.csv')  # Replace 'file1.csv' with your first file name
df2 = pd.read_csv('HTMDEC_MasterTable_Iterations.csv', encoding='latin1')  # You can also try 'iso-8859-1' or 'cp1252'

# Specify the column you want to add from df2 to df1
column_to_add = 'SFE_calc'  # Replace 'YourColumnName' with the actual column name

# Add the specified column from df2 to df1
df1[column_to_add] = df2[column_to_add]

# Display the updated df1
print(df1)

# Optionally, save the updated DataFrame to a new CSV file
df1.to_csv('HTMDEC_MasterTable_Interpolated_Orange_Iterations_BBC_with_SFEcalc.csv', index=False)