#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 15:45:39 2024

@author: attari.v
"""

#pip install neo4j

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from neo4j import GraphDatabase

# 

# Import the function from data_preprocessing.py (ensure this is correct)
#from data_preprocessing import process_and_split_data

# %%

# file_name = 'NbCrVWZr_data_stoic_creep_equil_filtered_v2.pkl'
# df = pd.read_pickle(file_name)

# # Display the first few rows of the DataFrame
# print(df.head()) #

# # %%

# # Load the .npy file for the GDB nodes
# nodes_file_path = 'nimplex_GF_5_20_nodes.npy'
# node_data = np.load(nodes_file_path)

# # Convert the numpy array to a DataFrame for better readability
# nodes_df = pd.DataFrame(node_data)

# # Show first 10 rows of nodes data
# print("First 10 rows of the nodes array:")
# print(nodes_df.head(10))

# # Print the number of nodes
# num_rows = node_data.shape[0]
# print(f'Number of nodes: {num_rows}')

# # Compare the number of nodes to the free AuraDB Neo4j Database Limit
# print(f'Conservative number of nodes for full GDB: {num_rows*4}')
# print(f'Node Limit for AuraDB Free: {200000}')

# %%

df = pd.read_csv('../../input_data/v2/NbCrVWZr_data_stoic_creep_equil_filtered_v2_IQR_filtered.csv')

# Define input and output columns
input_columns = ['Nb', 'Cr', 'V', 'W', 'Zr']
output_columns = ['1000 Min Creep NH [1/s]']
    
# Show the UMAP Vertices w/ respect to each pure element

# Input the elements
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

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Property to plot
#plot_prop = 'SCHEIL ST'
#plot_prop = 'Pugh_Ratio_PRIOR'
#plot_prop = 'YS 1500C PRIOR'
#plot_prop = 'YS 1000C PRIOR'
plot_prop = 'Kou Criteria'
#plot_prop = 'Cr'

# Create figure
plt.figure(figsize=(5, 4))

# Plot all points in grey
plt.scatter(df['umap0'], df['umap1'], color='grey', s=20)

# Plot a property of the remaining single phase BCC compositions
filtered_df = df.sort_values(by=plot_prop)
scatter = plt.scatter(filtered_df['umap0'], filtered_df['umap1'], c=filtered_df[plot_prop])
plt.colorbar(scatter, label=plot_prop)
plt.axis('off')
plt.show()

# Print % compositions remaining
ratio_remaining = len(filtered_df) / len(df)
print(f"Ratio of points remaining after filtering for single phase BCC: {ratio_remaining:.4f}")

# Calculate mean and quartiles
mean_value = filtered_df[plot_prop].mean()
q1 = filtered_df[plot_prop].quantile(0.25)
q3 = filtered_df[plot_prop].quantile(0.75)

# Create the KDE plot
plt.figure(figsize=(8, 4))
sns.kdeplot(filtered_df[plot_prop], fill=True)

# Mark the mean
plt.axvline(mean_value, color='r', linestyle='--', label=f'Mean: {mean_value:.2f}')

# Mark the quartile range
plt.axvline(q1, color='g', linestyle='--', label=f'1st Quartile: {q1:.2f}')
plt.axvline(q3, color='b', linestyle='--', label=f'3rd Quartile: {q3:.2f}')

# Customize the plot with titles and labels
plt.title('KDE Plot of '+plot_prop)
plt.xlabel(plot_prop)
plt.ylabel('Density')
plt.legend()

# Show the plot
plt.show()