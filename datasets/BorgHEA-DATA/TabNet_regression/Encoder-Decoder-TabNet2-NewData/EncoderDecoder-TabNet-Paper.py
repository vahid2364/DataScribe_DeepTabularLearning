#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 12:29:46 2024

@author: attari.v
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
from pytorch_tabnet.tab_model import TabNetRegressor
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler#, RobustScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Lambda, Input, Dense, Dropout, Add, BatchNormalization, LayerNormalization, LeakyReLU, ELU, Layer
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.losses import MeanSquaredLogarithmicError
import matplotlib.pyplot as plt
from scipy.special import expit  # Sigmoid function
import scipy.stats as stats
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error, r2_score, explained_variance_score
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan

# Import the function from data_preprocessing.py (ensure this is correct)
#from data_preprocessing import process_and_split_data


os.makedirs('scales', exist_ok=True)
os.makedirs('images', exist_ok=True)

plt.rcParams['font.size'] = 18

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

# %%

# Function to test normality and homoscedasticity of residuals for multiple input columns
def test_residuals(df, x_cols, y_col):
    """
    Test for normality and homoscedasticity of residuals in a linear regression model.
	•	This function allows you to input multiple columns as independent variables while keeping a single dependent variable (output column).
	•	The function handles fitting the model, calculating residuals, and performing the normality and homoscedasticity tests.
    •   This is useful when you have multiple features influencing your target variable and want to ensure that the residuals meet the assumptions of linear regression.

    Parameters:
    df (pd.DataFrame): DataFrame containing the data
    x_cols (list of str): Names of the columns to be used as the independent variables (inputs)
    y_col (str): Name of the column to be used as the dependent variable (output)

    Returns:
    None: Prints the test results and shows plots for diagnostic checks.

    """

    # Fit a linear regression model
    X = sm.add_constant(df[x_cols])  # Add constant to the model
    model = sm.OLS(df[y_col], X).fit()

    # Get residuals
    residuals = model.resid

    # ---- 1. Test for Normality ----

    # 1.1 Visual Test: Histogram of residuals
    plt.figure(figsize=(10, 5))
    sns.histplot(residuals, kde=True)
    plt.title('Histogram of Residuals')
    plt.show()

    # 1.2 Visual Test: Q-Q Plot
    sm.qqplot(residuals, line='s')
    plt.title('Q-Q Plot of Residuals')
    plt.show()

    # 1.3 Statistical Test: Shapiro-Wilk Test for Normality
    shapiro_test = stats.shapiro(residuals)
    print(f'Shapiro-Wilk Test: Test Statistic = {shapiro_test[0]}, p-value = {shapiro_test[1]}')

    # ---- 2. Test for Homoscedasticity ----

    # 2.1 Visual Test: Residuals vs. Fitted Values Plot
    fitted_values = model.fittedvalues
    plt.figure(figsize=(10, 5))
    plt.scatter(fitted_values, residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Residuals vs. Fitted Values')
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.show()

    # 2.2 Statistical Test: Breusch-Pagan Test for Homoscedasticity
    bp_test = het_breuschpagan(residuals, X)
    labels = ['Lagrange multiplier statistic', 'p-value', 'f-value', 'f p-value']
    print('Breusch-Pagan Test Results:')
    print(dict(zip(labels, bp_test)))

# %%

# Sigmoid function
def sigmoid_transform(x):
    return 1 / (1 + np.exp(-x))

# Function to scale and de-scale data with flexible scaling and transformation methods
# including log1p, sigmoid, square root, and cube root transformations
def scale_data(df, input_columns, output_columns, 
               apply_sc=False, scaling_method='minmax', 
               apply_pt=False, pt_method='yeo-johnson', 
               apply_qt=False, qt_method='normal', 
               apply_log1p=False, apply_sigmoid=False, 
               apply_sqrt=True, sqrt_constant=1, 
               apply_cbrt=False, cbrt_constant=1):
    
    # Copy the data
    transformed_data = df.copy()

    # Separate inputs and outputs first
    inputs = transformed_data[input_columns].to_numpy()
    outputs = transformed_data[output_columns].to_numpy()

    # Apply log1p transformation if requested
    if apply_log1p:
        for idx in input_columns:
            inputs[:, input_columns.index(idx)] = np.log1p(df[idx])
        for idx in output_columns:
            outputs[:, output_columns.index(idx)] = np.log1p(df[idx])

        # Save log1p transformed data
        os.makedirs('scales', exist_ok=True)
        joblib.dump(inputs, 'scales/log1p_inputs.save')
        joblib.dump(outputs, 'scales/log1p_outputs.save')

    # Apply Sigmoid transformation if requested
    if apply_sigmoid:
        for idx in input_columns:
            inputs[:, input_columns.index(idx)] = expit(df[idx])
        for idx in output_columns:
            outputs[:, output_columns.index(idx)] = expit(df[idx])

        # Save sigmoid transformed data
        joblib.dump(inputs, 'scales/sigmoid_inputs.save')
        joblib.dump(outputs, 'scales/sigmoid_outputs.save')

    # Apply PowerTransformer if requested
    if apply_pt:
        if pt_method == 'yeo-johnson':
            pt_inputs = PowerTransformer(method='yeo-johnson')
            pt_outputs = PowerTransformer(method='yeo-johnson')
        elif pt_method == 'box-cox':
            pt_inputs = PowerTransformer(method='box-cox')
            pt_outputs = PowerTransformer(method='box-cox')
        else:
            raise ValueError(f"Unknown PowerTransformer method: {pt_method}. Choose 'yeo-johnson' or 'box-cox'.")
        
        # Apply Power Transformer to inputs and outputs
        inputs = pt_inputs.fit_transform(inputs)
        outputs = pt_outputs.fit_transform(outputs)
        
        # Save PowerTransformer data
        joblib.dump(pt_inputs, 'scales/power_transformer_inputs_scaler.save')
        joblib.dump(pt_outputs, 'scales/power_transformer_outputs_scaler.save')

    # Apply QuantileTransformer if requested
    if apply_qt:
        if qt_method == 'normal':
            qt_inputs = QuantileTransformer(output_distribution='normal', n_quantiles=1000)
            qt_outputs = QuantileTransformer(output_distribution='normal', n_quantiles=1000)
        elif qt_method == 'uniform':
            qt_inputs = QuantileTransformer(output_distribution='uniform', n_quantiles=1000)
            qt_outputs = QuantileTransformer(output_distribution='uniform', n_quantiles=1000)
        else:
            raise ValueError(f"Unknown QuantileTransformer method: {qt_method}. Choose 'normal' or 'uniform'.")
        
        # Apply Quantile Transformer to inputs and outputs
        inputs = qt_inputs.fit_transform(inputs)
        outputs = qt_outputs.fit_transform(outputs)
        
        # Save QuantileTransformer data
        joblib.dump(qt_inputs, 'scales/quantile_transformer_inputs_scaler.save')
        joblib.dump(qt_outputs, 'scales/quantile_transformer_outputs_scaler.save')

    # Apply square root transformation √(constant - x) if requested
    if apply_sqrt:
        for idx in input_columns:
            inputs[:, input_columns.index(idx)] = np.sqrt(np.maximum(0, sqrt_constant - df[idx]))
        for idx in output_columns:
            outputs[:, output_columns.index(idx)] = np.sqrt(np.maximum(0, sqrt_constant - df[idx]))

        # Save sqrt transformed data
        joblib.dump(inputs, 'scales/sqrt_inputs.save')
        joblib.dump(outputs, 'scales/sqrt_outputs.save')

    # Apply cube root transformation ³√(constant - x) if requested
    if apply_cbrt:
        for idx in input_columns:
            inputs[:, input_columns.index(idx)] = np.cbrt(cbrt_constant - df[idx])
        for idx in output_columns:
            outputs[:, output_columns.index(idx)] = np.cbrt(cbrt_constant - df[idx])

        # Save cbrt transformed data
        joblib.dump(inputs, 'scales/cbrt_inputs.save')
        joblib.dump(outputs, 'scales/cbrt_outputs.save')

    # Apply scaling if requested
    if apply_sc:
        # Initialize the scalers based on the scaling method chosen
        if scaling_method == 'minmax':
            input_scaler = MinMaxScaler()
            output_scaler = MinMaxScaler()
        elif scaling_method == 'standard':
            input_scaler = StandardScaler()
            output_scaler = StandardScaler()
        else:
            raise ValueError(f"Unknown scaling method: {scaling_method}. Choose 'minmax' or 'standard'.")
    
        # Scale the inputs and outputs
        inputs_scaled = input_scaler.fit_transform(inputs)
        outputs_scaled = output_scaler.fit_transform(outputs)
    
        # Save the scalers and scaled data
        joblib.dump(input_scaler, 'scales/input_scaler.save')
        joblib.dump(output_scaler, 'scales/output_scaler.save')
        joblib.dump(inputs_scaled, 'scales/scaled_inputs.save')
        joblib.dump(outputs_scaled, 'scales/scaled_outputs.save')
        
    else:
        inputs_scaled = inputs
        outputs_scaled = outputs
        input_scaler = None
        output_scaler = None
    
    return inputs_scaled, outputs_scaled, input_scaler, output_scaler, transformed_data, pt_inputs if apply_pt else None, pt_outputs if apply_pt else None, qt_inputs if apply_qt else None, qt_outputs if apply_qt else None

# # Function to scale and de-scale data with flexible scaling and transformation methods
# # including log1p, sigmoid, square root, and cube root transformations
# def scale_data(df, input_columns, output_columns, 
#                apply_sc=False, scaling_method='minmax', 
#                apply_pt=False, pt_method='yeo-johnson', 
#                apply_qt=False, qt_method='normal', 
#                apply_log1p=False, apply_sigmoid=False, 
#                apply_sqrt=True, sqrt_constant=1, 
#                apply_cbrt=False, cbrt_constant=1):
    
#     # Copy the data
#     transformed_data = df.copy()
    
#     # Apply log1p transformation if requested
#     if apply_log1p:
#         for idx in df.columns:
#             transformed_data[idx] = np.log1p(df[idx])

#     # Apply Sigmoid transformation if requested
#     if apply_sigmoid:
#         for idx in df.columns:
#             transformed_data[idx] = expit(df[idx])  # Sigmoid function

#     # Apply PowerTransformer if requested
#     if apply_pt:
#         if pt_method == 'yeo-johnson':
#             pt = PowerTransformer(method='yeo-johnson')
#         elif pt_method == 'box-cox':
#             pt = PowerTransformer(method='box-cox')
#         else:
#             raise ValueError(f"Unknown PowerTransformer method: {pt_method}. Choose 'yeo-johnson' or 'box-cox'.")
        
#         # Apply Power Transformer to selected columns
#         for idx in df.columns:
#             transformed_data[idx] = pt.fit_transform(df[[idx]]).ravel()

#     # Apply QuantileTransformer if requested
#     if apply_qt:
#         if qt_method == 'normal':
#             qt = QuantileTransformer(output_distribution='normal', n_quantiles=1000)
#         elif qt_method == 'uniform':
#             qt = QuantileTransformer(output_distribution='uniform', n_quantiles=1000)
#         else:
#             raise ValueError(f"Unknown QuantileTransformer method: {qt_method}. Choose 'normal' or 'uniform'.")
        
#         # Apply Quantile Transformer to selected columns
#         for idx in df.columns:
#             transformed_data[idx] = qt.fit_transform(transformed_data[[idx]]).ravel()

#     # Apply square root transformation √(constant - x) if requested
#     if apply_sqrt:
#         #for col in input_columns + output_columns:
#         for idx in df.columns:
#             transformed_data[idx] = np.sqrt(np.maximum(0, sqrt_constant - df[idx]))  # Ensure non-negative inside square root

#     # Apply cube root transformation ³√(constant - x) if requested
#     if apply_cbrt:
#         for idx in df.columns:
#             transformed_data[idx] = np.cbrt(cbrt_constant - df[idx])

#     # Separate inputs and outputs
#     inputs = transformed_data[input_columns].to_numpy()
#     outputs = transformed_data[output_columns].to_numpy()

#     if apply_sc:
#         # Initialize the scalers based on the scaling method chosen
#         if scaling_method == 'minmax':
#             input_scaler = MinMaxScaler()
#             output_scaler = MinMaxScaler()
#         elif scaling_method == 'standard':
#             input_scaler = StandardScaler()
#             output_scaler = StandardScaler()
#         else:
#             raise ValueError(f"Unknown scaling method: {scaling_method}. Choose 'minmax' or 'standard'.")
    
#         # Scale the inputs and outputs
#         inputs_scaled = input_scaler.fit_transform(inputs)
#         outputs_scaled = output_scaler.fit_transform(outputs)
    
#         # Create the 'scales' directory if it doesn't exist
#         os.makedirs('scales', exist_ok=True)
        
#         # Save the scalers and transformers if applied
#         joblib.dump(input_scaler, 'scales/input_scaler.save')
#         joblib.dump(output_scaler, 'scales/output_scaler.save')
#         if apply_pt:
#             joblib.dump(pt, 'scales/power_transformer.save')
#         if apply_qt:
#             joblib.dump(qt, 'scales/quantile_transformer.save')
            
#     else:
#         inputs_scaled = inputs
#         outputs_scaled = outputs
#         input_scaler = []
#         output_scaler = []
        
#     return inputs_scaled, outputs_scaled, input_scaler, output_scaler, transformed_data, pt if apply_pt else None, qt if apply_qt else None


# Inverse Sigmoid function
def sigmoid_inverse_transform(x):
    return -np.log((1 / x) - 1)

# Function to descale data with flexible options for transformers (separate for inputs and outputs)
def descale_data(scaled_data, 
                 input_scaler=None, output_scaler=None,
                 apply_dsc=False, 
                 apply_qt=False, qt_inputs=None, qt_outputs=None, 
                 apply_pt=False, pt_inputs=None, pt_outputs=None, 
                 apply_log1p=False, 
                 apply_sigmoid=False,
                 data_type='input'):
    
    descaled_data = scaled_data
    
    # Inverse transform using input or output scaler if requested
    if apply_dsc:
        if data_type == 'input' and input_scaler is not None:
            descaled_data = input_scaler.inverse_transform(scaled_data)
        elif data_type == 'output' and output_scaler is not None:
            descaled_data = output_scaler.inverse_transform(scaled_data)

    # If QuantileTransformer was applied, inverse transform using separate QT for inputs and outputs
    if apply_qt:
        if data_type == 'input' and qt_inputs is not None:
            descaled_data = qt_inputs.inverse_transform(descaled_data)
        elif data_type == 'output' and qt_outputs is not None:
            descaled_data = qt_outputs.inverse_transform(descaled_data)
    
    # If PowerTransformer was applied, inverse transform using separate PT for inputs and outputs
    if apply_pt:
        if data_type == 'input' and pt_inputs is not None:
            descaled_data = pt_inputs.inverse_transform(descaled_data)
        elif data_type == 'output' and pt_outputs is not None:
            descaled_data = pt_outputs.inverse_transform(descaled_data)

    # If log1p was applied, reverse the log1p transformation using expm1
    if apply_log1p:
        descaled_data = np.expm1(descaled_data)

    # If Sigmoid transformation was applied, reverse the Sigmoid transformation
    if apply_sigmoid:
        descaled_data = sigmoid_inverse_transform(descaled_data)

    return descaled_data

# Load and split the data only once
def load_and_split_data():
    df = pd.read_csv('../../input_data/v2/NbCrVWZr_data_stoic_creep_equil_filtered_v2_IQR_filtered.csv')
    input_columns = ['Nb', 'Cr', 'V', 'W', 'Zr']
    output_columns = ['Kou Criteria']
    X_train, X_test, y_train, y_test = process_and_split_data(
        df, 
        input_columns, 
        output_columns, 
        threshold=1e-9, 
        apply_sc=True, scaling_method='minmax', 
        apply_qt=True, qt_method='uniform', 
        apply_log1p=False, 
        apply_sqrt=False, 
        test_size=0.10, 
        random_state=42
    )
    return X_train, X_test, y_train, y_test

# %% Architecture 1

# Define the learning rate schedule function
def step_decay_schedule(initial_lr=0.0002, decay_factor=0.98, step_size=30):
    def schedule(epoch, lr):
        return initial_lr * (decay_factor ** np.floor(epoch / step_size))
    return LearningRateScheduler(schedule)

# Define the cosine annealing learning rate schedule function
def cosine_annealing_schedule(initial_lr=3.0, T_max=50, eta_min=0):
    def schedule(epoch):
        return eta_min + (initial_lr - eta_min) * (1 + np.cos(np.pi * epoch / T_max)) / 2
    return LearningRateScheduler(schedule)

# Define the learning rate warm-up and cosine annealing schedule function
def lr_warmup_cosine_annealing(initial_lr=3.0, warmup_epochs=10, T_max=50, eta_min=0.01):
    def schedule(epoch):
        if epoch < warmup_epochs:
            return initial_lr * (epoch / warmup_epochs)
        else:
            cosine_epoch = epoch - warmup_epochs
            return eta_min + (initial_lr - eta_min) * (1 + np.cos(np.pi * cosine_epoch / T_max)) / 2
    return LearningRateScheduler(schedule)

# %% Load Data and Scale

# Load and split the data
#X_train, X_test, y_train, y_test = load_and_split_data()

# Instantiate the Power Transformer (Yeo-Johnson is more general, works with positive and negative values)
#pt = PowerTransformer(method='yeo-johnson')
#qt = QuantileTransformer(output_distribution='normal')

# Load the CSV file

# Set the random seed for reproducibility
np.random.seed(45)
tf.random.set_seed(45)

# Load the CSV file
csv_file_path = '../../Borg_df_updated.csv'
df = pd.read_csv(csv_file_path, usecols=lambda column: column not in ['Unnamed: 0.3'])

# Define input and output columns
input_columns = df.columns[:30]
output_columns = df.columns[36:37] # Remaining columns

# Drop columns with all zeros
df = df.loc[:, ~(df == 0).all()]

print("\nDataFrame after dropping all-zero columns:")
print(df)

columns_to_keep = input_columns.tolist() + output_columns.tolist()

df = df[columns_to_keep]
df = df.dropna()
output_column_names = output_columns.tolist()



# %%

# Run the function on the DataFrame
#test_residuals(df, input_columns, output_columns)

# %%

print('Inspect original data')

plt.figure(figsize=(8,6))
for col,col_label in zip(output_columns,output_column_names):
    sns.kdeplot(df[col], label=col_label, fill=True, log_scale=True)
plt.legend(loc='upper left')
plt.xlabel('Data')
plt.savefig('images/Kdensity-OutputFeatures-LogScale.jpg')
plt.show()



# %%

plt.figure(figsize=(8,6))
for col in output_columns:
    sns.kdeplot(df[col], label=col, fill=True, log_scale=False)
plt.legend()
plt.xlabel('Min Creep NH [1/s]')
plt.savefig('images/Kdensity-OutputFeatures.jpg')
plt.show()

# %% split the data 

## To scale with sigmoid transformation
#inputs_scaled, outputs_scaled, input_scaler, output_scaler, pt, qt = scale_data(df, input_columns, output_columns, apply_sc= False, apply_log1p=True, apply_qt=False, qt_method='uniform')

# Apply square root and cube root transformations along with scaling
#inputs_scaled, outputs_scaled, input_scaler, output_scaler, transformed_data, pt, qt 
inputs_scaled, outputs_scaled, input_scaler, output_scaler, transformed_data, pt_inputs, pt_outputs, qt_inputs, qt_outputs = scale_data(
    df, 
    input_columns, 
    output_columns, 
    apply_sc=True, scaling_method='minmax', 
    apply_pt=False, pt_method='yeo-johnson', 
    apply_qt=False, qt_method='uniform', 
    apply_sigmoid=False, 
    apply_sqrt=False, sqrt_constant=10, 
    apply_cbrt=False, cbrt_constant=50
)

print('Inspect scaled data')
#test_residuals(transformed_data, input_columns, output_columns)

# %%

plt.figure(figsize=(8,6))
for idx, col in enumerate(output_columns):
    sns.kdeplot(outputs_scaled[:,idx], label=col, fill=True, log_scale=False)
plt.legend()
#plt.xlabel(str(output_columns)+' - Scaled')
plt.savefig('images/Kdensity-OutputFeatures-scaled.jpg')
plt.show()

# de-scaling the data
outputs_descaled = descale_data(outputs_scaled,
                 input_scaler=input_scaler, output_scaler=output_scaler,
                 apply_dsc=True, 
                 apply_qt=False, qt_inputs=qt_inputs, qt_outputs=qt_outputs, 
                 apply_pt=False, pt_inputs=None, pt_outputs=None, 
                 apply_log1p=False, 
                 apply_sigmoid=False,
                 data_type='output'
                 )

#     outputs_scaled, 
#     scaler=output_scaler, 
#     apply_dsc=True, 
#     apply_qt=True, qt_inputs=qt_inputs, qt_outputs=qt_outputs,
#     apply_pt=False, pt_inputs=pt_inputs, pt_outputs=pt_outputs,
#     apply_log1p=False,
#     apply_sigmoid=False
# )


# %%

print('Inspect scaled and descaled data')

# Plot the original and descaled distributions side by side
plt.figure(figsize=(10,4))

# Original data
plt.subplot(1, 2, 1)
sns.kdeplot(df[output_columns], label='Original', fill=True, log_scale=False)
plt.legend()
plt.title('Original Data Distribution')

# Descaled data
plt.subplot(1, 2, 2)
sns.kdeplot(outputs_descaled[:,:], label='Descaled', fill=True, log_scale=False)
plt.legend()
plt.title('Descaled Data Distribution')

plt.tight_layout()
plt.savefig('images/Original_vs_Descaled_Distribution.jpg')
plt.show()

# %% Train and Test Data Split

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(inputs_scaled, outputs_scaled, test_size=0.15, random_state=45 )
#X_train, X_test, y_train, y_test = train_test_split(inputs_scaled, outputs_scaled, test_size=0.10, random_state=30, stratify=pd.cut(df['1000 Min Creep NH [1/s]'], bins=[1e-2, 1, 4, 100]) )

# Print the shapes to verify
print("Training inputs shape:", X_train.shape)
print("Training outputs shape:", y_train.shape)
print("Test inputs shape:", X_test.shape)
print("Test outputs shape:", y_test.shape)


# %%

# TabNet Model Initialization

# Set the optimizer function based on the selected optimizer
optimizer_fn = torch.optim.RMSprop

n_d = 24
n_a = 32
gamma = 7.98e-4

tabnet_model = tabnet_model = TabNetRegressor(
            optimizer_fn=optimizer_fn,
            optimizer_params=dict(lr=4.2e-3),
            n_d=n_d,
            n_a=n_a,
            n_steps=3,
            gamma=gamma,
        )
        
        
# Reshape y_train and y_test to be 2D arrays
#y_train_reshaped = y_train.reshape(-1, 1)
#y_test_reshaped = y_test.reshape(-1, 1)


# Fit the TabNet model
tabnet_model.fit(
    X_train=X_train,
    y_train=y_train,
    eval_set=[(X_test, y_test)],
    eval_metric=['rmse'],  # Specify your evaluation metric
    max_epochs=50,
    patience=10,
    batch_size=72,
    virtual_batch_size=32,
    num_workers=0,
    drop_last=False,
)


# %%


import matplotlib.pyplot as plt

# Extract values from history
train_loss = tabnet_model.history['loss']
val_rmse = tabnet_model.history['val_0_rmse']  # Replace with the correct key or attribute

# Plot
plt.figure(figsize=(8, 5))
plt.plot(train_loss, label='Training Loss', color='blue', linewidth=2)
if val_rmse:
    plt.plot(val_rmse, label='Validation RMSE', color='orange', linewidth=2)

plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Metrics', fontsize=14)
plt.title('Training Loss and Validation RMSE', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig('training_validation_plot.png', dpi=300, transparent=True)
plt.show()

# Check if history is available
if tabnet_model.history is not None:
    history = tabnet_model.history

    # Extract training and validation loss
    train_loss = history['loss']
    val_loss = history['val_0_rmse']

    # Plotting the loss curves
    plt.figure(figsize=(8, 6))
    plt.plot(train_loss, label='Training Loss', color='blue')
    plt.plot(val_loss, label='Validation Loss', color='red', linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print("No history available for plotting.")

# %% Evaluate the autoencoder

# Evaluate the autoencoder
#loss = tabnet_model.evaluate(X_test, y_test)
#print(f"Test loss: {loss}")

# Make predictions
predictions_scaled = tabnet_model.predict(X_test)

# de-scaling the predictions
predictions = descale_data(predictions_scaled,
                 input_scaler=input_scaler, output_scaler=output_scaler,
                 apply_dsc=True, 
                 apply_qt=False, qt_inputs=qt_inputs, qt_outputs=qt_outputs, 
                 apply_pt=False, pt_inputs=None, pt_outputs=None, 
                 apply_log1p=False, 
                 apply_sigmoid=False,
                 data_type='output'
                 )

# de-scaling the y_test
y_test_original = descale_data(y_test,
                 input_scaler=input_scaler, output_scaler=output_scaler,
                 apply_dsc=True, 
                 apply_qt=False, qt_inputs=qt_inputs, qt_outputs=qt_outputs, 
                 apply_pt=False, pt_inputs=None, pt_outputs=None, 
                 apply_log1p=False, 
                 apply_sigmoid=False,
                 data_type='output'
                 )

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test_original, predictions)
print("Mean Squared Error (MSE):", mse)

# # Calculate MSE for each feature
mse_per_feature = mean_squared_error(y_test_original, predictions, multioutput='raw_values')

# Print the MSE for each feature
for i, mse in enumerate(mse_per_feature):
    print(f"Mean Squared Error for feature {i}: {mse}")

#test_loss = tabnet_model.evaluate(X_test, y_test)
test_mae = mean_absolute_error(predictions, y_test_original)
test_mse = mean_squared_error(predictions, y_test_original)
test_r2 = r2_score(predictions, y_test_original)
test_ev = explained_variance_score(predictions, y_test_original)

print(f'Test MAE: {test_mae}')
print(f'Test MSE: {test_mse}')
print(f'Test R²: {test_r2}')
print(f'Test Explained Variance: {test_ev}')
#print(f'Duration: {duration} seconds')


# %% QQ PLOTs

os.makedirs('QQplot', exist_ok=True)

# Function to plot QQ plots
def plot_qq(original, reconstructed, n=5):
    plt.figure(figsize=(7, 7))
    ax = plt.subplot(1, 1, 1)

    # Plot the QQ plot for the original data
    stats.probplot(original.flatten(), plot=ax)
    # Get the first line (corresponding to original data) and set its label
    ax.get_lines()[0].set_label('Original Data')

    # Plot the QQ plot for the reconstructed data
    stats.probplot(reconstructed.flatten(), plot=ax)
    # Get the second line (corresponding to reconstructed data) and set its label
    ax.get_lines()[2].set_color('red')
    ax.get_lines()[2].set_label('Reconstructed Data')

    # Add the legend
    plt.legend()

    # Set the Y-axis limit
    plt.ylim([0, original.flatten().max() * 1.05])

    # Adjust the layout and save the plot
    plt.tight_layout()
    plt.savefig('QQplot/qq_scaled_data.jpg')
    plt.show()

# Function to plot QQ plot for original and reconstructed data
def plot_qq_all(original, reconstructed):
    plt.figure(figsize=(7, 6))
    ax = plt.subplot(1, 1, 1)

    # Plot the QQ plot for the original data
    stats.probplot(original.flatten(), plot=ax)
    # Get the first line (corresponding to original data) and set its label
    ax.get_lines()[0].set_label('Original Data')

    # Plot the QQ plot for the reconstructed data
    stats.probplot(reconstructed.flatten(), plot=ax)
    # Get the second line (corresponding to reconstructed data) and set its label

    # Modify the markers for reconstructed data (unfilled red circles)
    ax.get_lines()[2].set_color('red')
    ax.get_lines()[2].set_marker('o')   # Use 'o' for circles
    ax.get_lines()[2].set_markerfacecolor('None')  # Make the face blank
    ax.get_lines()[2].set_markeredgecolor('red')  # Red outline
    ax.get_lines()[2].set_label('Reconstructed Data (Log Scale)')

    # Add label for the theoretical line (ideal fit for a normal distribution)
    ax.get_lines()[1].set_color('green')  # Change color to differentiate if desired
    ax.get_lines()[1].set_label('Ideal Fit (Normal Distribution)')

    # Add the legend
    # Add the legend
    plt.legend(loc='upper left', fontsize=13)

    # Set the Y-axis limit
    plt.ylim([0, original.flatten().max() * 1.05])

    # Adjust the layout and save the plot
    plt.tight_layout()
    plt.savefig('QQplot/qq.jpg', dpi=300)
    plt.show()

# Function to plot QQ plot for original and reconstructed data on a log scale
def plot_qq_all_log(original, reconstructed):
    plt.figure(figsize=(7, 6))
    ax = plt.subplot(1, 1, 1)

    # Safeguard: Apply log transformation with a shift for non-positive values
    original_log_safe = np.where(original.flatten() <= 0, 1e-10, original.flatten())
    reconstructed_log_safe = np.where(reconstructed.flatten() <= 0, 1e-10, reconstructed.flatten())

    # Plot the QQ plot for the log-transformed original data
    stats.probplot(np.log(original_log_safe), plot=ax)
    # Get the first line (corresponding to original data) and set its label
    ax.get_lines()[0].set_label('Original Data (Log Scale)')

    # Plot the QQ plot for the log-transformed reconstructed data
    stats.probplot(np.log(reconstructed_log_safe), plot=ax)
    # Modify the markers for reconstructed data (unfilled red circles)
    ax.get_lines()[2].set_color('red')
    ax.get_lines()[2].set_marker('o')   # Use 'o' for circles
    ax.get_lines()[2].set_markerfacecolor('None')  # Make the face blank
    ax.get_lines()[2].set_markeredgecolor('red')  # Red outline
    ax.get_lines()[2].set_label('Reconstructed Data (Log Scale)')

    # Add label for the theoretical line (ideal fit for a normal distribution)
    ax.get_lines()[1].set_color('green')  # Change color to differentiate if desired
    ax.get_lines()[1].set_label('Ideal Fit (Normal Distribution)')

    # Add the legend
    plt.legend(loc='upper left', fontsize=14)

    # Adjust the Y-axis limits if needed
    plt.xlim([-1, 4])
    plt.ylim([-32, np.log(original_log_safe).max() * 1.75])

    ax = plt.gca()
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    
    # Adjust the layout and save the plot
    plt.tight_layout()
    plt.savefig('QQplot/qq_log.jpg', dpi=300)
    plt.show()
    
# Plot the QQ plots for a random subset of test data
plot_qq_all(y_test, predictions_scaled)

plot_qq_all_log(y_test_original, predictions  )

plot_qq(y_test_original, predictions)
# Plot the QQ plot for all test data points
#plot_qq_all(y_test_original, predictions)

# %% predictions vs data

os.makedirs('parityplots-scaled', exist_ok=True)

for idx in range(outputs_scaled.shape[1]):
    
    plt.figure(figsize=(7, 6))

    # Annotate MSE on the plot
    mse = mean_squared_error(y_test[:, idx], predictions_scaled[:, idx])
    r2 = r2_score(y_test[:, idx], predictions_scaled[:, idx])
    plt.text(0.05, 0.95, f'MSE: {mse:.3f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
    plt.text(0.05, 0.92, f'r$^2$ : {r2 :.3f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

    ## Plot predictions vs actual outputs for the first output feature
    plt.scatter(y_test[:, idx], predictions_scaled[:, idx], color='gray')
    #plt.plot([np.min(y_test[:, idx]),np.max(y_test[:, idx]) ], [np.min(predictions_scaled[:, idx]),np.max(predictions_scaled[:, idx]) ], c='black')
    plt.plot([0,1], [0,1], c='black')

    plt.xlabel('Actual Outputs')
    plt.ylabel('Predicted Outputs')

    # Increase box line width to 2.5
    ax = plt.gca()
    ax.spines['top'].set_linewidth(2.5)
    ax.spines['right'].set_linewidth(2.5)
    ax.spines['bottom'].set_linewidth(2.5)
    ax.spines['left'].set_linewidth(2.5)
    
    plt.tight_layout()
    plt.savefig('parityplots-scaled/scatterplot_'+str(idx)+'.jpg')
    

os.makedirs('parityplots-original', exist_ok=True)

for idx in range(outputs_scaled.shape[1]):

    plt.figure(figsize=(7, 6))

    try:
        # Safeguard log against zero or negative values
        y_test_log_safe = np.where(y_test_original[:, idx] <= 0, 1e-40, y_test_original[:, idx])
        predictions_log_safe = np.where(predictions[:, idx] <= 0, 1e-40, predictions[:, idx])
        
        mse = mean_squared_error(y_test_log_safe, predictions_log_safe)
        r2 = r2_score(y_test_log_safe, predictions_log_safe)
        # Calculate MSLE and log R²
        msle = mean_squared_log_error(y_test_log_safe, predictions_log_safe)
        log_r2 = r2_score(np.log(y_test_log_safe), np.log(predictions_log_safe))
        
        # Calculate GMAE, SMAPE, MASE, and RMSPE
        gmae = np.mean(np.abs(np.log(y_test_log_safe) - np.log(predictions_log_safe)))
        smape = np.mean(2 * np.abs(predictions[:, idx] - y_test_original[:, idx]) / (np.abs(predictions[:, idx]) + np.abs(y_test_original[:, idx]))) * 100
        mase = mse / np.mean(np.abs(np.diff(y_test_original[:, idx])))  # Simplified MASE
        rmspe = np.sqrt(np.mean(np.square((y_test_original[:, idx] - predictions[:, idx]) / y_test_original[:, idx])))
        
        # Calculate RMSLE
        rmsle = np.sqrt(msle)

        # Annotate MSLE, RMSLE, and log R² on the plot
        plt.text(0.45, 0.99, f'R$^2$ : {r2:.3f}', transform=plt.gca().transAxes, fontsize=14, verticalalignment='top')
        plt.text(0.05, 0.99, f'MSE: {mse:.3g}', transform=plt.gca().transAxes, fontsize=14, verticalalignment='top')
        plt.text(0.05, 0.95, f'MSLE: {msle:.3g}', transform=plt.gca().transAxes, fontsize=14, verticalalignment='top')
        plt.text(0.05, 0.91, f'RMSLE: {rmsle:.3g}', transform=plt.gca().transAxes, fontsize=14, verticalalignment='top')
        plt.text(0.05, 0.87, f'log R$^2$ : {log_r2:.3f}', transform=plt.gca().transAxes, fontsize=14, verticalalignment='top')
        plt.text(0.05, 0.82, f'GMAE: {gmae:.3g}', transform=plt.gca().transAxes, fontsize=14, verticalalignment='top')
        plt.text(0.05, 0.78, f'SMAPE: {smape:.2f}%', transform=plt.gca().transAxes, fontsize=14, verticalalignment='top')
        plt.text(0.05, 0.74, f'MASE: {mase:.3g}', transform=plt.gca().transAxes, fontsize=14, verticalalignment='top')
        plt.text(0.05, 0.70, f'RMSPE: {rmspe:.3g}', transform=plt.gca().transAxes, fontsize=14, verticalalignment='top')
    
    except ValueError:
        # In case of error, fall back to normal MSE and r²
        mse = mean_squared_error(y_test_original[:, idx], predictions[:, idx])
        r2 = r2_score(y_test_original[:, idx], predictions[:, idx])
        
        # Calculate GMAE, SMAPE, MASE, and RMSPE
        gmae = np.mean(np.abs(np.log(y_test_log_safe) - np.log(predictions_log_safe)))
        smape = np.mean(2 * np.abs(predictions[:, idx] - y_test_original[:, idx]) / (np.abs(predictions[:, idx]) + np.abs(y_test_original[:, idx]))) * 100
        mase = mse / np.mean(np.abs(np.diff(y_test_original[:, idx])))  # Simplified MASE
        rmspe = np.sqrt(np.mean(np.square((y_test_original[:, idx] - predictions[:, idx]) / y_test_original[:, idx])))

        # Annotate MSE, GMAE, SMAPE, MASE, and RMSPE on the plot
        plt.text(0.05, 0.95, f'MSE: {mse:.3g}', transform=plt.gca().transAxes, fontsize=14, verticalalignment='top')
        plt.text(0.05, 0.91, f'R$^2$ : {r2:.3f}', transform=plt.gca().transAxes, fontsize=14, verticalalignment='top')
        plt.text(0.05, 0.87, f'GMAE: {gmae:.3g}', transform=plt.gca().transAxes, fontsize=14, verticalalignment='top')
        plt.text(0.05, 0.82, f'SMAPE: {smape:.2f}%', transform=plt.gca().transAxes, fontsize=14, verticalalignment='top')
        plt.text(0.05, 0.78, f'MASE: {mase:.3g}', transform=plt.gca().transAxes, fontsize=14, verticalalignment='top')
        plt.text(0.05, 0.74, f'RMSPE: {rmspe:.3g}', transform=plt.gca().transAxes, fontsize=14, verticalalignment='top')

    # Plot predictions vs actual outputs
    plt.scatter(y_test_original[:, idx], predictions[:, idx])
    plt.plot([np.min(y_test_original[:, idx]), np.max(y_test_original[:, idx])], 
             [np.min(y_test_original[:, idx]), np.max(y_test_original[:, idx])], 
             c='black')

    # Set x and y scales to logarithmic
    plt.xscale('log')
    plt.yscale('log')

    ax = plt.gca()
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    
    # Set labels and title
    plt.xlabel(f'Actual Outputs: {output_columns[idx]}')
    plt.ylabel('Predicted Outputs')
    plt.tight_layout()

    # Save the figure
    plt.savefig(f'parityplots-original/scatterplot_{idx}.jpg')
    plt.show()

# %%

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
import numpy as np
import pandas as pd

# Function to calculate metrics for each output feature
def generate_metrics_dataframe(predictions, y_test_original, output_columns):
    # Initialize metrics
    mse_per_feature = mean_squared_error(y_test_original, predictions, multioutput='raw_values')
    mae_per_feature = mean_absolute_error(y_test_original, predictions, multioutput='raw_values')
    r2_per_feature = r2_score(y_test_original, predictions, multioutput='raw_values')

    # Initialize results dictionary
    results_dict = {
        "Model Name": ["TabNet"],  # Model name column
    }

    # Iterate over each feature and calculate metrics
    for i, col in enumerate(output_columns):
        # Check if log R² is applicable
        if 'creep' in col.lower():  # For "creep" features
            log_y_test = np.log1p(y_test_original[:, i])  # log(1 + y) to handle zeros
            log_predictions = np.log1p(predictions[:, i])
            log_r2 = r2_score(log_y_test, log_predictions)
            results_dict[col] = [
                f"MSE: {mse_per_feature[i]:.4g} (R$^2$: {r2_per_feature[i]:.4g}), log R$^2$: {log_r2:.4g}"
            ]
        else:
            # For other features
            results_dict[col] = [
                f"MSE: {mse_per_feature[i]:.4g} (R$^2$: {r2_per_feature[i]:.4g})"
            ]

    # Create a DataFrame
    df_results = pd.DataFrame(results_dict)

    return df_results


# Generate the metrics DataFrame
df_results = generate_metrics_dataframe(predictions, y_test_original, output_columns)

# Convert the DataFrame to LaTeX
latex_table = df_results.to_latex(
    index=False, 
    column_format='|l|c|c|c|c|c|',  # Adjust column format
    escape=False, 
    multicolumn=True, 
    caption="Model Performance Metrics for Each Output Feature"
)

# Print LaTeX table
print(latex_table)



