#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 14:13:11 2024

@author: attari.v
"""

import os
import numpy as np
import pandas as pd
#import tensorflow as tf
#from tensorflow.keras.layers import Lambda, Input, Dense, BatchNormalization, Dropout, Add, LeakyReLU, ELU
#from tensorflow.keras.models import Model
#from tensorflow.keras import backend as K
#from tensorflow.keras.losses import MeanSquaredError

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler#, RobustScaler
#from tensorflow.keras.models import Model
#from tensorflow.keras.layers import Lambda, Input, Dense, Dropout, Add, BatchNormalization, LayerNormalization, LeakyReLU, ELU, Layer
#from tensorflow.keras.optimizers import Adadelta
#from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
#from tensorflow.keras.regularizers import l1_l2
import matplotlib.pyplot as plt
from scipy.special import expit  # Sigmoid function
import scipy.stats as stats
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan


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
    
    # Apply log1p transformation if requested
    if apply_log1p:
        for idx in df.columns:
            transformed_data[idx] = np.log1p(df[idx])

    # Apply Sigmoid transformation if requested
    if apply_sigmoid:
        for idx in df.columns:
            transformed_data[idx] = expit(df[idx])  # Sigmoid function

    # Apply PowerTransformer if requested
    if apply_pt:
        if pt_method == 'yeo-johnson':
            pt = PowerTransformer(method='yeo-johnson')
        elif pt_method == 'box-cox':
            pt = PowerTransformer(method='box-cox')
        else:
            raise ValueError(f"Unknown PowerTransformer method: {pt_method}. Choose 'yeo-johnson' or 'box-cox'.")
        
        # Apply Power Transformer to selected columns
        for idx in df.columns:
            transformed_data[idx] = pt.fit_transform(df[[idx]]).ravel()

    # Apply QuantileTransformer if requested
    if apply_qt:
        if qt_method == 'normal':
            qt = QuantileTransformer(output_distribution='normal', n_quantiles=1000)
        elif qt_method == 'uniform':
            qt = QuantileTransformer(output_distribution='uniform', n_quantiles=1000)
        else:
            raise ValueError(f"Unknown QuantileTransformer method: {qt_method}. Choose 'normal' or 'uniform'.")
        
        # Apply Quantile Transformer to selected columns
        for idx in df.columns:
            transformed_data[idx] = qt.fit_transform(transformed_data[[idx]]).ravel()

    # Apply square root transformation √(constant - x) if requested
    if apply_sqrt:
        #for col in input_columns + output_columns:
        for idx in df.columns:
            transformed_data[idx] = np.sqrt(np.maximum(0, sqrt_constant - df[idx]))  # Ensure non-negative inside square root

    # Apply cube root transformation ³√(constant - x) if requested
    if apply_cbrt:
        for idx in df.columns:
            transformed_data[idx] = np.cbrt(cbrt_constant - df[idx])

    # Separate inputs and outputs
    inputs = transformed_data[input_columns].to_numpy()
    outputs = transformed_data[output_columns].to_numpy()

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
    
        # Create the 'scales' directory if it doesn't exist
        os.makedirs('scales', exist_ok=True)
        
        # Save the scalers and transformers if applied
        joblib.dump(input_scaler, 'scales/input_scaler.save')
        joblib.dump(output_scaler, 'scales/output_scaler.save')
        if apply_pt:
            joblib.dump(pt, 'scales/power_transformer.save')
        if apply_qt:
            joblib.dump(qt, 'scales/quantile_transformer.save')
            
    else:
        inputs_scaled = inputs
        outputs_scaled = outputs
        input_scaler = []
        output_scaler = []
        
    return inputs_scaled, outputs_scaled, input_scaler, output_scaler, transformed_data, pt if apply_pt else None, qt if apply_qt else None

# Inverse Sigmoid function
def sigmoid_inverse_transform(x):
    return -np.log((1 / x) - 1)

# Function to descale data with flexible options for transformers and sigmoid/log1p
def descale_data(scaled_data, scaler, apply_dsc=False, apply_qt=False, apply_pt=False, apply_log1p=False, apply_sigmoid=False):
    
    descaled_data = scaled_data
    
    if apply_dsc:
        # Inverse transform using MinMaxScaler (or StandardScaler)
        descaled_data = scaler.inverse_transform(scaled_data)
    
    # If QuantileTransformer was applied, inverse transform using QuantileTransformer
    if apply_qt:
        descaled_data = qt.inverse_transform(descaled_data)
    
    # If PowerTransformer was applied, inverse transform using PowerTransformer
    if apply_pt:
        descaled_data = pt.inverse_transform(descaled_data)

    # If log1p was applied, reverse the log1p transformation using expm1
    if apply_log1p:
        descaled_data = np.expm1(descaled_data)

    # If Sigmoid transformation was applied, reverse the Sigmoid transformation
    if apply_sigmoid:
        descaled_data = sigmoid_inverse_transform(descaled_data)

    return descaled_data

# %% Architecture 1

# # Define the learning rate schedule function
# def step_decay_schedule(initial_lr=0.001, decay_factor=0.98, step_size=20):
#     def schedule(epoch, lr):
#         return initial_lr * (decay_factor ** np.floor(epoch / step_size))
#     return LearningRateScheduler(schedule)

# # Define the cosine annealing learning rate schedule function
# def cosine_annealing_schedule(initial_lr=3.0, T_max=50, eta_min=0):
#     def schedule(epoch):
#         return eta_min + (initial_lr - eta_min) * (1 + np.cos(np.pi * epoch / T_max)) / 2
#     return LearningRateScheduler(schedule)

# # Define the learning rate warm-up and cosine annealing schedule function
# def lr_warmup_cosine_annealing(initial_lr=3.0, warmup_epochs=10, T_max=50, eta_min=0.01):
#     def schedule(epoch):
#         if epoch < warmup_epochs:
#             return initial_lr * (epoch / warmup_epochs)
#         else:
#             cosine_epoch = epoch - warmup_epochs
#             return eta_min + (initial_lr - eta_min) * (1 + np.cos(np.pi * cosine_epoch / T_max)) / 2
#     return LearningRateScheduler(schedule)

# %% Load Data and Scale

# Instantiate the Power Transformer (Yeo-Johnson is more general, works with positive and negative values)
pt = PowerTransformer(method='yeo-johnson')
qt = QuantileTransformer(output_distribution='normal')

# Set the random seed for reproducibility
np.random.seed(42)
#tf.random.set_seed(42)

# Load the CSV file
csv_file_path = '../input_data/NbCrVWZr_data_stoic_creep_equil_filtered_v2_IQR_filtered.csv'  # Replace with your CSV file path
df = pd.read_csv(csv_file_path, usecols=lambda column: column not in ['Unnamed: 0.3'])

# Define the remaining features
columns_to_keep = [
    'Nb', 'Cr', 'V', 'W', 'Zr', '1000 Min Creep NH [1/s]'   # EQ 1273K Density (g/cc), EQ 1273K THCD (W/mK), YS 1000C PRIOR, 1000 Min Creep NH [1/s], Kou Criteria
]


df = df[columns_to_keep]

#df.to_csv('df.csv', index=False)

df['1000 Min Creep NH [1/s]'] = df['1000 Min Creep NH [1/s]']

# Define input and output columns
input_columns = df.columns[:5]  # First 5 columns
output_columns = df.columns[5:] # Remaining columns

#
# Series used for threshold
threshold_series = df['1000 Min Creep NH [1/s]']

# Define the threshold
threshold = 1e-9

# # Split the DataFrame
df = df[threshold_series > threshold]
df_below_threshold = df[threshold_series <= threshold]

# print("DataFrame with values above threshold:")
# print(df)

# print("\nDataFrame with values below or equal to threshold:")
# print(df_below_threshold)

df['1000 Min Creep NH [1/s]'] = df['1000 Min Creep NH [1/s]']*1e6


# %%

# Run the function on the DataFrame
test_residuals(df, input_columns, output_columns)

# %%

print('Inspect original data')

plt.figure(figsize=(8,6))
sns.kdeplot(df[output_columns], label=output_columns, fill=True, log_scale=True)
plt.legend()
plt.xlabel(output_columns)
plt.savefig('images/Kdensity-OutputFeatures-LogScale.jpg')
plt.show()

# %%

plt.figure(figsize=(8,6))
sns.kdeplot(df[output_columns], label=output_columns, fill=True, log_scale=False)
plt.legend()
plt.xlabel('Min Creep NH [1/s]')
plt.savefig('images/Kdensity-OutputFeatures.jpg')
plt.show()



# %% split the data 

## To scale with sigmoid transformation
#inputs_scaled, outputs_scaled, input_scaler, output_scaler, pt, qt = scale_data(df, input_columns, output_columns, apply_sc= False, apply_log1p=True, apply_qt=False, qt_method='uniform')

# Apply square root and cube root transformations along with scaling
inputs_scaled, outputs_scaled, input_scaler, output_scaler, transformed_data, pt, qt = scale_data(
    df, 
    input_columns, 
    output_columns, 
    apply_sc=True, scaling_method='minmax', 
    apply_pt=False, pt_method='yeo-johnson', 
    apply_qt=True, qt_method='uniform', 
    apply_log1p=False, 
    apply_sigmoid=False, 
    apply_sqrt=False, sqrt_constant=10, 
    apply_cbrt=False, cbrt_constant=50
)

print('Inspect scaled data')
test_residuals(transformed_data, input_columns, output_columns)

# %%

plt.figure(figsize=(8,6))
sns.kdeplot(outputs_scaled[:,0], label=output_columns, fill=True, log_scale=False)
plt.legend()
plt.xlabel(str(output_columns)+' - Scaled')
plt.savefig('images/Kdensity-OutputFeatures-scaled.jpg')
plt.show()

# Example of de-scaling the data
outputs_descaled = descale_data(
    outputs_scaled, 
    output_scaler, 
    apply_dsc=True, 
    apply_pt=False, 
    apply_qt=True, 
    apply_log1p=False
)

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
sns.kdeplot(outputs_descaled[:,0], label='Descaled', fill=True, log_scale=False)
plt.legend()
plt.title('Descaled Data Distribution')

plt.tight_layout()
plt.savefig('images/Original_vs_Descaled_Distribution.jpg')
plt.show()


# %% Train and Test Data Split

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(inputs_scaled, outputs_scaled, test_size=0.10, random_state=42 )
#X_train, X_test, y_train, y_test = train_test_split(inputs_scaled, outputs_scaled, test_size=0.10, random_state=30, stratify=pd.cut(df['1000 Min Creep NH [1/s]'], bins=[1e-2, 1, 4, 100]) )

# Print the shapes to verify
print("Training inputs shape:", X_train.shape)
print("Training outputs shape:", y_train.shape)
print("Test inputs shape:", X_test.shape)
print("Test outputs shape:", y_test.shape)

# Reshape input for CNN to (input_dim, 1) to add the channel dimension
X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))  # Add channel dimension
X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))      # Add channel dimension

# %% Model Construction

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
from pytorch_tabnet.tab_model import TabNetRegressor

# TabNet Model Initialization
tabnet_model = TabNetRegressor(optimizer_fn=torch.optim.Adam, optimizer_params=dict(lr=2e-2),)

# Reshape y_train and y_test to be 2D arrays
y_train_reshaped = y_train.reshape(-1, 1)
y_test_reshaped = y_test.reshape(-1, 1)

# Fit the TabNet model
history = tabnet_model.fit(
    X_train=X_train,
    y_train=y_train,
    eval_set=[(X_test, y_test)],
    #eval_metric=['rmse'],  # Specify your evaluation metric
    max_epochs=200,
    patience=20,
    batch_size=64,
    virtual_batch_size=32,
    num_workers=0,
    drop_last=False,
)


# %% Loss vs Epoch

# Check if history is available
if tabnet_model.history is not None:
    history = tabnet_model.history

    # Extract training and validation loss
    train_loss = history['loss']
    val_loss = history['val_0_mse']

    # Plotting the loss curves
    plt.figure(figsize=(8, 6))
    plt.plot(train_loss, label='Training Loss', color='blue')
    plt.plot(val_loss, label='Validation Loss', color='red', linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss.jpg')
    plt.show()
else:
    print("No history available for plotting.")
    
# %% Evaluate the autoencoder

# Make predictions
predictions_scaled = tabnet_model.predict(X_test)

# de-scaling the predictions
predictions = descale_data(
    predictions_scaled, 
    output_scaler, 
    apply_dsc=True, 
    apply_pt=False, 
    apply_qt=True, 
    apply_log1p=False
)
#predictions = descale_data(predictions_scaled, output_scaler, qt, pt)

# de-scaling the y_test
y_test_original = descale_data(
    y_test, 
    output_scaler, 
    apply_dsc=True, 
    apply_pt=False, 
    apply_qt=True, 
    apply_log1p=False
)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test_original, predictions)
print("Mean Squared Error (MSE):", mse)

# # Calculate MSE for each feature
mse_per_feature = mean_squared_error(y_test_original, predictions, multioutput='raw_values')

# Print the MSE for each feature
for i, mse in enumerate(mse_per_feature):
    print(f"Mean Squared Error for feature {i}: {mse}")

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
    plt.savefig('QQplot/qq.jpg')
    plt.show()

# Plot the QQ plots for a random subset of test data
plot_qq_all(y_test, predictions_scaled)

plot_qq_all(y_test_original, predictions  )

#plot_qq_all(y_test, predictions)
# Plot the QQ plot for all test data points
#plot_qq_all(y_test_original, predictions)

# %% predictions vs data

os.makedirs('scatterplots-scaled', exist_ok=True)

for idx in range(outputs_scaled.shape[1]):
    
    plt.figure(figsize=(7, 7))

    # Annotate MSE on the plot
    mse = mean_squared_error(y_test[:, idx], predictions_scaled[:, idx])
    r2 = r2_score(y_test[:, idx], predictions_scaled[:, idx])
    plt.text(0.05, 0.95, f'MSE: {mse:.3f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
    plt.text(0.05, 0.92, f'r$^2$ : {r2 :.3f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')


    ## Plot predictions vs actual outputs for the first output feature
    plt.scatter(y_test[:, idx], predictions_scaled[:, idx])
    #plt.plot([np.min(y_test[:, idx]),np.max(y_test[:, idx]) ], [np.min(predictions_scaled[:, idx]),np.max(predictions_scaled[:, idx]) ], c='black')
    plt.plot([0,1], [0,1], c='black')

    plt.xlabel('Actual Outputs')
    plt.ylabel('Predicted Outputs')
    plt.title('Actual vs Predicted') #  - '+str(output_columns[idx]))
    plt.tight_layout()
    plt.savefig('scatterplots-scaled/scatterplot_'+str(idx)+'.jpg')
    #plt.show()
    

os.makedirs('scatterplots-original', exist_ok=True)

for idx in range(outputs_scaled.shape[1]):

    plt.figure(figsize=(7, 7))

    # Annotate MSE on the plot
    mse = mean_squared_error(y_test_original[:, idx], predictions[:, idx])
    r2 = r2_score(y_test_original[:, idx], predictions[:, idx])
    plt.text(0.05, 0.95, f'MSE: {mse:.3f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
    plt.text(0.05, 0.92, f'r$^2$ : {r2 :.3f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

    ## Plot predictions vs actual outputs for the first output feature
    plt.scatter(y_test_original[:, idx], predictions[:, idx])
    plt.plot([np.min(y_test_original[:, idx]),np.max(y_test_original[:, idx]) ], [np.min(y_test_original[:, idx]),np.max(y_test_original[:, idx]) ], c='black')

    #plt.xlim([np.min(y_test_original[:, idx])*0.95,np.max(y_test_original[:, idx])*1.05 ])
    #plt.ylim([np.min(predictions[:, idx])*0.95,np.max(predictions[:, idx])*1.05 ])
    
    plt.xlabel('Actual Outputs:'+str(output_columns[idx]))
    plt.ylabel('Predicted Outputs')
    #plt.title('Actual vs Predicted Outputs - '+str(output_columns[idx]))
    plt.tight_layout()
    plt.savefig('scatterplots-original/scatterplot_'+str(idx)+'.jpg')
    plt.show()


