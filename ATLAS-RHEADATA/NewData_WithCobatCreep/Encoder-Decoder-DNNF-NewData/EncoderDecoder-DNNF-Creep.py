#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 14:41:42 2024

@author: attari.v
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 14:47:08 2024

@author: attari.v
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler#, RobustScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Lambda, Input, Dense, Dropout, Add, BatchNormalization, LayerNormalization, LeakyReLU, ELU, Layer
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
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

# %% Architecture 1

# Define the learning rate schedule function
def step_decay_schedule(initial_lr=0.001, decay_factor=0.98, step_size=20):
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

# Instantiate the Power Transformer (Yeo-Johnson is more general, works with positive and negative values)
pt = PowerTransformer(method='yeo-johnson')
qt = QuantileTransformer(output_distribution='normal')

# Set the random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load the CSV file
#csv_file_path = 'input_data/NbCrVWZr_data_stoic_creep_equil_filtered_v2_IQR_filtered.csv'  # Replace with your CSV file path
#csv_file_path = '../../input_data/IQR_dataframe-cobalt.csv'  # Replace with your CSV file path
#csv_file_path = '../../input_data/tc_df.csv'  # Replace with your CSV file path
csv_file_path = '../../input_data/v3/IQR_dataframe-NbCrVWZr_data_stoic_creep_equil_v3.csv'  # Replace with your CSV file path
df = pd.read_csv(csv_file_path, usecols=lambda column: column not in ['Unnamed: 0.3'])

columns_to_keep = [
#    'Nb', 'Cr', 'V', 'W', 'Zr', 'Creep Merit', '500 Min Creep CB [1/s]', '1300 Min Creep CB [1/s]'   # EQ 1273K Density (g/cc), EQ 1273K THCD (W/mK), YS 1000C PRIOR, 1000 Min Creep NH [1/s], Kou Criteria
    'Nb', 'Cr', 'V', 'W', 'Zr', 'YS 1000C PRIOR', 'EQ 1273K THCD (W/mK)', 'EQ 1273K Density (g/cc)', 'Kou Criteria', '1300 Min Creep CB [1/s]',   # EQ 1273K Density (g/cc), EQ 1273K THCD (W/mK), YS 1000C PRIOR, 1000 Min Creep NH [1/s], Kou Criteria
#    'Nb', 'Cr', 'V', 'W', 'Zr', 'Creep Merit', '25 Min Creep CB [1/s]', '500 Min Creep CB [1/s]', '1300 Min Creep CB [1/s]', '1500 Min Creep CB [1/s]'   # EQ 1273K Density (g/cc), EQ 1273K THCD (W/mK), YS 1000C PRIOR, 1000 Min Creep NH [1/s], Kou Criteria
#    'Nb', 'Cr', 'V', 'W', 'Zr', '1300 Min Creep CB [1/s]'   # EQ 1273K Density (g/cc), EQ 1273K THCD (W/mK), YS 1000C PRIOR, 1000 Min Creep NH [1/s], Kou Criteria
]

df = df[columns_to_keep]

# Define input and output columns
input_columns = df.columns[:5]  # First 5 columns
output_columns = df.columns[5:] # Remaining columns


# %% Remove data if needed

# # Series used for threshold
# threshold_series = df[output_columns]

# # Define the threshold
# threshold = 1e-3

# # Filter the DataFrame based on the threshold
# df_above_threshold = df[threshold_series.ge(threshold).all(axis=1)]  # all columns greater than or equal to threshold
# df_below_threshold = df[threshold_series.le(threshold).any(axis=1)]  # any column less than or equal to threshold

# # Output the results
# print("DataFrame with values above threshold:")
# print(df_above_threshold)

# print("\nDataFrame with values below or equal to threshold:")
# print(df_below_threshold)

# df = df_below_threshold

# %%

# Run the function on the DataFrame
#test_residuals(df, input_columns, output_columns)

# %%

print('Inspect original data')

plt.figure(figsize=(8,6))
for col in output_columns:
    sns.kdeplot(df[col], label=col, fill=True, log_scale=True)
plt.legend()
plt.xlabel('Data')
plt.savefig('images/Kdensity-OutputFeatures-LogScale.jpg')
plt.show()

# %%

plt.figure(figsize=(8,6))
for col in output_columns:
    sns.kdeplot(df[col], label=col, fill=True, log_scale=False)
plt.legend()
plt.xlabel('Data')
plt.savefig('images/Kdensity-OutputFeatures.jpg')
plt.show()

# %% split the data 

## To scale with sigmoid transformation
#inputs_scaled, outputs_scaled, input_scaler, output_scaler, pt, qt = scale_data(df, input_columns, output_columns, apply_sc= False, apply_log1p=True, apply_qt=False, qt_method='uniform')

# Apply square root and cube root transformations along with scaling
inputs_scaled, outputs_scaled, input_scaler, output_scaler, transformed_data, pt_inputs, pt_outputs, qt_inputs, qt_outputs = scale_data(
    df, 
    input_columns, 
    output_columns, 
    apply_sc=True, scaling_method='minmax', 
    apply_pt=False, pt_method='yeo-johnson', 
    apply_qt=True, qt_method='uniform', 
    apply_sigmoid=False, 
    apply_sqrt=False, sqrt_constant=10, 
    apply_cbrt=False, cbrt_constant=50
)

print('Inspect scaled data')
#test_residuals(transformed_data, input_columns, output_columns)

plt.figure(figsize=(8,6))
for idx, col in enumerate(output_columns):
    sns.kdeplot(outputs_scaled[:,idx], label=col, fill=True, log_scale=False)
plt.legend()
#plt.xlabel(str(output_columns)+' - Scaled')
plt.savefig('images/Kdensity-OutputFeatures-scaled.jpg')
plt.show()

# %%

# de-scaling the data
outputs_descaled = descale_data(outputs_scaled,
                 input_scaler=input_scaler, output_scaler=output_scaler,
                 apply_dsc=True, 
                 apply_qt=True, qt_inputs=qt_inputs, qt_outputs=qt_outputs, 
                 apply_pt=False, pt_inputs=None, pt_outputs=None, 
                 apply_log1p=False, 
                 apply_sigmoid=False,
                 data_type='output'
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
sns.kdeplot(outputs_descaled[:,:], label='Descaled', fill=True, log_scale=False)
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

# %% Model Construction

# Import the functions from dnf_model.py
from dnf_model import create_dnnf_encoder, create_dnnf_decoder

input_dim = X_train.shape[1]  # Number of input features
output_dim = y_train.shape[1]  # Number of output features
latent_dim = 64  # Latent space dimension

# Create DNNF Encoder and Decoder
encoder = create_dnnf_encoder(input_dim=input_dim, latent_dim=latent_dim, num_conjunctions=10, conjunction_units=112, dropout_rate=0.1, l2_reg=1e-6)
decoder = create_dnnf_decoder(output_dim=output_dim, latent_dim=latent_dim, num_conjunctions=10, conjunction_units=112, dropout_rate=0.1, l2_reg=1e-6)

# DNNF model combining encoder and decoder
input_data = Input(shape=(input_dim,))
encoded = encoder(input_data)
decoded_output = decoder(encoded)

dnnf_autoencoder = Model(inputs=input_data, outputs=decoded_output)

# Compile the model
# Define learning rate
learning_rate = 5.4e-4  # You can adjust this value as needed
# Create the Adam optimizer with the specified learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
# Compile the model using the optimizer with the specified learning rate
dnnf_autoencoder.compile(optimizer=optimizer, loss='mse')

# %%

# Print model summary
encoder.summary()
decoder.summary()

dnnf_autoencoder.summary()

print('model parameters', dnnf_autoencoder.count_params())  # in Keras/TensorFlow

# %%

def mse_metric(y_true, y_pred):
    return tf.keras.losses.MeanSquaredError()(y_true, y_pred)

def mae_metric(y_true, y_pred):
    return tf.keras.losses.MeanAbsoluteError()(y_true, y_pred)

def bce_metric(y_true, y_pred):
    return tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)

def combined_loss(y_true, y_pred):
    # Compute Mean Squared Error for all outputs
    mse_loss = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
    
    # Compute Mean Absolute Error for all outputs
    mae_loss = tf.keras.losses.MeanAbsoluteError()(y_true, y_pred)
    
    # Compute Binary Cross-Entropy for all outputs
    bce_loss = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    
    # Combine the losses
    total_loss = mse_loss + mae_loss + bce_loss
    return total_loss

# %%

#def weighted_mse(y_true, y_pred, weights):
#    squared_diff = tf.square(y_true - y_pred)
#    weighted_squared_diff = squared_diff * weights
#    return tf.reduce_mean(weighted_squared_diff)

## Define weights (adjust according to the importance of each feature)
#weights = np.array([1.0, 1.0, 1.0, 1.0, 2.0])  # Example weights

## Custom loss function
#def custom_loss(y_true, y_pred):
#    return weighted_mse(y_true, y_pred, weights)

# Define a custom loss function
def combined_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    mae = tf.keras.losses.MeanAbsoluteError()(y_true, y_pred)
    return bce + mae

def mse_metric(y_true, y_pred):
    return tf.keras.losses.MeanSquaredError()(y_true, y_pred)

# # Define the custom loss function combining MSE and KL Divergence
# def custom_loss(y_true, y_pred):
#     weight = 0.5
#     mse = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
#     kl_divergence = tf.keras.losses.KLDivergence()(y_true, y_pred)
    
#     return weight * mse + (1-weight) * kl_divergence  # You can adjust the weights of mse and kl_divergence if needed

# %%

# Set the initial learning rate for Adadelta optimizer
#initial_learning_rate = 2
#optimizer = Adadelta(learning_rate=initial_learning_rate)

# Compile the model with Adadelta optimizer and custom loss
#autoencoder.compile(optimizer=optimizer, loss=custom_loss)
#autoencoder.compile(optimizer='SGD', loss='huber_loss')
#autoencoder.compile(optimizer='SGD', loss='binary_crossentropy', metrics=['accuracy'])
#autoencoder.compile(optimizer='AdaDelta', loss='mae')  
#autoencoder.compile(optimizer='SGD', loss=combined_loss,metrics=[bce_metric, mse_metric])
#dnnf_autoencoder.compile(optimizer='SGD', loss='mse')
#autoencoder.compile(optimizer='SGD', loss='mean_squared_error')
#autoencoder.compile(optimizer='SGD', loss=custom_loss)
#autoencoder.compile(optimizer='adam', loss=combined_loss)
#autoencoder.compile(optimizer='SGD', loss=combined_loss, metrics=[mse_metric, mae_metric, bce_metric])

# %%

# Compile the model with Adam optimizer
#from tensorflow.keras.optimizers import Adam
#optimizer = Adam(learning_rate=0.001)
#autoencoder.compile(optimizer=optimizer, loss='mae')


# %%

# callbacks
# learning_rate_scheduler = step_decay_schedule(initial_lr=initial_learning_rate, decay_factor=0.5, step_size=10)
# early_stopping = EarlyStopping(monitor='loss', patience=40, restore_best_weights=True)
# model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='loss')

# # Fit the model with the learning rate scheduler, early stopping, and model checkpoint
# history = autoencoder.fit(X_train, y_train, epochs=250, batch_size=16, validation_split=0.2, callbacks=[learning_rate_scheduler, early_stopping, model_checkpoint])


# callbacks
callbacks = [
    step_decay_schedule(),
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=False),
    ModelCheckpoint(filepath='autoencoder_model_final.keras', monitor='val_loss', save_best_only=True, mode='min')  # Save the best weights
    ]

# Define the learning rate scheduler callback
#warmup_cosine_schedule = lr_warmup_cosine_annealing(initial_lr=3.0, warmup_epochs=10, T_max=50, eta_min=0.01)

# Train the autoencoder
history = dnnf_autoencoder.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.1,
    shuffle=True,
    callbacks=[callbacks],
    verbose=1
)

# %% Loss vs Epoch

# Plot the losses
plt.figure(figsize=(8, 4))
epochs = range(len(history.history['loss']))

# # Combined loss
# plt.plot(epochs, history.history['loss'], label='Total Loss', color='blue')

# # Individual losses
# plt.plot(epochs, history.history['mse_metric'], label='MSE Loss', color='green')
# plt.plot(epochs, history.history['mae_metric'], label='MAE Loss', color='red')
# plt.plot(epochs, history.history['bce_metric'], label='BCE Loss', color='purple')

# Validation losses (if available)
if 'val_loss' in history.history:
    plt.plot(epochs, history.history['loss'], '-', label='Training Total Loss', color='black')
    plt.plot(epochs, history.history['val_loss'], '--', label='Validation Total Loss', color='blue')
    #plt.plot(epochs, history.history['val_mse_metric'], '--', label='Val MSE Loss', color='green')
    #plt.plot(epochs, history.history['val_mae_metric'], '--', label='Val MAE Loss', color='red')
    #plt.plot(epochs, history.history['val_bce_metric'], '--', label='Val BCE Loss', color='purple')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.savefig('loss_DNNF.jpg', dpi=300)
    plt.show()

# %% Evaluate the autoencoder

# Evaluate the autoencoder
loss = dnnf_autoencoder.evaluate(X_test, y_test)
print(f"Test loss: {loss}")

# Make predictions
predictions_scaled = dnnf_autoencoder.predict(X_test)

# de-scaling the predictions
predictions = descale_data(predictions_scaled,
                 input_scaler=input_scaler, output_scaler=output_scaler,
                 apply_dsc=True, 
                 apply_qt=True, qt_inputs=qt_inputs, qt_outputs=qt_outputs, 
                 apply_pt=False, pt_inputs=None, pt_outputs=None, 
                 apply_log1p=False, 
                 apply_sigmoid=False,
                 data_type='output'
                 )

# de-scaling the y_test
y_test_original = descale_data(y_test,
                 input_scaler=input_scaler, output_scaler=output_scaler,
                 apply_dsc=True, 
                 apply_qt=True, qt_inputs=qt_inputs, qt_outputs=qt_outputs, 
                 apply_pt=False, pt_inputs=None, pt_outputs=None, 
                 apply_log1p=False, 
                 apply_sigmoid=False,
                 data_type='output'
                 )

#y_test_original = descale_data(y_test, output_scaler, qt, pt)

# # Inverse transform the predictions to original scale
# predictions = output_scaler.inverse_transform(predictions_scaled)
# #predictions = np.expm1(predictions)
# predictions = pt.inverse_transform(predictions)

# y_test_original = output_scaler.inverse_transform(y_test)
# #y_test_original = np.expm1(y_test_original)
# y_test_original = pt.inverse_transform(y_test_original)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test_original, predictions)
print("Mean Squared Error (MSE):", mse)

# # Calculate MSE for each feature
mse_per_feature = mean_squared_error(y_test_original, predictions, multioutput='raw_values')

# Print the MSE for each feature
for i, mse in enumerate(mse_per_feature):
    print(f"Mean Squared Error for feature {i}: {mse}")

test_loss = dnnf_autoencoder.evaluate(X_test, y_test)
test_mae = mean_absolute_error(predictions, y_test_original)
test_mse = mean_squared_error(predictions, y_test_original)
test_r2 = r2_score(predictions, y_test_original)
test_ev = explained_variance_score(predictions, y_test_original)

print(f'Test Loss: {test_loss}')
print(f'Test MAE: {test_mae}')
print(f'Test MSE: {test_mse}')
print(f'Test R²: {test_r2}')
print(f'Test Explained Variance: {test_ev}')
#print(f'Duration: {duration} seconds')


# %% QQ PLOTs

from Parity_Plots import plot_qq, plot_parity

# Plot QQ plots for scaled data
plot_qq(y_test, predictions_scaled, 'QQplot/qq_scaled_data.jpg')

# Plot QQ plots for original data
plot_qq(y_test_original, predictions, 'QQplot/qq_original_data.jpg')

# Plot scatter plots for scaled data
plot_parity(y_test, predictions_scaled, output_columns, 'parityplots-scaled', log_scale=False)

# Plot scatter plots for original data in log scale
plot_parity(y_test_original, predictions, output_columns, 'parityplots-original', log_scale=True)

# %%

# function to calculate metrics and generate the DataFrame
def generate_metrics_dataframe(predictions, y_test_original, output_columns):
    # Calculate metrics for each feature
    mse_per_feature = mean_squared_error(y_test_original, predictions, multioutput='raw_values')
    mae_per_feature = mean_absolute_error(y_test_original, predictions, multioutput='raw_values')
    r2_per_feature = r2_score(y_test_original, predictions, multioutput='raw_values')
    ev_per_feature = explained_variance_score(y_test_original, predictions, multioutput='raw_values')
    
    # Create a DataFrame to hold the results
    results_dict = {
        "Model Name": ["Encoder-Decoder (DNNF)"],
    }
    
    for i, col in enumerate(output_columns):
        results_dict[f"{col}"] = [f"MSE: {mse_per_feature[i]:.4g} (R$^2$: {r2_per_feature[i]:.3g})"]
    
    df_results = pd.DataFrame(results_dict)
    
    return df_results

# Assuming output_columns is a list of your feature names
#output_columns = ['Density 1000C', 'THCD 1000C (W/mK)', 'Yield Strength 1000C', 'Cobalt Creep 1000C', 'Kou Criterion']

# Generate the DataFrame
df_results = generate_metrics_dataframe(predictions, y_test_original, output_columns)

# Print the DataFrame to LaTeX format
latex_table = df_results.to_latex(index=False, column_format='l|ccccc', escape=False)
print(latex_table)

# Save the LaTeX table to a .txt file
with open("results_table.txt", "w") as text_file:
    text_file.write(latex_table)

print("Table saved to results_table.txt")

