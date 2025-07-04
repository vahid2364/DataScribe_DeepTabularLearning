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
import re
import numpy as np
import pandas as pd
import tensorflow as tf
#from tensorflow.keras.layers import Lambda, Input, Den se, BatchNormalization, Dropout, Add, LeakyReLU, ELU
#from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.losses import MeanSquaredError

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

##

# Import the function from data_preprocessing.py (ensure this is correct)
from data_preprocessing import process_and_split_data

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
            qt_inputs = QuantileTransformer(output_distribution='normal', n_quantiles=100)
            qt_outputs = QuantileTransformer(output_distribution='normal', n_quantiles=100)
        elif qt_method == 'uniform':
            qt_inputs = QuantileTransformer(output_distribution='uniform', n_quantiles=100)
            qt_outputs = QuantileTransformer(output_distribution='uniform', n_quantiles=100)
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

####
####

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
def step_decay_schedule(initial_lr=0.0003, decay_factor=0.98, step_size=30):
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

# Set the random seed for reproducibility
np.random.seed(45)
tf.random.set_seed(45)

# Load the CSV file
csv_file_path = '../data/Borg_df_updated.csv'
df = pd.read_csv(csv_file_path, usecols=lambda column: column not in ['Unnamed: 0.3'])




# Define input and output columns
input_columns = df.columns[:30]
input_columns = input_columns.drop(['B', 'Nd', 'Ga', 'Ag', 'Cu', 'C'])
output_columns = df.columns[ 33:38 ] # Remaining columns
output_columns = output_columns.drop(['PROPERTY: Test temperature ($^\circ$C)'])


#output_columns = output_columns.drop(['PROPERTY: Test temperature ($^\circ$C)','PROPERTY: Elongation (%)', 'PROPERTY: Elongation plastic (%)',
#                                      'PROPERTY: Exp. Young modulus (GPa)'])





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
X_train, X_test, y_train, y_test = train_test_split(inputs_scaled, outputs_scaled, test_size=0.1, random_state=45 )
#X_train, X_test, y_train, y_test = train_test_split(inputs_scaled, outputs_scaled, test_size=0.10, random_state=30, stratify=pd.cut(df['1000 Min Creep NH [1/s]'], bins=[1e-2, 1, 4, 100]) )

# Print the shapes to verify
print("Training inputs shape:", X_train.shape)
print("Training outputs shape:", y_train.shape)
print("Test inputs shape:", X_test.shape)
print("Test outputs shape:", y_test.shape)

# %% Model Construction

# Import the functions from dnf_model.py
from FullyDense_Model import create_complex_encoder, create_complex_decoder

#Best parameters:  {'latent_dim': 224, 'lambda': 6.887973446742919e-05, 'drop_out_rate': 0.2, 'alpha': 0.05169504979411294, 
#                   'learning_rate': 0.0012927298824231088, 'optimizer': 'adam', 'num_layers_encoder': 3, 'num_layers_decoder': 4, 
#                   'neurons_encoder_layer_0': 64, 'neurons_encoder_layer_1': 480, 'neurons_encoder_layer_2': 512, 
#                   'neurons_decoder_layer_0': 256,'neurons_decoder_layer_1': 64, 'neurons_decoder_layer_2': 384, 
#                   'neurons_decoder_layer_3': 448, 'batch_size': 80, 'epochs': 50}


#Best parameters:  {'lambda': 5.144546749006667e-05, 'drop_out_rate': 0.1, 
#                   'alpha': 0.019625854823285042, 'learning_rate': 0.0016841615283576471, 
#                   'optimizer': 'adam', 'num_layers_encoder': 4, 'num_layers_decoder': 3, 
#                   'encoder_neurons_layer_1': 80, 'latent_dim': 736, 'batch_size': 72, 
#                   'epochs': 30}

# Best for minmax only
#Best parameters:  {'lambda': 6.41095869803012e-05, 'drop_out_rate': 0.1, 
#                   'alpha': 0.0947518640556042, 'learning_rate': 0.0009608671487918118,
#                   'optimizer': 'adam', 'num_layers_encoder': 3, 'num_layers_decoder': 3,
#                   'encoder_neurons_layer_1': 96, 'latent_dim': 864, 'batch_size': 88, 'epochs': 30}

#	user_attrs_neurons_per_layer_decoder	user_attrs_neurons_per_layer_encoder
#62	[384, 192, 96]	[96, 192, 384]


# # Set latent dimension smaller than input dimension
# latent_dim = 564

# # Get input and output dimensions
# input_dim = X_train.shape[1]   # Number of input features
# output_dim = y_train.shape[1]  # Number of output features


# # Create the encoder and decoder models
# encoder = create_complex_encoder(input_dim=input_dim, latent_dim=latent_dim, 
#                                  num_layers=3, 
#                                  neurons_per_layer=[384,192,96], 
#                                  lamb=6.41e-5, alp=0.094, rate=0.1)                 

# # Create a decoder with 6 layers, specified neurons per layer, lamb=1e-5, alpha=0.2, dropout rate 0.2
# decoder = create_complex_decoder(output_dim=output_dim, latent_dim=latent_dim, 
#                                  num_layers=3, 
#                                  neurons_per_layer=[96,192,384], 
#                                  lamb=6.41e-5, alp=0.094, rate=0.1)



#
# Best parameters:  {'lambda': 0.00010392747572554459, 'drop_out_rate': 0.2, 'alpha': 0.1697328901149096, 
#                    'learning_rate': 0.0033705750463278315, 'optimizer': 'adam', 'num_layers_encoder': 2, 
#                    'num_layers_decoder': 4, 'encoder_neurons_layer_1': 80, 'latent_dim': 320, 'batch_size': 32, 'epochs': 30}
# Best validation loss:  0.011589953095044212
#	user_attrs_encoder_decoder_config
#60	{'neurons_per_layer_encoder': [80, 160], 'neurons_per_layer_decoder': [160, 80, 40, 20]}

# Set latent dimension smaller than input dimension
latent_dim = 320

# Get input and output dimensions
input_dim = X_train.shape[1]   # Number of input features
output_dim = y_train.shape[1]  # Number of output features


# Create the encoder and decoder models
encoder = create_complex_encoder(input_dim=input_dim, latent_dim=latent_dim, 
                                 num_layers=2, 
                                 neurons_per_layer=[80,160], 
                                 lamb=0.0001, alp=0.169, rate=0.2)                 

# Create a decoder with 6 layers, specified neurons per layer, lamb=1e-5, alpha=0.2, dropout rate 0.2
decoder = create_complex_decoder(output_dim=output_dim, latent_dim=latent_dim, 
                                 num_layers=4, 
                                 neurons_per_layer=[160, 80, 40, 20], 
                                 lamb=0.0001, alp=0.169, rate=0.2)

#encoder = create_complex_encoder(input_dim, latent_dim, num_layers=4, neurons_per_layer=None, lamb=1e-6, alp=0.1, rate=0.1)
#decoder = create_complex_decoder(output_dim, latent_dim, num_layers=4, neurons_per_layer=None, lamb=1e-6, alp=0.1, rate=0.1)

# Create the autoencoder model
autoencoder_input = Input(shape=(input_dim,))
encoded = encoder(autoencoder_input)
decoded = decoder(encoded)
FD_EncoderDecoder = Model(inputs=autoencoder_input, outputs=decoded)

# %%

# Print model summary
encoder.summary()
decoder.summary()

FD_EncoderDecoder.summary()

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

msle_loss = MeanSquaredLogarithmicError()

# Set the initial learning rate for Adadelta optimizer
#initial_learning_rate = 2
#optimizer = Adadelta(learning_rate=initial_learning_rate)

# Compile the model with Adadelta optimizer and custom loss
#autoencoder.compile(optimizer=optimizer, loss=custom_loss)
#autoencoder.compile(optimizer='SGD', loss='huber_loss')
#autoencoder.compile(optimizer='SGD', loss='binary_crossentropy', metrics=['accuracy'])
#autoencoder.compile(optimizer='AdaDelta', loss='mae')  
#autoencoder.compile(optimizer='SGD', loss=combined_loss,metrics=[bce_metric, mse_metric])
FD_EncoderDecoder.compile(optimizer='adam', loss='mse')
#autoencoder.compile(optimizer='SGD', loss='mean_squared_error')
#autoencoder.compile(optimizer='SGD', loss=custom_loss)
#autoencoder.compile(optimizer='adam', loss=combined_loss)
#autoencoder.compile(optimizer='SGD', loss=combined_loss, metrics=[mse_metric, mae_metric, bce_metric])

# %%

# callbacks
# learning_rate_scheduler = step_decay_schedule(initial_lr=initial_learning_rate, decay_factor=0.5, step_size=10)
# early_stopping = EarlyStopping(monitor='loss', patience=40, restore_best_weights=True)
# model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='loss')

# # Fit the model with the learning rate scheduler, early stopping, and model checkpoint
# history = autoencoder.fit(X_train, y_train, epochs=250, batch_size=16, validation_split=0.2, callbacks=[learning_rate_scheduler, early_stopping, model_checkpoint])


# Define custom callback to save the model only at the last epoch
class SaveAtLastEpoch(tf.keras.callbacks.Callback):
    def __init__(self, filepath):
        super(SaveAtLastEpoch, self).__init__()
        self.filepath = filepath

    def on_epoch_end(self, epoch, logs=None):
        if epoch == self.params['epochs'] - 1:  # Check if it's the last epoch
            self.model.save(self.filepath)
            print(f"Model saved at the last epoch: {epoch + 1}")

# Callbacks
callbacks = [
    step_decay_schedule(),
    EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=False),
    ModelCheckpoint(filepath='autoencoder_model_final.keras', monitor='val_loss', save_best_only=True, mode='min')  # Save the best weights
    #SaveAtLastEpoch('autoencoder_model_final.keras')  # Save model at the last epoch
]

# Define the learning rate scheduler callback
#warmup_cosine_schedule = lr_warmup_cosine_annealing(initial_lr=3.0, warmup_epochs=10, T_max=50, eta_min=0.01)

# Train the autoencoder
history = FD_EncoderDecoder.fit(
    X_train, y_train,
    epochs=1050,
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
    plt.plot(epochs, history.history['loss'], '-', label='Training Total Loss', color='black', lw=3)
    plt.plot(epochs, history.history['val_loss'], '--', label='Validation Total Loss', color='blue', lw=3)
    #plt.plot(epochs, history.history['val_mse_metric'], '--', label='Val MSE Loss', color='green')
    #plt.plot(epochs, history.history['val_mae_metric'], '--', label='Val MAE Loss', color='red')
    #plt.plot(epochs, history.history['val_bce_metric'], '--', label='Val BCE Loss', color='purple')
    #plt.yscale('log')
    plt.xlabel('Epochs', fontsize=25)
    plt.ylabel('Loss', fontsize=25)
    
    plt.tick_params(axis='both', which='major', labelsize=25)  # 'both' for x and y, 'major' for major ticks
    
    #plt.yscale('log')
    #plt.ylim([0,0.18])

    #plt.xlabel('Epochs')
    #plt.ylabel('Loss')
    #plt.title('Training and Validation Losses')
    plt.legend(fontsize=25)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('loss.jpg', dpi=300, transparent=True)
    plt.show()

# %% Evaluate the autoencoder

# Evaluate the autoencoder
loss = FD_EncoderDecoder.evaluate(X_test, y_test)
print(f"Test loss: {loss}")

# Make predictions
predictions_scaled = FD_EncoderDecoder.predict(X_test)

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

# Make predictions
predictions_scaled_training = FD_EncoderDecoder.predict(X_train)

y_train_original = descale_data(predictions_scaled_training,
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

test_loss = FD_EncoderDecoder.evaluate(X_test, y_test)
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
    #plt.xlim([-1, 4])
    #plt.ylim([0, np.log(original_log_safe).max() * 1.05])

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
    plt.text(0.05, 0.95, f'MSE: {mse:.3f}', transform=plt.gca().transAxes, fontsize=15, verticalalignment='top')
    plt.text(0.05, 0.91, f'r$^2$ : {r2 :.3f}', transform=plt.gca().transAxes, fontsize=15, verticalalignment='top')
    
    ## Plot predictions vs actual outputs for the first output feature
    plt.scatter(y_test[:, idx], predictions_scaled[:, idx], color='gray')
    plt.scatter(y_train[:, idx], predictions_scaled_training[:, idx], color='blue')
    #plt.plot([np.min(y_test[:, idx]),np.max(y_test[:, idx]) ], [np.min(predictions_scaled[:, idx]),np.max(predictions_scaled[:, idx]) ], c='black')
    plt.plot([0,1], [0,1], c='black')

    plt.xlabel(f'Actual: {output_columns[idx]}')
    plt.ylabel('Predicted')

    # Increase box line width to 2.5
    ax = plt.gca()
    ax.spines['top'].set_linewidth(2.5)
    ax.spines['right'].set_linewidth(2.5)
    ax.spines['bottom'].set_linewidth(2.5)
    ax.spines['left'].set_linewidth(2.5)
    
    plt.tight_layout()
    plt.savefig('parityplots-scaled/scatterplot_'+str(idx)+'_FDN_MPEA.jpg', dpi=300)
    

os.makedirs('parityplots-original', exist_ok=True)

for idx in range(outputs_scaled.shape[1]):

    plt.figure(figsize=(7, 6))

    try:
        
        # Safeguard log against zero or negative values
        y_test_log_safe = np.where(y_test_original[:, idx] <= 0, 1e-10, y_test_original[:, idx])
        predictions_log_safe = np.where(predictions[:, idx] <= 0, 1e-10, predictions[:, idx])
        
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
        plt.text(0.05, 0.95, f'MSLE: {msle:.2g}', transform=plt.gca().transAxes, fontsize=14, verticalalignment='top')
        plt.text(0.05, 0.91, f'RMSLE: {rmsle:.2g}', transform=plt.gca().transAxes, fontsize=14, verticalalignment='top')
        plt.text(0.05, 0.87, f'log R$^2$ : {log_r2:.2f}', transform=plt.gca().transAxes, fontsize=14, verticalalignment='top')
        plt.text(0.05, 0.82, f'GMAE: {gmae:.2g}', transform=plt.gca().transAxes, fontsize=14, verticalalignment='top')
        plt.text(0.05, 0.78, f'SMAPE: {smape:.2f}%', transform=plt.gca().transAxes, fontsize=14, verticalalignment='top')
        plt.text(0.05, 0.74, f'MASE: {mase:.3g}', transform=plt.gca().transAxes, fontsize=14, verticalalignment='top')
        plt.text(0.05, 0.70, f'RMSPE: {rmspe:.2g}', transform=plt.gca().transAxes, fontsize=14, verticalalignment='top')
    
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
        plt.text(0.05, 0.95, f'MSE: {mse:.2g}', transform=plt.gca().transAxes, fontsize=14, verticalalignment='top')
        plt.text(0.05, 0.91, f'R$^2$ : {r2:.2f}', transform=plt.gca().transAxes, fontsize=14, verticalalignment='top')
        plt.text(0.05, 0.87, f'GMAE: {gmae:.2g}', transform=plt.gca().transAxes, fontsize=14, verticalalignment='top')
        plt.text(0.05, 0.82, f'SMAPE: {smape:.2f}%', transform=plt.gca().transAxes, fontsize=14, verticalalignment='top')
        plt.text(0.05, 0.78, f'MASE: {mase:.3g}', transform=plt.gca().transAxes, fontsize=14, verticalalignment='top')
        plt.text(0.05, 0.74, f'RMSPE: {rmspe:.2g}', transform=plt.gca().transAxes, fontsize=14, verticalalignment='top')

    # Plot predictions vs actual outputs
    plt.scatter(y_test_original[:, idx], predictions[:, idx])
    plt.plot([np.min(y_test_original[:, idx]), np.max(y_test_original[:, idx])], 
             [np.min(y_test_original[:, idx]), np.max(y_test_original[:, idx])], 
             c='black')

    # Set x and y scales to logarithmic
    #plt.xscale('log')
    #plt.yscale('log')

    ax = plt.gca()
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    
    # Set labels and title
    plt.xlabel(f'Actual: {output_columns[idx]}')
    plt.ylabel('Predicted')
    plt.tight_layout()

    # Save the figure
    plt.savefig(f'parityplots-original/scatterplot_original_{idx}_FDN_MPEA.jpg', dpi=300)
    plt.show()

# %%

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score

# Example function to calculate metrics and generate the DataFrame
def generate_metrics_dataframe(predictions, y_test_original, output_columns):
    # Calculate metrics for each feature
    mse_per_feature = mean_squared_error(y_test_original, predictions, multioutput='raw_values')
    mae_per_feature = mean_absolute_error(y_test_original, predictions, multioutput='raw_values')
    r2_per_feature = r2_score(y_test_original, predictions, multioutput='raw_values')
    ev_per_feature = explained_variance_score(y_test_original, predictions, multioutput='raw_values')
    
    # Initialize results dictionary
    results_dict = {
        "Model Name": ["Encoder-Decoder (FDN)"],
    }

    # Loop through each feature and compute regular metrics and log R^2 for creep features
    for i, col in enumerate(output_columns):
        if 'creep' in col.lower():  # Check if the feature is a creep-related feature
            # Apply log transformation and compute R^2 for log-transformed data
            log_y_test_original = np.log1p(y_test_original[:, i])  # log(1 + y) to handle potential zeros
            log_predictions = np.log1p(predictions[:, i])
            log_r2 = r2_score(log_y_test_original, log_predictions)
            results_dict[f"{col}"] = [f"MSE: {mse_per_feature[i]:.4g} (R$^2$: {r2_per_feature[i]:.4g}), log R$^2$: {log_r2:.4g}"]
        else:
            results_dict[f"{col}"] = [f"MSE: {mse_per_feature[i]:.4g} (R$^2$: {r2_per_feature[i]:.4g})"]
    
    # Create a DataFrame to hold the results
    df_results = pd.DataFrame(results_dict)
    
    return df_results

# Example Usage:
# Assuming predictions and y_test_original are your model's predictions and true values
# Assuming output_columns is a list of your feature names

# Example output_columns
#output_columns = ['Density 1000C', 'THCD 1000C (W/mK)', 'Yield Strength 1000C', 'Cobalt Creep 1000C', 'Kou Criterion']

# Generate the DataFrame
df_results = generate_metrics_dataframe(predictions, y_test_original, output_columns)

# Print the DataFrame to LaTeX format
latex_table = df_results.to_latex(index=False, column_format='l|ccccc', escape=False)
print(latex_table)

# %%

from lime.lime_tabular import LimeTabularExplainer
import numpy as np

# Ensure X_train and X_test are in the required format (numpy arrays)
X_train = X_train.values if hasattr(X_train, 'values') else X_train
X_test = X_test.values if hasattr(X_test, 'values') else X_test

# Initialize the LIME explainer
explainer = LimeTabularExplainer(
    training_data=X_train,          # Training data (numpy array)
    feature_names=input_columns,    # Column names
    mode='regression'               # Regression task
)

# Select a single instance for explanation
instance = X_test[0]  # First test sample

# Get prediction from the model
model_prediction = FD_EncoderDecoder.predict(instance.reshape(1, -1))

# Explain the prediction
explanation = explainer.explain_instance(
    data_row=instance,              # The selected instance
    predict_fn=lambda x: FD_EncoderDecoder.predict(x).flatten()  # Model's prediction function
)

# Visualize the explanation
explanation.show_in_notebook(show_table=True)

# Save LIME explanation to an HTML file
explanation.save_to_file('lime_explanation.html')

# Open the HTML file in your default web browser
import webbrowser
webbrowser.open('lime_explanation.html')
