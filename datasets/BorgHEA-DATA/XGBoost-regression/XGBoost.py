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
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l1_l2
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
def step_decay_schedule(initial_lr=0.1, decay_factor=0.98, step_size=20):
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

csv_file_path = '../Borg_df_updated.csv'
df = pd.read_csv(csv_file_path, usecols=lambda column: column not in ['Unnamed: 0.3'])

# Define input and output columns
input_columns = df.columns[:30]
#output_columns = df.columns[36:37] # Remaining columns
input_columns = input_columns.drop(['B', 'Nd', 'Ga', 'Ag', 'Cu', 'C'])
output_columns = df.columns[ 33:38 ] # Remaining columns
output_columns = output_columns.drop(['PROPERTY: Test temperature ($^\circ$C)'])

# Drop columns with all zeros
df = df.loc[:, ~(df == 0).all()]

columns_to_keep = input_columns.tolist() + output_columns.tolist()

df = df[columns_to_keep]

df = df.dropna()

output_column_names = output_columns.tolist()

print("\nDataFrame after dropping all-zero columns:")
print(df)

# %% Remove data if needed



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
    apply_qt=False, qt_method='uniform', 
    apply_sigmoid=False, 
    apply_sqrt=False, sqrt_constant=10, 
    apply_cbrt=False, cbrt_constant=50
)

print('Inspect scaled data')

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

# Example of de-scaling the data
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
X_train, X_test, y_train, y_test = train_test_split(inputs_scaled, outputs_scaled, test_size=0.10, random_state=42 )
#X_train, X_test, y_train, y_test = train_test_split(inputs_scaled, outputs_scaled, test_size=0.10, random_state=30, stratify=pd.cut(df['1000 Min Creep NH [1/s]'], bins=[1e-2, 1, 4, 100]) )

# Print the shapes to verify
print("Training inputs shape:", X_train.shape)
print("Training outputs shape:", y_train.shape)
print("Test inputs shape:", X_test.shape)
print("Test outputs shape:", y_test.shape)

# %% Model Construction

from xgboost_model import xgboost_model  # Import the function from the file where it's defined

# Call the XGBoost model function with custom parameters
#model, evals_result, mse = xgboost_model(X_train, y_train, n_estimators=200, max_depth=8)

# %%

#xgboost_model.summary()

# %%

# Create the model
model = xgboost_model(n_estimators=550, max_depth=80, learning_rate=0.009)

#model.summary()

# Prepare evaluation sets to monitor training and test loss
eval_set = [(X_train, y_train), (X_test, y_test)]

# Fit the model with evaluation metric
model.fit(
    X_train, y_train,
    eval_metric="rmse",
    eval_set=eval_set,
    early_stopping_rounds=10,
    verbose=True
)

# Make predictions and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# %% Loss vs Epoch


plt.figure(figsize=(8, 4))

# Plot training and test RMSE
epochs = len(model.evals_result()['validation_0']['rmse'])
x_axis = range(0, epochs)
plt.plot(x_axis, model.evals_result()['validation_0']['rmse'], label='Train')
plt.plot(x_axis, model.evals_result()['validation_1']['rmse'], label='Test')
plt.xlabel('Epochs')
plt.ylabel('RMSE')
plt.title('XGBoost Training and Test Loss')
plt.legend()
#plt.grid(True)
plt.show()

# %% Evaluate the autoencoder

# Make predictions
predictions_scaled = model.predict(X_test)

# Assuming predictions_scaled is a NumPy array and qt is the fitted QuantileTransformer
#feature_names = ['feature1']  # replace with your actual feature names

#predictions_scaled = predictions_scaled.reshape(-1,1)

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

# Optionally, calculate Root Mean Squared Error (RMSE)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"Root Mean Squared Error: {rmse}")

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

from Parity_Plots import plot_qq, plot_parity

# Plot QQ plots for scaled data
plot_qq(y_test, predictions_scaled, 'QQplot/qq_scaled_data.jpg')

# Plot QQ plots for original data
plot_qq(y_test_original, predictions, 'QQplot/qq_original_data.jpg')

# Plot scatter plots for scaled data
plot_parity(y_test, predictions_scaled, output_columns, 'scatterplots-scaled', log_scale=False)

# Plot scatter plots for original data in log scale
plot_parity(y_test_original, predictions, output_columns, 'scatterplots-original', log_scale=False)

# %%

# Example function to calculate metrics and generate the DataFrame
def generate_metrics_dataframe(predictions, y_test_original, output_columns):
    # Calculate metrics for each feature
    mse_per_feature = mean_squared_error(y_test_original, predictions, multioutput='raw_values')
    mae_per_feature = mean_absolute_error(y_test_original, predictions, multioutput='raw_values')
    r2_per_feature = r2_score(y_test_original, predictions, multioutput='raw_values')
    ev_per_feature = explained_variance_score(y_test_original, predictions, multioutput='raw_values')
    
    # Create a DataFrame to hold the results
    results_dict = {
        "Model Name": ["XGBoost"],
    }
    
    for i, col in enumerate(output_columns):
        results_dict[f"{col}"] = [f"MSE: {mse_per_feature[i]:.4g} (R$^2$: {r2_per_feature[i]:.4g})"]
    
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
