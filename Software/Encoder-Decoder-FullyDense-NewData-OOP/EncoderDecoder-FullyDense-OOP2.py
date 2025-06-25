#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 11:21:11 2024

@author: attari.v
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, QuantileTransformer
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
from FullyDense_Model import create_complex_encoder, create_complex_decoder
from tensorflow.keras import backend as K

# Sigmoid function
def sigmoid_transform(x):
    return 1 / (1 + np.exp(-x))

# Inverse Sigmoid function
def sigmoid_inverse_transform(x):
    return -np.log((1 / x) - 1)

# Scaling Class
class Scaling:
    def __init__(self, input_columns, output_columns):
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.input_scaler = None
        self.output_scaler = None
        self.pt_inputs = None
        self.pt_outputs = None
        self.qt_inputs = None
        self.qt_outputs = None

    # Function to scale data
    def scale_data(self, df, 
                   apply_sc=False, scaling_method='minmax', 
                   apply_pt=False, pt_method='yeo-johnson', 
                   apply_qt=False, qt_method='normal', 
                   apply_log1p=False, apply_sigmoid=False, 
                   apply_sqrt=False, sqrt_constant=1, 
                   apply_cbrt=False, cbrt_constant=1):

        # Copy the data
        transformed_data = df.copy()

        # Separate inputs and outputs
        inputs = transformed_data[self.input_columns].to_numpy()
        outputs = transformed_data[self.output_columns].to_numpy()

        # Apply log1p transformation if requested
        if apply_log1p:
            inputs = np.log1p(inputs)
            outputs = np.log1p(outputs)

        # Apply Sigmoid transformation if requested
        if apply_sigmoid:
            inputs = sigmoid_transform(inputs)
            outputs = sigmoid_transform(outputs)

        # Apply PowerTransformer if requested
        if apply_pt:
            print('pt applied')
            self.pt_inputs = PowerTransformer(method=pt_method)
            self.pt_outputs = PowerTransformer(method=pt_method)
            inputs = self.pt_inputs.fit_transform(inputs)
            outputs = self.pt_outputs.fit_transform(outputs)
            joblib.dump(self.pt_inputs, 'scales/power_transformer_inputs.save')
            joblib.dump(self.pt_outputs, 'scales/power_transformer_outputs.save')

        # Apply QuantileTransformer if requested
        if apply_qt:
            print('qt applied')
            self.qt_inputs = QuantileTransformer(output_distribution=qt_method, n_quantiles=1000)
            self.qt_outputs = QuantileTransformer(output_distribution=qt_method, n_quantiles=1000)
            inputs = self.qt_inputs.fit_transform(inputs)
            outputs = self.qt_outputs.fit_transform(outputs)
            joblib.dump(self.qt_inputs, 'scales/quantile_transformer_inputs.save')
            joblib.dump(self.qt_outputs, 'scales/quantile_transformer_outputs.save')

        # Apply square root transformation if requested
        # Apply square root transformation √(constant - x) if requested
        if apply_sqrt:
            inputs = np.sqrt(np.maximum(0, sqrt_constant - inputs))
            outputs = np.sqrt(np.maximum(0, sqrt_constant - outputs))

        # Apply cube root transformation if requested
        if apply_cbrt:
            inputs = np.cbrt(cbrt_constant - inputs)
            outputs = np.cbrt(cbrt_constant - outputs)

        # Apply MinMaxScaler or StandardScaler if requested
        if apply_sc:
            if scaling_method == 'minmax':
                self.input_scaler = MinMaxScaler()
                self.output_scaler = MinMaxScaler()
            else:
                raise ValueError(f"Unknown scaling method: {scaling_method}. Choose 'minmax'.")
    
            inputs = self.input_scaler.fit_transform(inputs)
            outputs = self.output_scaler.fit_transform(outputs)
            joblib.dump(self.input_scaler, 'scales/input_scaler.save')
            joblib.dump(self.output_scaler, 'scales/output_scaler.save')

        return inputs, outputs

    # Function to inverse scale data
    def inverse_scale_data(self, inputs_scaled, outputs_scaled, 
                           apply_log1p=False, apply_sigmoid=False):
        
        inputs_inv, outputs_inv = inputs_scaled, outputs_scaled
    
        # Inverse scaling for inputs and outputs using standard scalers
        if self.input_scaler:
            print('input_scaler de-scaled')
            inputs_inv = self.input_scaler.inverse_transform(inputs_scaled)
        if self.output_scaler:
            outputs_inv = self.output_scaler.inverse_transform(outputs_scaled)
    
        # Apply inverse QuantileTransformer if used
        if self.qt_inputs:
            print('qt_inputs de-scaled')
            inputs_inv = self.qt_inputs.inverse_transform(inputs_inv)
        if self.qt_outputs:
            outputs_inv = self.qt_outputs.inverse_transform(outputs_inv)
    
        # Apply inverse PowerTransformer if used
        if self.pt_inputs:
            inputs_inv = self.pt_inputs.inverse_transform(inputs_inv)
        if self.pt_outputs:
            outputs_inv = self.pt_outputs.inverse_transform(outputs_inv)
    
        # Reverse log1p transformation if applied
        if apply_log1p:
            inputs_inv = np.expm1(inputs_inv)
            outputs_inv = np.expm1(outputs_inv)
    
        # Reverse sigmoid transformation if applied
        if apply_sigmoid:
            inputs_inv = sigmoid_inverse_transform(inputs_inv)
            outputs_inv = sigmoid_inverse_transform(outputs_inv)
    
        return inputs_inv, outputs_inv


def rmse(y_true, y_pred):
    return K.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))

#    msle_loss = MeanSquaredLogarithmicError()


# Model Handler Class
class AutoencoderModel:
    def __init__(self, input_dim, output_dim, latent_dim=192):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.encoder = None
        self.decoder = None
        self.autoencoder = None

    def build_encoder_decoder(self):
        # Use your original code to create the encoder and decoder
        self.encoder = create_complex_encoder(
            input_dim=self.input_dim, 
            latent_dim=self.latent_dim, 
            num_layers=2, 
            neurons_per_layer=[128, 256], 
            lamb=3.69e-4, alp=0.0164, rate=0.1
        )

        self.decoder = create_complex_decoder(
            output_dim=self.output_dim, 
            latent_dim=self.latent_dim, 
            num_layers=2, 
            neurons_per_layer=[256, 128], 
            lamb=3.69e-4, alp=0.0164, rate=0.1
        )
        
    def compile_autoencoder(self):
        # Create the autoencoder model
        autoencoder_input = Input(shape=(self.input_dim,))
        encoded = self.encoder(autoencoder_input)
        decoded = self.decoder(encoded)
        self.autoencoder = Model(inputs=autoencoder_input, outputs=decoded)

        # Compile the autoencoder with Adam optimizer and MSE loss
        self.autoencoder.compile(optimizer='adam', loss='mse')
        #self.autoencoder.compile(optimizer='adam', loss='mse', metrics=[rmse])
        #self.autoencoder.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredLogarithmicError())

    def train(self, X_train, y_train, epochs=50, batch_size=96, validation_split=0.1, callbacks=None):
        # Train the autoencoder model
        return self.autoencoder.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                                    validation_split=validation_split, callbacks=callbacks)

    def evaluate(self, X_test, y_test):
        # Evaluate the model
        return self.autoencoder.evaluate(X_test, y_test)

    def predict(self, X_test):
        # Make predictions
        return self.autoencoder.predict(X_test)

# Visualization and Evaluation Class
class Visualizer:
    @staticmethod
    def plot_kde(data, columns, log_scale=False, filename='kde_plot.jpg'):
        """
        Universal function to plot KDE for Pandas DataFrame, NumPy arrays, or lists.
        
        Parameters:
        - data: The input data, can be a Pandas DataFrame, NumPy array, or list.
        - columns: List of column names (for DataFrame) or indices (for arrays/lists).
        - log_scale: Boolean, whether to plot the KDE on a logarithmic scale.
        - filename: The filename for saving the plot.
        """
        
        plt.figure(figsize=(8, 6))
        
        # Check if the input data is a DataFrame
        if isinstance(data, pd.DataFrame):
            for col in columns:
                sns.kdeplot(data[col], label=col, fill=True, log_scale=log_scale)
        
        # If it's a NumPy array or a list
        elif isinstance(data, np.ndarray) or isinstance(data, list):
            # Convert to NumPy array for easier indexing
            data = np.array(data)
                        
            for i, col in enumerate(columns):
                sns.kdeplot(data[:, i], label=f"Column {col}", fill=True, log_scale=log_scale)
        
        else:
            raise TypeError("Input data must be a Pandas DataFrame, NumPy array, or list.")
        
        plt.legend()
        plt.savefig(filename)
        plt.show()

    @staticmethod
    def plot_loss(history, filename='loss_plot.jpg', log_scale=False):
        plt.figure(figsize=(8, 4))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.yscale('log')
        plt.legend()
        plt.savefig(filename)
        plt.show()

    @staticmethod    
    def scatter_plot(y_true, y_pred, idx=0, filename='scatter_plot.jpg', log_scale=False):
        # Make a copy to avoid modifying the original arrays
        y_true = y_true.copy()
        y_pred = y_pred.copy()
    
        if log_scale:
            min_val = np.min(y_true[:, idx])
            epsilon = 1e-12  # Small shift for near-zero or negative values
    
            if min_val >= 0:
                # Safe to use log1p
                y_true_log = np.log1p(y_true[:, idx])
                y_pred_log = np.log1p(y_pred[:, idx])
                print(f"[Info] Applied log1p since min(y_true) = {min_val:.3e}")
            
            elif min_val > -epsilon:
                # Safe to use log(x + epsilon)
                y_true_log = np.log(y_true[:, idx] + epsilon)
                y_pred_log = np.log(y_pred[:, idx] + epsilon)
                print(f"[Info] Applied log(x + epsilon) since min(y_true) = {min_val:.3e}")
            
            else:
                # Unsafe to apply log — mask invalid values
                mask = y_true[:, idx] > -1
                y_true_log = np.log1p(y_true[mask, idx])
                y_pred_log = np.log1p(y_pred[mask, idx])
                print(f"[Warning] Detected y_true < -1 for idx={idx}. Masked before log1p.")
            
            # Compute log-scale metrics
            mse = mean_squared_error(y_true_log, y_pred_log)
            r2 = r2_score(y_true_log, y_pred_log)
    
            # Plot log values
            plt.figure(figsize=(7, 7))
            plt.scatter(y_true_log, y_pred_log, label='Predictions')
            plt.plot([min(y_true_log), max(y_true_log)],
                     [min(y_true_log), max(y_true_log)],
                     c='black', label='Perfect Fit')
            plt.xlabel('Actual Outputs (Log Scale)')
            plt.ylabel('Predicted Outputs (Log Scale)')
    
        else:
            # Compute normal metrics
            mse = mean_squared_error(y_true[:, idx], y_pred[:, idx])
            r2 = r2_score(y_true[:, idx], y_pred[:, idx])
    
            # Plot original values
            plt.figure(figsize=(7, 7))
            plt.scatter(y_true[:, idx], y_pred[:, idx], label='Predictions')
            plt.plot([min(y_true[:, idx]), max(y_true[:, idx])],
                     [min(y_true[:, idx]), max(y_true[:, idx])],
                     c='black', label='Perfect Fit')
            plt.xlabel('Actual Outputs')
            plt.ylabel('Predicted Outputs')
    
        # Annotate the plot with MSE and R²
        plt.text(0.05, 0.95, f'MSE: {mse:.3g}', transform=plt.gca().transAxes,
                 fontsize=15, verticalalignment='top')
        plt.text(0.05, 0.90, f'R$^2$: {r2:.3f}', transform=plt.gca().transAxes,
                 fontsize=15, verticalalignment='top')
    
        plt.tight_layout()
        plt.savefig(filename)
        plt.show()
    
# Main Execution Logic
if __name__ == "__main__":
    
    os.makedirs('results', exist_ok=True)
    os.makedirs('scales', exist_ok=True)
    
    # Paths and column definitions
    csv_file_path = '../../input_data/v3/IQR_dataframe-NbCrVWZr_data_stoic_creep_equil_v3.csv'  # Replace with your CSV file path
    input_columns = ['Nb', 'Cr', 'V', 'W', 'Zr']
    output_columns = ['Creep Merit'] # 'Creep Merit','25 Min Creep CB [1/s]','1000 Min Creep CB [1/s]'
    sanitized_output_columns = [col.replace(' ', '_').replace('[', '').replace(']', '').replace('/', '_') for col in output_columns]

    # Load data
    df = pd.read_csv(csv_file_path)

    Visualizer.plot_kde(df, output_columns , log_scale=True, filename='results/kde_plot_'+ '_'.join(sanitized_output_columns)+'.jpg')

    # --- Optional Thresholding Parameters ---
    apply_threshold = False  # Set to False to skip thresholding
    threshold_column = '1500 Min Creep CB [1/s]'
    threshold_value = 1e-3  # Customize this threshold as needed
    inclusive = False       # Use True for <= threshold, False for < threshold
    
    # --- Apply Thresholding if Enabled ---
    if apply_threshold:
        threshold_series = df[threshold_column]
        if inclusive:
            df = df[threshold_series <= threshold_value]
        else:
            df = df[threshold_series < threshold_value]
        df_below_threshold = df.copy()
        print(f"[Info] Applied threshold on '{threshold_column}' with value {threshold_value} (inclusive={inclusive})")
    else:
        df_below_threshold = df.copy()
        print("[Info] Skipped thresholding")

    Visualizer.plot_kde(df, output_columns , log_scale=False, filename='results/kde_plot_'+ '_'.join(sanitized_output_columns)+'.jpg')

    # Scaling and Data Preprocessing
    scaling = Scaling(input_columns=input_columns, output_columns=output_columns)
    
    # Apply scaling (for example, using MinMaxScaler)
    inputs_scaled, outputs_scaled = scaling.scale_data(
        df, 
        apply_sc=False, scaling_method='minmax', 
        apply_qt=True, qt_method='uniform', 
        apply_sqrt=False, sqrt_constant=1, 
        apply_log1p=False, 
        apply_sigmoid=False
    )

    Visualizer.plot_kde(outputs_scaled, output_columns, log_scale=True, filename='results/kde_plot_scaled_'+ '_'.join(sanitized_output_columns)+'.jpg')  
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(inputs_scaled, outputs_scaled, test_size=0.1, random_state=42)

    # Build and Train Autoencoder Model
    autoencoder_model = AutoencoderModel(input_dim=X_train.shape[1], output_dim=y_train.shape[1], latent_dim=512)
    autoencoder_model.build_encoder_decoder()
    autoencoder_model.compile_autoencoder()
    #autoencoder_model.autoencoder.summary()    
    autoencoder_model.encoder.summary()    
    autoencoder_model.decoder.summary()    
    
    
    
    # Define the learning rate schedule function
    def step_decay_schedule(initial_lr=1.26e-4, decay_factor=0.98, step_size=30):
        def schedule(epoch, lr):
            return initial_lr * (decay_factor ** np.floor(epoch / step_size))
        return LearningRateScheduler(schedule)
    
    callbacks = [
        step_decay_schedule(),
        EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=False),
        ModelCheckpoint(filepath='results/autoencoder_model_final_'+ '_'.join(output_columns)+'.keras', monitor='val_loss', save_best_only=True, mode='min')  # Save the best weights
    ]
    
    # Train the model
    history = autoencoder_model.train(X_train, y_train, epochs=150, batch_size=96, callbacks=callbacks)

    # Predict using the model
    predictions = autoencoder_model.predict(X_test)

    # Inverse scaling for test and predictions
    X_test_inv, y_test_inv = scaling.inverse_scale_data(X_test, y_test, apply_log1p=False)
    _, predictions_inv = scaling.inverse_scale_data(X_test, predictions)

    # Visualize Loss
    Visualizer.plot_loss(history, filename='results/loss_plot_'+ '_'.join(sanitized_output_columns)+'.jpg')

    # Visualize Predictions
    Visualizer.scatter_plot(y_test, predictions, filename='results/scatter_plot_scaled_'+ '_'.join(sanitized_output_columns)+'.jpg', log_scale=False)
    Visualizer.scatter_plot(y_test_inv, predictions_inv, filename='results/scatter_plot_org_'+ '_'.join(sanitized_output_columns)+'.jpg', log_scale=True)