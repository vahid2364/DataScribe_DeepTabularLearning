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

# Data Preprocessing Class
class DataPreprocessor:
    def __init__(self, csv_file_path, input_columns, output_columns):
        self.csv_file_path = csv_file_path
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.df = pd.read_csv(self.csv_file_path)
        self.X = self.df[self.input_columns].values
        self.y = self.df[self.output_columns].values
        self.input_scaler = None
        self.output_scaler = None
        self.pt = None
        self.qt = None

    def scale_data(self, apply_sc=True, scaling_method='minmax', apply_qt=False, qt_method='uniform'):
        if apply_sc:
            if scaling_method == 'minmax':
                self.input_scaler = MinMaxScaler()
                self.output_scaler = MinMaxScaler()
            # Apply scaling
            self.X = self.input_scaler.fit_transform(self.X)
            self.y = self.output_scaler.fit_transform(self.y)
            joblib.dump(self.input_scaler, 'scales/input_scaler.save')
            joblib.dump(self.output_scaler, 'scales/output_scaler.save')

        if apply_qt:
            self.qt = QuantileTransformer(output_distribution=qt_method)
            self.X = self.qt.fit_transform(self.X)
            self.y = self.qt.fit_transform(self.y)
            joblib.dump(self.qt, 'scales/quantile_transformer.save')

        return self.X, self.y

    def inverse_scale_data(self, X_scaled, y_scaled):
        X_inv = self.input_scaler.inverse_transform(X_scaled) if self.input_scaler else X_scaled
        y_inv = self.output_scaler.inverse_transform(y_scaled) if self.output_scaler else y_scaled
        return X_inv, y_inv

    def train_test_split(self, test_size=0.1, random_state=42):
        return train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)

# Model Handler Class
class AutoencoderModel:
    def __init__(self, input_dim, output_dim, latent_dim=128):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.encoder = None
        self.decoder = None
        self.autoencoder = None

    def build_encoder(self):
        input_layer = tf.keras.Input(shape=(self.input_dim,))
        encoded = tf.keras.layers.Dense(256, activation='relu')(input_layer)
        encoded = tf.keras.layers.Dense(self.latent_dim, activation='relu')(encoded)
        self.encoder = tf.keras.Model(input_layer, encoded)

    def build_decoder(self):
        latent_input = tf.keras.Input(shape=(self.latent_dim,))
        decoded = tf.keras.layers.Dense(256, activation='relu')(latent_input)
        decoded = tf.keras.layers.Dense(self.output_dim, activation='linear')(decoded)
        self.decoder = tf.keras.Model(latent_input, decoded)

    def compile_autoencoder(self):
        autoencoder_input = tf.keras.Input(shape=(self.input_dim,))
        encoded = self.encoder(autoencoder_input)
        decoded = self.decoder(encoded)
        self.autoencoder = tf.keras.Model(autoencoder_input, decoded)
        self.autoencoder.compile(optimizer='adam', loss='mse')

    def train(self, X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, callbacks=None):
        return self.autoencoder.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                                    validation_split=validation_split, callbacks=callbacks)

    def evaluate(self, X_test, y_test):
        return self.autoencoder.evaluate(X_test, y_test)

    def predict(self, X_test):
        return self.autoencoder.predict(X_test)

# Visualization and Evaluation Class
class Visualizer:
    @staticmethod
    def plot_kde(df, columns, log_scale=False, filename='kde_plot.jpg'):
        plt.figure(figsize=(8, 6))
        for col in columns:
            sns.kdeplot(df[col], label=col, fill=True, log_scale=log_scale)
        plt.legend()
        plt.savefig(filename)
        plt.show()

    @staticmethod
    def plot_loss(history):
        plt.figure(figsize=(8, 4))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.legend()
        plt.savefig('loss_plot.jpg')
        plt.show()

    @staticmethod
    def scatter_plot(y_true, y_pred, idx=0, filename='scatter_plot.jpg'):
        plt.figure(figsize=(7, 7))
        plt.scatter(y_true[:, idx], y_pred[:, idx])
        plt.plot([min(y_true[:, idx]), max(y_true[:, idx])], [min(y_pred[:, idx]), max(y_pred[:, idx])], c='black')
        plt.xlabel('Actual Outputs')
        plt.ylabel('Predicted Outputs')
        plt.tight_layout()
        plt.savefig(filename)
        plt.show()

# Main Execution Logic
if __name__ == "__main__":
    # Paths and column definitions
    csv_file_path = '../../input_data/v3/IQR_dataframe-NbCrVWZr_data_stoic_creep_equil_v3.csv'  # Replace with your CSV file path
    input_columns = ['Nb', 'Cr', 'V', 'W', 'Zr']
    output_columns = ['Kou Criteria']

    # Data Preprocessing
    data_preprocessor = DataPreprocessor(csv_file_path, input_columns, output_columns)
    X_train, X_test, y_train, y_test = data_preprocessor.train_test_split()

    # Model Creation and Training
    autoencoder_model = AutoencoderModel(input_dim=X_train.shape[1], output_dim=y_train.shape[1])
    autoencoder_model.build_encoder()
    autoencoder_model.build_decoder()
    autoencoder_model.compile_autoencoder()

    # Train the autoencoder
    history = autoencoder_model.train(X_train, y_train)

    # Visualize Loss
    Visualizer.plot_loss(history)

    # Predictions and Evaluation
    predictions = autoencoder_model.predict(X_test)
    Visualizer.scatter_plot(y_test, predictions)

    # Inverse scaling and replotting
    _, y_test_original = data_preprocessor.inverse_scale_data(X_test, y_test)
    _, predictions_original = data_preprocessor.inverse_scale_data(X_test, predictions)
    Visualizer.scatter_plot(y_test_original, predictions_original, filename='scatter_plot_original.jpg')