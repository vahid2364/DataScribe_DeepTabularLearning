#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 01:12:01 2024

@author: attari.v
"""

import json
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from modules.FullyDense_Model import create_complex_encoder, create_complex_decoder

def read_optuna_parameters(file_path, trial_index):
    """
    Reads Optuna trial parameters from a CSV file.

    Parameters:
    - file_path (str): Path to the Optuna CSV file.
    - trial_index (int): Index of the trial to extract parameters.

    Returns:
    - dict: Extracted parameters for the selected trial.
    """
    optuna_trials = pd.read_csv(file_path)
    params = optuna_trials.iloc[trial_index, :]
    
    # Extract parameters
    trial_params = {
        "latent_dim": int(params.params_latent_dim),
        "alpha": params.params_alpha,
        "lambda": params.params_lambda,
        "dropout_rate": params.params_drop_out_rate,
        "learning_rate": params.params_learning_rate,
        "batch_size": int(params.params_batch_size),
        "encoder_num_layers": params.params_num_layers_encoder,
        "decoder_num_layers": params.params_num_layers_decoder,
        "encoder_neurons": json.loads(params.user_attrs_neurons_per_layer_encoder),
        "decoder_neurons": json.loads(params.user_attrs_neurons_per_layer_decoder),
    }
    
    # Nicely display the parameters
    print("Trial Parameters:")
    for key, value in trial_params.items():
        print(f"  {key}: {value}")
    
    return trial_params


def build_autoencoder(params, input_dim, output_dim):
    """
    Builds an encoder-decoder model based on given parameters.

    Parameters:
    - params (dict): Parameters for the encoder and decoder.
    - input_dim (int): Dimension of the input data.
    - output_dim (int): Dimension of the output data.

    Returns:
    - Model: Compiled autoencoder model.
    """
    encoder = create_complex_encoder(
        input_dim=input_dim, 
        latent_dim=params["latent_dim"], 
        num_layers=params["encoder_num_layers"], 
        neurons_per_layer=params["encoder_neurons"], 
        lamb=params["lambda"], 
        alp=params["alpha"], 
        rate=params["dropout_rate"]
    )

    decoder = create_complex_decoder(
        output_dim=output_dim, 
        latent_dim=params["latent_dim"], 
        num_layers=params["decoder_num_layers"], 
        neurons_per_layer=params["decoder_neurons"], 
        lamb=params["lambda"], 
        alp=params["alpha"], 
        rate=params["dropout_rate"]
    )

    # Create the autoencoder model
    autoencoder_input = Input(shape=(input_dim,))
    encoded = encoder(autoencoder_input)
    decoded = decoder(encoded)
    autoencoder = Model(inputs=autoencoder_input, outputs=decoded)

    # Compile the model
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder, encoder, decoder


def print_model_summary(encoder, decoder, autoencoder):
    """
    Prints the summaries of the encoder, decoder, and autoencoder.

    Parameters:
    - encoder (Model): Encoder model.
    - decoder (Model): Decoder model.
    - autoencoder (Model): Autoencoder model.
    """
    print("Encoder Summary:")
    encoder.summary()
    print("\nDecoder Summary:")
    decoder.summary()
    print("\nAutoencoder Summary:")
    autoencoder.summary()


# Custom Loss Functions and Metrics
def mse_metric(y_true, y_pred):
    return tf.keras.losses.MeanSquaredError()(y_true, y_pred)

def mae_metric(y_true, y_pred):
    return tf.keras.losses.MeanAbsoluteError()(y_true, y_pred)

def bce_metric(y_true, y_pred):
    return tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)

def combined_loss(y_true, y_pred):
    mse_loss = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
    mae_loss = tf.keras.losses.MeanAbsoluteError()(y_true, y_pred)
    bce_loss = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    return mse_loss + mae_loss + bce_loss