#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 13:00:55 2024

@author: attari.v
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, LeakyReLU


def create_complex_encoder(input_dim, latent_dim, num_layers=5, neurons_per_layer=None, lamb=1e-6, alp=0.1, rate=0.1):
    """
    Creates an encoder model with dynamic number of layers and neurons per layer.
    
    Parameters:
    - input_dim: The input dimension.
    - latent_dim: The output dimension of the encoder (the size of the latent space).
    - num_layers: Number of hidden layers.
    - neurons_per_layer: List of number of neurons for each layer. If None, a default pattern will be used.
    - lamb: Regularization parameter (L2 regularization).
    - alp: LeakyReLU alpha parameter.
    - rate: Dropout rate.
    
    Returns:
    - encoder: The Keras model for the encoder.
    """
    input_layer = Input(shape=(input_dim,))
    
    if neurons_per_layer is None:
        # Default neurons per layer if not provided
        neurons_per_layer = [2056, 1024, 512, 256, 128][:num_layers]
    
    x = input_layer
    for i in range(num_layers):
        x = Dense(neurons_per_layer[i], kernel_regularizer=tf.keras.regularizers.l2(lamb))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=alp)(x)
        x = Dropout(rate)(x)
    
    encoded_output = Dense(latent_dim, activation='linear')(x)
    encoder = Model(inputs=input_layer, outputs=encoded_output)
    return encoder


def create_complex_decoder(output_dim, latent_dim, num_layers=5, neurons_per_layer=None, lamb=1e-6, alp=0.1, rate=0.1):
    """
    Creates a decoder model with dynamic number of layers and neurons per layer.
    
    Parameters:
    - output_dim: The output dimension of the decoder.
    - latent_dim: The input dimension to the decoder (the size of the latent space).
    - num_layers: Number of hidden layers.
    - neurons_per_layer: List of number of neurons for each layer. If None, a default pattern will be used.
    - lamb: Regularization parameter (L2 regularization).
    - alp: LeakyReLU alpha parameter.
    - rate: Dropout rate.
    
    Returns:
    - decoder: The Keras model for the decoder.
    """
    encoded_input = Input(shape=(latent_dim,))
    
    if neurons_per_layer is None:
        # Default neurons per layer if not provided
        neurons_per_layer = [128, 256, 512, 1024, 2056][:num_layers]
    
    x = encoded_input
    for i in range(num_layers):
        x = Dense(neurons_per_layer[i], kernel_regularizer=tf.keras.regularizers.l2(lamb))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=alp)(x)
        x = Dropout(rate)(x)
    
    decoded_output = Dense(output_dim, activation='sigmoid')(x)
    decoder = Model(inputs=encoded_input, outputs=decoded_output)
    return decoder