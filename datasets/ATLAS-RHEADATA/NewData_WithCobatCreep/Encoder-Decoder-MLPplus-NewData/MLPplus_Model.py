#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 13:00:55 2024

@author: attari.v
"""

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, LeakyReLU, Reshape
from tensorflow.keras.models import Model

# Custom Ghost Batch Normalization Layer
def GhostBatchNormalization(ghost_batch_size=32, momentum=0.99, epsilon=1e-3):
    def batchnorm_with_ghost_batch(inputs):
        # Reshape the input to create ghost batches of shape [ghost_batch_size, batch_size // ghost_batch_size, ...]
        shape = tf.shape(inputs)
        ghost_batches = shape[0] // ghost_batch_size
        
        reshaped_inputs = tf.reshape(inputs, [ghost_batches, ghost_batch_size] + inputs.shape[1:])
        
        # Apply batch normalization to each ghost batch
        normalized = tf.keras.layers.BatchNormalization(momentum=momentum, epsilon=epsilon)(reshaped_inputs)
        
        # Reshape back to the original shape
        return tf.reshape(normalized, shape)
    
    return batchnorm_with_ghost_batch

# Custom Ghost Batch Normalization Layer
def GhostBatchNormalization(vbatch, **kwargs):
    return BatchNormalization(**kwargs)

# Function to create a complex encoder with Ghost Batch Normalization
def create_complex_encoder_with_gbn(input_dim, latent_dim, num_layers=5, neurons_per_layer=None, lamb=1e-6, alp=0.1, rate=0.1, vbatch=32):
    """
    Creates an encoder model with dynamic number of layers and neurons per layer.
    Ghost Batch Normalization (GBN) is applied.

    Parameters:
    - input_dim: The input dimension.
    - latent_dim: The output dimension of the encoder (the size of the latent space).
    - num_layers: Number of hidden layers.
    - neurons_per_layer: List of number of neurons for each layer. If None, a default pattern will be used.
    - lamb: Regularization parameter (L2 regularization).
    - alp: LeakyReLU alpha parameter.
    - rate: Dropout rate.
    - vbatch: Virtual batch size for Ghost Batch Normalization.

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
        x = GhostBatchNormalization(vbatch=vbatch)(x)
        x = LeakyReLU(alpha=alp)(x)
        x = Dropout(rate)(x)
    
    encoded_output = Dense(latent_dim, activation='linear')(x)
    encoder = Model(inputs=input_layer, outputs=encoded_output)
    return encoder

# Function to create a complex decoder with Ghost Batch Normalization
def create_complex_decoder_with_gbn(output_dim, latent_dim, num_layers=5, neurons_per_layer=None, lamb=1e-6, alp=0.1, rate=0.1, vbatch=32):
    """
    Creates a decoder model with dynamic number of layers and neurons per layer.
    Ghost Batch Normalization (GBN) is applied.

    Parameters:
    - output_dim: The output dimension of the decoder.
    - latent_dim: The input dimension to the decoder (the size of the latent space).
    - num_layers: Number of hidden layers.
    - neurons_per_layer: List of number of neurons for each layer. If None, a default pattern will be used.
    - lamb: Regularization parameter (L2 regularization).
    - alp: LeakyReLU alpha parameter.
    - rate: Dropout rate.
    - vbatch: Virtual batch size for Ghost Batch Normalization.

    Returns:
    - decoder: The Keras model for the decoder.
    """
    latent_inputs = Input(shape=(latent_dim,))
    
    if neurons_per_layer is None:
        # Default neurons per layer if not provided
        neurons_per_layer = [128, 256, 512, 1024, 2056][:num_layers]
    
    x = latent_inputs
    for i in range(num_layers):
        x = Dense(neurons_per_layer[i], kernel_regularizer=tf.keras.regularizers.l2(lamb))(x)
        x = GhostBatchNormalization(vbatch=vbatch)(x)
        x = LeakyReLU(alpha=alp)(x)
        x = Dropout(rate)(x)
    
    decoded_output = Dense(output_dim, activation='sigmoid')(x)
    decoder = Model(inputs=latent_inputs, outputs=decoded_output)
    return decoder
