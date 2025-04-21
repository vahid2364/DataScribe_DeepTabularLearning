#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 15:47:45 2024

@author: attari.v

Updated to use Transformer-based Encoder-Decoder architecture for tabular data.
"""

import tensorflow as tf
from tensorflow.keras.models import Model
#from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, Add, GlobalAveragePooling1D

#import tensorflow as tf
#from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, Add

def transformer_encoder(input_dim, d_model, num_heads, ff_dim, num_layers=3, rate=0.1):
    """
    Creates a Transformer-based encoder model with multi-head attention layers.
    
    Parameters:
    - input_dim: The input dimension.
    - d_model: Dimensionality of the output space of the encoder (latent space).
    - num_heads: Number of attention heads.
    - ff_dim: Dimensionality of the feedforward network.
    - num_layers: Number of transformer encoder layers.
    - rate: Dropout rate.

    Returns:
    - encoder: The Keras model for the encoder.
    """
    input_layer = Input(shape=(input_dim,))
    x = Dense(d_model)(input_layer)
    
    for _ in range(num_layers):
        # Multi-Head Attention block
        attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
        attn_output = Dropout(rate)(attn_output)
        x = Add()([x, attn_output])  # Residual connection
        x = LayerNormalization(epsilon=1e-6)(x)
        
        # Feedforward network
        ff_output = Dense(ff_dim, activation="relu")(x)
        ff_output = Dense(d_model)(ff_output)
        ff_output = Dropout(rate)(ff_output)
        x = Add()([x, ff_output])  # Residual connection
        x = LayerNormalization(epsilon=1e-6)(x)
    
    # No pooling layer; output directly
    encoder = Model(inputs=input_layer, outputs=x)
    return encoder

from tensorflow.keras.layers import Softmax

def transformer_decoder(output_dim, d_model, num_heads, ff_dim, num_layers=3, rate=0.1, task="regression"):
    """
    Creates a Transformer-based decoder model with multi-head attention layers.

    Parameters:
    - output_dim: The output dimension of the decoder.
    - d_model: Dimensionality of the input space to the decoder (latent space).
    - num_heads: Number of attention heads.
    - ff_dim: Dimensionality of the feedforward network.
    - num_layers: Number of transformer decoder layers.
    - rate: Dropout rate.
    - task: Type of task ("regression", "binary", or "multi-class").

    Returns:
    - decoder: The Keras model for the decoder.
    """
    encoded_input = Input(shape=(d_model,))
    x = encoded_input
    
    for _ in range(num_layers):
        # Multi-Head Attention block for self-attention
        attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
        attn_output = Dropout(rate)(attn_output)
        x = Add()([x, attn_output])  # Residual connection
        x = LayerNormalization(epsilon=1e-6)(x)
        
        # Feedforward network
        ff_output = Dense(ff_dim, activation="relu")(x)
        ff_output = Dense(d_model)(ff_output)
        ff_output = Dropout(rate)(ff_output)
        x = Add()([x, ff_output])  # Residual connection
        x = LayerNormalization(epsilon=1e-6)(x)
    
    # Set activation function based on task type
    if task == "regression":
        decoded_output = Dense(output_dim, activation='linear')(x)
    elif task == "binary":
        decoded_output = Dense(output_dim, activation='sigmoid')(x)
    elif task == "multi-class":
        # Ensure `softmax` is applied along the last axis explicitly
        decoded_output = Dense(output_dim)(x)
        decoded_output = Softmax(axis=-1)(decoded_output)
    else:
        raise ValueError("Unsupported task type. Use 'regression', 'binary', or 'multi-class'.")
        
    print('encoded_input', encoded_input)
    print('decoded_output', decoded_output)
    
    decoder = Model(inputs=encoded_input, outputs=decoded_output)
    return decoder


# # Example usage:
# input_dim = 1024  # Example input dimension
# latent_dim = 128  # Latent space dimension
# output_dim = 1    # Example output dimension (e.g., regression or classification task)
# d_model = 64      # Dimension of the model's output in the transformer layers
# num_heads = 4     # Number of attention heads
# ff_dim = 128      # Feedforward network dimension
# num_layers = 3    # Number of layers in encoder and decoder
# rate = 0.1        # Dropout rate

# # Instantiate the encoder and decoder models
# encoder = transformer_encoder(input_dim=input_dim, d_model=d_model, num_heads=num_heads, ff_dim=ff_dim, num_layers=num_layers, rate=rate)

# # Example usage
# output_dim = 1  # For binary classification, set output_dim=1. For multi-class, set output_dim to the number of classes.
# task = "binary"  # Change to "multi-class" or "regression" based on your task
# decoder = transformer_decoder(output_dim=output_dim, d_model=d_model, num_heads=num_heads, ff_dim=ff_dim, num_layers=num_layers, rate=rate, task=task)


# #decoder = transformer_decoder(output_dim=output_dim, d_model=d_model, num_heads=num_heads, ff_dim=ff_dim, num_layers=num_layers, rate=rate)

# # Define the inputs and outputs for the full encoder-decoder model
# encoder_inputs = Input(shape=(input_dim,))
# decoder_inputs = encoder(encoder_inputs)
# decoder_outputs = decoder(decoder_inputs)

# # Combine encoder and decoder into a single model
# transformer_model = Model(inputs=encoder_inputs, outputs=decoder_outputs)
# transformer_model.compile(optimizer="adam", loss="mse")  # Example loss function; use appropriate one based on task