#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 17:51:04 2024

@author: attari.v
"""

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, Add, Concatenate, Lambda
from tensorflow.keras.models import Model

def tab_transformer(input_dim_cat, input_dim_cont, d_model, num_heads, ff_dim, num_layers=3, rate=0.1):
    """
    Creates a TabTransformer model for tabular data with categorical and continuous features.

    Parameters:
    - input_dim_cat: The number of categorical features (each feature will be embedded).
    - input_dim_cont: The number of continuous features.
    - d_model: Dimensionality of the embedding and model layers.
    - num_heads: Number of attention heads in the Transformer.
    - ff_dim: Dimensionality of the feedforward network.
    - num_layers: Number of Transformer encoder layers.
    - rate: Dropout rate.

    Returns:
    - model: The Keras model for TabTransformer.
    """
    # Inputs for categorical and continuous features
    input_cat = Input(shape=(input_dim_cat,), name='categorical_features')
    input_cont = Input(shape=(input_dim_cont,), name='continuous_features')
    
    # Column Embedding for categorical features
    embeddings = [Dense(d_model)(Lambda(lambda x: tf.expand_dims(x[:, i], -1))(input_cat)) for i in range(input_dim_cat)]
    x_cat = tf.stack(embeddings, axis=1)  # Shape: (batch_size, num_categorical_features, d_model)

    # Transformer Encoder applied to categorical embeddings
    for _ in range(num_layers):
        # Multi-Head Attention
        attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x_cat, x_cat)
        attn_output = Dropout(rate)(attn_output)
        x_cat = Add()([x_cat, attn_output])  # Residual connection
        x_cat = LayerNormalization(epsilon=1e-6)(x_cat)
        
        # Feed Forward Network
        ff_output = Dense(ff_dim, activation="relu")(x_cat)
        ff_output = Dense(d_model)(ff_output)
        ff_output = Dropout(rate)(ff_output)
        x_cat = Add()([x_cat, ff_output])  # Residual connection
        x_cat = LayerNormalization(epsilon=1e-6)(x_cat)

    # Flatten the Transformer output for categorical features
    x_cat_flat = tf.reshape(x_cat, (-1, input_dim_cat * d_model))
    
    # Layer Normalization for continuous features
    x_cont = LayerNormalization()(input_cont)
    
    # Concatenate processed categorical and continuous features
    x = Concatenate()([x_cat_flat, x_cont])
    
    # Multi-Layer Perceptron (MLP) for final prediction
    x = Dense(128, activation="relu")(x)
    x = Dropout(rate)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(rate)(x)
    output = Dense(1, activation="linear", name="output")(x)  # Adjust output units/activation for your task

    # Define the model
    model = Model(inputs=[input_cat, input_cont], outputs=output)
    return model

# Example usage:
input_dim_cat = 5  # Number of categorical features
input_dim_cont = 5  # Number of continuous features
d_model = 64        # Dimension of the model layers
num_heads = 4       # Number of attention heads
ff_dim = 128        # Feedforward layer dimension
num_layers = 3      # Number of transformer encoder layers
rate = 0.1          # Dropout rate

# Instantiate and compile the model
tab_transformer_model = tab_transformer(input_dim_cat, input_dim_cont, d_model, num_heads, ff_dim, num_layers, rate)
tab_transformer_model.compile(optimizer="adam", loss="mse")  # Use an appropriate loss for your task
tab_transformer_model.summary()