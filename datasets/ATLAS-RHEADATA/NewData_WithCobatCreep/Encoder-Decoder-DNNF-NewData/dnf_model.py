#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 21:38:04 2024

@author: attari.v
"""

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, Activation, Add, Layer
from tensorflow.keras.models import Model

# Custom DNF Block
class DNFBlock(Layer):
    def __init__(self, units, dropout_rate=0.2, l2_reg=1e-4, **kwargs):
        super(DNFBlock, self).__init__(**kwargs)
        self.units = units
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg  # L2 regularization parameter

    def build(self, input_shape):
        # Apply L2 regularization in the Dense layers
        self.linear_separators = Dense(self.units, activation='linear', 
                                       kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg))
        self.batch_norm = BatchNormalization()
        self.dropout = Dropout(self.dropout_rate)

    def call(self, inputs):
        x = self.linear_separators(inputs)
        x = Activation('relu')(x)  # Linear separators with ReLU activation (conjunction)
        x = self.batch_norm(x)
        x = self.dropout(x)
        return x

# DNNF Encoder
def create_dnnf_encoder(input_dim, latent_dim, num_conjunctions=10, conjunction_units=64, dropout_rate=0.2, l2_reg=1e-4):
    input_layer = Input(shape=(input_dim,))
    x = input_layer

    # Construct multiple DNF blocks (AND conditions followed by OR)
    conjunctions = []
    for _ in range(num_conjunctions):
        conj = DNFBlock(conjunction_units, dropout_rate=dropout_rate, l2_reg=l2_reg)(x)
        conjunctions.append(conj)

    # OR logic: combine the conjunction outputs by adding them up
    x = Add()(conjunctions)

    # Output layer for the latent space
    encoded_output = Dense(latent_dim, activation='linear', kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
    encoder = Model(inputs=input_layer, outputs=encoded_output, name="dnnf_encoder")
    return encoder

# DNNF Decoder
def create_dnnf_decoder(output_dim, latent_dim, num_conjunctions=10, conjunction_units=64, dropout_rate=0.2, l2_reg=1e-4):
    latent_inputs = Input(shape=(latent_dim,))
    x = latent_inputs

    # Construct DNF blocks for decoding
    conjunctions = []
    for _ in range(num_conjunctions):
        conj = DNFBlock(conjunction_units, dropout_rate=dropout_rate, l2_reg=l2_reg)(x)
        conjunctions.append(conj)

    # OR logic: combine the conjunction outputs by adding them up
    x = Add()(conjunctions)

    # Output layer for reconstruction
    decoded_output = Dense(output_dim, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
    decoder = Model(inputs=latent_inputs, outputs=decoded_output, name="dnnf_decoder")
    return decoder