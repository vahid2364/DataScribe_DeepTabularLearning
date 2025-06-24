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
    def __init__(self, units, dropout_rate=0.2, **kwargs):
        super(DNFBlock, self).__init__(**kwargs)
        self.units = units
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.linear_separators = Dense(self.units, activation='linear')
        self.batch_norm = BatchNormalization()
        self.dropout = Dropout(self.dropout_rate)

    def call(self, inputs):
        x = self.linear_separators(inputs)
        x = Activation('relu')(x)  # Linear separators with ReLU activation (conjunction)
        x = self.batch_norm(x)
        x = self.dropout(x)
        return x

# DNNF Encoder
def create_dnnf_encoder(input_dim, latent_dim, num_conjunctions=10, conjunction_units=64, dropout_rate=0.2):
    input_layer = Input(shape=(input_dim,))
    x = input_layer

    # Construct multiple DNF blocks (AND conditions followed by OR)
    conjunctions = []
    for _ in range(num_conjunctions):
        conj = DNFBlock(conjunction_units, dropout_rate=dropout_rate)(x)
        conjunctions.append(conj)

    # OR logic: combine the conjunction outputs by adding them up
    x = Add()(conjunctions)

    # Output layer for the latent space
    encoded_output = Dense(latent_dim, activation='linear')(x)
    encoder = Model(inputs=input_layer, outputs=encoded_output, name="dnnf_encoder")
    return encoder

# DNNF Decoder
def create_dnnf_decoder(output_dim, latent_dim, num_conjunctions=10, conjunction_units=64, dropout_rate=0.2):
    latent_inputs = Input(shape=(latent_dim,))
    x = latent_inputs

    # Construct DNF blocks for decoding
    conjunctions = []
    for _ in range(num_conjunctions):
        conj = DNFBlock(conjunction_units, dropout_rate=dropout_rate)(x)
        conjunctions.append(conj)

    # OR logic: combine the conjunction outputs by adding them up
    x = Add()(conjunctions)

    # Output layer for reconstruction
    decoded_output = Dense(output_dim, activation='sigmoid')(x)
    decoder = Model(inputs=latent_inputs, outputs=decoded_output, name="dnnf_decoder")
    return decoder

