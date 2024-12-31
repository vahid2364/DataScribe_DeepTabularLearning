#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 13:00:55 2024

@author: attari.v
"""


import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, LeakyReLU


def create_complex_encoder(input_dim, latent_dim, num_layers=4, neurons_per_layer=None, lamb=1e-6, rate=0.1, alp=0.1):
    if neurons_per_layer is None:
        # Default neuron sizes if none are provided
        neurons_per_layer = [2056, 1024, 512, 256]

    input_layer = Input(shape=(input_dim,))
    x = input_layer

    for i in range(num_layers):
        x = Dense(neurons_per_layer[i], kernel_regularizer=tf.keras.regularizers.l2(lamb))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(negative_slope=alp)(x)
        x = Dropout(rate)(x)

    encoded_output = Dense(latent_dim, activation='linear')(x)
    encoder = Model(inputs=input_layer, outputs=encoded_output)
    return encoder


def create_complex_decoder(output_dim, latent_dim, num_layers=4, neurons_per_layer=None, lamb=1e-6, rate=0.1, alp=0.1):
    if neurons_per_layer is None:
        # Default neuron sizes if none are provided
        neurons_per_layer = [256, 512, 1024, 2056]

    encoded_input = Input(shape=(latent_dim,))
    x = encoded_input

    for i in range(num_layers):
        x = Dense(neurons_per_layer[i], kernel_regularizer=tf.keras.regularizers.l2(lamb))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(negative_slope=alp)(x)
        x = Dropout(rate)(x)

    decoded_output = Dense(output_dim, activation='sigmoid')(x)
    decoder = Model(inputs=encoded_input, outputs=decoded_output)
    return decoder