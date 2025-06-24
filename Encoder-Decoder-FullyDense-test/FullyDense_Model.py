#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 13:00:55 2024

@author: attari.v
"""


import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, LeakyReLU


def create_complex_encoder(input_dim, latent_dim):
    input_layer = Input(shape=(input_dim,))
    lamb = 1e-6
    rate = 0.1
    alp  = 0.1

    x = Dense(2056, kernel_regularizer=tf.keras.regularizers.l2(lamb))(input_layer)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=alp)(x)
    x = Dropout(rate)(x)
    
    x = Dense(1024, kernel_regularizer=tf.keras.regularizers.l2(lamb))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=alp)(x)
    x = Dropout(rate)(x)
    
    x = Dense(512, kernel_regularizer=tf.keras.regularizers.l2(lamb))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=alp)(x)
    x = Dropout(rate)(x)
    
    x = Dense(256, kernel_regularizer=tf.keras.regularizers.l2(lamb))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=alp)(x)
    x = Dropout(rate)(x)
    
    x = Dense(128, kernel_regularizer=tf.keras.regularizers.l2(lamb))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=alp)(x)
    x = Dropout(rate)(x)
    
    # x = Dense(64, kernel_regularizer=tf.keras.regularizers.l2(lamb))(x)
    # x = BatchNormalization()(x)
    # x = LeakyReLU(alpha=alp)(x)
    # x = Dropout(rate)(x)
    
    encoded_output = Dense(latent_dim, activation='linear')(x)
    encoder = Model(inputs=input_layer, outputs=encoded_output)
    return encoder

def create_complex_decoder(output_dim, latent_dim):
    encoded_input = Input(shape=(latent_dim,))
    lamb = 1e-6
    rate = 0.1
    alp  = 0.1
    
    x = Dense(128, kernel_regularizer=tf.keras.regularizers.l2(lamb))(encoded_input)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=alp)(x)
    x = Dropout(rate)(x)
    
    x = Dense(256, kernel_regularizer=tf.keras.regularizers.l2(lamb))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=alp)(x)
    x = Dropout(rate)(x)
    
    x = Dense(512, kernel_regularizer=tf.keras.regularizers.l2(lamb))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=alp)(x)
    x = Dropout(rate)(x)
    
    x = Dense(1024, kernel_regularizer=tf.keras.regularizers.l2(lamb))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=alp)(x)
    x = Dropout(rate)(x)

    x = Dense(2056, kernel_regularizer=tf.keras.regularizers.l2(lamb))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=alp)(x)
    x = Dropout(rate)(x)
    
    decoded_output = Dense(output_dim, activation='sigmoid')(x)
    decoder = Model(inputs=encoded_input, outputs=decoded_output)
    return decoder