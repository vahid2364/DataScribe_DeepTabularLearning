#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 20:03:08 2024

@author: attari.v
"""


from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, LeakyReLU, Layer
import tensorflow as tf

# Custom Sampling Layer for VAE
class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a latent variable."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seed_generator = 1337  # Directly use a seed integer

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = K.shape(z_mean)[0]  # Use Keras backend to get the shape
        dim = K.shape(z_mean)[1]    # Use Keras backend to get the shape
        epsilon = tf.random.normal(shape=(batch, dim), seed=self.seed_generator)  # Use tf.random for sampling
        return z_mean + K.exp(0.5 * z_log_var) * epsilon  # Use K.exp for the exponential calculation
    

def create_variational_encoder(input_dim, latent_dim, layer_sizes=[2056, 1024, 512, 256], lamb=1e-3, rate=0.2, alpha=0.1):
    input_layer = Input(shape=(input_dim,))  # The input layer for tabular data
    x = input_layer

    # Build the dense layers
    for idx, layer_size in enumerate(layer_sizes):
        x = Dense(layer_size, kernel_regularizer=tf.keras.regularizers.l2(lamb))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(negative_slope=alpha)(x)  # LeakyReLU with alpha
        x = Dropout(rate)(x)

    # Latent space outputs
    z_mean = Dense(latent_dim, name="z_mean")(x)
    z_log_var = Dense(latent_dim, name="z_log_var")(x)

    # Sampling layer to get the latent space representation
    z = Sampling()([z_mean, z_log_var])

    # Return inputs and outputs instead of the encoder model
    return input_layer, [z_mean, z_log_var, z]

def create_variational_decoder(output_dim, latent_dim, layer_sizes=[256, 512, 1024, 2056], lamb=1e-3, rate=0.2, alpha=0.1):
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = latent_inputs

    # Build the dense layers
    for layer_size in layer_sizes:
        x = Dense(layer_size, kernel_regularizer=tf.keras.regularizers.l2(lamb))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(negative_slope=alpha)(x)  # LeakyReLU with alpha
        x = Dropout(rate)(x)

    # Final output layer
    decoded_output = Dense(output_dim, activation='sigmoid')(x)

    # Return the inputs and outputs instead of the decoder model
    return latent_inputs, decoded_output

# %% 

from tensorflow import keras  # Import Keras from TensorFlow

class VAE(keras.Model):  # Inherit from keras.Model
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    # Define the call() method for inference
    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        return self.decoder(z)

    def train_step(self, data):
        # Unpack the input data. Assumes data is a tuple (x, y).
        x, y = data

        with tf.GradientTape() as tape:
            # Get latent variables from encoder
            z_mean, z_log_var, z = self.encoder(x)
            # Get predictions (reconstructed output) from decoder
            reconstruction = self.decoder(z)

            # Compute reconstruction loss (for regression, you can use MSE or MAE)
            reconstruction_loss = K.mean(
                K.sum(
                    keras.losses.mean_squared_error(y, reconstruction),  # MSE for regression
                    axis=-1
                )
            )

            # Compute KL divergence loss
            kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=1)
            kl_loss = K.mean(kl_loss)

            # Total loss
            total_loss = reconstruction_loss + kl_loss

        # Compute gradients and apply them
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        # Return the tracked metrics
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    
    def test_step(self, data):
        x, y = data
        z_mean, z_log_var, z = self.encoder(x)
        reconstruction = self.decoder(z)
    
        # Compute reconstruction loss
        reconstruction_loss = K.mean(
            keras.losses.mean_squared_error(y, reconstruction)
        )
    
        # Compute KL divergence loss
        kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=1)
        kl_loss = K.mean(kl_loss)
    
        # Total loss
        total_loss = reconstruction_loss + kl_loss
    
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }