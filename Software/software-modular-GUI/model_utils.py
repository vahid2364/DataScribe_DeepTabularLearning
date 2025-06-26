#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 02:37:12 2025

@author: attari.v
"""

# model_utils.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
from FullyDense_Model import create_complex_encoder, create_complex_decoder
from tensorflow.keras.callbacks import Callback

class AutoencoderModel:
    def __init__(self, input_dim, output_dim, latent_dim=192,
                 encoder_layers=2, encoder_neurons=[128, 256],
                 decoder_layers=2, decoder_neurons=[256, 128]):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.encoder_layers = encoder_layers
        self.encoder_neurons = encoder_neurons
        self.decoder_layers = decoder_layers
        self.decoder_neurons = decoder_neurons
        self.encoder = None
        self.decoder = None
        self.autoencoder = None

    def build_encoder_decoder(self):
        self.encoder = create_complex_encoder(
            input_dim=self.input_dim,
            latent_dim=self.latent_dim,
            num_layers=self.encoder_layers,
            neurons_per_layer=self.encoder_neurons,
            lamb=3.69e-4,
            alp=0.0164,
            rate=0.1
        )

        self.decoder = create_complex_decoder(
            output_dim=self.output_dim,
            latent_dim=self.latent_dim,
            num_layers=self.decoder_layers,
            neurons_per_layer=self.decoder_neurons,
            lamb=3.69e-4,
            alp=0.0164,
            rate=0.1
        )

    def compile_autoencoder(self):
        autoencoder_input = Input(shape=(self.input_dim,))
        encoded = self.encoder(autoencoder_input)
        decoded = self.decoder(encoded)
        self.autoencoder = Model(inputs=autoencoder_input, outputs=decoded)
        self.autoencoder.compile(optimizer='adam', loss='mse')

    def train(self, X_train, y_train, epochs=50, batch_size=96, validation_split=0.1, callbacks=None):
        return self.autoencoder.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks
        )

    def evaluate(self, X_test, y_test):
        return self.autoencoder.evaluate(X_test, y_test)

    def predict(self, X_test):
        return self.autoencoder.predict(X_test)

    def default_callbacks(self, output_name):
        return [
            EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=False),
            ModelCheckpoint(
                filepath=f'results/autoencoder_model_final_{output_name}.keras',
                monitor='val_loss',
                save_best_only=True,
                mode='min'
            )
        ]
    

class StreamlitProgressCallback(Callback):
    def __init__(self, epochs, progress_bar, status_text):
        super().__init__()
        self.epochs = epochs
        self.progress_bar = progress_bar
        self.status_text = status_text

    def on_epoch_end(self, epoch, logs=None):
        progress = int((epoch + 1) / self.epochs * 100)
        self.progress_bar.progress(progress)
        self.status_text.text(f"Epoch {epoch + 1}/{self.epochs} - Loss: {logs.get('loss'):.4f}")

def step_decay_schedule(initial_lr=1.26e-4, decay_factor=0.98, step_size=30):
    def schedule(epoch, lr):
        return initial_lr * (decay_factor ** np.floor(epoch / step_size))
    return LearningRateScheduler(schedule)