#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 01:28:08 2024

@author: attari.v
"""

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler

# Custom Callback
class SaveAtLastEpoch(tf.keras.callbacks.Callback):
    """
    Custom callback to save the model at the last epoch.
    """
    def __init__(self, filepath):
        super(SaveAtLastEpoch, self).__init__()
        self.filepath = filepath

    def on_epoch_end(self, epoch, logs=None):
        if epoch == self.params['epochs'] - 1:  # Check if it's the last epoch
            self.model.save(self.filepath)
            print(f"Model saved at the last epoch: {epoch + 1}")


def step_decay_schedule(initial_lr=0.0003, decay_factor=0.98, step_size=10):
    """
    Learning rate scheduler for step decay.
    """
    def schedule(epoch, lr):
        return initial_lr * (decay_factor ** (epoch // step_size))
    return LearningRateScheduler(schedule)


def train_autoencoder(
    model, 
    X_train, 
    y_train, 
    epochs=100, 
    batch_size=32, 
    validation_split=0.1, 
    validation_data=None,  # Add validation_data as a parameter
    learning_rate=0.001, 
    patience=30, 
    checkpoint_filepath="autoencoder_model_final.keras"
):
    """
    Trains an autoencoder model using specified callbacks.

    Parameters:
    - model (Model): The compiled autoencoder model.
    - X_train (ndarray): Training input data.
    - y_train (ndarray): Training output data.
    - epochs (int): Number of training epochs.
    - batch_size (int): Batch size for training.
    - validation_split (float): Fraction of training data to use for validation.
    - validation_data (tuple): Tuple (X_test, y_test) for validation data. Overrides validation_split if provided.
    - learning_rate (float): Initial learning rate for the model.
    - patience (int): Number of epochs with no improvement for early stopping.
    - checkpoint_filepath (str): Filepath to save the best model weights.

    Returns:
    - History: Training history object.
    """
    # Define callbacks
    callbacks = [
        step_decay_schedule(initial_lr=learning_rate),
        EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=False),
        ModelCheckpoint(filepath=checkpoint_filepath, monitor='val_loss', save_best_only=True, mode='min')
    ]

    # Determine validation argument
    fit_kwargs = {}
    if validation_data is not None:
        fit_kwargs['validation_data'] = validation_data
    else:
        fit_kwargs['validation_split'] = validation_split

    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=callbacks,
        verbose=1,
        **fit_kwargs  # Use appropriate validation argument
    )

    return history

