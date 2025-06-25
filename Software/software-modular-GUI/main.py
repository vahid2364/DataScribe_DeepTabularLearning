#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 02:23:30 2025

@author: attari.v
"""

# main.py
from config import CONFIG
from data_utils import load_and_filter_data, apply_scaling
from model_utils import AutoencoderModel, step_decay_schedule
from visualize import Visualizer

import os
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    os.makedirs('results', exist_ok=True)
    os.makedirs('scales', exist_ok=True)

    cfg = CONFIG
    output_name = '_'.join(cfg['output_columns']).replace(' ', '_').replace('[', '').replace(']', '').replace('/', '_')

    # Load and preprocess data
    df = load_and_filter_data(cfg)
    Visualizer.plot_kde(df, cfg['output_columns'], log_scale=True, filename=f'results/kde_input_{output_name}.jpg')

    (X, y), scaler = apply_scaling(df, cfg, cfg['input_columns'], cfg['output_columns'])
    Visualizer.plot_kde(y, cfg['output_columns'], log_scale=False, filename=f'results/kde_scaled_{output_name}.jpg')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # Train model
    model = AutoencoderModel(input_dim=X_train.shape[1], output_dim=y_train.shape[1], latent_dim=cfg['model']['latent_dim'])
    model.build_encoder_decoder()
    model.compile_autoencoder()

    callbacks = [
        step_decay_schedule(),
        *model.default_callbacks(output_name)
    ]
    
    history = model.train(X_train, y_train, epochs=cfg['model']['epochs'], batch_size=cfg['model']['batch_size'], callbacks=callbacks)

    # Evaluate and visualize
    preds = model.predict(X_test)
    _, y_test_inv = scaler.inverse_scale_data(X_test, y_test, apply_log1p=cfg['scaling']['apply_log1p'])
    _, preds_inv = scaler.inverse_scale_data(X_test, preds, apply_log1p=cfg['scaling']['apply_log1p'])

    Visualizer.plot_loss(history, filename=f'results/loss_{output_name}.jpg')
    Visualizer.scatter_plot(y_test, preds, filename=f'results/scatter_scaled_{output_name}.jpg', log_scale=False)
    Visualizer.scatter_plot(y_test_inv, preds_inv, filename=f'results/scatter_original_{output_name}.jpg', log_scale=True)
