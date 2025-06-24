#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 13:55:30 2024

@author: attari.v
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
from pytorch_tabnet.tab_model import TabNetRegressor
import matplotlib.pyplot as plt

# Generate synthetic data (you can replace this with your own dataset)
np.random.seed(42)
n_samples = 1000
n_features = 5

X = np.random.rand(n_samples, n_features)  # Features
y = np.random.rand(n_samples, 1) * 100  # Target variable (for regression)

# Train-Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# Convert to numpy arrays (TabNet expects float32)
X_train_scaled = X_train_scaled.astype(np.float32)
X_test_scaled = X_test_scaled.astype(np.float32)
y_train_scaled = y_train_scaled.astype(np.float32).reshape(-1)
y_test_scaled = y_test_scaled.astype(np.float32).reshape(-1)

# TabNet Model Initialization
tabnet_model = TabNetRegressor(optimizer_fn=torch.optim.Adam, optimizer_params=dict(lr=2e-3),)

# Reshape y_train and y_test to be 2D arrays
y_train_reshaped = y_train.reshape(-1, 1)
y_test_reshaped = y_test.reshape(-1, 1)

# Fit the TabNet model
history = tabnet_model.fit(
    X_train=X_train_scaled,
    y_train=y_train_reshaped,
    eval_set=[(X_test_scaled, y_test_reshaped)],
    #eval_metric=['rmse'],  # Specify your evaluation metric
    max_epochs=200,
    patience=10,
    batch_size=64,
    virtual_batch_size=32,
    num_workers=0,
    drop_last=False,
)

# Predict and evaluate the model
y_pred_scaled = tabnet_model.predict(X_test_scaled)

# Rescale the predictions back to the original scale
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
y_test_orig = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1))

# Evaluate the model
mse = mean_squared_error(y_test_orig, y_pred)
mae = mean_absolute_error(y_test_orig, y_pred)
r2 = r2_score(y_test_orig, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
print(f"R-squared: {r2}")

# %%


# Check if history is available
if tabnet_model.history is not None:
    history = tabnet_model.history

    # Extract training and validation loss
    train_loss = history['loss']
    val_loss = history['val_0_mse']

    # Plotting the loss curves
    plt.figure(figsize=(8, 6))
    plt.plot(train_loss, label='Training Loss', color='blue')
    plt.plot(val_loss, label='Validation Loss', color='red', linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print("No history available for plotting.")