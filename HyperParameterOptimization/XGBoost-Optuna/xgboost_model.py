#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 21:38:04 2024

@author: attari.v
"""

import xgboost as xgb

# Define the XGBoost model creation function
def xgboost_model(random_state=42, **xgb_params):
    """
    Function to create an XGBoost model with common parameters.

    Parameters:
    random_state : int, optional
        Random seed for model initialization.
    **xgb_params : dict, optional
        Additional parameters to pass to the XGBoost model.

    Returns:
    model : XGBRegressor
        Untrained XGBoost model ready for training.
    """
    # Initialize the XGBoost regressor with a wide range of common parameters
    model = xgb.XGBRegressor(
        n_estimators=xgb_params.get('n_estimators', 100),  # Number of boosting rounds
        max_depth=xgb_params.get('max_depth', 6),  # Maximum depth of a tree
        learning_rate=xgb_params.get('learning_rate', 0.1),  # Learning rate (eta)
        subsample=xgb_params.get('subsample', 0.8),  # Subsample ratio of the training instance
        colsample_bytree=xgb_params.get('colsample_bytree', 0.8),  # Subsample ratio of columns when constructing each tree
        colsample_bylevel=xgb_params.get('colsample_bylevel', 1.0),  # Subsample ratio of columns for each level
        colsample_bynode=xgb_params.get('colsample_bynode', 1.0),  # Subsample ratio of columns for each node
        gamma=xgb_params.get('gamma', 0),  # Minimum loss reduction required to make a further partition on a leaf node
        min_child_weight=xgb_params.get('min_child_weight', 1),  # Minimum sum of instance weight (hessian) needed in a child
        reg_alpha=xgb_params.get('reg_alpha', 0),  # L1 regularization term on weights
        reg_lambda=xgb_params.get('reg_lambda', 1),  # L2 regularization term on weights
        scale_pos_weight=xgb_params.get('scale_pos_weight', 1),  # Control the balance of positive and negative weights
        objective=xgb_params.get('objective', 'reg:squarederror'),  # Objective function (use 'binary:logistic' for binary classification)
        booster=xgb_params.get('booster', 'gbtree'),  # Type of booster to use: gbtree, gblinear, dart
        tree_method=xgb_params.get('tree_method', 'auto'),  # Tree construction algorithm (e.g., auto, exact, approx, hist, gpu_hist)
        eval_metric=xgb_params.get('eval_metric', 'rmse'),  # Evaluation metric (rmse, mae, logloss, etc.)
        verbosity=xgb_params.get('verbosity', 1),  # Verbosity (0: silent, 1: warning, 2: info, 3: debug)
        random_state=random_state,  # Seed for random number generation
        max_delta_step=xgb_params.get('max_delta_step', 0),  # Maximum delta step we allow each tree's weight estimation to be
        base_score=xgb_params.get('base_score', 0.5),  # Initial prediction score
        grow_policy=xgb_params.get('grow_policy', 'depthwise')  # Growth policy: depthwise or lossguide
    )

    return model