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
    Function to create an XGBoost model.

    Parameters:
    random_state : int, optional
        Random seed for model initialization.
    **xgb_params : dict, optional
        Additional parameters to pass to the XGBoost model.

    Returns:
    model : XGBRegressor
        Untrained XGBoost model ready for training.
    """
    # Initialize the XGBoost regressor with optional parameters
    model = xgb.XGBRegressor(
        n_estimators=xgb_params.get('n_estimators', 100),
        max_depth=xgb_params.get('max_depth', 6),
        learning_rate=xgb_params.get('learning_rate', 0.1),
        subsample=xgb_params.get('subsample', 0.8),
        colsample_bytree=xgb_params.get('colsample_bytree', 0.8),
        objective=xgb_params.get('objective', 'reg:squarederror'),
        eval_metric='rmse',      
        early_stopping_rounds=10,
        random_state=random_state
    )

    return model