#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 02:31:41 2025

@author: attari.v
"""

# data_utils.py
import pandas as pd
from scaling_utils import Scaling

def load_and_filter_data(config):
    df = pd.read_csv(config['csv_path'])

    if config['thresholding']['apply']:
        col = config['thresholding']['column']
        val = config['thresholding']['value']
        inclusive = config['thresholding']['inclusive']
        if inclusive:
            df = df[df[col] <= val]
        else:
            df = df[df[col] < val]
        print(f"[Info] Thresholding applied on {col} with value {val} (inclusive={inclusive})")
    else:
        print("[Info] Thresholding skipped")

    return df

def apply_scaling(df, config, input_columns, output_columns):
    scaler = Scaling(input_columns=input_columns, output_columns=output_columns)
    inputs_scaled, outputs_scaled = scaler.scale_data(
        df,
        apply_sc=config['scaling']['apply_sc'],
        scaling_method=config['scaling']['scaling_method'],
        apply_pt=config['scaling'].get('apply_pt', False),
        pt_method=config['scaling'].get('pt_method', 'yeo-johnson'),
        apply_qt=config['scaling']['apply_qt'],
        qt_method=config['scaling']['qt_method'],
        apply_log1p=config['scaling']['apply_log1p'],
        apply_sigmoid=config['scaling']['apply_sigmoid'],
        apply_sqrt=config['scaling']['apply_sqrt'],
        sqrt_constant=config['scaling']['sqrt_constant'],
        apply_cbrt=config['scaling'].get('apply_cbrt', False),
        cbrt_constant=config['scaling'].get('cbrt_constant', 1),
    )
    return (inputs_scaled, outputs_scaled), scaler