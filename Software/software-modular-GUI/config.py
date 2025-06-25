#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 02:30:45 2025

@author: attari.v
"""

CONFIG = {
    'csv_path': '../../input_data/v3/IQR_dataframe-NbCrVWZr_data_stoic_creep_equil_v3.csv',
    'input_columns': ['Nb', 'Cr', 'V', 'W', 'Zr'],
    'output_columns': ['Creep Merit'],
    'thresholding': {
        'apply': False,
        'column': '1500 Min Creep CB [1/s]',
        'value': 1e-3,
        'inclusive': False
    },
    'scaling': {
        'apply_sc': False,
        'scaling_method': 'minmax',
        'apply_qt': True,
        'qt_method': 'uniform',
        'apply_log1p': False,
        'apply_sigmoid': False,
        'apply_sqrt': False,
        'sqrt_constant': 1
    },
    'model': {
        'latent_dim': 512,
        'epochs': 150,
        'batch_size': 96
    }
}