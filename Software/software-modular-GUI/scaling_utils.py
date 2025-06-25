#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 02:35:10 2025

@author: attari.v
"""

# scaling_utils.py
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, QuantileTransformer

# Sigmoid function
def sigmoid_transform(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_inverse_transform(x):
    return -np.log((1 / x) - 1)

class Scaling:
    def __init__(self, input_columns, output_columns):
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.input_scaler = None
        self.output_scaler = None
        self.pt_inputs = None
        self.pt_outputs = None
        self.qt_inputs = None
        self.qt_outputs = None

    def scale_data(self, df, 
                   apply_sc=False, scaling_method='minmax', 
                   apply_pt=False, pt_method='yeo-johnson', 
                   apply_qt=False, qt_method='normal', 
                   apply_log1p=False, apply_sigmoid=False, 
                   apply_sqrt=False, sqrt_constant=1, 
                   apply_cbrt=False, cbrt_constant=1):

        transformed_data = df.copy()
        inputs = transformed_data[self.input_columns].to_numpy()
        outputs = transformed_data[self.output_columns].to_numpy()

        if apply_log1p:
            inputs = np.log1p(inputs)
            outputs = np.log1p(outputs)

        if apply_sigmoid:
            inputs = sigmoid_transform(inputs)
            outputs = sigmoid_transform(outputs)

        if apply_pt:
            print('pt applied')
            self.pt_inputs = PowerTransformer(method=pt_method)
            self.pt_outputs = PowerTransformer(method=pt_method)
            inputs = self.pt_inputs.fit_transform(inputs)
            outputs = self.pt_outputs.fit_transform(outputs)
            joblib.dump(self.pt_inputs, 'scales/power_transformer_inputs.save')
            joblib.dump(self.pt_outputs, 'scales/power_transformer_outputs.save')

        if apply_qt:
            print('qt applied')
            self.qt_inputs = QuantileTransformer(output_distribution=qt_method, n_quantiles=1000)
            self.qt_outputs = QuantileTransformer(output_distribution=qt_method, n_quantiles=1000)
            inputs = self.qt_inputs.fit_transform(inputs)
            outputs = self.qt_outputs.fit_transform(outputs)
            joblib.dump(self.qt_inputs, 'scales/quantile_transformer_inputs.save')
            joblib.dump(self.qt_outputs, 'scales/quantile_transformer_outputs.save')

        if apply_sqrt:
            inputs = np.sqrt(np.maximum(0, sqrt_constant - inputs))
            outputs = np.sqrt(np.maximum(0, sqrt_constant - outputs))

        if apply_cbrt:
            inputs = np.cbrt(cbrt_constant - inputs)
            outputs = np.cbrt(cbrt_constant - outputs)

        if apply_sc:
            if scaling_method == 'minmax':
                self.input_scaler = MinMaxScaler()
                self.output_scaler = MinMaxScaler()
            else:
                raise ValueError(f"Unknown scaling method: {scaling_method}. Choose 'minmax'.")

            inputs = self.input_scaler.fit_transform(inputs)
            outputs = self.output_scaler.fit_transform(outputs)
            joblib.dump(self.input_scaler, 'scales/input_scaler.save')
            joblib.dump(self.output_scaler, 'scales/output_scaler.save')

        return inputs, outputs

    def inverse_scale_data(self, inputs_scaled, outputs_scaled, 
                           apply_log1p=False, apply_sigmoid=False):

        inputs_inv, outputs_inv = inputs_scaled, outputs_scaled

        if self.input_scaler:
            print('input_scaler de-scaled')
            inputs_inv = self.input_scaler.inverse_transform(inputs_scaled)
        if self.output_scaler:
            outputs_inv = self.output_scaler.inverse_transform(outputs_scaled)

        if self.qt_inputs:
            print('qt_inputs de-scaled')
            inputs_inv = self.qt_inputs.inverse_transform(inputs_inv)
        if self.qt_outputs:
            outputs_inv = self.qt_outputs.inverse_transform(outputs_inv)

        if self.pt_inputs:
            inputs_inv = self.pt_inputs.inverse_transform(inputs_inv)
        if self.pt_outputs:
            outputs_inv = self.pt_outputs.inverse_transform(outputs_inv)

        if apply_log1p:
            inputs_inv = np.expm1(inputs_inv)
            outputs_inv = np.expm1(outputs_inv)

        if apply_sigmoid:
            inputs_inv = sigmoid_inverse_transform(inputs_inv)
            outputs_inv = sigmoid_inverse_transform(outputs_inv)

        return inputs_inv, outputs_inv