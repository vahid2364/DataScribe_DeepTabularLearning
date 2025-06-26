#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 02:43:05 2025

@author: attari.v
"""

# visualize.py
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

class Visualizer:
    @staticmethod
    def plot_kde(data, columns, log_scale=False, filename='kde_plot.jpg'):
        plt.figure(figsize=(8, 5))

        if isinstance(data, pd.DataFrame):
            for col in columns:
                sns.kdeplot(data[col], label=col, fill=True, log_scale=log_scale)

        elif isinstance(data, (np.ndarray, list)):
            data = np.array(data)
            for i, col in enumerate(columns):
                sns.kdeplot(data[:, i], label=f"Column {col}", fill=True, log_scale=log_scale)

        else:
            raise TypeError("Input data must be a Pandas DataFrame, NumPy array, or list.")

        plt.legend()
        plt.savefig(filename)
        #plt.show()

    @staticmethod
    def plot_loss(history, filename='loss_plot.jpg', log_scale=False):
        plt.figure(figsize=(8, 4))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        if log_scale:
            plt.yscale('log')
        plt.legend()
        plt.savefig(filename, dpi=300)
        #plt.show()

    @staticmethod
    def scatter_plot(y_true, y_pred, idx=0, filename='scatter_plot.jpg', log_scale=False):
        y_true = y_true.copy()
        y_pred = y_pred.copy()

        if log_scale:
            min_val = np.min(y_true[:, idx])
            epsilon = 1e-12

            if min_val >= 0:
                y_true_log = np.log1p(y_true[:, idx])
                y_pred_log = np.log1p(y_pred[:, idx])
                print(f"[Info] Applied log1p since min(y_true) = {min_val:.3e}")
            elif min_val > -epsilon:
                y_true_log = np.log(y_true[:, idx] + epsilon)
                y_pred_log = np.log(y_pred[:, idx] + epsilon)
                print(f"[Info] Applied log(x + epsilon) since min(y_true) = {min_val:.3e}")
            else:
                mask = y_true[:, idx] > -1
                y_true_log = np.log1p(y_true[mask, idx])
                y_pred_log = np.log1p(y_pred[mask, idx])
                print(f"[Warning] Detected y_true < -1 for idx={idx}. Masked before log1p.")

            mse = mean_squared_error(y_true_log, y_pred_log)
            r2 = r2_score(y_true_log, y_pred_log)

            plt.figure(figsize=(7, 7))
            plt.scatter(y_true_log, y_pred_log, label='Predictions')
            plt.plot([min(y_true_log), max(y_true_log)], [min(y_true_log), max(y_true_log)], c='black', label='Perfect Fit')
            plt.xlabel('Actual Outputs (Log Scale)')
            plt.ylabel('Predicted Outputs (Log Scale)')

        else:
            mse = mean_squared_error(y_true[:, idx], y_pred[:, idx])
            r2 = r2_score(y_true[:, idx], y_pred[:, idx])

            plt.figure(figsize=(7, 7))
            plt.scatter(y_true[:, idx], y_pred[:, idx], label='Predictions')
            plt.plot([min(y_true[:, idx]), max(y_true[:, idx])], [min(y_true[:, idx]), max(y_true[:, idx])], c='black', label='Perfect Fit')
            plt.xlabel('Actual Outputs')
            plt.ylabel('Predicted Outputs')

        plt.text(0.05, 0.95, f'MSE: {mse:.3g}', transform=plt.gca().transAxes, fontsize=15, verticalalignment='top')
        plt.text(0.05, 0.90, f'R$^2$: {r2:.3f}', transform=plt.gca().transAxes, fontsize=15, verticalalignment='top')

        plt.tight_layout()
        plt.savefig(filename)
        #plt.show()
        
# visualize_plotly.py
import plotly.graph_objects as go
import plotly.express as px
import scipy.stats as stats

class PlotlyVisualizer:
    @staticmethod
    def plot_kde(data, columns, log_scale=False):
        fig = go.Figure()

        if isinstance(data, pd.DataFrame):
            for col in columns:
                values = data[col].dropna()
                density = stats.gaussian_kde(values)
                x = np.linspace(np.min(values), np.max(values), 500)
                y = density(x)
                if log_scale:
                    y = np.log1p(y)
                fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=str(col)))
        elif isinstance(data, (np.ndarray, list)):
            data = np.array(data)
            for i, col in enumerate(columns):
                values = data[:, i]
                density = stats.gaussian_kde(values)
                x = np.linspace(np.min(values), np.max(values), 500)
                y = density(x)
                if log_scale:
                    y = np.log1p(y)
                fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=f"Column {col}"))
        else:
            raise TypeError("Input data must be a Pandas DataFrame, NumPy array, or list.")

        fig.update_layout(
            title="Kernel Density Estimate (KDE)",
            xaxis_title="Value",
            yaxis_title="Density (log scale)" if log_scale else "Density",
            template="plotly_white"
        )
        return fig

    @staticmethod
    def plot_kde_px(data, column, log_scale=False):
        fig = px.histogram(data, x=column, nbins=200, marginal="rug", histnorm='density')
        fig.update_traces(opacity=0.6)
        fig.update_layout(
            title=f"KDE for {column}",
            yaxis_title="Density",
            xaxis_title=column
        )
        return fig

    @staticmethod
    def plot_loss(history, log_scale=False):
        loss = history.history['loss']
        val_loss = history.history.get('val_loss', [])

        fig = go.Figure()
        fig.add_trace(go.Scatter(y=loss, mode='lines', name='Training Loss'))
        if val_loss:
            fig.add_trace(go.Scatter(y=val_loss, mode='lines', name='Validation Loss'))

        fig.update_layout(
            title="Training and Validation Loss",
            xaxis_title="Epoch",
            yaxis_title="Loss (log)" if log_scale else "Loss",
            yaxis_type="log" if log_scale else "linear",
            template="plotly_white"
        )
        return fig

    @staticmethod
    def scatter_plot(y_true, y_pred, idx=0, log_scale=False):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        if log_scale:
            epsilon = 1e-12
            y_true_vals = np.log1p(np.maximum(y_true[:, idx], epsilon))
            y_pred_vals = np.log1p(np.maximum(y_pred[:, idx], epsilon))
        else:
            y_true_vals = y_true[:, idx]
            y_pred_vals = y_pred[:, idx]

        mse = mean_squared_error(y_true_vals, y_pred_vals)
        r2 = r2_score(y_true_vals, y_pred_vals)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=y_true_vals,
            y=y_pred_vals,
            mode='markers',
            name='Predictions',
            marker=dict(size=5, color='blue', opacity=0.5)
        ))

        min_val = min(np.min(y_true_vals), np.min(y_pred_vals))
        max_val = max(np.max(y_true_vals), np.max(y_pred_vals))

        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Fit',
            line=dict(color='black', dash='dash')
        ))

        fig.update_layout(
            title="Predicted vs. Actual Scatter Plot",
            xaxis_title="Actual" + (" (log1p)" if log_scale else ""),
            yaxis_title="Predicted" + (" (log1p)" if log_scale else ""),
            annotations=[
                dict(x=0.05, y=0.95, xref='paper', yref='paper',
                     text=f"MSE: {mse:.3g}<br>R²: {r2:.3f}",
                     showarrow=False, align='left',
                     font=dict(size=12))
            ],
            template="plotly_white"
        )

        return fig