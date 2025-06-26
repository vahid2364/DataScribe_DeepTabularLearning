#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 02:47:14 2025
@author: attari.v
"""

# app.py
import streamlit as st
import pandas as pd
import numpy as np
import os

from scaling_utils import Scaling
from model_utils import AutoencoderModel, step_decay_schedule, StreamlitProgressCallback
from visualize import Visualizer
from visualize import PlotlyVisualizer
from sklearn.model_selection import train_test_split

os.makedirs("results", exist_ok=True)

st.set_page_config(page_title="Autoencoder Trainer", layout="wide")

st.title("🔍 DataScribe Encoder-Decoder Tabular Data Learning")

# --- Upload CSV File ---
st.sidebar.header("Step 1: Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.session_state["df"] = df
elif "df" in st.session_state:
    df = st.session_state["df"]
else:
    df = None
    st.info("Please upload a CSV file to begin.")

if df is not None:
    st.write("### Data Preview")
    st.dataframe(df.describe())

    st.sidebar.header("Step 2: Define Columns")
    input_columns = st.sidebar.multiselect("Input Features", df.columns.tolist(), key="input_cols")
    output_columns = st.sidebar.multiselect("Output Targets", df.columns.tolist(), key="output_cols")

    if input_columns and output_columns:
        # --- Optional Thresholding ---
        st.sidebar.header("Step 3: Thresholding (Optional)")
        apply_threshold = st.sidebar.checkbox("Apply threshold filter", value=False)
        threshold_column = st.sidebar.selectbox("Column for Thresholding", df.columns.tolist())
        threshold_value = st.sidebar.number_input("Threshold Value", value=1e-3)
        inclusive = st.sidebar.checkbox("Inclusive (<=)", value=False)

        if apply_threshold:
            if inclusive:
                df = df[df[threshold_column] <= threshold_value]
            else:
                df = df[df[threshold_column] < threshold_value]
            st.success(f"Filtered using {threshold_column} with threshold {threshold_value:.2e}")

        st.sidebar.header("Step 4: Scaling Options")
        apply_qt = st.sidebar.checkbox("QuantileTransformer", value=True)
        qt_method = st.sidebar.selectbox("QT Distribution", ['normal', 'uniform'])
        apply_log1p = st.sidebar.checkbox("Log1p Transform", value=False)
        apply_sigmoid = st.sidebar.checkbox("Sigmoid Transform", value=False)
        apply_sc = st.sidebar.checkbox("MinMax Scaling", value=False)

        if st.sidebar.button("📊 Apply Scaling"):
            scaler = Scaling(input_columns=input_columns, output_columns=output_columns)
            X_scaled, y_scaled = scaler.scale_data(
                df,
                apply_sc=apply_sc,
                scaling_method='minmax',
                apply_qt=apply_qt,
                qt_method=qt_method,
                apply_log1p=apply_log1p,
                apply_sigmoid=apply_sigmoid
            )
        
            st.session_state["X_scaled"] = X_scaled
            st.session_state["y_scaled"] = y_scaled
            st.session_state["scaler"] = scaler
            st.session_state["scaling_applied"] = True
            st.session_state["apply_log1p"] = apply_log1p
            st.session_state["apply_sigmoid"] = apply_sigmoid
        
            kde_orig_path = "results/kde_original.jpg"
            kde_scaled_path = "results/kde_scaled.jpg"
        
            Visualizer.plot_kde(df, output_columns, log_scale=True, filename=kde_orig_path)
            Visualizer.plot_kde(y_scaled, output_columns, log_scale=False, filename=kde_scaled_path)

            st.session_state["kde_orig_path"] = kde_orig_path
            st.session_state["kde_scaled_path"] = kde_scaled_path
        
        if "kde_orig_path" in st.session_state and "kde_scaled_path" in st.session_state:
            col1, col2 = st.columns(2)
        
            with col1:
                st.image(st.session_state["kde_orig_path"], caption="Original KDE", use_container_width=True)
        
            with col2:
                st.image(st.session_state["kde_scaled_path"], caption="Scaled KDE", use_container_width=True) 
        
        #     fig_kde_orig = PlotlyVisualizer.plot_kde(df, output_columns, log_scale=True)
        #     fig_kde_scaled = PlotlyVisualizer.plot_kde(y_scaled, output_columns, log_scale=False)
                        
        #     # Generate KDE figures using Plotly
        #     st.session_state["fig_kde_orig"] = PlotlyVisualizer.plot_kde(df, output_columns, log_scale=True)
        #     st.session_state["fig_kde_scaled"] = PlotlyVisualizer.plot_kde(y_scaled, output_columns, log_scale=False)
                    
        # # --- Display the Plotly KDE plots ---
        # if "fig_kde_orig" in st.session_state and "fig_kde_scaled" in st.session_state:
        #     col1, col2 = st.columns(2)
        
        #     with col1:
        #         st.plotly_chart(st.session_state["fig_kde_orig"], use_container_width=True)
        
        #     with col2:
        #         st.plotly_chart(st.session_state["fig_kde_scaled"], use_container_width=True)
                
                
        #     # Only one output column at a time is supported in plot_kde_px
        #     output_col = output_columns[0] if output_columns else None
            
        #     if output_col:
        #         # Generate KDE figures using Plotly Express
        #         fig_kde_orig = PlotlyVisualizer.plot_kde_px(df, output_col, log_scale=True)
        #         fig_kde_scaled = PlotlyVisualizer.plot_kde_px(pd.DataFrame(y_scaled, columns=output_columns), output_col, log_scale=False)
            
        #         # Store in session
        #         st.session_state["fig_kde_orig"] = fig_kde_orig
        #         st.session_state["fig_kde_scaled"] = fig_kde_scaled
            
        #     # --- Display the Plotly KDE plots ---
        # if "fig_kde_orig" in st.session_state and "fig_kde_scaled" in st.session_state:
        #     col1, col2 = st.columns(2)
        
        #     with col1:
        #         st.plotly_chart(st.session_state["fig_kde_orig"], use_container_width=True)
        
        #     with col2:
        #         st.plotly_chart(st.session_state["fig_kde_scaled"], use_container_width=True)

        if st.session_state.get("scaling_applied", False):
            X_scaled = st.session_state["X_scaled"]
            y_scaled = st.session_state["y_scaled"]
            scaler = st.session_state["scaler"]
            
            st.sidebar.header("Step 5: Train-Test Split")
            test_size = st.sidebar.slider("Test Size (%)", min_value=5, max_value=50, value=10, step=5) / 100.0
            
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=test_size, random_state=42)

            st.sidebar.header("Step 5: Autoencoder Config")
            latent_dim = st.sidebar.slider("Latent Dim", 32, 1024, 192, step=32)
            epochs = st.sidebar.slider("Epochs", 10, 300, 150, step=10)
            batch_size = st.sidebar.selectbox("Batch Size", [32, 64, 96, 128], index=2)

            st.sidebar.header("Step 6: Network Architecture")
            num_encoder_layers = st.sidebar.slider("Encoder Layers", 1, 5, 2)
            encoder_neurons = st.sidebar.text_input("Encoder Neurons (comma-separated)", "128,256")
            encoder_neurons = [int(n.strip()) for n in encoder_neurons.split(",") if n.strip().isdigit()]

            num_decoder_layers = st.sidebar.slider("Decoder Layers", 1, 5, 2)
            decoder_neurons = st.sidebar.text_input("Decoder Neurons (comma-separated)", "256,128")
            decoder_neurons = [int(n.strip()) for n in decoder_neurons.split(",") if n.strip().isdigit()]

            if st.button("🚀 Train Model"):
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                st.write("### Training Autoencoder...")

                model = AutoencoderModel(
                    input_dim=X_train.shape[1],
                    output_dim=y_train.shape[1],
                    latent_dim=latent_dim,
                    encoder_layers=num_encoder_layers,
                    encoder_neurons=encoder_neurons,
                    decoder_layers=num_decoder_layers,
                    decoder_neurons=decoder_neurons
                )
                model.build_encoder_decoder()
                model.compile_autoencoder()
                
                callbacks = [
                    step_decay_schedule(),
                    StreamlitProgressCallback(epochs, progress_bar, status_text),
                    *model.default_callbacks("streamlit_run")
                ]

                #callbacks = [step_decay_schedule()] + model.default_callbacks("streamlit_run")
                history = model.train(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=callbacks)

                st.success("Model training complete.")
                PlotlyVisualizer.plot_loss(history, filename="results/loss_streamlit_run.jpg", log_scale=False)

                #if os.path.exists("results/loss_streamlit_run.jpg"):
                #    st.image("results/loss_streamlit_run.jpg", caption="Loss Plot", use_container_width=True)

                y_pred = model.predict(X_test)
                _, y_test_inv = scaler.inverse_scale_data(X_test, y_test, apply_log1p=apply_log1p, apply_sigmoid=apply_sigmoid)
                _, y_pred_inv = scaler.inverse_scale_data(X_test, y_pred, apply_log1p=apply_log1p, apply_sigmoid=apply_sigmoid)

                PlotlyVisualizer.scatter_plot(y_test, y_pred, filename="results/scatter_scaled.jpg", log_scale=False)
                PlotlyVisualizer.scatter_plot(y_test_inv, y_pred_inv, filename="results/scatter_original.jpg", log_scale=True)

                st.session_state["model_trained"] = True
                st.session_state["y_test_inv"] = y_test_inv
                st.session_state["y_pred_inv"] = y_pred_inv

# --- Persisted Outputs ---
if st.session_state.get("model_trained", False):
    if os.path.exists("results/loss_streamlit_run.jpg"):
        st.image("results/loss_streamlit_run.jpg", caption="Loss Plot", use_container_width=True)


    col1, col2 = st.columns(2)
    
    with col1:
        if os.path.exists("results/scatter_scaled.jpg"):
            st.image("results/scatter_scaled.jpg", caption="Scatter Plot (Scaled Space)", use_container_width=True)    
    with col2:
        if os.path.exists("results/scatter_original.jpg"):
            st.image("results/scatter_original.jpg", caption="Scatter Plot (Original Space, Log)", use_container_width=True)



    download_df = pd.DataFrame({
        "True": st.session_state["y_test_inv"][:, 0],
        "Pred": st.session_state["y_pred_inv"][:, 0]
    })
    st.download_button("📥 Download Predictions CSV", data=download_df.to_csv(index=False), file_name="predictions.csv", mime="text/csv")