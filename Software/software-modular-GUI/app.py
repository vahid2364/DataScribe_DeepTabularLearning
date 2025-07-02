#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 02:47:14 2025
@author: attari.v
"""

# app.py
import streamlit as st
#import streamlit.web.bootstrap

import pandas as pd
import numpy as np
import os
import base64

from scaling_utils import Scaling
from model_utils import AutoencoderModel, step_decay_schedule, StreamlitProgressCallback
from visualize import Visualizer
from visualize import PlotlyVisualizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import plot_model

st.set_page_config(page_title="DataScribe GUI", layout="wide")

# CSS styling
st.markdown("""
    <style>
    /* Main app background and font */
    .block-container {
        /* background-color: #FFFBEC; */
        font-family: 'Segoe UI', sans-serif;
    }

    h1, h2, h3 {
        color: #1A1A1A;
        font-weight: 500;
    }
    
    h1 {
        font-size: 24px;  /* Change this value as desired */
    }

    /* Sidebar titles */
    .css-1d391kg, .css-1v0mbdj {
        color: #5B3E40;
        font-weight: bold;
    }
    
    /* Elegant button styling */
    .stButton > button {
        background: linear-gradient(135deg, #f44336, #e53935);
        color: #fff;
        font-size: 15px;
        font-weight: 600;
        padding: 0.55em 1.3em;
        border: none;
        border-radius: 4px;
        box-shadow: 0 6px 10px rgba(0,0,0,0.2);
        transition: all 0.3s ease-in-out;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #d32f2f, #c62828);
        box-shadow: 0 8px 16px rgba(0,0,0,0.25);
        transform: translateY(-2px);
    }

    /* File uploader styling */
    .stFileUploader > label div {
        background: linear-gradient(135deg, #F2545B, #E54457) !important;
        color: white !important;
        font-weight: bold;
        border-radius: 6px;
        padding: 8px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.08);
        transition: 0.3s ease;
    }

    .stFileUploader > label div:hover {
        background: linear-gradient(135deg, #D94350, #C13E48) !important;
        box-shadow: 0 6px 12px rgba(0,0,0,0.12);
    }

    /* Download button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #3A86FF, #2E6FE3);
        color: white;
        font-size: 14px;
        font-weight: 600;
        padding: 0.5em 1.2em;
        border-radius: 6px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease-in-out;
    }

    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #2F6FE4, #265CC4);
        box-shadow: 0 6px 10px rgba(0,0,0,0.15);
        transform: translateY(-1px);
    }

    /* Center buttons inside the sidebar */
    div[data-testid="stSidebar"] .stButton {
        display: flex;
        justify-content: center;
    }

    input[type="number"] {
        font-size: 20px !important;
    }
    
    </style>
""", unsafe_allow_html=True)


#if __name__ == "__main__":
#    streamlit.web.bootstrap.run(__file__, '', [], {})
    
file_path = "assets/app-icon.png"
with open(file_path, "rb") as f:
    data = f.read()
    encoded = base64.b64encode(data).decode()

st.markdown(
    f"""
    <div style='text-align: center;'>
        <img src='data:image/png;base64,{encoded}' width='150'>
    </div>
    """,
    unsafe_allow_html=True
)


os.makedirs("results", exist_ok=True)

st.markdown("""
    <h1 style='font-size: 40px; text-align: center; font-weight: bold;'>
        🔍 DataScribe Encoder-Decoder Tabular Data Learning
    </h1>
""", unsafe_allow_html=True)

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
        qt_method = st.sidebar.selectbox("QT Distribution", ['uniform', 'normal'])
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
            
            st.write("### Target Data Kernel Desnsity Estimates...")
                
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
            random_state = st.sidebar.slider("Random State", min_value=1, max_value=255, value=42, step=1) 
            
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=test_size, random_state=random_state)

            st.sidebar.header("Step 6: Model Architecture")
            num_encoder_layers = st.sidebar.slider("Encoder Layers", 1, 5, 2)
            encoder_neurons = st.sidebar.text_input("Encoder Neurons (comma-separated)", "128,256")
            encoder_neurons = [int(n.strip()) for n in encoder_neurons.split(",") if n.strip().isdigit()]

            num_decoder_layers = st.sidebar.slider("Decoder Layers", 1, 5, 2)
            decoder_neurons = st.sidebar.text_input("Decoder Neurons (comma-separated)", "256,128")
            decoder_neurons = [int(n.strip()) for n in decoder_neurons.split(",") if n.strip().isdigit()]

            latent_dim = st.sidebar.slider("Latent Dim", 32, 1024, 192, step=32)
            
            # Model Architecture
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
            
            #import io
            #from contextlib import redirect_stdout
            
            ## Capture model summary
            #string_buffer = io.StringIO()
            #with redirect_stdout(string_buffer):
            #    model.autoencoder.summary()
            #summary_string = string_buffer.getvalue()
            
            # Display in Streamlit
            #st.subheader("Model Architecture Summary")
            #st.code(summary_string, language="text")
            
            # Save model plot
            with st.container(border=True):
                # Save model plots
                #plot_model(model.autoencoder, to_file="results/model_architecture.png", show_shapes=True, show_layer_names=False)
                plot_model(model.encoder, to_file="results/encoder_architecture.png", show_shapes=True, show_layer_names=False, rankdir='LR')
                plot_model(model.decoder, to_file="results/decoder_architecture.png", show_shapes=True, show_layer_names=False, rankdir='LR')
            
                # Show main autoencoder plot
                #st.image("results/model_architecture.png", caption="Model Architecture", width=350)
            
                # Show encoder & decoder side by side
                st.write("## Encoder & Decoder Architectures")
                st.image("results/encoder_architecture.png", caption="Encoder Architecture", use_container_width=True)
                st.image("results/decoder_architecture.png", caption="Decoder Architecture", use_container_width=True)
            
            st.sidebar.header("Step 7: Training Config")
            epochs = st.sidebar.slider("Epochs", 10, 300, 150, step=10)
            batch_size = st.sidebar.selectbox("Batch Size", [32, 64, 96, 128], index=2)
            
            if st.sidebar.button("🚀 Train Model"):
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                st.write("### Training Autoencoder...")
                
                callbacks = [
                    step_decay_schedule(),
                    StreamlitProgressCallback(epochs, progress_bar, status_text),
                    *model.default_callbacks("streamlit_run")
                ]

                #callbacks = [step_decay_schedule()] + model.default_callbacks("streamlit_run")
                history = model.train(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=callbacks)

                st.success("Model training complete.")
                Visualizer.plot_loss(history, filename="results/loss_streamlit_run.jpg", log_scale=False)
                #PlotlyVisualizer.plot_loss(history, log_scale=False)

                #if os.path.exists("results/loss_streamlit_run.jpg"):
                #    st.image("results/loss_streamlit_run.jpg", caption="Loss Plot", use_container_width=True)

                y_pred = model.predict(X_test)
                _, y_test_inv = scaler.inverse_scale_data(X_test, y_test, apply_log1p=apply_log1p, apply_sigmoid=apply_sigmoid)
                _, y_pred_inv = scaler.inverse_scale_data(X_test, y_pred, apply_log1p=apply_log1p, apply_sigmoid=apply_sigmoid)

                Visualizer.scatter_plot(y_test, y_pred, filename="results/scatter_scaled.jpg", log_scale=False)
                Visualizer.scatter_plot(y_test_inv, y_pred_inv, filename="results/scatter_original.jpg", log_scale=True)
                #PlotlyVisualizer.scatter_plot(y_test, y_pred, log_scale=False)
                #PlotlyVisualizer.scatter_plot(y_test_inv, y_pred_inv, log_scale=True)

                st.session_state["model_trained"] = True
                st.session_state["y_test_inv"] = y_test_inv
                st.session_state["y_pred_inv"] = y_pred_inv
                

# --- Persisted Outputs ---
if st.session_state.get("model_trained", False):
    #if os.path.exists("results/loss_streamlit_run.jpg"):
    #    st.image("results/loss_streamlit_run.jpg", caption="Loss Plot", width=900)


    col1, col2, col3 = st.columns([0.7, 0.7, 0.7], gap='large')
    
    with col1:
        if os.path.exists("results/loss_streamlit_run.jpg"):
            st.image("results/loss_streamlit_run.jpg", caption="Training History", use_container_width=True)
    with col2:
        if os.path.exists("results/scatter_scaled.jpg"):
            st.image("results/scatter_scaled.jpg", caption="Scatter Plot (Scaled Space)", use_container_width=True)    
    with col3:
        if os.path.exists("results/scatter_original.jpg"):
            st.image("results/scatter_original.jpg", caption="Scatter Plot (Original Space, Log)", use_container_width=True)

    # --- User Prediction ---
    if st.session_state.get("model_trained", False):
        # --- Predict on new user inputs
        st.header("🔎 Predict for New Inputs")
        scaler = st.session_state["scaler"]
        input_values = []
        
        st.markdown("<div style='font-size:20px; font-weight:600;'>Enter values for input features:</div>", unsafe_allow_html=True)
        
        for i in range(0, len(input_columns), 3):
            cols = st.columns(3)
            for j, feature in enumerate(input_columns[i:i+3]):
                with cols[j]:
                    st.markdown(f"<div style='font-size:24px; font-weight:600'>{feature}</div>", unsafe_allow_html=True)
                    val = st.number_input(f"{feature}", value=float(df[feature].mean()), key=f"{feature}_input", label_visibility="collapsed")
                    input_values.append(val)
        
        input_array = np.array(input_values).reshape(1, -1)
        
        # Scale exactly like during training
        input_scaled = input_array
        if scaler.pt_inputs:
            input_scaled = scaler.pt_inputs.transform(input_scaled)
        if scaler.qt_inputs:
            input_scaled = scaler.qt_inputs.transform(input_scaled)
        if scaler.input_scaler:
            input_scaled = scaler.input_scaler.transform(input_scaled)
        if apply_log1p:
            input_scaled = np.log1p(input_scaled)
        if apply_sigmoid:
            input_scaled = 1 / (1 + np.exp(-input_scaled))
        
        # Predict
        pred_scaled = model.predict(input_scaled)
        
        # Inverse scale to original space
        pred_output = pred_scaled
        if scaler.output_scaler:
            pred_output = scaler.output_scaler.inverse_transform(pred_output)
        if scaler.qt_outputs:
            pred_output = scaler.qt_outputs.inverse_transform(pred_output)
        if scaler.pt_outputs:
            pred_output = scaler.pt_outputs.inverse_transform(pred_output)
        if apply_log1p:
            pred_output = np.expm1(pred_output)
        if apply_sigmoid:
            pred_output = -np.log((1 / pred_output) - 1)
        
        st.success(f"Predicted {output_columns[0]}: {pred_output.flatten()[0]:.5f}")
        

    

    download_df = pd.DataFrame({
        "True": st.session_state["y_test_inv"][:, 0],
        "Pred": st.session_state["y_pred_inv"][:, 0]
    })
    st.download_button("📥 Download Predictions CSV", data=download_df.to_csv(index=False), file_name="predictions.csv", mime="text/csv")