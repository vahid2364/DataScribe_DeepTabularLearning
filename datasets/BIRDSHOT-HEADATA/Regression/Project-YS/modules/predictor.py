import os
import pandas as pd
import tensorflow as tf
import joblib
from modules.preprocessing import descale_data


def load_model_and_scalers(weight_folder, scale_folder):
    """
    Load the model and associated scalers from the specified folders.

    Parameters:
    weight_folder : str
        Path to the folder containing the model weights.
    scale_folder : str
        Path to the folder containing the scalers.

    Returns:
    model : keras.Model
        Loaded autoencoder model.
    input_scaler : scaler object
        Scaler for input data.
    output_scaler : scaler object
        Scaler for output data.
    """
    model_path = os.path.join(weight_folder, 'autoencoder_model_final.keras')
    input_scaler_path = os.path.join(scale_folder, 'input_scaler.save')
    output_scaler_path = os.path.join(scale_folder, 'output_scaler.save')

    # Load model and scalers
    model = tf.keras.models.load_model(model_path)
    input_scaler = joblib.load(input_scaler_path)
    output_scaler = joblib.load(output_scaler_path)

    return model, input_scaler, output_scaler


def make_predictions(model, input_scaler, output_scaler, input_data):
    """
    Use the model to make predictions and descale the output.

    Parameters:
    model : keras.Model
        Trained autoencoder model.
    input_scaler : scaler object
        Scaler for input data.
    output_scaler : scaler object
        Scaler for output data.
    input_data : array-like
        Input data for prediction.

    Returns:
    predictions_scaled : array-like
        Scaled predictions.
    predictions_descaled : array-like
        Predictions on the original scale.
    """
    # Scale inputs, predict, and descale outputs
    predictions_scaled = model.predict(input_scaler.transform(input_data))
    predictions_descaled = descale_data(
        scaled_data=predictions_scaled,
        input_scaler=input_scaler,
        output_scaler=output_scaler,
        apply_dsc=True,
        apply_qt=False,
        apply_pt=False,
        apply_log1p=False,
        apply_sigmoid=False,
        data_type='output',
    )
    return predictions_scaled, predictions_descaled


def compare_predictions(input_data, output_data, predictions_descaled, output_columns, save_path):
    """
    Compare actual and predicted data side by side and save the results.

    Parameters:
    input_data : array-like
        Input data used for predictions.
    output_data : array-like
        Actual output data.
    predictions_descaled : array-like
        Predictions on the original scale.
    output_columns : list
        Names of the output columns.
    save_path : str
        File path to save the comparison DataFrame.

    Returns:
    comparison_df : pandas.DataFrame
        DataFrame containing original outputs and predictions.
    """
    # Create DataFrames for actual and predicted outputs
    original_df = pd.DataFrame(output_data, columns=output_columns)
    predictions_df = pd.DataFrame(predictions_descaled, columns=[f'Predicted {col}' for col in output_columns])

    # Combine the actual and predicted outputs side by side
    comparison_df = pd.concat([original_df, predictions_df], axis=1)

    # Save to CSV
    comparison_df.to_csv(save_path, index=False)
    return comparison_df


def predict_and_compare(df, input_columns, output_columns, weight_folder, scale_folder, save_path):
    """
    Load the model, make predictions, and compare them with actual data.

    Parameters:
    df : pandas.DataFrame
        DataFrame containing the input and output data.
    input_columns : list
        List of input column names.
    output_columns : list
        List of output column names.
    weight_folder : str
        Path to the folder containing the model weights.
    scale_folder : str
        Path to the folder containing the scalers.
    save_path : str
        File path to save the comparison DataFrame.

    Returns:
    comparison_df : pandas.DataFrame
        DataFrame containing original outputs and predictions.
    """
    # Load the model and scalers
    model, input_scaler, output_scaler = load_model_and_scalers(weight_folder, scale_folder)

    # Separate inputs and outputs from the DataFrame
    input_data = df[input_columns].to_numpy()
    output_data = df[output_columns].to_numpy()

    # Make predictions
    predictions_scaled, predictions_descaled = make_predictions(model, input_scaler, output_scaler, input_data)

    # Compare predictions with actual data
    comparison_df = compare_predictions(input_data, output_data, predictions_descaled, output_columns, save_path)
    return comparison_df