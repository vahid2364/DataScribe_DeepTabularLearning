import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import joblib
from sklearn.metrics import mean_absolute_error, r2_score

from scaling_utils2 import scale_data, descale_data 

# %% Origianl Data

#csv_file_path = 'IQR_dataframe-.csv'  # Replace with your CSV file path
csv_file_path = '../../input_data/v3/IQR_dataframe-NbCrVWZr_data_stoic_creep_equil_v3.csv'  # Replace with your CSV file path

df = pd.read_csv(csv_file_path, usecols=lambda column: column not in ['Unnamed: 0.3'])

#df['PROP 500C CTE (1/K)'] = df['PROP 500C CTE (1/K)']*1e6
#df['PROP 1000C CTE (1/K)'] = df['PROP 1000C CTE (1/K)']*1e6
#df['PROP 1500C CTE (1/K)'] = df['PROP 1500C CTE (1/K)']*1e6

# Define the remaining features
columns_to_keep = [
    'Nb', 'Cr', 'V', 'W', 'Zr',
#    'PROP LT (K)', 'PROP ST (K)', 
#     'PROP 500C CTE (1/K)', 'PROP 1000C CTE (1/K)', 'PROP 1500C CTE (1/K)', 
#     'EQ 1273K MAX BCC', 'EQ 1523K MAX BCC',
#     'EQ 1273K SUM BCC', 'EQ 1523K SUM BCC',
#     'EQ 1273K THCD (W/mK)', 'EQ 1523K THCD (W/mK)', 
#     'EQ 1273K Density (g/cc)', 'EQ 1523K Density (g/cc)',
#     'EQ 2/3*ST THCD (W/mK)', 'EQ 2/3*ST Density (g/cc)',  
#     'YS 1000C PRIOR','YS 1500C PRIOR',
#     'Pugh_Ratio_PRIOR', 
     'Creep Merit', 
     '25 Min Creep CB [1/s]', 
     '500 Min Creep CB [1/s]', 
     '1300 Min Creep CB [1/s]', 
     '1500 Min Creep CB [1/s]', 
#    'SCHEIL ST',
#    'SCHEIL LT', 
#    'Kou Criteria'
]

df = df[columns_to_keep]

# Define input and output columns
input_columns = df.columns[:5]  # First 5 columns
output_columns = df.columns[5:] # Remaining columns

# %% EncoderDecoder Input:     'Nb', 'Cr', 'V', 'W', 'Zr',

# Example alloy chemistry - replace this with desired alloys
conditional_parameters = np.array(df.iloc[0:10,0:5])

# %%

#import FullyDense_Model

EncoderDecoderI = tf.keras.models.load_model('../Encoder-Decoder-FullyDense-NewData/autoencoder_model_final.keras')

input_scalerI  = joblib.load('../Encoder-Decoder-FullyDense-NewData/scales/input_scaler.save')
output_scalerI = joblib.load('../Encoder-Decoder-FullyDense-NewData/scales/output_scaler.save')
qt_inputs = joblib.load('../Encoder-Decoder-FullyDense-NewData/scales/quantile_transformer_outputs_scaler.save')
qt_outputs= joblib.load('../Encoder-Decoder-FullyDense-NewData/scales/quantile_transformer_outputs_scaler.save')

# Use the encoder-Decoder to transform the input data to the output space
predictions_scaled = EncoderDecoderI.predict(qt_inputs.transform(input_scalerI.transform(conditional_parameters)))

# Inverse transform the predictions to original scale
predictions_descaled = descale_data(
    predictions_scaled, 
    input_scaler=input_scalerI, output_scaler=output_scalerI,
    apply_dsc=True, 
    apply_qt=True, qt_inputs=None, qt_outputs=qt_outputs, 
    apply_pt=False, pt_inputs=None, pt_outputs=None, 
    apply_log1p=False, 
    apply_sigmoid=False,
    data_type='output'
    )

print("Decoded data shape:", predictions_scaled.shape)

min_length = 10
# Trimming both arrays to match the minimum length
predictions_scaled_trimmed = predictions_scaled[:min_length,:]
predictions_descaled_trimmed = predictions_descaled[:min_length,:]

# Convert both arrays to DataFrames for easy comparison of features side by side
df_scaled = pd.DataFrame(predictions_scaled_trimmed, columns=[f'Scaled Feature {i+1}' for i in range(predictions_scaled.shape[1])])
df_descaled = pd.DataFrame(predictions_descaled_trimmed, columns=[f'Original Feature {i+1}' for i in range(predictions_descaled.shape[1])])

# Combine both DataFrames side by side
comparison_df = pd.concat([df.iloc[:min_length,5:], df_descaled], axis=1)

# Display the comparison DataFrame
print(comparison_df)

# Display the comparison DataFrame
#import ace_tools as tools; tools.display_dataframe_to_user(name="Scaled vs Original Features", dataframe=comparison_df)




