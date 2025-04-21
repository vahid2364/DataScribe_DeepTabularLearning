import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import joblib
from sklearn.metrics import mean_absolute_error, r2_score

from scaling_utils import scale_data, descale_data 

# %% Origianl Data

csv_file_path = 'IQR_dataframe-cobalt.csv'  # Replace with your CSV file path
df = pd.read_csv(csv_file_path, usecols=lambda column: column not in ['Unnamed: 0.3'])

df['PROP 500C CTE (1/K)'] = df['PROP 500C CTE (1/K)']*1e6
df['PROP 1000C CTE (1/K)'] = df['PROP 1000C CTE (1/K)']*1e6
df['PROP 1500C CTE (1/K)'] = df['PROP 1500C CTE (1/K)']*1e6

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

# %% Load Model IV: Outputs: Cobalt Creep

#    '1500 Min Creep CB [1/s]', 

import dnf_model 

EncoderDecoderI = tf.keras.models.load_model('Encoder-Decoder-DNNF-NewData/autoencoder_model_epoch_629_loss.keras')

input_scalerI  = joblib.load('Encoder-Decoder-DNNF-NewData/scales/input_scaler.save')
output_scalerI = joblib.load('Encoder-Decoder-DNNF-NewData/scales/output_scaler.save')
qt = joblib.load('Encoder-Decoder-DNNF-NewData/scales/quantile_transformer.save')

# Use the encoder-Decoder to transform the input data to the output space
predictions_scaled = EncoderDecoderI.predict(qt.transform(input_scalerI.transform(conditional_parameters)))

pause

# Inverse transform the predictions to original scale
predictions_descaled = descale_data(
    predictions_scaled, 
    output_scalerI, 
    apply_dsc=True, 
    apply_pt=False,  
    apply_qt=True, qt=qt,
    apply_log1p=False
)

print("Decoded data shape:", predictions_scaled.shape)
print("Decoded data (scale):", predictions_scaled)
print("(1500 Min Creep CB [1/s]) (original scale):\n", predictions_descaled)

print(df.iloc[0:9,5])

