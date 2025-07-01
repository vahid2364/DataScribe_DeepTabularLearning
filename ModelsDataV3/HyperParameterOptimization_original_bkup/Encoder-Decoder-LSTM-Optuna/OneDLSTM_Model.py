import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, BatchNormalization, Dropout, Reshape, RepeatVector, TimeDistributed
from tensorflow.keras.models import Model

# LSTM Encoder with decreasing LSTM units
def create_lstm_encoder(input_dim, latent_dim, lstm_units=64, dropout_rate=0.2):
    input_layer = Input(shape=(input_dim, 1))  # Input for LSTM requires (input_dim, 1) shape
    
    # LSTM Layers with decreasing units
    x = LSTM(lstm_units, activation='relu', return_sequences=True)(input_layer)
    x = Dropout(dropout_rate)(x)

    x = LSTM(lstm_units // 2, activation='relu', return_sequences=False)(x)
    x = Dropout(dropout_rate)(x)
    
    # Latent space output
    encoded_output = Dense(latent_dim, activation='relu')(x)
    
    # Create encoder model
    encoder = Model(inputs=input_layer, outputs=encoded_output, name="lstm_encoder")
    return encoder

# LSTM Decoder with increasing LSTM units
def create_lstm_decoder(output_dim, latent_dim, lstm_units=32, dropout_rate=0.2):
    latent_inputs = Input(shape=(latent_dim,))
    
    # Fully connected layer to reshape into LSTM input shape
    x = Dense(lstm_units, activation='relu')(latent_inputs)  # Apply ReLU here
    x = Reshape((1, lstm_units))(x)  # Reshape to (1, lstm_units)

    # LSTM Layers with increasing units
    x = LSTM(lstm_units, activation='relu', return_sequences=True)(x)
    x = Dropout(dropout_rate)(x)

    x = LSTM(lstm_units * 2, activation='relu', return_sequences=True)(x)
    x = Dropout(dropout_rate)(x)

    # Fully connected layer for final reconstruction
    decoded_output = TimeDistributed(Dense(output_dim, activation='linear'))(x)  # Apply linear for regression
    
    # Create decoder model
    decoder = Model(inputs=latent_inputs, outputs=decoded_output, name="lstm_decoder")
    return decoder