import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, Dense, BatchNormalization, Dropout, Flatten, Add, Activation, Layer, MaxPooling1D, Reshape
from tensorflow.keras.models import Model
import numpy as np

# 1D-CNN Encoder
def create_cnn_encoder(input_dim, latent_dim, filters=64, kernel_size=3, dropout_rate=0.2):

    input_layer = Input(shape=(input_dim, 1))  # Input for 1D-CNN requires (input_dim, 1) shape
    
    # Convolutional Layers
    x = Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(dropout_rate)(x)
    
    x = Conv1D(filters=filters * 2, kernel_size=kernel_size, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(dropout_rate)(x)

    # Flatten the CNN output for fully connected layer
    x = Flatten()(x)

    # Latent space output
    encoded_output = Dense(latent_dim, activation='linear')(x)
    
    # Create encoder model
    encoder = Model(inputs=input_layer, outputs=encoded_output, name="cnn_encoder")
    return encoder

# # 1D-CNN Decoder
# def create_cnn_decoder(output_dim, latent_dim, filters=64, kernel_size=3, dropout_rate=0.2):
#     latent_inputs = Input(shape=(latent_dim,))
    
#     # Calculate the number of elements required for reshaping
#     reshape_size = output_dim * 1  # Since you want to reshape to (output_dim, 1)

#     # Fully connected layer to reshape into CNN input shape
#     x = Dense(reshape_size)(latent_inputs)  # Make sure this produces the right number of elements
#     x = Reshape((output_dim, 1))(x)  # Reshape to (output_dim, 1)

#     # Convolutional layers for decoding
#     x = Conv1D(filters=filters * 2, kernel_size=kernel_size, activation='relu', padding='same')(x)
#     x = BatchNormalization()(x)
#     x = Dropout(dropout_rate)(x)

#     x = Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', padding='same')(x)
#     x = BatchNormalization()(x)
#     x = Dropout(dropout_rate)(x)

#     # Flatten the output and fully connected layer for final reconstruction
#     decoded_output = Dense(output_dim, activation='linear')(x)  # Linear activation for regression
    
#     # Create decoder model
#     decoder = Model(inputs=latent_inputs, outputs=decoded_output, name="cnn_decoder")
#     return decoder

def create_cnn_decoder(output_dim, latent_dim, filters=64, kernel_size=3, dropout_rate=0.2):
    latent_inputs = Input(shape=(latent_dim,))
    
    # Calculate the number of elements required for reshaping
    reshape_size = output_dim * 1  # Since you want to reshape to (output_dim, 1)

    # Fully connected layer to reshape into CNN input shape
    x = Dense(reshape_size)(latent_inputs)  # Ensure this produces the right number of elements
    x = Reshape((output_dim, 1))(x)  # Reshape to (output_dim, 1)

    # Convolutional layers for decoding
    x = Conv1D(filters=filters * 2, kernel_size=kernel_size, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    x = Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    # Flatten the output so it's 2D (batch_size, output_dim) for regression
    x = Flatten()(x)

    # Fully connected layer for final reconstruction
    decoded_output = Dense(output_dim, activation='linear')(x)  # Linear activation for regression
    
    # Create decoder model
    decoder = Model(inputs=latent_inputs, outputs=decoded_output, name="cnn_decoder")
    return decoder