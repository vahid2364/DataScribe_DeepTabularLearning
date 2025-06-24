
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, ELU, Layer, Lambda
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint

# # Custom Sampling Layer for VAE
# class Sampling(Layer):
#     def call(self, inputs):
#         z_mean, z_log_var = inputs
#         batch = K.shape(z_mean)[0]
#         dim = K.int_shape(z_mean)[1]
#         epsilon = K.random_normal(shape=(batch, dim))  # Standard normal noise
#         return z_mean + K.exp(0.5 * z_log_var) * epsilon

# # VAE Loss (only the reconstruction loss, we will handle KL separately)
# def vae_loss(y_true, y_pred):
#     reconstruction_loss = MeanSquaredError()(y_true, y_pred)
#     return reconstruction_loss

# Define KL loss as a custom function
def kl_loss(z_mean, z_log_var):
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    return kl_loss

# Sampling function using the reparameterization trick
def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))  # Standard normal noise
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

# Create the encoder for VAE
def create_variational_encoder(input_dim, latent_dim, layer_sizes=[2056, 1024, 512, 256], lamb=1e-3, rate=0.2, alpha=0.1):
    input_layer = Input(shape=(input_dim,))
    x = input_layer

    # Build the hidden layers
    for layer_size in layer_sizes:
        x = Dense(layer_size, kernel_regularizer=tf.keras.regularizers.l2(lamb))(x)
        x = BatchNormalization()(x)
        x = ELU(alpha=alpha)(x)
        x = Dropout(rate)(x)
    
    # Latent space outputs
    z_mean = Dense(latent_dim, name="z_mean")(x)
    z_log_var = Dense(latent_dim, name="z_log_var")(x)

    # Sampling layer using the reparameterization trick
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    # Return encoder model with both z_mean and z_log_var
    encoder = Model(input_layer, [z_mean, z_log_var, z], name="encoder")
    return encoder

# Create the decoder for VAE
def create_variational_decoder(output_dim, latent_dim, layer_sizes=[256, 512, 1024, 2056], lamb=1e-3, rate=0.2, alpha=0.1):
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = latent_inputs

    # Build the layers
    for layer_size in layer_sizes:
        x = Dense(layer_size, kernel_regularizer=tf.keras.regularizers.l2(lamb))(x)
        x = BatchNormalization()(x)
        x = ELU(alpha=alpha)(x)
        x = Dropout(rate)(x)

    # Final layer
    decoded_output = Dense(output_dim, activation='sigmoid')(x)

    decoder = Model(inputs=latent_inputs, outputs=decoded_output, name="decoder")
    return decoder

