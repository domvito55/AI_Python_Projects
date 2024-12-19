# First, configure GPU and imports
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("GPU Name: ", tf.test.gpu_device_name())

# Configure mixed precision for faster training on A100
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

# Configure memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

import numpy as np
import matplotlib.pyplot as plt
from time import time

# ------------------------- a. Get the data:
# ----- 1. Import and load the 'fashion_mnist' dataset from TensorFlow. Using 2
# dictionaries store the fashion_mnist datasets into train_firstname and
# test_firstname, where firstname is your firstname. The first 60,000 data
# samples will be stored in train_firstname directory with keys 'images' and
# 'labels', which will contain the images and labels for supervised learning.
# The next 10,000 data samples will be stored in test_firstname directory
# with keys 'images' and 'labels', which will contain the images and labels
# for supervised learning
# For more info checkout:
# https://keras.io/api/datasets/fashion_mnist/#load_data-function
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

train_matheus = {
    'images': x_train,
    'labels': y_train
}

test_matheus = {
    'images': x_test,
    'labels': y_test
}

# ------------------------ b. Data Pre-preprocessing
# ----- 1. Normalize the pixal values in the dataset to a range between 0-1.
# Store result back into unsupervised_firstname['images'] and
# supervised_firstname['images']
train_matheus['images'] = train_matheus['images'].astype('float32') / 255.0
test_matheus['images'] = test_matheus['images'].astype('float32') / 255.0

# ----- 2. Display (print) the shape of the train_firstname['images'],
# test_firstname['images'].
print("Training data shape:", train_matheus['images'].shape)
print("Testing data shape:", test_matheus['images'].shape)

# adding the channel dimension (1), so the convulutional layers will work
train_matheus['images'] = np.expand_dims(train_matheus['images'], axis=-1)
test_matheus['images'] = np.expand_dims(test_matheus['images'], axis=-1)

# ------- c. Build Variational Autoencoder with latent dimension size of 2
# ----- 1. Implement a customer layer named SampleLayer that extends the
# tf.keras.layers.layer class. The customer layer will sample the latent space
# of the encoder for the decoder. For more info checkout:
# https://www.tensorflow.org/tutorials/customization/custom_layers
class SampleLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SampleLayer, self).__init__(**kwargs)

    # i. The call function takes as input the mean and standard deviation
    # as a list. 
    def call(self, inputs):
        z_mean, z_log_var = inputs
        # From one of the input use tf.shape to calculate the batch size and
        #  dimension of the input.
        batch_size = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]

        # Generate random noise with the sample dimension as the batch size
        # and dimension of the input from a standard normal distribution using
        # tf.keras.backend.random_normal.
        epsilon = tf.random.normal(shape=(batch_size, dim), dtype=z_mean.dtype)
        # Generate samples z using the following formula:
        # 𝑧 = 𝜇 + 𝜎 ⨀ 𝜖
        # where 𝜇 𝑎𝑛𝑑 𝜎 are the mean and standard deviation from
        # the input and 𝜖 is the generated random noise.
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def compute_output_shape(self, input_shape):
        return input_shape[0]

class KLDivergenceLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(KLDivergenceLayer, self).__init__(**kwargs)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        # kl_loss = -0.5 * tf.reduce_mean(z_mu - tf.square(z_mu) - tf.exp(z_log_sigma) + 1)
        kl_loss = -0.5 * tf.reduce_mean(
            z_mean - tf.square(z_mean) - tf.exp(z_log_var) + 1,
            axis=-1
        )
        self.add_loss(kl_loss)
        return inputs

# ----- 2. Use TensorFlow's Model() [For more info, reference:
# https://www.tensorflow.org/api_docs/python/tf/keras/Model] to build
# the encoder section of the variational autoencoder with the following
# architecture:
# --- i. Input = Size of input image, store the layer as input_img
input_img = tf.keras.layers.Input(shape=(28, 28, 1))

# --- ii. Layer 1 = Convolution with 32 kernels with window size 3x3, a
# 'relu' activation function, and 'same' padding
x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)

# --- iii. Layer 2 = Convolution with 64 kernels with window size 3x3, a
# 'relu' activation function, 'same' padding, stride of 2x2
x = tf.keras.layers.Conv2D(64, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)

# --- iv. Layer 3 = Convolution with 64 kernels with window size 3x3, a
# 'relu' activation function, and 'same' padding
x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)

# --- v. Layer 4 = Convolution with 64 kernels with window size 3x3, a
# 'relu' activation function, and 'same' padding
x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)

# --- vi. Layer 5 = Full connected layer with 32 neurons and 'relu'
# activation (Note: Input to fully connected layer should be flatten
# first)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(32, activation='relu')(x)

# --- vii. LatentSpace.A = Full connected layer with 2 neurons, store layer
# as z_mu_firstname
z_mu_matheus = tf.keras.layers.Dense(2)(x)

# --- viii. LatentSpace.B = Full connected layer with 2 neurons, store layer
# as z_log_sigma_firstname
z_log_sigma_matheus = tf.keras.layers.Dense(2)(x)

# --- ix. Output = SampleLayer defined Step C.1, store layer as z_firstname
# Apply the KLDivergenceLayer before sampling
[z_mu_matheus, z_log_sigma_matheus] = KLDivergenceLayer()([z_mu_matheus, z_log_sigma_matheus])

# Apply the SampleLayer
z_matheus = SampleLayer()([z_mu_matheus, z_log_sigma_matheus])

# Create encoder model
encoder = tf.keras.Model(input_img, [z_mu_matheus, z_log_sigma_matheus, z_matheus])

# ----- 3. Display (print) a summary of the model using summary(). Draw a diagram
# illustrating the structure of the neural network model, making note of the
# size of each layer (# of neurons), number of weights in each layer and the
# unique connection between the latent space layer and the sample output
# layer.
encoder.summary()

# ----- 4. Use TensorFlow's Model() [For more info, reference:
# https://www.tensorflow.org/api_docs/python/tf/keras/Model] to build
# the decoder section of the variational autoencoder (store as
# decoder_firstname) with the following architecture:
# --- i. Input = Size of latent dimension
latent_input = tf.keras.layers.Input(shape=(2,))

# --- ii. Layer 1 = Fully connected layer, the number of neurons should be
# same as the output shape of Layer 4 in the encoder (i.e. the
# flatten input dimension for layer 5)
x = tf.keras.layers.Dense(14 * 14 * 64)(latent_input)

# --- iii. Layer 2 = Use tf.keras.layers.reshape to reshape the tensor as an
# image. The dimension of the reshape should be the same as Layer
# 4 in the encoder
x = tf.keras.layers.Reshape((14, 14, 64))(x)

# --- iv. Layer 3 = Use tf.keras.layers.Conv2DTranspose to add a
# transposed convolution layer with 32 kernels with window size
# 3x3, a 'relu' activation function, 'same' padding, stride of 2x2. For
# more info, reference:
# https://keras.io/api/layers/convolution_layers/convolution2d_tra
# nspose/
x = tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=(2, 2),
                                     activation='relu', padding='same')(x)

# --- v. Layer 4 = Convolution with 1 kernels with window size 3x3, a
# sigmoid activation function, and 'same' padding
decoder_output = tf.keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

# Create decoder model
decoder_matheus = tf.keras.Model(latent_input, decoder_output)

# ----- 5. Display (print) a summary of the model using summary(). Draw a diagram
# illustrating the structure of the neural network model, making note of the
# size of each layer (# of neurons), number of weights in each layer.
decoder_matheus.summary()

# ----- 6. Use TensorFlow's Model() [For more info, reference:
# https://www.tensorflow.org/api_docs/python/tf/keras/Model] to build a
# variational autoencoders (store model as vae_firstname) from the input
# layer of the encoder in Step C.2.i and output of the decoder(Note: use
# built model decoder_firstname from C.4 and z_firstname from Step C.2 to
# define output y, which will be the output of the variational autoencoder)
vae_output = decoder_matheus(z_matheus)
vae_matheus = tf.keras.Model(input_img, vae_output)

# ----- 7. Display (print) a summary of the model using summary(). Draw a diagram
# illustrating the structure of the neural network model, making note of the
# size of each layer (# of neurons), number of weights in each layer.
vae_matheus.summary()

# ---------------- d. Define the KL divergence using the following line,
# making sure to replace z_mu and z_log_sigma, as defined from Step C.2:
# kl_loss = -0.5 * tf.reduce_mean(z_mu - tf.square(z_mu) - tf.exp(z_log_sigma) + 1)

# ---------------- e. Use model.add_loss() to add the KL loss function defined
# in Step D. For more info reference:
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer#add_loss

# Compile the model with 'adam' optimizer, and 'mean_square_error' loss
# function
vae_matheus.compile(optimizer='adam',
                   loss='mean_squared_error')

# ------------------------ g. Use TensorFlow's fit() and the
# train_firstname['images' dataset to train the VAE with 10 epochs and batch
# size of 256.
# Note: Training a generative model is computationally expensive and can take
# several minutes to a couple hours to complete training. Therefore, during
# development you should train the model using a very limited subset of the data.
# Once you are confident there is no errors in your code, you can run the code and
# train the model using the full dataset.
print("Starting training...")
start_time = time()

batch_size = 256
history = vae_matheus.fit(train_matheus['images'], 
                         train_matheus['images'],
                         epochs=10,
                         batch_size=batch_size,
                         shuffle=True)

print(f"Training completed in {(time() - start_time)/60:.2f} minutes")

# ---------------------- h. Review sample code below and generate 10x10 samples
# from the VAE model using the decoder.
# import tensorflow_probability as tfp
# n = 4
# figure_size = 28
# norm = tfp.distributions.Normal(0, 1)
# grid_x = norm.quantile(np.linspace(0.05, 0.95, n))
# grid_y = norm.quantile(np.linspace(0.05, 0.95, n))
# figure = np.zeros((figure_size*n, figure_size*n))
# for i, yi in enumerate(grid_x):
# for j, xi in enumerate(grid_y):
# z_sample = np.array([[xi, yi]])
# z_sample = np.tile(z_sample, batch_size).reshape(batch_size, 2)
# x_decoded = decoder.predict(z_sample, batch_size=batch_size)
# img = x_decoded[0].reshape(figure_size, figure_size)
# figure[i * figure_size: (i + 1) * figure_size,
# j * figure_size: (j + 1) * figure_size] = img
# plt.figure(figsize=(20, 20))
# plt.imshow(figure)
# plt.show()

import tensorflow_probability as tfp

n = 10
figure_size = 28
norm = tfp.distributions.Normal(0, 1)
grid_x = norm.quantile(np.linspace(0.05, 0.95, n))
grid_y = norm.quantile(np.linspace(0.05, 0.95, n))
figure = np.zeros((figure_size*n, figure_size*n))

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        z_sample = np.tile(z_sample, batch_size).reshape(batch_size, 2)
        x_decoded = decoder_matheus.predict(z_sample, batch_size=batch_size)
        digit = x_decoded[0].reshape(figure_size, figure_size)
        figure[i * figure_size: (i + 1) * figure_size,
               j * figure_size: (j + 1) * figure_size] = digit
plt.figure(figsize=(10, 10))
plt.imshow(figure)
plt.show()

# i. Display (plot) the latent space of z_mu of the test dataset
# ----- 1. Using TensorFlow's Model() and layers from the encoder (see Step C.2)
# build a model to generate the latent space for z_mu.
encoder_mu = tf.keras.Model(input_img, z_mu_matheus)

# ----- 2. Use the model to predict the encoded latent space of the test dataset
mu_test = encoder_mu.predict(test_matheus['images'])

# ----- 3. Use matplotlib.pyplot.scatter to plot the latent space. For more info,
# reference:
# https://matplotlib.org/3.5.1/api/_as_gen/matplotlib.pyplot.scatter.html.
# Your plot should look something like the following.
plt.figure(figsize=(5, 5))
scatter = plt.scatter(mu_test[:, 0], mu_test[:, 1], c=test_matheus['labels'], 
                     cmap='tab10', marker='o', alpha=0.5)
plt.colorbar(scatter)
plt.title('Latent Space Visualization')
plt.xlabel('z[0]')
plt.ylabel('z[1]')
plt.show()