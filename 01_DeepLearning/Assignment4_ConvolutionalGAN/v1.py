import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

############################ a. Get the data:
print("a. Getting the data...")
##### 1. Import and load the 'fashion_mnist' dataset from TensorFlow. Using 2 
# dictionaries store the fashion_mnist datasets into ds1_firstname and 
# ds2_firstname, where firstname is your firstname. The first 60,000 data 
# samples will be stored in ds1_firstname directory with keys 'images' and 
# 'labels', which will contain the images and labels of the dataset. The next 
# 10,000 data samples will be stored in ds2_firstname directory with keys 
# 'images' and 'labels', which will contain the images and labels of the 
# dataset
# For more info checkout: 
# https://keras.io/api/datasets/fashion_mnist/#load_data-function
(train_images, train_labels), (test_images, test_labels) =\
                                     tf.keras.datasets.fashion_mnist.load_data()

ds1_matheus = {
    'images': train_images,
    'labels': train_labels
}

ds2_matheus = {
    'images': test_images,
    'labels': test_labels
}

####################### b. Dataset Pre-preprocessing
print("b. Dataset Pre-preprocessing...")
##### 1. Normalize the pixal values in the dataset to a range between -1 to 1.
# Store result back into ds1_firstname['images'] and 
# ds2_firstname['images']
ds1_matheus['images'] = (ds1_matheus['images'].astype('float32') - 127.5) / 127.5
ds2_matheus['images'] = (ds2_matheus['images'].astype('float32') - 127.5) / 127.5

###### 2. Display (print) the shape of the ds1_firstname['images'], 
# ds2_firstname['images'].
print("ds1_matheus['images'] shape:", ds1_matheus['images'].shape)
print("ds2_matheus['images'] shape:", ds2_matheus['images'].shape)

# Add channel dimension for CNN
ds1_matheus['images'] = ds1_matheus['images'][..., np.newaxis]
ds2_matheus['images'] = ds2_matheus['images'][..., np.newaxis]

print("After adding channel dimension:")
print("ds1_matheus['images'] shape:", ds1_matheus['images'].shape)
print("ds2_matheus['images'] shape:", ds2_matheus['images'].shape)

###### 3. Using np.concatenate, create a new dataset named dataset_firstname.
# The dataset will contain pants images (class label 1) from ds1_firstname 
# and ds2_firstname. For more info checkout 
# https://numpy.org/doc/stable/reference/generated/numpy.concatenate.
# html)
pants_train = ds1_matheus['images'][ds1_matheus['labels'] == 1]
pants_test = ds2_matheus['images'][ds2_matheus['labels'] == 1]
dataset_matheus = np.concatenate([pants_train, pants_test])

##### 4. Display (print) the shape of the dataset_firstname. Note: The dataset 
# should have a total of 7000 images.
print("dataset_matheus shape:", dataset_matheus.shape)

###### 5. Display (plot) the first 12 images from the dataset using matplotlip. 
# Remove xticks and yticks when plotting the image. Plot the images using 
# a figure size of 8x8 and a subplot dimension of 4x3
plt.figure(figsize=(8, 8))
for i in range(12):
    plt.subplot(4, 3, i+1)
    plt.imshow(dataset_matheus[i, :, :, 0], cmap='gray')
    plt.axis('off')
plt.show()

##### 6. Using Tensorflow's Dataset from_tensor_slices(), shuffle(), and batch 
# create training dataset called train_dataset_firstname from the 
# dataset_firstname. The training dataset will shuffle all 7000 images and 
# have a batch size of 256.
train_dataset_matheus = tf.data.Dataset.from_tensor_slices(dataset_matheus)\
    .shuffle(7000)\
    .batch(256)

################### c. Build the Generator Model of the GAN
print("c. Building the Generator Model of the GAN...")
##### 1. Use TensorFlow's Sequential() to build a CNN mode (name the model 
# generator_model_firstname) with the following architecture:
def make_generator_model():
    model = tf.keras.Sequential([
        # i. Input = Vector with dimension size 100
        tf.keras.layers.Input(shape=(100,), batch_size=256),
        # ii. 1st Layer = Fully connected Layer with 7*7*256 neurons and no 
        # bias term
        tf.keras.layers.Dense(7*7*256, use_bias=False),
        # iii. 2nd Layer = Batch Normalization
        tf.keras.layers.BatchNormalization(),
        # iv. 3rd Layer = Leaky ReLU activation
        tf.keras.layers.LeakyReLU(),
        # v. 4th Layer = Transposed Convolution Layer with 128 kernels with 
        # window size 5x5, no bias, 'same' padding, stride of 1x1. Note: 
        # Input to the Transposed Layer should first be reshaped to 
        # (7,7,256). For more info, reference: 
        # https://keras.io/api/layers/convolution_layers/convolution2d_tra
        # nspose/
        tf.keras.layers.Reshape((7, 7, 256)),
        tf.keras.layers.Conv2DTranspose(128,
                                        (5, 5),
                                        strides=(1, 1),
                                        padding='same',
                                        use_bias=False
                                        ),
        # vi. 5th Layer = Batch Normalization
        tf.keras.layers.BatchNormalization(),
        # vii. 6th Layer = Leaky ReLU
        tf.keras.layers.LeakyReLU(),
        # viii. 7th Layer = Transposed Convolution Layer with 64 kernels with 
        # window size 5x5, no bias, 'same' padding, stride of 2x2.
        tf.keras.layers.Conv2DTranspose(64,
                                        (5, 5),
                                        strides=(2, 2),
                                        padding='same',
                                        use_bias=False
                                        ),
        # ix. 8th Layer = Batch Normalization
        tf.keras.layers.BatchNormalization(),
        # x. 9th Layer = Leaky ReLU
        tf.keras.layers.LeakyReLU(),
        # xi. 7th Layer = Transposed Convolution Layer with 1 kernels with 
        # window size 5x5, no bias, 'same' padding, stride of 2x2, and tanh 
        # activation
        tf.keras.layers.Conv2DTranspose(1,
                                        (5, 5),
                                        strides=(2, 2),
                                        padding='same',
                                        use_bias=False,
                                        activation='tanh'
                                        )
    ])
    return model

generator_model_matheus = make_generator_model()

##### 2. Display (print) a summary of the model using summary(). Draw a diagram 
# illustrating the structure of the neural network model, making note of the 
# size of each layer (# of neurons) and number of weights in each layer. 
# Note: The generator model should output an image the same dimension 
# as the dataset
generator_model_matheus.summary()

###################### d. Sample untrained generator
print("d. Sample untrained generator...")
###### 1. Using Tensorflow's random.normal(), create a sample vector with 
# dimension size 100.
noise = tf.random.normal([1, 100])

###### 2. Generate an image from generator_model_firstname. Ensure training is 
# disabled.
generated_image = generator_model_matheus(noise, training=False)

##### 3. Display (plot) the generated image using matplot lib.
plt.figure(figsize=(4, 4))
plt.imshow(generated_image[0, :, :, 0], cmap='gray')
plt.axis('off')
plt.show()

# I GUESS THERE IS A TYPO HERE!!! THIS PART SHOULD CREATE A DISCRIMINATOR MODEL
# NOT A GENERATOR MODEL, RIGHT? --------------------------------------------- #

############## e. Build the Generator Model of the GAN
print("e. Building the Discriminator Model of the GAN...")
###### 1. Use TensorFlow's Sequential() to build a CNN mode (name the model 
# generator_model_firstname) with the following architecture:
def make_discriminator_model():
    model = tf.keras.Sequential([
        # i. Input = Image
        tf.keras.layers.Input(shape=(28, 28, 1), batch_size=256),
        # ii. 1st Layer = Convolution with 64 filter kernels with window size 
        # 5x5, stride of 2x2, and 'same' padding
        tf.keras.layers.Conv2D(64,
                               (5, 5),
                               strides=(2, 2),
                               padding='same'
                               ),
        # iii. 2nd Layer = Leaky ReLU activation
        tf.keras.layers.LeakyReLU(),
        # iv. 3rd Layer = Dropout with rate of 0.3
        tf.keras.layers.Dropout(0.3),
        # v. 4th Layer = Convolution with 128 filter kernels with window size 
        # 5x5, stride of 2x2, and 'same' padding
        tf.keras.layers.Conv2D(128,
                               (5, 5),
                               strides=(2, 2),
                               padding='same'
                               ),
        # vi. 5th Layer = Leaky ReLU activation
        tf.keras.layers.LeakyReLU(),
        # vii. 6th Layer = Dropout with rate of 0.3
        tf.keras.layers.Dropout(0.3),
        # viii. 7th Layer = Transposed Convolution Layer with 64 kernels with 
        # window size 5x5, no bias, 'same' padding, stride of 2x2.
        tf.keras.layers.Conv2DTranspose(64,
                                      (5, 5),
                                      strides=(2, 2),
                                      padding='same',
                                      use_bias=False),
        # ix. 8th Layer = Batch Normalization
        tf.keras.layers.BatchNormalization(),
        # x. 9th Layer = Leaky ReLU
        tf.keras.layers.LeakyReLU(),
        # xi. Output = 1 (Note: Input to the output should be flatten first)
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1)
    ])
    return model

discriminator_model_matheus = make_discriminator_model()

##### 2. Display (print) a summary of the model using summary(). Draw a diagram 
# illustrating the structure of the neural network model, making note of the 
# size of each layer (# of neurons) and number of weights in each layer.
discriminator_model_matheus.summary()

########################## f. Implement Training
print("f. Implementing Training...")
##### 1. Create a loss function using Tensorflow's BinaryCrossentropy() and
# call it cross_entropy_firstname. Make sure to set from_logits=True. This loss 
# function will be used to calculate the loss for the generator and 
# discriminator. For more info checkout: 
# https://www.tensorflow.org/api_docs/python/tf/keras/losses/BinaryCros
# sentropy
cross_entropy_matheus = tf.keras.losses.BinaryCrossentropy(from_logits=True)

##### 2. Using Tensorflow's optimizers, create a generator and discriminator 
# optimizer. Both optimizers will use Adam optimizers and should have the 
# name generator_optimizer_firstname and 
# discriminator_optimizer_firstname respectively.
generator_optimizer_matheus = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer_matheus = tf.keras.optimizers.Adam(1e-4)

##### 3. Create a tensorflow function using tf.function and call it training_step. 
# The function takes a batch of images as input and updates the 
# discriminator and generator using the optimizer and calculating the
# gradients from the calculated the losses. For more info checkout: 
# https://www.tensorflow.org/api_docs/python/tf/function. The function 
# should be similar to the following code snippet below (Examine the code 
# and make the necessary adjustment):
@tf.function
def training_step(images):
    noise = tf.random.normal([256, 100])
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator_model_matheus(noise, training=True)
        
        real_output = discriminator_model_matheus(images, training=True)
        fake_output = discriminator_model_matheus(generated_images, training=True)
        
        gen_loss = cross_entropy_matheus(tf.ones_like(fake_output), fake_output)
        disc_loss = cross_entropy_matheus(tf.ones_like(real_output), real_output) + \
                   cross_entropy_matheus(tf.zeros_like(fake_output), fake_output)
    
    gradients_of_generator = gen_tape.gradient(gen_loss,
                                               generator_model_matheus.trainable_variables
                                               )
    gradients_of_discriminator = disc_tape.gradient(disc_loss,
                                                    discriminator_model_matheus.trainable_variables
                                                    )
    
    generator_optimizer_matheus.apply_gradients(zip(gradients_of_generator,
                                                    generator_model_matheus.trainable_variables
                                                    )
                                                )
    discriminator_optimizer_matheus.apply_gradients(zip(gradients_of_discriminator,
                                                        discriminator_model_matheus.trainable_variables
                                                        )
                                                    )

##### g. Using the train_dataset_firstname from Step b.6 and the training
# function defined in Step f.3, train the models in batches with 10 epochs. Use
# Python's time module to calculate and display (print) how long each epoch
# takes.
# Note: 
# GAN's are trained typically on tens of thousands to hundreds of thousands 
# samples with large number of epochs. In your report, calculate and explain how 
# long it would take to train the same model using 70,000 training samples on 100 
# epochs using your current hardware.
EPOCHS = 10
epoch_times = []  # Array to store time for each epoch

for epoch in range(EPOCHS):
    start = time.time()
    
    for image_batch in train_dataset_matheus:
        training_step(image_batch)
    
    epoch_time = time.time() - start
    epoch_times.append(epoch_time)
    print(f'Time for epoch {epoch + 1} is {epoch_time} sec')

# Calculate estimations for 70,000 samples
average_epoch_time = sum(epoch_times) / len(epoch_times)
scaling_factor = 70000 / 7000
estimated_epoch_time = average_epoch_time * scaling_factor
total_estimated_time = estimated_epoch_time * 100  # For 100 epochs

print("\nTraining Time Analysis:")
print(f"Current average epoch time: {average_epoch_time:.2f} seconds")
print(f"Estimated time per epoch with 70,000 samples: {estimated_epoch_time:.2f} seconds")
print(f"Total estimated time for 100 epochs: ({total_estimated_time/3600:.2f} hours)")

##################### h. Visualized Trained Generator
##### 1. Using Tensorflow's random.normal(), create 16 sample vectors, each with 
# the dimension size of 100.
noise = tf.random.normal([16, 100])

##### 2. Generate an image from generator_model_firstname. Ensure training is 
# disabled.
generated_images = generator_model_matheus(noise, training=False)

##### 3. Normalize the pixels in the generated images by multiplying each pixel
# by 127.5 and adding 127.5 to each pixel.
generated_images = generated_images * 127.5 + 127.5

plt.figure(figsize=(4, 4))
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.imshow(generated_images[i, :, :, 0], cmap='gray')
    plt.axis('off')
plt.show()