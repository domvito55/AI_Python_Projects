#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 02:46:05 2024

@author: mathteixeira55
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 15:12:20 2024

@author: AriyaAgnihothri
"""

#: Generative Adversarial Network

#imports
from  keras.datasets import fashion_mnist
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model

"a. Get the data:"
(image_train, label_train), (image_test, label_test) = fashion_mnist.load_data()

print(image_train.shape)
print(label_train.shape)
print(image_test.shape)
print(label_test.shape)

ds1_ariya = {'images': image_train[:60000], 'labels': label_train[:60000]}
ds2_ariya = {'images': image_test[:10000], 'labels': label_test[:10000]}
"b. Dataset Pre-preprocessing"
#1. Normalize
ds1_ariya['images']= (ds1_ariya['images'] / 127.5)-1.0
ds2_ariya['images']= (ds2_ariya['images'] / 127.5)-1.

#2. Display
print(ds1_ariya['images'].shape)
print(ds2_ariya['images'].shape)

#3. Pants Images
# Extracting the numpy arrays for images and labels
labels_ds1 = ds1_ariya['labels']
labels_ds2 = ds2_ariya['labels']

images_ds1 = ds1_ariya['images']
images_ds2 = ds2_ariya['images']

# Using boolean masking to filter pants (label == 1)
pants_ds1 = images_ds1[labels_ds1 == 1]  # Filter images with label 1
pants_ds2 = images_ds2[labels_ds2 == 1]  # Same for test dataset

dataset_ariya = np.concatenate([pants_ds1, pants_ds2], axis=0)

#4. Display
print("Shape of dataset of images: ",dataset_ariya.shape)

print("Number of pants images in training dataset:", pants_ds1.shape[0])
print("Number of pants images in test dataset:", pants_ds2.shape[0])

#5. Display (plot) the first 12 images
plt.figure(figsize=(8, 8))
for i in range(12):
    plt.subplot(4, 3, i + 1)
    plt.imshow(dataset_ariya[i], cmap='gray')
    plt.axis('off')  # Remove xticks and yticks
plt.show()

#6.Create training dataset
train_dataset_ariya = tf.data.Dataset.from_tensor_slices(dataset_ariya)
train_dataset_ariya = train_dataset_ariya.shuffle(buffer_size=7000).batch(256)
print(train_dataset_ariya)
"c. Build the Generator Model of the GAN"
# Define the generator model
def build_generator_model():
    model = tf.keras.Sequential(name='generator_model_ariya')
    model.add(layers.Dense(units=7*7*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())

    model.add(layers.LeakyReLU())







    model.add(layers.Reshape((7, 7, 256)))
    model.add(layers.Conv2DTranspose(filters=128,
                                     kernel_size=5,
                                     strides=1,
                                     padding='same',
                                     use_bias=False
                                     ))

    model.add(layers.BatchNormalization())

    model.add(layers.LeakyReLU())



    model.add(layers.Conv2DTranspose(filters=64,
                                     kernel_size=5,
                                     strides=2,
                                     padding='same',
                                     use_bias=False
                                     ))
    
    model.add(layers.BatchNormalization())

    model.add(layers.LeakyReLU())





    model.add(layers.Conv2DTranspose(filters=1,
                                     kernel_size=5,
                                     strides=2,
                                     padding='same',
                                     use_bias=False,
                                     activation='tanh'
                                     ))

    return model

# Build the generator model
generator_model_ariya = build_generator_model()





# Display summary of the generator model
generator_model_ariya.summary()

# plot_model(generator_model_ariya, to_file='generator_model.png', show_shapes=True, show_layer_names=True)
# encoder_plot = plt.imread('generator_model.png')
# plt.figure(figsize=(10, 10))
# plt.imshow(encoder_plot)
# plt.axis('off')
# plt.show()

"d. Sample untrained generator"
#Generate a sample vector(100)


sample_vector = tf.random.normal(shape=(1, 100)) 



# Generate an image(untrain)
generated_image = generator_model_ariya(sample_vector, training=False)  # Set training to False

# Plot the image

plt.imshow(generated_image[0], cmap='gray')  # Assuming batch size is 1
plt.axis('off')  # Hide axes
plt.show()

"e. Build the Generator Model of the GAN"



def build_discriminator_model():
    model = tf.keras.Sequential(name='discriminator_model_ariya')




    model.add(layers.Conv2D(filters=64,
                            kernel_size=5,
                            strides=2,
                            padding='same',
                            input_shape=(28, 28, 1)))
    
    
    model.add(layers.LeakyReLU())
    
    model.add(layers.Dropout(0.3))



    model.add(layers.Conv2D(filters=128,
                            kernel_size=5,
                            strides=2,
                            padding='same')
                            )

    model.add(layers.LeakyReLU())

    model.add(layers.Dropout(0.3))



    model.add(layers.Conv2DTranspose(filters=64,
                                     kernel_size=5,
                                     strides=2,
                                     padding='same',
                                     use_bias=False))

    model.add(layers.BatchNormalization())

    model.add(layers.LeakyReLU())


    model.add(layers.Flatten())
    model.add(layers.Dense(units=1))

    return model

# Build the discriminator model
discriminator_model_ariya = build_discriminator_model()






# Display summary of the discriminator model
discriminator_model_ariya.summary()

# Plot model
# plot_model(discriminator_model_ariya, to_file='discrimniator_model.png', show_shapes=True, show_layer_names=True)
# encoder_plot = plt.imread('discrimniator_model.png')
# plt.figure(figsize=(10, 10))
# plt.imshow(encoder_plot)
# plt.axis('off')
# plt.show()

"f. Implement Training"









cross_entropy_ariya = tf.keras.losses.BinaryCrossentropy(from_logits=True)








generator_optimizer_ariya = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer_ariya = tf.keras.optimizers.Adam(1e-4)








@tf.function
def training_step(images):
    noise = tf.random.normal([256, 100])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator_model_ariya(noise, training=True)

        real_output = discriminator_model_ariya(images, training=True)
        fake_output = discriminator_model_ariya(generated_images, training=True)

        gen_loss = cross_entropy_ariya(tf.ones_like(fake_output), fake_output)
        disc_loss = cross_entropy_ariya(tf.ones_like(real_output), real_output) + cross_entropy_ariya(tf.zeros_like(fake_output), fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss,
                                                generator_model_ariya.trainable_variables
                                                )
    gradients_of_discriminator = disc_tape.gradient(disc_loss,
                                                     discriminator_model_ariya.trainable_variables
                                                     )

    generator_optimizer_ariya.apply_gradients(zip
                                              (gradients_of_generator, generator_model_ariya.trainable_variables
                                               )
                                               )
    discriminator_optimizer_ariya.apply_gradients(zip(gradients_of_discriminator,
                                                      discriminator_model_ariya.trainable_variables
                                                      )
                                                      )
        
"g. Train the models in batches"
import time

epochs = 10




for epoch in range(epochs):
    start_time = time.time()

    for batch_images in train_dataset_ariya:
        training_step(batch_images)

    epoch_time = time.time() - start_time
    print(f"Epoch {epoch + 1}/{epochs} - Duration: {epoch_time:.2f}s")












"h. Visualized Trained Generator"
sample_vectors = tf.random.normal(shape=(16, 100))




generated_images = generator_model_ariya(sample_vectors, training=False)




generated_images = (generated_images + 1.0) * 127.5 

plt.figure(figsize=(8, 8))
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.imshow(generated_images[i, :, :, 0], cmap='gray')
    plt.axis('off')
plt.show()
