import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
# Exercise #1: Autoencoders and Transfer Learning (100 marks)
# Requirements:
##### a. Get the data:
### 1. Import and load the 'fashion_mnist' dataset from TensorFlow.
print("##### a. Loading data #####\n")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()


### 2. Using dictionaries store the fashion_mnist datasets into
# unsupervised_firstname and supervised_firstname, where firstname is
# your firstname. The first 60,000 data samples will be stored in
# unsupervised_firstname directory with one key 'images', which will
# contain the images for unsupervised learning. The next 10,000 data
# samples will be stored in supervised_firstname directory with keys
# 'images' and 'labels', which will contain the images and labels for
# supervised learning
# For more info checkout:
# https://keras.io/api/datasets/fashion_mnist/#load_data-function
unsupervised_Matheus = {'images': x_train[:60000]}
supervised_Matheus = {'images': x_test, 'labels': y_test}


##### b. Data Pre-preprocessing
print("\n##### b. Data pre processing #####\n")
### 1. Normalize the pixal values in the dataset to a range between 0-1.
# Store result back into unsupervised_firstname['images'] and
# supervised_firstname['images']
unsupervised_Matheus['images'] = unsupervised_Matheus['images'].astype('float32') / 255.0
supervised_Matheus['images'] = supervised_Matheus['images'].astype('float32') / 255.0


### 2. Using tenflow's build in method to_cateogircal() to one-hot encode
# the labels. Store results back into supervised_firstname['labels'].
# For more info checkout:
# https://www.tensorflow.org/api_docs/python/tf/keras/utils/to_categorical
supervised_Matheus['labels'] = tf.keras.utils.to_categorical(
    supervised_Matheus['labels'])


### 3. Display (print) the shape of the unsupervised_firstname['images'],
# supervised_firstname['images'] and supervised_firstname['labels'].
print("Unsupervised images shape:", unsupervised_Matheus['images'].shape)
print("Supervised images shape:", supervised_Matheus['images'].shape)
print("Supervised labels shape:", supervised_Matheus['labels'].shape)


##### c. Data Preparation (Training, Validation, Testing)
print("\n##### c. Preparation #####\n")
### 1. Using Sklearn's train_test_split() method split the unsupervised dataset
# into training (57,000 samples) and validation (3,000 samples). Set the
# random seed to be the last two digits of your student ID number. Store
# the training and validation data in a dataframe named:
# unsupervised_train_firstname and unsupervised_val_firstname for the
# feature (predictors) of the training and validation data respectively.
unsupervised_train_Matheus, unsupervised_val_Matheus = train_test_split(
    unsupervised_Matheus['images'], train_size=57000, test_size=3000,
    random_state=4
)


### 2. Using Sklearn's train_test_split() method randomly discard 7,000
# samples from the supervised dataset. Set the random seed to be the last two
# digits of your student ID number.
_, supervised_data, _, supervised_labels = train_test_split(
    supervised_Matheus['images'], supervised_Matheus['labels'], 
    test_size=3000, random_state=4
)


### 3. Using Sklearn's train_test_split() method split the remaining
# supervised dataset (3,000 samples) into training (1800), validation(600)
# and testing(600). Set the random seed to be the last two digits of
# your student ID number. Store the datasets in a dataframe named:
# x_train_firstname, x_val_firstname, and x_test_firstname for the
# feature (predictors) and the training labels y_train_firstname,
# y_val_firstname, and y_test_firstname.
x_train_Matheus, x_temp, y_train_Matheus, y_temp = train_test_split(
    supervised_data, supervised_labels, train_size=1800, test_size=1200,
    random_state=4
)
x_val_Matheus, x_test_Matheus, y_val_Matheus, y_test_Matheus = train_test_split(
    x_temp, y_temp, train_size=600, test_size=600, random_state=4
)


### 4. Display (print) the shape of the unsupervised_train_firstname,
# unsupervised_val_firstname, x_train_firstname, x_val_firstname,
# x_test_firstname, y_train_firstname, y_val_firstname, and y_test_firstname.
print("Unsupervised train shape:", unsupervised_train_Matheus.shape)
print("Unsupervised validation shape:", unsupervised_val_Matheus.shape)
print("Supervised train shape:", x_train_Matheus.shape)
print("Supervised validation shape:", x_val_Matheus.shape)
print("Supervised test shape:", x_test_Matheus.shape)
print("Supervised train labels shape:", y_train_Matheus.shape)
print("Supervised validation labels shape:", y_val_Matheus.shape)
print("Supervised test labels shape:", y_test_Matheus.shape)


##### d. Build, Train, and Validate a baseline CNN Model
print("\n##### d. Baseline Model #####\n")
### 1. Use TensorFlow's Sequential() to build a CNN mode (name the
# model cnn_v1_model_firstname) with the following architecture:
cnn_v1_model_Matheus = models.Sequential([
# i. Input = Set based on image size of the fashion MNIST dataset.
    layers.Input(shape=(28, 28, 1)),
# ii. 1st Layer = Convolution with 16 filter kernels with window size 3x3,
# a 'relu' activation function, 'same' padding, and a stride of 2.
    layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2),
# iii. 3rd Layer = Convolution with 8 filter kernels with window size 3x3,
# a 'relu' activation function, 'same' padding, and a stride of 2.
    layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2),
# iv. 4th Layer = Full connected layer with 100 neurons (Note: Input to
# fully connected layer should be flatten first)
    layers.Flatten(),
    layers.Dense(100, activation='relu'),
# v. Output = Set output size using info identified in Step b.3 and a
# softmax activation function
    layers.Dense(10, activation='softmax')
])


### 2. Compile the model with 'adam' optimizer, 'cateogrical_crossentropy'
# loss function, 'accuracy' metric
cnn_v1_model_Matheus.compile(optimizer='adam', loss='categorical_crossentropy',
                             metrics=['accuracy'])


### 3. Display (print) a summary of the model using summary(). Draw a
# diagram illustrating the structure of the neural network model, making
# note of the size of each layer (# of neurons) and number of weights in
# each layer.
cnn_v1_model_Matheus.summary()


### 4. Using TensorFlow's fit() and the training/validation supervised
# dataset to train and validate the cnn model with 10 epochs and batch size
# of 256. Store training/validation results in cnn_v1_history_firstname.
cnn_v1_history_Matheus = cnn_v1_model_Matheus.fit(
    x_train_Matheus.reshape(-1, 28, 28, 1), y_train_Matheus,
    validation_data=(x_val_Matheus.reshape(-1, 28, 28, 1), y_val_Matheus),
    epochs=10, batch_size=256
)


##### e. Test and analyze the baseline model
print("\n##### e. Baseline Model Analysis #####\n")
### 1. Display (plot) the Training Vs Validation Accuracy of the baseline
# CNN Model as a line graph using matplotlib. Provide proper axis labels,
# title and a legend. Use different line color's for training and
# validation accuracy. Compare and analyze the training and validation
# accuracy in your report.
plt.figure(figsize=(10, 6))
plt.plot(cnn_v1_history_Matheus.history['accuracy'], label='Training Accuracy')
plt.plot(cnn_v1_history_Matheus.history['val_accuracy'],
         label='Validation Accuracy')
plt.title('Training vs Validation Accuracy - Baseline CNN')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

### 2. Evaluate the cnn model with the test dataset using
# Tensorflow's evaluate() and display (Print) the test accuracy. Compare
# and discuss the test accuracy to the validation accuracy in your report
test_loss, test_accuracy = cnn_v1_model_Matheus.evaluate(
    x_test_Matheus.reshape(-1, 28, 28, 1), y_test_Matheus)
print(f"Test accuracy: {test_accuracy}")


### 3. Create predictions on the test dataset using TensorFlow's
# predict(). Name in the predictions cnn_predictions_firstname.
cnn_predictions_Matheus = cnn_v1_model_Matheus.predict(
    x_test_Matheus.reshape(-1, 28, 28, 1))


### 4. Display (plot) the confusion matrix of the test prediction
# using matplotlib, seaborn, and sklearn's confusion matrix. For more info
# checkout the following:
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
# and https://seaborn.pydata.org/generated/seaborn.heatmap.html
cm = confusion_matrix(np.argmax(y_test_Matheus, axis=1),
                      np.argmax(cnn_predictions_Matheus, axis=1))
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Baseline CNN')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


### 5. Analyze and discuss the confusion matrix in your report

##### f. Add random noise to unsupervised dataset
print("\n##### f. Noisy Dataset #####\n")
### 1. Using tf.random.normal(), add random noise to the training and
# validation unsupervised dataset with a noise factor of 0.2. Set the
# random seed to be the last two digits of your student ID number.
# Store results into x_train_noisy_firstname and x_val_noisy_firstname.
# For more info, reference:
# https://www.tensorflow.org/api_docs/python/tf/random/normal
noise_factor = 0.2
x_train_noisy_Matheus = unsupervised_train_Matheus + tf.random.normal(
    shape=unsupervised_train_Matheus.shape, mean=0, stddev=1, seed=4
) * noise_factor

x_val_noisy_Matheus = unsupervised_val_Matheus + tf.random.normal(
    shape=unsupervised_val_Matheus.shape, mean=0, stddev=1, seed=4
) * noise_factor


### 2. Using tf.clip_by_value(), clip the values of the noisy dataset to
# a range between 0 and 1. Store results back into x_train_noisy_firstname
# and x_val_noisy_firstname. For more info, reference:
# https://www.tensorflow.org/api_docs/python/tf/clip_by_value
x_train_noisy_Matheus = tf.clip_by_value(x_train_noisy_Matheus, 0., 1.)
x_val_noisy_Matheus = tf.clip_by_value(x_val_noisy_Matheus, 0., 1.)


### 3. Display (plot) the first 10 images from the x_val_noisy_firstname
# using matplotlib. Remove xticks and yticks when plotting the image.
# Plot noisy images
plt.figure(figsize=(10, 4))
for i in range(10):
    ax = plt.subplot(2, 5, i+1)
    plt.imshow(tf.squeeze(x_val_noisy_Matheus[i]), cmap='gray')
    plt.axis('off')
plt.show()


##### g. Build and pretrain Autoencoder
print("\n##### g. Training Auto encoder #####\n")
# 1. Use TensorFlow's Model()
# [For more info, reference:
# https://www.tensorflow.org/api_docs/python/tf/keras/Model] to build
# an autoencoder mode (name the autoencoder_firstname) with the
# following architecture:
# i. Input = Set based on image size of the fashion MNIST dataset. Store
# layer as inputs_firstname.
inputs_Matheus = layers.Input(shape=(28, 28, 1))
# ii. Encoder Section (Store layers as e_firstname)
# 1. Convolution with 16 filter kernels with window size 3x3, a
# 'relu' activation function, 'same' padding, and a stride of 2.
e_Matheus = layers.Conv2D(16, (3, 3), activation='relu', padding='same',
                          strides=2)(inputs_Matheus)
# 2. Convolution with 8 filter kernels with window size 3x3, a
# 'relu' activation function, 'same' padding, and a stride of 2.
e_Matheus = layers.Conv2D(8, (3, 3), activation='relu', padding='same',
                          strides=2)(e_Matheus)
# iii. Decoder Section (Store layers as d_firstname)
# 1. Transposed Convolution with 8 filter kernels with window size 3x3,
# a 'relu' activation function, 'same' padding, and a stride of 2.
# For more info reference:
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2DTranspose
d_Matheus = layers.Conv2DTranspose(8, (3, 3), activation='relu',
                                   padding='same', strides=2)(e_Matheus)
# 2. Transposed Convolution with 16 filter kernels with window size 3x3,
# a 'relu' activation function, 'same' padding, and a stride of 2.
d_Matheus = layers.Conv2DTranspose(16, (3, 3), activation='relu',
                                   padding='same', strides=2)(d_Matheus)
# 3. Convolution with 1 filter kernels with window size 3x3, a
# 'sigmoid' activation function, and 'same' padding.
d_Matheus = layers.Conv2D(1, (3, 3), activation='sigmoid',
                          padding='same')(d_Matheus)

autoencoder_Matheus = models.Model(inputs_Matheus, d_Matheus)


### 2.
# Compile the model with 'adam' optimizer, and 'mean squared error'
# loss function.
autoencoder_Matheus.compile(optimizer='adam', loss='mse')


### 3. Display (print) a summary of the model using summary(). Draw a
# diagram illustrating the structure of the neural network model, making
# note of the size of each layer (# of neurons) and number of weights
# in each layer.
autoencoder_Matheus.summary()


### 4. Using TensorFlow's fit() and the training/validation unsupervised
# dataset to train and validate the cnn model with 10 epochs, batch size of
# 256 and shuffle set to True. Use the noisy images from step f as input
# and original images from step c.1 as output.
autoencoder_history = autoencoder_Matheus.fit(
    tf.expand_dims(x_train_noisy_Matheus, axis=-1),
    tf.expand_dims(unsupervised_train_Matheus, axis=-1),
    validation_data=(
        tf.expand_dims(x_val_noisy_Matheus, axis=-1),
        tf.expand_dims(unsupervised_val_Matheus, axis=-1)
    ),
    epochs=10, batch_size=256, shuffle=True
)


### 5. Create predictions on the unsupervised_val_firstname dataset
# using TensorFlow's predict(). Name in the predictions
# autoencoder_predictions_firstname.
autoencoder_predictions_Matheus = autoencoder_Matheus.predict(
    unsupervised_val_Matheus.reshape(-1, 28, 28, 1))


### 6. Display (plot) the first 10 predicted images from step 5 the
# using matplotlib. Remove xticks and yticks when plotting the image.
# Note: You can use Numpy's mean() to remove the remove the 3rd axis
# to properly plot the predicted images. For more info reference:
# https://numpy.org/doc/stable/reference/generated/numpy.mean.html
plt.figure(figsize=(10, 4))
for i in range(10):
    ax = plt.subplot(2, 5, i+1)
    plt.imshow(autoencoder_predictions_Matheus[i].reshape(28, 28), cmap='gray')
    plt.axis('off')
plt.show()


##### h. Build and perform transfer learning on a CNN with the Autoencoder
print("\n##### h. Transfer learning #####\n")
### 1. Use TensorFlow's Model() [For more info, reference:
# https://www.tensorflow.org/api_docs/python/tf/keras/Model] to build a
# cnn mode (name the cnn_v2_firstname) with the following architecture:
# i. Input = Transferred from Autoencoder. See step g.1.i
# ii. 1st layer = Transferred from encoder section of Autoencoder (step g.1.ii)
encoder = models.Model(inputs_Matheus, e_Matheus)
cnn_v2_Matheus = models.Sequential([
    encoder,
# iii. 2nd layer = Full connected layer with 100 neurons (Note: Input to
# fully connected layer should be flatten first)
    layers.Flatten(),
    layers.Dense(100, activation='relu'),
# iv. Output = Set output size using info identified in Step b.3 and a
# softmax activation function
    layers.Dense(10, activation='softmax')
])


### 2. Compile the model with 'adam' optimizer, 'cateogrical_crossentropy'
# loss function, 'accuracy' metric
cnn_v2_Matheus.compile(optimizer='adam', loss='categorical_crossentropy',
                       metrics=['accuracy'])


### 3. Display (print) a summary of the model using summary(). Draw a
# diagram illustrating the structure of the neural network model, making
# note of the size of each layer (# of neurons) and number of weights in
# each layer.
cnn_v2_Matheus.summary(expand_nested=True)


### 4. Using TensorFlow's fit() and the training/validation supervised
# dataset to train and validate the cnn model with 10 epochs and batch size
# of 256. Store training/validation results in cnn_v2_history_firstname.
cnn_v2_history_Matheus = cnn_v2_Matheus.fit(
    x_train_Matheus.reshape(-1, 28, 28, 1), y_train_Matheus,
    validation_data=(x_val_Matheus.reshape(-1, 28, 28, 1), y_val_Matheus),
    epochs=10, batch_size=256
)


# i. Test and analyze the pretrained CNN model
print("\n##### i. Pretrained CNN Analysis #####\n")
# 1. Display (plot) the Training Vs Validation Accuracy of the pretrained
# CNN Model as a line graph using matplotlib. Provide proper axis labels,
# title and a legend. Use different line color's for training and
# validation accuracy. Compare and analyze the training and validation
# accuracy in your report.
plt.figure(figsize=(10, 6))
plt.plot(cnn_v2_history_Matheus.history['accuracy'], label='Training Accuracy')
plt.plot(cnn_v2_history_Matheus.history['val_accuracy'],
         label='Validation Accuracy')
plt.title('Training vs Validation Accuracy - Pretrained CNN')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# 2. Evaluate the cnn model with the test dataset using Tensorflow's
# evaluate() and display (Print) the test accuracy. Compare and discuss
# the test accuracy to the validation accuracy in your report
test_loss, test_accuracy = cnn_v2_Matheus.evaluate(
    x_test_Matheus.reshape(-1, 28, 28, 1), y_test_Matheus)
print(f"Test accuracy: {test_accuracy}")


# 3. Create predictions on the test dataset using TensorFlow's predict().
# Name in the predictions cnn_predictions_firstname.
cnn_predictions_Matheus = cnn_v2_Matheus.predict(
    x_test_Matheus.reshape(-1, 28, 28, 1))

# 4. Display (plot) the confusion matrix of the test prediction
# using matplotlib, seaborn, and sklearn's confusion matrix. For more
# info checkout the following:
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
# and https://seaborn.pydata.org/generated/seaborn.heatmap.html
cm = confusion_matrix(np.argmax(y_test_Matheus, axis=1),
                      np.argmax(cnn_predictions_Matheus, axis=1))
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Pretrained CNN')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# 5. Analyze and discuss the confusion matrix in your report

##### j. Compare the performance of the baseline CNN model to the
print("\n##### j. Pretrained CNN vs Baseline Analysis #####\n")
# pretrained model in your report
### 1. Display (plot) the Validation Accuracy of the Baseline vs the
# Pretrained model as a line graph using matplotlib. Provide proper axis
# labels, title and a legend. Use different line color's for the baseline
# and pretrained accuracy. Compare and analyze the validation accuracy in
# your report.
plt.figure(figsize=(10, 6))
plt.plot(cnn_v1_history_Matheus.history['val_accuracy'], label='Baseline CNN')
plt.plot(cnn_v2_history_Matheus.history['val_accuracy'], label='Pretrained CNN')
plt.title('Validation Accuracy: Baseline vs Pretrained CNN')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

### 2. Compare and analyze the test accuracy in your report.