import tensorflow as tf
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

##### a. Get the data: #####
### 1. Import and load the 'fashion_mnist' dataset from TensorFlow. Using 2 
# dictionaries with keys 'images' and 'labels', store the fashion_mnist 
# datasets into train_firstname and test_firstname, where firstname is your 
# firstname. train_firstname will contain the images and labels of the 
# training data from 'fashion_mnist' and test_firstname will contain the 
# images and labels of the testing data from 'fasion_mnists'. For more info 
# checkout: https://keras.io/api/datasets/fashion_mnist/#load_datafunction
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_matheus = {'images': train_images, 'labels': train_labels}
test_matheus = {'images': test_images, 'labels': test_labels}

##### b. Initial Exploration #####
### 1. Display (print) the size of the training and testing dataset
print("Training dataset size:", train_matheus['images'].shape[0])
print("Testing dataset size:", test_matheus['images'].shape[0])

### 2. Display (print) the image resolution (dimension) of the input images.
print("Image resolution:", train_matheus['images'].shape[1:])

### 3. Display (print) the largest pixel value in the dataset using numpy.amax(). 
# For more info checkout: 
# https://numpy.org/doc/stable/reference/generated/numpy.amax.html
print("Largest pixel value in train dataset:", np.amax(train_matheus['images']))
print("Largest pixel value in test dataset:", np.amax(test_matheus['images']))

##### c. Data Pre-preprocessing #####
### 1. Normalize the pixel values in the dataset to a range between 0-1 using 
# the info identified in Step b. Store result back into 
# train_firstname['images'] and test_firstname['images']
train_matheus['images'] = train_matheus['images'] / 255.0
test_matheus['images'] = test_matheus['images'] / 255.0

### 2. Using tenflow's build in method to_cateogircal() to one-hot encode the 
# labels. Store results back into train_firstname['labels'] and 
# test_firstname['labels']. For more info checkout: 
# https://www.tensorflow.org/api_docs/python/tf/keras/utils/to_categorical
train_matheus['labels'] = tf.keras.utils.to_categorical(train_matheus['labels'])
test_matheus['labels'] = tf.keras.utils.to_categorical(test_matheus['labels'])

### 3. Display (print) the shape of the train_firstname['labels'] and 
# test_firstname['labels']. Take note of the number of possible labels in the 
# dataset
print("Shape of train labels:", train_matheus['labels'].shape)
print("Shape of test labels:", test_matheus['labels'].shape)

##### d. Visualization #####
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
def get_label_name(label):
    return class_names[label]
### 1. Create a function that displays (plots) an image with its true label using 
# matplotlib. Remove xticks and yticks when plotting the image.
def display_image(image, label):
    plt.imshow(image, cmap='gray')
    label = np.argmax(label)
    plt.title(f"Label: {label} - {get_label_name(label)}")
    plt.xticks([])
    plt.yticks([])

### 2. Using the function created in Step d.1, plot the first 12 data samples in 
# the training dataset using a figure size of 8x8 and a subplot dimension of 
# 4x3
plt.figure(figsize=(8, 8))
for i in range(12):
    plt.subplot(4, 3, i+1)
    display_image(train_matheus['images'][i], train_matheus['labels'][i])
plt.tight_layout()
plt.show()

##### e. Training Data Preparation #####
### 1. Using Sklearn's train_test_split() method split the training dataset in 80% 
# training and 20% validation. Set the random seed to be the last two 
# digits of your student ID number. Store the training data in a dataframe 
# named: x_train_firstname for the feature (predictors) and the training 
# labels y_train_firstname. Store the validation data as follows: 
# x_val_firstname and y_val_firstname
x_train_matheus, x_val_matheus, y_train_matheus, y_val_matheus = train_test_split(
    train_matheus['images'], train_matheus['labels'], 
    test_size=0.2, random_state=4
)

##### f. Build, Train, and Validate CNN Model #####
### 1. Use TensorFlow's Sequential() to build a CNN mode (name the model 
# cnn_model_firstname) with the following architecture:
cnn_model_matheus = tf.keras.Sequential([
    # i. Input = Set using info identified in Step b.
    tf.keras.layers.Input(shape=(28, 28, 1)),
    # ii. 1st Layer = Convolution with 32 filter kernels with window size 
    # 3x3 and a 'relu' activation function
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    # iii. 2nd Layer = Max Pooling with window size 2x2
    tf.keras.layers.MaxPooling2D((2, 2)),
    # iv. 3rd Layer = Convolution with 32 filter kernels with window size 
    # 3x3 and a 'relu' activation function
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    # v. 4th Layer = Max Pooling with window size 2x2
    tf.keras.layers.MaxPooling2D((2, 2)),
    # vi. 5th Layer = Full connected layer with 100 neurons (Note: Input to 
    # fully connected layer should be flatten first)
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation='relu'),
    # vii. Output = Set output size using info identified in Step c.3 and a 
    # softmax activation function
    tf.keras.layers.Dense(10, activation='softmax')
])

### 2. Compile the model with 'adam' optimizer, 'cateogrical_crossentropy' loss 
# function, 'accuracy' metric
cnn_model_matheus.compile(optimizer='adam',
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])

### 3. Display (print) a summary of the model using summary(). Draw a diagram 
# illustrating the structure of the neural network model, making note of the 
# size of each layer (# of neurons) and number of weights in each layer.
cnn_model_matheus.summary()

### 4. Using TensorFlow's fit() to train and validate the cnn model with 8 epochs 
# and batch size of 256. Store training/validation results in 
# cnn_history_firstname.
cnn_history_matheus = cnn_model_matheus.fit(
    x_train_matheus, y_train_matheus,
    epochs=8,
    batch_size=256,
    validation_data=(x_val_matheus, y_val_matheus)
)

##### g. Test and analyze the model #####
### 1. Display (plot) the Training Vs Validation Accuracy of the CNN Model as a 
# line graph using matplotlib. Provide proper axis labels, title and a legend. 
# Use different line color's for training and validation accuracy. Compare 
# and analyze the training and validation accuracy in your report.
plt.figure(figsize=(10, 6))
plt.plot(cnn_history_matheus.history['accuracy'], label='Training Accuracy',
         color='blue')
plt.plot(cnn_history_matheus.history['val_accuracy'],
         label='Validation Accuracy', color='red')
plt.title('Training vs Validation Accuracy', fontsize=20)
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.legend(fontsize=14)
plt.xticks(fontsize=14)  # Set font size for x-axis ticks
plt.yticks(fontsize=14)  # Set font size for y-axis ticks
plt.show()

### 2. Evaluate the cnn model with the test dataset using Tensorflow's 
# evaluate() and display (Print) the test accuracy. Compare and discuss the 
# test accuracy to the validation accuracy in your report
test_loss, test_accuracy = cnn_model_matheus.evaluate(test_matheus['images'],
                                                      test_matheus['labels'])
print(f"Test accuracy: {test_accuracy}")

### 3. Create predictions on the test dataset using TensorFlow's predict(). Name 
# in the predictions cnn_predictions_firstname.
cnn_predictions_matheus = cnn_model_matheus.predict(test_matheus['images'])

### 4. Create a function that plots the probability distribution of the predictions 
# as a histogram using matplotlib. The function takes in the true label of 
# the image and an array with the probability distribution. Probability of 
# true labels are colored in green and predicted labels are colored in blue. 
def plot_probability_distribution(true_label, probabilities):
    predict = np.argmax(probabilities)
    # Plot the reference green bar
    bars = plt.bar(true_label, 1, width=0.9, color='green')
    # Overlay the probability distribution with standard bar width
    prob_bars = plt.bar(range(10), probabilities, color='grey', alpha=0.7,
                        width=0.5)
    prob_bars = plt.bar(predict, probabilities[predict], color='blue', alpha=0.7,
                        width=0.5)
    # Set x and y axis limits
    plt.xlim([-0.5, 9.5])
    plt.ylim([0, 1])
    plt.xticks(range(10), range(10))  # Ensures the x-axis shows integer labels from 0 to 9
    # Add labels and title
    plt.xlabel('Class')
    plt.ylabel('Probability')
    plt.title(f"True Label: {true_label}")

### 5. Using the created function in Step d.1 and g.4. display (plot) the first 4 
# images from the test dataset starting from the last 2 digits of your 
# student number (i.e. if last 2 digits is 23, then display images 24-27) with 
# their prediction probability distribution. For example:
plt.figure(figsize=(12, 6))
start_index = 4
for i in range(start_index, start_index + 4):
    # First subplot: Display the image
    plt.subplot(2, 4, (i - start_index) * 2 + 1)
    display_image(test_matheus['images'][i], test_matheus['labels'][i])
    # Second subplot: Display the probability distribution
    plt.subplot(2, 4, (i - start_index) * 2 + 2)
    plot_probability_distribution(np.argmax(test_matheus['labels'][i]), cnn_predictions_matheus[i])
# Adjust layout to ensure no overlaps
plt.tight_layout()
plt.show()

### 6. Analyze and discuss the prediction probability distribution in your report

### 7. Display (plot) the confusion matrix of the test prediction using matplotlib, 
# seaborn, and sklearn's confusion matrix. For more info checkout the 
# following: https://scikitlearn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.h
# tml and https://seaborn.pydata.org/generated/seaborn.heatmap.html
y_true = np.argmax(test_matheus['labels'], axis=1)
y_pred = np.argmax(cnn_predictions_matheus, axis=1)
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(12, 12))
sns.heatmap(cm, annot=True, fmt='d', square=True, cmap='Blues',
            annot_kws={"size": 14})
plt.title('Confusion Matrix', fontsize=20)
plt.xlabel('Predicted Label', fontsize=16)
plt.ylabel('True Label', fontsize=16)
plt.xticks(fontsize=14)  # Set font size for x-axis ticks
plt.yticks(fontsize=14)  # Set font size for y-axis ticks
plt.show()

### 8. Analyze and discuss the confusion matrix in your report

##### h. Build,Train,Validate,Test and Analyze RNN Model #####
### 1. Repeat Steps f and g for an RNN model with the following architecture
rnn_model_matheus = tf.keras.Sequential([
    # i. Input = Set using info identified in Step b (Note: you can consider
    # image height as the timestep in the RNN).
    # Input: 28 timesteps (rows), each with 28 features (columns)
    tf.keras.layers.Input(shape=(28, 28)),
    # ii. 1st Layer = LSTM with hidden state size 128 units
    tf.keras.layers.LSTM(128),
    # iii. Output = Set output size using info identified in Step c.3 and a 
    # softmax activation function
    tf.keras.layers.Dense(10, activation='softmax')
])
# Compile the model
rnn_model_matheus.compile(optimizer='adam',
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])

# Display model summary
rnn_model_matheus.summary()

# Train and validate the model
rnn_history_matheus = rnn_model_matheus.fit(
    x_train_matheus, y_train_matheus,
    epochs=8,
    batch_size=256,
    validation_data=(x_val_matheus, y_val_matheus)
)

# Plot the curves
plt.figure(figsize=(10, 6))
plt.plot(rnn_history_matheus.history['accuracy'], label='Training Accuracy',
         color='blue')
plt.plot(rnn_history_matheus.history['val_accuracy'],
         label='Validation Accuracy', color='red')
plt.title('RNN Training vs Validation Accuracy', fontsize=20)
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.legend(fontsize=14)
plt.xticks(fontsize=14)  # Set font size for x-axis ticks
plt.yticks(fontsize=14)  # Set font size for y-axis ticks
plt.show()

# Evaluate the model on test data
rnn_test_loss, rnn_test_accuracy = rnn_model_matheus.evaluate(test_matheus['images'], test_matheus['labels'])
print(f"RNN Test accuracy: {rnn_test_accuracy}")

# Predictons
rnn_predictions_matheus = rnn_model_matheus.predict(test_matheus['images'])

# Plot predictions
plt.figure(figsize=(12, 6))
start_index = 4
for i in range(start_index, start_index + 4):
    # First subplot: Display the image
    plt.subplot(2, 4, (i - start_index) * 2 + 1)
    display_image(test_matheus['images'][i], test_matheus['labels'][i])
    # Second subplot: Display the probability distribution
    plt.subplot(2, 4, (i - start_index) * 2 + 2)
    plot_probability_distribution(np.argmax(test_matheus['labels'][i]), rnn_predictions_matheus[i])
# Adjust layout to ensure no overlaps
plt.tight_layout()
plt.show()

# confusion matrix
y_true = np.argmax(test_matheus['labels'], axis=1)
y_pred = np.argmax(rnn_predictions_matheus, axis=1)
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(12, 12))
sns.heatmap(cm, annot=True, fmt='d', square=True, cmap='Blues',
            annot_kws={"size": 14})
plt.title('Confusion Matrix', fontsize=20)
plt.xlabel('Predicted Label', fontsize=16)
plt.ylabel('True Label', fontsize=16)
plt.xticks(fontsize=14)  # Set font size for x-axis ticks
plt.yticks(fontsize=14)  # Set font size for y-axis ticks
plt.show()
