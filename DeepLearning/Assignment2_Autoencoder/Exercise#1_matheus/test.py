import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# a. Get the data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

unsupervised_Matheus = {'images': x_train[:60000]}
supervised_Matheus = {'images': x_test, 'labels': y_test}

# b. Data Pre-preprocessing
# 1. Normalize pixel values
unsupervised_Matheus['images'] = unsupervised_Matheus['images'].astype('float32') / 255.0
supervised_Matheus['images'] = supervised_Matheus['images'].astype('float32') / 255.0

# 2. One-hot encode labels
supervised_Matheus['labels'] = tf.keras.utils.to_categorical(supervised_Matheus['labels'])

# 3. Display shapes
print("Unsupervised images shape:", unsupervised_Matheus['images'].shape)
print("Supervised images shape:", supervised_Matheus['images'].shape)
print("Supervised labels shape:", supervised_Matheus['labels'].shape)

# c. Data Preparation
# 1. Split unsupervised data
unsupervised_train_Matheus, unsupervised_val_Matheus = train_test_split(
    unsupervised_Matheus['images'], train_size=57000, test_size=3000, random_state=4
)

# 2. Discard 7,000 samples from supervised data
_, supervised_data, _, supervised_labels = train_test_split(
    supervised_Matheus['images'], supervised_Matheus['labels'], 
    test_size=3000, random_state=4
)

# 3. Split supervised data
x_train_Matheus, x_temp, y_train_Matheus, y_temp = train_test_split(
    supervised_data, supervised_labels, train_size=1800, test_size=1200,
    random_state=4
)
x_val_Matheus, x_test_Matheus, y_val_Matheus, y_test_Matheus = train_test_split(
    x_temp, y_temp, train_size=600, test_size=600, random_state=4
)

# 4. Display shapes
print("Unsupervised train shape:", unsupervised_train_Matheus.shape)
print("Unsupervised validation shape:", unsupervised_val_Matheus.shape)
print("Supervised train shape:", x_train_Matheus.shape)
print("Supervised validation shape:", x_val_Matheus.shape)
print("Supervised test shape:", x_test_Matheus.shape)
print("Supervised train labels shape:", y_train_Matheus.shape)
print("Supervised validation labels shape:", y_val_Matheus.shape)
print("Supervised test labels shape:", y_test_Matheus.shape)

# d. Build, Train, and Validate a baseline CNN Model
cnn_v1_model_Matheus = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2),
    layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2),
    layers.Flatten(),
    layers.Dense(100, activation='relu'),
    layers.Dense(10, activation='softmax')
])

cnn_v1_model_Matheus.compile(optimizer='adam', loss='categorical_crossentropy',
                             metrics=['accuracy'])

cnn_v1_model_Matheus.summary()

# Train the model
cnn_v1_history_Matheus = cnn_v1_model_Matheus.fit(
    x_train_Matheus.reshape(-1, 28, 28, 1), y_train_Matheus,
    validation_data=(x_val_Matheus.reshape(-1, 28, 28, 1), y_val_Matheus),
    epochs=10, batch_size=256
)

# e. Test and analyze the baseline model
# 1. Plot training vs validation accuracy
plt.figure(figsize=(10, 6))
plt.plot(cnn_v1_history_Matheus.history['accuracy'], label='Training Accuracy')
plt.plot(cnn_v1_history_Matheus.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training vs Validation Accuracy - Baseline CNN')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# 2. Evaluate on test set
test_loss, test_accuracy = cnn_v1_model_Matheus.evaluate(
    x_test_Matheus.reshape(-1, 28, 28, 1), y_test_Matheus)
print(f"Test accuracy: {test_accuracy}")

# 3. Make predictions
cnn_predictions_Matheus = cnn_v1_model_Matheus.predict(x_test_Matheus.reshape(-1, 28, 28, 1))

# 4. Plot confusion matrix
cm = confusion_matrix(np.argmax(y_test_Matheus, axis=1), np.argmax(cnn_predictions_Matheus, axis=1))
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Baseline CNN')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# f. Add random noise to unsupervised dataset
noise_factor = 0.2
x_train_noisy_Matheus = unsupervised_train_Matheus + tf.random.normal(
    shape=unsupervised_train_Matheus.shape, mean=0, stddev=1, seed=4
) * noise_factor

x_val_noisy_Matheus = unsupervised_val_Matheus + tf.random.normal(
    shape=unsupervised_val_Matheus.shape, mean=0, stddev=1, seed=4
) * noise_factor

x_train_noisy_Matheus = tf.clip_by_value(x_train_noisy_Matheus, 0., 1.)
x_val_noisy_Matheus = tf.clip_by_value(x_val_noisy_Matheus, 0., 1.)

# Plot noisy images
plt.figure(figsize=(10, 4))
for i in range(10):
    ax = plt.subplot(2, 5, i+1)
    plt.imshow(tf.squeeze(x_val_noisy_Matheus[i]), cmap='gray')
    plt.axis('off')
plt.show()


# g. Build and pretrain Autoencoder
inputs_Matheus = layers.Input(shape=(28, 28, 1))
e_Matheus = layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2)(inputs_Matheus)
e_Matheus = layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2)(e_Matheus)

d_Matheus = layers.Conv2DTranspose(8, (3, 3), activation='relu', padding='same', strides=2)(e_Matheus)
d_Matheus = layers.Conv2DTranspose(16, (3, 3), activation='relu', padding='same', strides=2)(d_Matheus)
d_Matheus = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(d_Matheus)

autoencoder_Matheus = models.Model(inputs_Matheus, d_Matheus)
autoencoder_Matheus.compile(optimizer='adam', loss='mse')

autoencoder_Matheus.summary()

# Train autoencoder
autoencoder_history = autoencoder_Matheus.fit(
    tf.expand_dims(x_train_noisy_Matheus, axis=-1),
    tf.expand_dims(unsupervised_train_Matheus, axis=-1),
    validation_data=(
        tf.expand_dims(x_val_noisy_Matheus, axis=-1),
        tf.expand_dims(unsupervised_val_Matheus, axis=-1)
    ),
    epochs=10, batch_size=256, shuffle=True
)

# Make predictions
autoencoder_predictions_Matheus = autoencoder_Matheus.predict(
    unsupervised_val_Matheus.reshape(-1, 28, 28, 1))

# Plot reconstructed images
plt.figure(figsize=(10, 4))
for i in range(10):
    ax = plt.subplot(2, 5, i+1)
    plt.imshow(autoencoder_predictions_Matheus[i].reshape(28, 28), cmap='gray')
    plt.axis('off')
plt.show()

# h. Build and perform transfer learning on a CNN with the Autoencoder
encoder = models.Model(inputs_Matheus, e_Matheus)
cnn_v2_Matheus = models.Sequential([
    encoder,
    layers.Flatten(),
    layers.Dense(100, activation='relu'),
    layers.Dense(10, activation='softmax')
])

cnn_v2_Matheus.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

cnn_v2_Matheus.summary(expand_nested=True)

# Train the model
cnn_v2_history_Matheus = cnn_v2_Matheus.fit(
    x_train_Matheus.reshape(-1, 28, 28, 1), y_train_Matheus,
    validation_data=(x_val_Matheus.reshape(-1, 28, 28, 1), y_val_Matheus),
    epochs=10, batch_size=256
)

# i. Test and analyze the pretrained CNN model
# 1. Plot training vs validation accuracy
plt.figure(figsize=(10, 6))
plt.plot(cnn_v2_history_Matheus.history['accuracy'], label='Training Accuracy')
plt.plot(cnn_v2_history_Matheus.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training vs Validation Accuracy - Pretrained CNN')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# 2. Evaluate on test set
test_loss, test_accuracy = cnn_v2_Matheus.evaluate(
    x_test_Matheus.reshape(-1, 28, 28, 1), y_test_Matheus)
print(f"Test accuracy: {test_accuracy}")

# 3. Make predictions
cnn_predictions_Matheus = cnn_v2_Matheus.predict(
    x_test_Matheus.reshape(-1, 28, 28, 1))

# 4. Plot confusion matrix
cm = confusion_matrix(np.argmax(y_test_Matheus, axis=1), np.argmax(cnn_predictions_Matheus, axis=1))
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Pretrained CNN')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# j. Compare performance
plt.figure(figsize=(10, 6))
plt.plot(cnn_v1_history_Matheus.history['val_accuracy'], label='Baseline CNN')
plt.plot(cnn_v2_history_Matheus.history['val_accuracy'], label='Pretrained CNN')
plt.title('Validation Accuracy: Baseline vs Pretrained CNN')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()