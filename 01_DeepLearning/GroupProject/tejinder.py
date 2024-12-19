#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 13:52:45 2024

@author: Tejinder
@id: 301232634
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error, mean_squared_error

import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import (Dense, Conv1D, LeakyReLU,
                                     Flatten, Dropout, BatchNormalization)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Set display options for better output formatting
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.colheader_justify', 'center')

# ------------------------- Configurable parameters ----------------------------
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATES = [0.001, 0.0005, 0.001]
FILTERS = [64, 32, 128]
KERNEL_SIZES = [3, 5, 3]
DROPOUT_RATES = [0.2, 0.3, 0.1]
PATIENCE = 10

# ------------------------- Load the dataset -----------------------------------
# Load the car_prices cars_dfset
file_path = './car_prices.csv'
try:
    cars_df = pd.read_csv(file_path, on_bad_lines='skip')
    print("File loaded successfully with problematic rows skipped.")
except Exception as e:
    print("Error loading file:", e)

# ----------------     Exploratory Data Analysis (EDA) -------------------------
# Display the first few rows
print("First 5 Rows of the Dataset:")
print(cars_df.head(), "\n")

# Dataset Info
print("Dataset Info:")
cars_df.info()
print("\n")

# Summary statistics
print("Summary Statistics (Numerical Columns):")
print(cars_df.describe(), "\n")

print("Summary Statistics (Categorical Columns):")
print(cars_df.describe(include='object'), "\n")

# Calculate missing values and percentages
missing_data = cars_df.isnull().sum()
missing_percentage = (missing_data / len(cars_df)) * 100

# Create a summary DataFrame
missing_summary = pd.DataFrame({
    "Missing Count": missing_data,
    "Missing Percentage": missing_percentage
})

# Display the summary DataFrame
print("Missing Data Summary:")
print(missing_summary)

# Data Types
print("Data Types:")
print(cars_df.dtypes, "\n")

# Convert 'saledate' to datetime
cars_df['saledate'] = pd.to_datetime(cars_df['saledate'], errors='coerce', utc=True)

# Calculate the 'age' column
cars_df['age'] = cars_df['saledate'].dt.year - cars_df['year']
# Display the first few rows to verify
print(cars_df[['saledate', 'year', 'age']].head(5))

#------------------------------ VISUALIZATION ----------------------------------
numeric_columns = cars_df.select_dtypes(include='number')
correlation_matrix = numeric_columns.corr()

# Plot: Correlation Heatmap
plt.figure(figsize=(10, 8))
corr_matrix = cars_df[['condition', 'odometer',
                       'year','age', 'sellingprice']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.title('Correlation Matrix', fontsize=16)
plt.show()

# Plot: Distribution of Selling Price
plt.figure(figsize=(10, 6))
sns.histplot(cars_df['sellingprice'], kde=True, bins=30, color='blue')
plt.title('Distribution of Selling Price', fontsize=16)
plt.xlabel('Selling Price', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Plot: Odometer vs Selling Price
plt.figure(figsize=(10, 6))
sns.scatterplot(x=cars_df['odometer'], y=cars_df['sellingprice'], alpha=0.6, color='green', edgecolor='w')
plt.title('Odometer vs Selling Price', fontsize=16)
plt.xlabel('Odometer (miles)', fontsize=12)
plt.ylabel('Selling Price', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Plot: Average Selling Price by Make
average_price_by_make = cars_df.groupby('make')['sellingprice'].mean().sort_values(ascending=False).head(15)
plt.figure(figsize=(10, 6))
average_price_by_make.plot(kind='bar', color='orange', alpha=0.7)
plt.title('Top 15 Brands by Average Selling Price', fontsize=16)
plt.xlabel('Make', fontsize=12)
plt.ylabel('Average Selling Price', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Scatterplot: Relationship between condition and selling price
plt.figure(figsize=(10, 6))
sns.regplot(x=cars_df['condition'], y=cars_df['sellingprice'], scatter_kws={'alpha':0.6}, line_kws={'color':'red'})
plt.title('Scatterplot: Condition vs Selling Price', fontsize=16)
plt.xlabel('Condition', fontsize=12)
plt.ylabel('Selling Price', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# -----------------------     Data Pre-processing     --------------------------
def preprocess_data(cars_df, relevant_cols, target_col, test_size=0.2, random_state=42):
    """
    Preprocess the car dataset for training a machine learning model.

    Parameters:
    - cars_df: DataFrame containing the dataset.
    - relevant_cols: List of column names to keep.
    - target_col: The name of the target column.
    - test_size: Fraction of the data to reserve for testing (default: 0.2).
    - random_state: Random seed for reproducibility (default: 42).

    Returns:
    - X_train, X_test, y_train, y_test: Processed and split features and target.
    - label_encoders: Dictionary of label encoders for categorical columns.
    - scaler: MinMaxScaler object used for normalization.
    """
    # Select relevant columns
    cars_df = cars_df[relevant_cols]

    # Drop rows with missing values in relevant columns
    cars_df.dropna(subset=relevant_cols, inplace=True)

    # Extract features from saledate
    cars_df['saledate'] = pd.to_datetime(cars_df['saledate'], errors='coerce')
    cars_df['sale_month'] = cars_df['saledate'].dt.month
    cars_df['sale_day_of_week'] = cars_df['saledate'].dt.dayofweek
    cars_df.drop('saledate', axis=1, inplace=True)

    # Label encode categorical features
    categorical_columns = ['make', 'model', 'trim', 'body', 'transmission']
##############    categorical_columns = cars_df.select_dtypes(include='number')
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        cars_df[col] = le.fit_transform(cars_df[col])
        label_encoders[col] = le  # Store encoder for potential inverse transformation

    # Normalize numerical features
    scaler = MinMaxScaler()
    numerical_columns = ['year', 'condition', 'odometer', 'sale_month', 'sale_day_of_week']
################ numeric_columns = cars_df.select_dtypes(include='number')
    cars_df[numerical_columns] = scaler.fit_transform(cars_df[numerical_columns])

    # Separate features and target
    X = cars_df.drop(target_col, axis=1).values
    y = cars_df[target_col].values

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test, label_encoders, scaler

# Define columns
relevant_cols = ['year', 'age', 'make', 'model', 'trim', 'body', 'transmission', 'condition', 'odometer', 'saledate', 'sellingprice']
target_col = 'sellingprice'

# Preprocess the dataset
X_train, X_test, y_train, y_test, label_encoders, scaler = preprocess_data(
    cars_df, relevant_cols, target_col
)

# Print shapes of the datasets
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# --------------------------     Model Building     ----------------------------
# ---------- 1. Supervised Learning: Design and implement your own CNN/RNN model
# to address the specific task (i.e. image classification/segmentation/anomaly
# detection/sentiment analysis) of your problem. Groups should experiment and
# analyze the performance of the architecture using varying hyperparameters
# (i.e. #layers, #neurons, #kernals).Groups should also experiment with
# different regularization and normalization optimization methods.

# Function to create a CNN model
def create_cnn_model(input_shape, filters, kernel_size, dropout_rate, learning_rate):
    """
    Create a CNN model for regression tasks.
    
    Parameters:
    - input_shape: Tuple, shape of the input data (timesteps, features).
    - filters: Int, number of filters for Conv1D layers.
    - kernel_size: Int, size of the convolution kernel.
    - dropout_rate: Float, dropout rate for regularization.
    - learning_rate: Float, learning rate for the optimizer.
    
    Returns:
    - Compiled Keras model.
    """
    model = Sequential()

    # Input layer
    model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', input_shape=input_shape, padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    # Hidden layers
    model.add(Conv1D(filters=filters * 2, kernel_size=kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate + 0.1))

    model.add(Conv1D(filters=filters * 4, kernel_size=kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate + 0.2))

    # Flatten and Dense layers
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))  # Output layer for regression

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])
    return model

# Early Stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True, verbose=1)

# Function to train and evaluate a model
def train_and_evaluate_model(model, model_name, X_train_cnn, y_train, X_test_cnn, y_test, epochs, batch_size):
    print(f"\nTraining {model_name}...\n")
    
    # Train the model
    history = model.fit(
        X_train_cnn, y_train,
        validation_data=(X_test_cnn, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate the model
    test_loss, test_mae = model.evaluate(X_test_cnn, y_test, verbose=0)
    print(f"{model_name} - Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}")
    
    # Plot training history
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    return history, test_loss, test_mae

# Reshape X_train and X_test for CNN input
X_train_cnn = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_cnn = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Define the input shape for the CNN
input_shape = (X_train_cnn.shape[1], X_train_cnn.shape[2])

# Create the CNN model
# Model 1
cnn_model_1 = create_cnn_model(input_shape, filters=FILTERS[0], kernel_size=KERNEL_SIZES[0], 
                                dropout_rate=DROPOUT_RATES[0], learning_rate=LEARNING_RATES[0])

# Display the model summary
cnn_model_1.summary()

# Model 2
cnn_model_2 = create_cnn_model(input_shape, filters=FILTERS[1], kernel_size=KERNEL_SIZES[1], 
                                dropout_rate=DROPOUT_RATES[1], learning_rate=LEARNING_RATES[1])
# Display the model summary
cnn_model_2.summary()

# Model 3
cnn_model_3 = create_cnn_model(input_shape, filters=FILTERS[2], kernel_size=KERNEL_SIZES[2], 
                                dropout_rate=DROPOUT_RATES[2], learning_rate=LEARNING_RATES[2])

# Display the model summary
cnn_model_3.summary()

# Experiment 1
history_1, test_loss_1, test_mae_1 = train_and_evaluate_model(
    cnn_model_1, "CNN Model 1", X_train_cnn, y_train, X_test_cnn, y_test, EPOCHS, BATCH_SIZE
)

# Experiment 2
history_2, test_loss_2, test_mae_2 = train_and_evaluate_model(
    cnn_model_2, "CNN Model 2", X_train_cnn, y_train, X_test_cnn, y_test, EPOCHS, BATCH_SIZE
)

# Experiment 3
history_3, test_loss_3, test_mae_3 = train_and_evaluate_model(
    cnn_model_3, "CNN Model 3", X_train_cnn, y_train, X_test_cnn, y_test, EPOCHS, BATCH_SIZE
)

"""-----------------------     Evaluation    -------------------------------"""

# Evaluate model on test data
test_loss_1, test_mae_1 = cnn_model_1.evaluate(X_test_cnn, y_test, verbose=1)
print(f"CNN Model 1 - Test Loss: {test_loss_1:.4f}, Test MAE: {test_mae_1:.4f}")

# Predict on the test set
y_pred = cnn_model_1.predict(X_test_cnn)

# Display first 10 predictions vs actual values
print("Predicted vs Actual Selling Prices:")
for i in range(10):
    print(f"Predicted: {y_pred[i][0]:.2f}, Actual: {y_test[i]:.2f}")

# Scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title("Predicted vs Actual Selling Prices")
plt.xlabel("Actual Selling Prices")
plt.ylabel("Predicted Selling Prices")
plt.show()



# Calculate R^2 score
r2 = r2_score(y_test, y_pred)
print(f"CNN Model 1 - R^2 Score: {r2:.4f}")


# Calculate MAPE
mape = np.mean(np.abs((y_test - y_pred.flatten()) / y_test)) * 100
print(f"CNN Model 1 - MAPE: {mape:.2f}%")

# Directory to save models
save_dir = '/Users/user/Documents/Deep Learning/Final Project/models'
os.makedirs(save_dir, exist_ok=True)

# Save Model 1 as .keras file
model_1_path = os.path.join(save_dir, 'cnn_model_1.keras')
cnn_model_1.save(model_1_path)

print(f"Model 1 saved at {model_1_path}")
# Save Model 1 as .h5 file
model_1_path = os.path.join(save_dir, 'cnn_model_1.h5')
cnn_model_1.save(model_1_path)

print(f"Model 1 saved at {model_1_path}")

# Define the path to the saved model
model_path = '/Users/user/Documents/Deep Learning/Final Project/models/cnn_model_1.keras'
# Load the .keras model
cnn_model_1 = load_model(model_path)
print("Loaded model from .keras file.")


# """-----------------------     Unsupervised    -----------------------------"""
# # Set random seed for reproducibility
# np.random.seed(42)
# tf.random.set_seed(42)

# # Parameters
# latent_dim = 100  # Latent space dimension (noise vector size)
# input_dim = X_train.shape[1]  # Number of features in the dataset
# epochs = 5000
# batch_size = 64
# sample_interval = 1000  # Save generated samples every 1000 epochs
# patience = 200  # Early stopping patience

# # Initialize arrays for tracking losses
# g_losses = []
# d_losses = []

# # 1. Build Generator
# def build_generator(latent_dim, output_dim):
#     model = Sequential([
#         Dense(128, input_dim=latent_dim),
#         LeakyReLU(alpha=0.2),
#         BatchNormalization(momentum=0.8),
#         Dense(256),
#         LeakyReLU(alpha=0.2),
#         BatchNormalization(momentum=0.8),
#         Dense(output_dim, activation='tanh')
#     ])
#     return model

# # 2. Build Discriminator
# def build_discriminator(input_dim):
#     model = Sequential([
#         Dense(256, input_dim=input_dim),
#         LeakyReLU(alpha=0.2),
#         Dropout(0.3),  # Added dropout for regularization
#         Dense(128),
#         LeakyReLU(alpha=0.2),
#         Dropout(0.3),
#         Dense(1, activation='sigmoid')
#     ])
#     return model

# # Instantiate Models
# generator = build_generator(latent_dim, input_dim)
# discriminator = build_discriminator(input_dim)

# # Compile Discriminator
# discriminator.compile(
#     optimizer=Adam(learning_rate=0.0004, beta_1=0.5),
#     loss='binary_crossentropy',
#     metrics=['accuracy']
# )

# # Build and Compile GAN
# discriminator.trainable = False  # Freeze discriminator during generator training
# gan = Sequential([generator, discriminator])
# gan.compile(
#     optimizer=Adam(learning_rate=0.0001, beta_1=0.5), 
#     loss='binary_crossentropy'
# )

# # Visualize Noise
# def visualize_noise(latent_dim, num_samples):
#     noise = np.random.normal(0, 1, (num_samples, latent_dim))
#     plt.figure(figsize=(8, 6))
#     plt.scatter(noise[:, 0], noise[:, 1], alpha=0.6, color='blue', edgecolor='k')
#     plt.title("Noise Distribution (First Two Dimensions)", fontsize=14)
#     plt.xlabel("Dimension 1", fontsize=12)
#     plt.ylabel("Dimension 2", fontsize=12)
#     plt.grid(True, linestyle='--', alpha=0.7)
#     plt.show()

# visualize_noise(latent_dim, num_samples=1000)

# # Training Loop
# best_g_loss = np.inf
# no_improvement_epochs = 0

# for epoch in range(epochs):
#     # Select random real data samples
#     idx = np.random.randint(0, X_train.shape[0], batch_size)
#     real_data = X_train[idx]

#     # Generate fake data samples
#     noise = np.random.normal(0, 1, (batch_size, latent_dim))
#     fake_data = generator.predict(noise)

#     # Add noise to labels
#     y_real = np.random.uniform(0.8, 1.2, (batch_size, 1))  # Noisy real labels
#     y_fake = np.random.uniform(0.0, 0.3, (batch_size, 1))  # Noisy fake labels
    
#     discriminator.reset_metrics()

#     # Train Discriminator
#     discriminator.trainable = True  # Enable discriminator training
#     try:
#         d_loss_real = discriminator.train_on_batch(real_data, y_real)
#         d_loss_fake = discriminator.train_on_batch(fake_data, y_fake)
#     except Exception as e:
#         print(f"Error during discriminator training: {e}")
#         continue
#     d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
#     discriminator.trainable = False  # Disable discriminator training

#     # Train Generator
#     noise = np.random.normal(0, 1, (batch_size, latent_dim))
#     g_loss = gan.train_on_batch(noise, y_real)

#     # Append losses
#     g_losses.append(g_loss)
#     d_losses.append(d_loss[0])

#     # Print progress
#     if epoch % 100 == 0:
#         print(f"Epoch {epoch}/{epochs} [D loss: {d_loss[0]:.4f}, acc.: {100 * d_loss[1]:.2f}%] [G loss: {g_loss:.4f}]")

# # Generate Synthetic Data
# num_samples = 1000
# noise = np.random.normal(0, 1, (num_samples, latent_dim))
# synthetic_data = generator.predict(noise)

# # Combine Synthetic and Original Data
# X_augmented = np.vstack((X_train, synthetic_data))
# y_augmented = np.hstack((y_train, np.random.choice(y_train, num_samples)))

# # Plot Losses
# plt.figure(figsize=(10, 6))
# plt.plot(range(len(g_losses)), g_losses, label="Generator Loss")
# plt.plot(range(len(d_losses)), d_losses, label="Discriminator Loss")
# plt.title("GAN Training Losses")
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.legend()
# plt.show()

# # Visualize Final Generated Data
# plt.figure(figsize=(8, 6))
# plt.scatter(synthetic_data[:, 0], synthetic_data[:, 1], alpha=0.6, color='orange', label='Synthetic Data')
# plt.scatter(X_train[:, 0], X_train[:, 1], alpha=0.4, color='blue', label='Original Data')
# plt.title("Original vs Synthetic Data (First Two Features)", fontsize=14)
# plt.xlabel("Feature 1", fontsize=12)
# plt.ylabel("Feature 2", fontsize=12)
# plt.legend()
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.show()

# # 8. Retrain CNN on Augmented Data
# X_augmented_cnn = X_augmented.reshape((X_augmented.shape[0], X_augmented.shape[1], 1))
# cnn_model = create_cnn_model(
#     input_shape=(X_augmented_cnn.shape[1], X_augmented_cnn.shape[2]),
#     filters=64,
#     kernel_size=3,
#     dropout_rate=0.2,
#     learning_rate=0.001
# )

# # Train the CNN with augmented data
# history, test_loss, test_mae = train_and_evaluate_model(
#     cnn_model, "CNN with Augmented Data", X_augmented_cnn, y_augmented, X_test_cnn, y_test, EPOCHS, BATCH_SIZE
# )

# def evaluate_model(y_true, y_pred):
#     """
#     Evaluate the model using various regression metrics.
    
#     Parameters:
#     - y_true: True target values.
#     - y_pred: Predicted target values.

#     Returns:
#     - metrics: Dictionary containing the evaluation metrics.
#     """
#     mae = mean_absolute_error(y_true, y_pred)
#     rmse = np.sqrt(mean_squared_error(y_true, y_pred))
#     mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
#     r2 = r2_score(y_true, y_pred)
    
#     return {
#         "MAE": mae,
#         "RMSE": rmse,
#         "MAPE (%)": mape,
#         "R²": r2
#     }

# # Predict on test data
# y_pred_before_aug = cnn_model_1.predict(X_test_cnn).flatten()  # Before augmentation
# y_pred_after_aug = cnn_model.predict(X_test_cnn).flatten()    # After augmentation

# # Evaluate both models
# metrics_before_aug = evaluate_model(y_test, y_pred_before_aug)
# metrics_after_aug = evaluate_model(y_test, y_pred_after_aug)

# # Display metrics
# import pandas as pd

# comparison_df = pd.DataFrame([metrics_before_aug, metrics_after_aug], 
#                              index=["Before Data Augmentation", "After Data Augmentation"])
# comparison_df.index.name = "Model"
# comparison_df.columns = ["MAE", "RMSE", "MAPE (%)", "R²"]

# print("Model Performance Comparison:")
# print(comparison_df)

# import matplotlib.pyplot as plt

# # Bar plot for metrics comparison
# comparison_df.plot(kind='bar', figsize=(10, 6), rot=0)
# plt.title("Model Performance Metrics Comparison", fontsize=16)
# plt.ylabel("Metric Value", fontsize=12)
# plt.legend(loc="best", fontsize=10)
# plt.grid(axis="y", linestyle="--", alpha=0.7)
# plt.tight_layout()
# plt.show()

# # Scatter plot: Predicted vs Actual Selling Prices
# plt.figure(figsize=(10, 6))
# plt.scatter(y_test, y_pred_before_aug, alpha=0.5, label="Before Augmentation", color="blue")
# plt.scatter(y_test, y_pred_after_aug, alpha=0.5, label="After Augmentation", color="orange")
# plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linestyle="--")
# plt.title("Predicted vs Actual Selling Prices", fontsize=16)
# plt.xlabel("Actual Selling Prices", fontsize=12)
# plt.ylabel("Predicted Selling Prices", fontsize=12)
# plt.legend()
# plt.grid(True, linestyle="--", alpha=0.7)
# plt.show()
