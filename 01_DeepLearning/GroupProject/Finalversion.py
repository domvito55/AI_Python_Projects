#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 13:52:45 2024
Car Price Prediction using CNN with save/load capabilities

@author: Tejinder
@id: 301232634
"""

# ---------------------------- Import Libraries -----------------------------------
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ------------------------- Configurable Parameters ------------------------------
# Model parameters
BATCH_SIZE = 32
EPOCHS = 50
MODEL_CONFIGS = [
    {
        'filters': 64,
        'kernel_size': 3,
        'dropout_rate': 0.2,
        'learning_rate': 0.001
    },
    {
        'filters': 32,
        'kernel_size': 5,
        'dropout_rate': 0.3,
        'learning_rate': 0.0005
    },
    {
        'filters': 128,
        'kernel_size': 3,
        'dropout_rate': 0.1,
        'learning_rate': 0.001
    }
]
PATIENCE = 10

# Create directory for saved models
os.makedirs('./saved_models', exist_ok=True)

# Display options for better output formatting
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.colheader_justify', 'center')

# ---------------------------- Data Loading ------------------------------------
def load_data(file_path):
    """Load and perform initial data checks."""
    try:
        df = pd.read_csv(file_path, on_bad_lines='skip')
        print("File loaded successfully with problematic rows skipped.")
        return df
    except Exception as e:
        print("Error loading file:", e)
        return None

# ------------------------- Exploratory Data Analysis --------------------------
def perform_eda(df):
    """Perform exploratory data analysis and create visualizations."""
    # Basic data info
    print("\nFirst 5 Rows of the Dataset:")
    print(df.head())
    
    print("\nDataset Info:")
    df.info()
    
    print("\nSummary Statistics (Numerical Columns):")
    print(df.describe())
    
    print("\nSummary Statistics (Categorical Columns):")
    print(df.describe(include='object'))
    
    # Missing values analysis
    missing_data = df.isnull().sum()
    missing_percentage = (missing_data / len(df)) * 100
    missing_summary = pd.DataFrame({
        "Missing Count": missing_data,
        "Missing Percentage": missing_percentage
    })
    print("\nMissing Data Summary:")
    print(missing_summary)
    
    # Create visualizations
    create_visualizations(df)
    
    return df

def create_visualizations(df):
    """Create and display analysis visualizations."""
    # Correlation Matrix
    plt.figure(figsize=(10, 8))
    corr_matrix = df[['condition', 'odometer', 'year', 'age', 'sellingprice']].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True)
    plt.title('Correlation Matrix', fontsize=16)
    plt.show()
    
    # Distribution of Selling Price
    plt.figure(figsize=(10, 6))
    sns.histplot(df['sellingprice'], kde=True, bins=30, color='blue')
    plt.title('Distribution of Selling Price', fontsize=16)
    plt.xlabel('Selling Price', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
    
    # Odometer vs Selling Price
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df['odometer'], y=df['sellingprice'], alpha=0.6, color='green')
    plt.title('Odometer vs Selling Price', fontsize=16)
    plt.xlabel('Odometer (miles)', fontsize=12)
    plt.ylabel('Selling Price', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

# ------------------------- Data Preprocessing --------------------------------
def preprocess_data(df, test_size=0.2, random_state=42):
    """Preprocess the car dataset using the successful approach."""
    # Define columns (using successful version's columns)
    relevant_cols = ['year', 'age', 'make', 'model', 'trim', 'body', 'transmission', 
                    'condition', 'odometer', 'saledate', 'sellingprice']
    
    # Select relevant columns
    df = df[relevant_cols]

    # Drop rows with missing values
    df.dropna(subset=relevant_cols, inplace=True)

    # Extract features from saledate
    df['saledate'] = pd.to_datetime(df['saledate'], errors='coerce')
    df['sale_month'] = df['saledate'].dt.month
    df['sale_day_of_week'] = df['saledate'].dt.dayofweek
    df.drop('saledate', axis=1, inplace=True)

    # Label encode categorical features
    categorical_columns = ['make', 'model', 'trim', 'body', 'transmission']
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    # Normalize numerical features
    scaler = MinMaxScaler()
    numerical_columns = ['year', 'condition', 'odometer', 'sale_month', 'sale_day_of_week']
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

    # Separate features and target
    X = df.drop('sellingprice', axis=1).values
    y = df['sellingprice'].values

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test, label_encoders, scaler

# ------------------------- Model Architecture --------------------------------
def create_cnn_model(input_shape, config):
    """Create a CNN model with the specified configuration."""
    model = Sequential()

    # Input layer
    model.add(Conv1D(filters=config['filters'], kernel_size=config['kernel_size'],
               activation='relu', input_shape=input_shape, padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(config['dropout_rate']))

    # Hidden layers
    model.add(Conv1D(filters=config['filters'] * 2, kernel_size=config['kernel_size'],
               activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(config['dropout_rate'] + 0.1))

    model.add(Conv1D(filters=config['filters'] * 4, kernel_size=config['kernel_size'],
               activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(config['dropout_rate'] + 0.2))

    # Flatten and Dense layers
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=config['learning_rate']),
                 loss='mse', metrics=['mae'])
    
    return model

# ------------------------- Model Management ----------------------------------
def save_history(history, model_name):
    """Save training history to JSON file."""
    history_path = os.path.join('./saved_models', f"{model_name}_history.json")
    with open(history_path, 'w') as f:
        history_dict = {}
        for key, value in history.history.items():
            history_dict[key] = [float(v) for v in value]
        json.dump(history_dict, f)

def load_history(model_name):
    """Load training history from JSON file."""
    history_path = os.path.join('./saved_models', f"{model_name}_history.json")
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            return json.load(f)
    return None

def load_or_create_model(model_name, input_shape, config):
    """Load existing model or create new one if it doesn't exist."""
    model_path = os.path.join('./saved_models', f"{model_name}.keras")
    
    if os.path.exists(model_path):
        print(f"Loading existing model: {model_path}")
        return load_model(model_path), load_history(model_name)
    else:
        print(f"Creating new model: {model_name}")
        return create_cnn_model(input_shape, config), None

# ------------------------- Model Training ------------------------------------
def train_and_evaluate_model(model_num, X_train_cnn, y_train, X_test_cnn, y_test, config):
    """Train and evaluate a single model with save/load capability."""
    model_name = f"model_{model_num}"
    print(f"\nProcessing {model_name} - Configuration {model_num}")
    
    # Load or create the model and its history
    model, existing_history = load_or_create_model(
        model_name,
        input_shape=(X_train_cnn.shape[1], 1),
        config=config
    )
    
    # Determine the initial_epoch from existing history
    if existing_history and 'loss' in existing_history:
        initial_epoch = len(existing_history['loss'])
    else:
        initial_epoch = 0
    
    print(f"Starting training from epoch {initial_epoch + 1} to {EPOCHS}.")

    # Create callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            f'./saved_models/{model_name}.keras',
            monitor='val_loss',
            save_best_only=True,  # Ensures only the best model is saved
            verbose=1
        )
    ]
    
    # Train the model
    history = model.fit(
        X_train_cnn, y_train,
        validation_data=(X_test_cnn, y_test),
        epochs=EPOCHS,            # Total epochs to train
        initial_epoch=initial_epoch,  # Resume from the last completed epoch
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    # Combine new history with existing history
    if existing_history:
        for key in history.history:
            history.history[key] = existing_history[key] + history.history[key]
    
    # Save the updated training history
    save_history(history, model_name)
    
    # Evaluate the model
    y_pred = model.predict(X_test_cnn)
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred.flatten()) / y_test)) * 100
    
    print(f"\nModel {model_num} Evaluation:")
    print(f"R² Score: {r2:.4f}")
    print(f"MAPE: {mape:.2f}%")
    
    return model, history

def compare_models():
    """Load and compare all available trained models."""
    histories = []
    model_names = []
    
    for i in range(1, 4):
        history = load_history(f"model_{i}")
        if history:
            histories.append(history)
            model_names.append(f"Model {i}")
    
    if len(histories) > 1:
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        for i, hist in enumerate(histories):
            plt.plot(hist['loss'], label=model_names[i])
        plt.title('Training Loss Comparison')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        for i, hist in enumerate(histories):
            plt.plot(hist['val_loss'], label=model_names[i])
        plt.title('Validation Loss Comparison')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    else:
        print("Not enough trained models found for comparison")

# ------------------------- Modular Execution Steps ----------------------------------
# ------------------------- Execution Steps ----------------------------------
"""
These steps should be run in separate cells in a Jupyter notebook for better control
and ability to save intermediate results
"""

# Step 1: Load and explore data
file_path = './car_prices.csv'
cars_df = load_data(file_path)

# Calculate age before EDA
cars_df['saledate'] = pd.to_datetime(cars_df['saledate'], utc=True)
cars_df['age'] = cars_df['saledate'].dt.year - cars_df['year']

cars_df = perform_eda(cars_df)

# Step 2: Preprocess data
X_train, X_test, y_train, y_test, encoder, scaler = preprocess_data(cars_df)

# Step 3: Reshape data for CNN
X_train_cnn = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_cnn = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Step 4: Train Model 1
# This will load existing model if available, or create new one if not
print("\nProcessing Model 1 - Conservative Configuration")
model1, history1 = train_and_evaluate_model(
    model_num=1,
    X_train_cnn=X_train_cnn,
    y_train=y_train,
    X_test_cnn=X_test_cnn,
    y_test=y_test,
    config=MODEL_CONFIGS[0]
)

# Step 5: Train Model 2
print("\nProcessing Model 2 - Moderate Configuration")
model2, history2 = train_and_evaluate_model(
    model_num=2,
    X_train_cnn=X_train_cnn,
    y_train=y_train,
    X_test_cnn=X_test_cnn,
    y_test=y_test,
    config=MODEL_CONFIGS[1]
)

# Step 6: Train Model 3
print("\nProcessing Model 3 - Aggressive Configuration")
model3, history3 = train_and_evaluate_model(
    model_num=3,
    X_train_cnn=X_train_cnn,
    y_train=y_train,
    X_test_cnn=X_test_cnn,
    y_test=y_test,
    config=MODEL_CONFIGS[2]
)

# Step 7: Compare all available models
# This can be run even if only some models are trained
print("\nComparing available trained models...")
compare_models()


# ------------------------- Feature Extraction and Transfer Learning ----------------------
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, BatchNormalization, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback

# ------------------------- History Saving/Loading -------------------------
def save_history(history, model_name):
    """Save training history (dict) to JSON file."""
    history_path = os.path.join('/content/drive/MyDrive/Colab Notebooks/saved_models', f"{model_name}_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f)

def load_history(model_name):
    """Load training history from JSON file, return dict or None."""
    history_path = os.path.join('/content/drive/MyDrive/Colab Notebooks/saved_models', f"{model_name}_history.json")
    if os.path.exists(history_path) and os.path.getsize(history_path) > 0:
        with open(history_path, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Corrupted history file {history_path}, starting fresh.")
                return None
    return None

# ------------------------- SaveHistoryCallback -------------------------
class SaveHistoryCallback(Callback):
    def __init__(self, model_name, history, save_history_func):
        super().__init__()
        self.model_name = model_name
        self.history = history if history is not None else {}
        self.save_history_func = save_history_func

    def on_epoch_end(self, epoch, logs=None):
        if logs:
            for key, value in logs.items():
                if key not in self.history:
                    self.history[key] = []
                self.history[key].append(value)
        self.save_history_func(self.history, self.model_name)

# ------------------------- Autoencoder Creation -------------------------
def create_autoencoder(input_shape, config):
    """Create autoencoder for feature extraction."""
    encoder = Sequential([
        Conv1D(config['filters'][0], config['kernel_size'], activation='relu', 
               padding='same', input_shape=input_shape),
        BatchNormalization(),
        Conv1D(config['filters'][1], config['kernel_size'], activation='relu', padding='same'),
        BatchNormalization(),
        Conv1D(config['filters'][2], config['kernel_size'], activation='relu', padding='same'),
        BatchNormalization(),
        Flatten(),
        Dense(config['encoding_dim'], activation='relu', name='encoder_output')
    ])
    
    decoder = Sequential([
        Dense(input_shape[0] * config['filters'][2], activation='relu'),
        Reshape((input_shape[0], config['filters'][2])),
        Conv1D(config['filters'][2], config['kernel_size'], activation='relu', padding='same'),
        BatchNormalization(),
        Conv1D(config['filters'][1], config['kernel_size'], activation='relu', padding='same'),
        BatchNormalization(),
        Conv1D(config['filters'][0], config['kernel_size'], activation='relu', padding='same'),
        BatchNormalization(),
        Conv1D(1, config['kernel_size'], activation='linear', padding='same')
    ])
    
    autoencoder = Sequential([encoder, decoder])
    autoencoder.compile(
        optimizer=Adam(learning_rate=config['learning_rate']), 
        loss='mse'
    )
    return autoencoder, encoder

def load_or_create_autoencoder(model_name, input_shape, config):
    """Load existing autoencoder or create new one if it doesn't exist."""
    model_path = os.path.join('/content/drive/MyDrive/Colab Notebooks/saved_models', f"{model_name}.keras")
    history = load_history(model_name)
    if os.path.exists(model_path):
        print(f"Loading existing autoencoder: {model_path}")
        autoencoder = load_model(model_path)
        # Extract encoder part
        encoder = Sequential(autoencoder.layers[0].layers)
        encoder.build(input_shape=(None, *input_shape))
        return autoencoder, encoder, history
    else:
        print(f"Creating new autoencoder: {model_name}")
        autoencoder, encoder = create_autoencoder(input_shape, config)
        return autoencoder, encoder, history

def train_autoencoder(model_name, X_train, config):
    """Train the autoencoder with incremental history saving."""
    input_shape = (X_train.shape[1], 1)
    autoencoder, encoder, existing_history = load_or_create_autoencoder(model_name, input_shape, config)

    # Determine initial_epoch
    if existing_history and 'loss' in existing_history:
        initial_epoch = len(existing_history['loss'])
    else:
        initial_epoch = 0

    save_history_callback = SaveHistoryCallback(model_name, existing_history, save_history)

    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=config['patience'],
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            os.path.join('/content/drive/MyDrive/Colab Notebooks/saved_models', f"{model_name}.keras"),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        save_history_callback
    ]

    history_obj = autoencoder.fit(
        X_train, X_train,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        validation_split=0.2,
        initial_epoch=initial_epoch,
        callbacks=callbacks,
        verbose=1
    )

    # After training completes, history is saved incrementally, no extra action needed.
    return autoencoder, encoder

# ------------------------- CNN Model (Transfer Learning) -------------------------
def create_cnn_model(input_shape, filters, kernel_size, dropout_rate, learning_rate):
    model = Sequential()
    model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', input_shape=input_shape, padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    model.add(Conv1D(filters=filters*2, kernel_size=kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate + 0.1))

    model.add(Conv1D(filters=filters*4, kernel_size=kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate + 0.2))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])
    return model

def train_and_evaluate_model(model, model_name, X_train, y_train, X_test, y_test, epochs, batch_size):
    # For simplicity, we can just train from scratch here or implement a similar history-saving mechanism
    # If desired, you can also implement the SaveHistoryCallback for this model.
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        ModelCheckpoint(os.path.join('/content/drive/MyDrive/Colab Notebooks/saved_models', f"{model_name}.keras"), monitor='val_loss', save_best_only=True, verbose=1)
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"\n{model_name} Evaluation:")
    print(f"R² Score: {r2:.4f}")

    return history

# ------------------------- Feature Extraction and Transfer Learning ----------------------
# Assuming X_train_cnn, X_test_cnn, y_train, y_test are already defined.

AUTOENCODER_CONFIG = {
    'filters': [32, 16, 8],
    'kernel_size': 3,
    'learning_rate': 0.001,
    'encoding_dim': X_train_cnn.shape[1] // 3,  # Reducing features to 1/3
    'batch_size': 64,
    'epochs': 50,
    'patience': 5
}

# Train or load autoencoder
print("\nStarting Feature Extraction...")
autoencoder, encoder = train_autoencoder('autoencoder', X_train_cnn, AUTOENCODER_CONFIG)

# Extract features
X_train_encoded = encoder.predict(X_train_cnn)
X_test_encoded = encoder.predict(X_test_cnn)

# Reshape for CNN input
X_train_encoded = X_train_encoded.reshape((X_train_encoded.shape[0], X_train_encoded.shape[1], 1))
X_test_encoded = X_test_encoded.reshape((X_test_encoded.shape[0], X_test_encoded.shape[1], 1))

EPOCHS = 50
BATCH_SIZE = 32

# Select the best model config (e.g., model 1 is at index 0)
best_model_config = MODEL_CONFIGS[0]

print("\nTraining model with extracted features...")
cnn_transfer = create_cnn_model(
    input_shape=(X_train_encoded.shape[1], 1),
    filters=best_model_config['filters'],
    kernel_size=best_model_config['kernel_size'],
    dropout_rate=best_model_config['dropout_rate'],
    learning_rate=best_model_config['learning_rate']
)

history_transfer = train_and_evaluate_model(
    cnn_transfer, 
    "Transfer_Learning_Model", 
    X_train_encoded, 
    y_train, 
    X_test_encoded, 
    y_test, 
    EPOCHS, 
    BATCH_SIZE
)
