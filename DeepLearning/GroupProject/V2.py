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
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
    sns.scatterplot(data=df, x='odometer', y='sellingprice', alpha=0.6, color='green')
    plt.title('Odometer vs Selling Price', fontsize=16)
    plt.xlabel('Odometer (miles)', fontsize=12)
    plt.ylabel('Selling Price', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()
    
    # Top 15 Brands by Average Price
    plt.figure(figsize=(10, 6))
    avg_price_by_make = df.groupby('make')['sellingprice'].mean().sort_values(ascending=False).head(15)
    avg_price_by_make.plot(kind='bar', color='orange')
    plt.title('Top 15 Brands by Average Selling Price', fontsize=16)
    plt.xlabel('Make', fontsize=12)
    plt.ylabel('Average Selling Price', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# ------------------------- Data Preprocessing --------------------------------
def preprocess_data(df, target_col='sellingprice', test_size=0.2, random_state=42):
    """Preprocess the car dataset incorporating EDA insights."""
    df = df.copy()
    
    # Handle datetime and create age feature
    df['saledate'] = pd.to_datetime(df['saledate'], utc=True)
    df['age'] = df['saledate'].dt.year - df['year']
    
    # Handle missing values based on correlation analysis
    important_cols = ['year', 'condition', 'odometer', 'make', 'model', 'transmission']
    df = df.dropna(subset=important_cols)
    
    # Transform target variable to handle skewness
    df['log_price'] = np.log1p(df[target_col])
    
    # Feature engineering
    df['sale_month'] = df['saledate'].dt.month
    df['sale_quarter'] = df['saledate'].dt.quarter
    
    # Create brand categories based on price analysis
    luxury_brands = ['rolls-royce', 'ferrari', 'lamborghini', 'bentley', 'aston-martin']
    premium_brands = ['tesla', 'porsche', 'bmw', 'mercedes-benz', 'audi']
    df['brand_category'] = df['make'].str.lower().apply(
        lambda x: 'luxury' if x in luxury_brands else 
                 'premium' if x in premium_brands else 'standard'
    )
    
    # Handle odometer readings
    df['odometer_log'] = np.log1p(df['odometer'])
    
    # Prepare features
    categorical_cols = ['make', 'model', 'trim', 'body', 'transmission', 'brand_category']
    numerical_cols = ['year', 'condition', 'odometer_log', 'sale_month', 'sale_quarter', 'age']
    
    # Encode and scale features
    X, y, encoder, scaler = encode_and_scale_features(
        df, categorical_cols, numerical_cols, 'log_price'
    )
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test, encoder, scaler

def encode_and_scale_features(df, categorical_cols, numerical_cols, target_col):
    """Encode categorical variables and scale numerical features."""
    # One-hot encode categorical variables
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    categorical_encoded = encoder.fit_transform(df[categorical_cols])
    categorical_features = pd.DataFrame(
        categorical_encoded,
        columns=encoder.get_feature_names_out(categorical_cols)
    )
    
    # Scale numerical features
    scaler = MinMaxScaler()
    numerical_features = pd.DataFrame(
        scaler.fit_transform(df[numerical_cols]),
        columns=numerical_cols
    )
    
    # Combine features
    X = pd.concat([numerical_features, categorical_features], axis=1)
    y = df[target_col]
    
    return X, y, encoder, scaler

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
    model_path = os.path.join('./saved_models', f"{model_name}.h5")
    
    if os.path.exists(model_path):
        print(f"Loading existing model: {model_path}")
        return load_model(model_path), load_history(model_name)
    else:
        print(f"Creating new model: {model_name}")
        return create_cnn_model(input_shape, config), None

# ------------------------- Model Architecture --------------------------------
def create_cnn_model(input_shape, config):
    """Create a CNN model with the specified configuration."""
    model = Sequential([
        # First Conv1D block
        Conv1D(filters=config['filters'], kernel_size=config['kernel_size'],
               activation='relu', input_shape=input_shape, padding='same'),
        BatchNormalization(),
        Dropout(config['dropout_rate']),
        
        # Second Conv1D block
        Conv1D(filters=config['filters']*2, kernel_size=config['kernel_size'],
               activation='relu', padding='same'),
        BatchNormalization(),
        Dropout(config['dropout_rate']),
        
        # Third Conv1D block
        Conv1D(filters=config['filters']*4, kernel_size=config['kernel_size'],
               activation='relu', padding='same'),
        BatchNormalization(),
        Dropout(config['dropout_rate']),
        
        # Dense layers
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=config['learning_rate']),
        loss='mse',
        metrics=['mae']
    )
    
    return model

# ------------------------- Model Training -----------------------------------
def train_model_with_history(model, X_train, y_train, X_test, y_test, model_name, existing_history=None):
    """Train model with support for continuing from previous history."""
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            f'./saved_models/{model_name}.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    if existing_history:
        for key in history.history:
            history.history[key] = existing_history[key] + history.history[key]
    
    save_history(history, model_name)
    
    return history

def evaluate_model(model, X_test, y_test, scaler):
    """Evaluate the model and print metrics."""
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nModel Evaluation Metrics:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2 Score: {r2:.4f}")

def train_and_evaluate_model(model_num, X_train_cnn, y_train, X_test_cnn, y_test, config):
    """Train and evaluate a single model with save/load capability."""
    model_name = f"model_{model_num}"
    print(f"\nProcessing {model_name} - Configuration {model_num}")
    
    model, existing_history = load_or_create_model(
        model_name,
        input_shape=(X_train_cnn.shape[1], 1),
        config=config
    )
    
    history = train_model_with_history(
        model, X_train_cnn, y_train, X_test_cnn, y_test,
        model_name, existing_history
    )
    
    evaluate_model(model, X_test_cnn, y_test, scaler)
    
    return model, history

# ------------------------- Model Comparison ---------------------------------
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
X_train_cnn = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_cnn = X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1))

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