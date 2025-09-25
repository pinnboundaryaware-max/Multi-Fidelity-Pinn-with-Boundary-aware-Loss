# -*- coding: utf-8 -*-

"""
!pip install tensorflow --quiet
print("TensorFlow installed.")

# Step 1: Install the required packages
!pip install xgboost pykrige

# Importing libraries and run the analysis
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import numpy as np

# Loading the dataset
try:
    df = pd.read_csv('data_full (2).csv')
    print("Successfully loaded 'data_full (2).csv'.")
except FileNotFoundError:
    print("ERROR: Make sure 'data_full (2).csv' is uploaded to your Colab session.")
    exit()

df.dropna(inplace=True)

# Features for Random Forest and XGBoost
features = ['surf_x', 'surf_y', 'surf_vx', 'surf_vy', 'surf_elv', 'surf_dhdt', 'surf_SMB']
target = 'track_bed_target'

X = df[features]
y = df[target]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Data prepared. Training on {len(X_train)} samples, testing on {len(X_test)} samples.\n")

# 1. Random Forest Regressor
print("Training Random Forest Regressor")
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
print(f"Random Forest MAE: {mae_rf:.4f}")
print(f"Random Forest R-squared: {r2_rf:.4f}")
print(f"Random Forest MSE: {mse_rf:.4f}")
print(f"Random Forest RMSE: {rmse_rf:.4f}\n")

# 2. XGBoost Regressor
print("Training XGBoost Regressor")
xgb = XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mse_xgb)
print(f"XGBoost MAE: {mae_xgb:.4f}")
print(f"XGBoost R-squared: {r2_xgb:.4f}")
print(f"XGBoost MSE: {mse_xgb:.4f}")
print(f"XGBoost RMSE: {rmse_xgb:.4f}\n")

# Model Comparison
print("Model Comparison")
results = {
    "Model": ["Random Forest", "XGBoost"],
    "MAE": [mae_rf, mae_xgb],
    "R-squared": [r2_rf, r2_xgb],
    "MSE": [mse_rf, mse_xgb],
    "RMSE": [rmse_rf, rmse_xgb]
}
results_df = pd.DataFrame(results)
print(results_df)

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# --------------------------
# Plot 1: Predicted vs. True Values
# --------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Predicted vs. True Values', fontsize=16)

def plot_predictions(ax, y_true, y_pred, title):
    """Generates a scatter plot of true vs. predicted values."""
    ax.scatter(y_true, y_pred, alpha=0.5, edgecolors='k', s=20)
    ax.plot([y_true.min(), y_true.max()],
            [y_true.min(), y_true.max()],
            '--', color='red', lw=2)
    ax.set_xlabel('True Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title(title)

# Generate plots for each model
plot_predictions(axes[0], y_test, y_pred_rf, 'Random Forest')
plot_predictions(axes[1], y_test, y_pred_xgb, 'XGBoost')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("predicted_vs_true.png")
plt.show()

# --------------------------
# Plot 2: Residual Plots
# --------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Residual Plots', fontsize=16)

def plot_residuals(ax, y_true, y_pred, title):
    """Generates a scatter plot of residuals."""
    residuals = y_true - y_pred
    ax.scatter(y_pred, residuals, alpha=0.5, edgecolors='k', s=20)
    ax.axhline(y=0, color='red', linestyle='--')
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Residuals (True - Predicted)')
    ax.set_title(title)

plot_residuals(axes[0], y_test, y_pred_rf, 'Random Forest')
plot_residuals(axes[1], y_test, y_pred_xgb, 'XGBoost')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("residual_plots.png")
plt.show()

#IMPORTS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
print("Libraries imported.")

# Loading the Dataset
try:
    df = pd.read_csv('data_full (2).csv')
    print("Successfully loaded 'data_full (2).csv'.")
except FileNotFoundError:
    print("ERROR: Could not find 'data_full (2).csv'. Please upload it to your Colab session.")
    exit()

# Data Preparation
df.dropna(inplace=True)

features = ['surf_x', 'surf_y', 'surf_vx', 'surf_vy', 'surf_elv', 'surf_dhdt', 'surf_SMB']
target = 'track_bed_target'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling the Data
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))

print(f"Data prepared and scaled: {len(X_train)} training samples, {len(X_test)} testing samples.\n")

# MULTI-LAYER PERCEPTRON (MLP) BASELINE
print("Building and Training MLP Model")

# Defining the model architecture
mlp_model = Sequential([

    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)
])

# Compiling the model
mlp_model.compile(optimizer='adam', loss='mae')

# Train the model
history_mlp = mlp_model.fit(
    X_train_scaled,
    y_train_scaled,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    verbose=0
)
print("MLP training complete.")

y_pred_scaled_mlp = mlp_model.predict(X_test_scaled)
y_pred_mlp = scaler_y.inverse_transform(y_pred_scaled_mlp)

# Calculate metrics for MLP
mae_mlp = mean_absolute_error(y_test, y_pred_mlp)
r2_mlp = r2_score(y_test, y_pred_mlp)
mse_mlp = mean_squared_error(y_test, y_pred_mlp)
rmse_mlp = np.sqrt(mse_mlp)


#1D CONVOLUTIONAL NEURAL NETWORK (CNN) BASELINE
print("\n--- Building and Training 1D CNN Model ---")


X_train_cnn = np.expand_dims(X_train_scaled, axis=2)
X_test_cnn = np.expand_dims(X_test_scaled, axis=2)

# Define the model architecture
cnn_model = Sequential([
    Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X_train_cnn.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(32, activation='relu'),
    Dense(1)
])

# Compile and train the model
cnn_model.compile(optimizer='adam', loss='mae')
history_cnn = cnn_model.fit(
    X_train_cnn,
    y_train_scaled,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    verbose=0
)
print("CNN training complete.")

y_pred_scaled_cnn = cnn_model.predict(X_test_cnn)
y_pred_cnn = scaler_y.inverse_transform(y_pred_scaled_cnn)

# Calculate metrics for CNN
mae_cnn = mean_absolute_error(y_test, y_pred_cnn)
r2_cnn = r2_score(y_test, y_pred_cnn)
mse_cnn = mean_squared_error(y_test, y_pred_cnn)
rmse_cnn = np.sqrt(mse_cnn)


# EVALUATE AND VISUALIZE RESULTS

print("\n--- Model Performance Metrics ---")

# Calculating the metrics
results = {
    "Model": ["MLP", "1D CNN"],
    "MAE": [mae_mlp, mae_cnn],
    "R-squared": [r2_mlp, r2_cnn],
    "MSE": [mse_mlp, mse_cnn],
    "RMSE": [rmse_mlp, rmse_cnn]

}
results_df = pd.DataFrame(results).set_index("Model")
print(results_df)

print("\n--- Generating Visualizations ---")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Neural Network Baseline Performance', fontsize=20)

# --- Predicted vs. True Plots ---
axes[0, 0].scatter(y_test, y_pred_mlp, alpha=0.6, edgecolors='k', s=25)
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red', lw=2)
axes[0, 0].set_title('MLP: Predicted vs. True', fontsize=14)
axes[0, 0].set_xlabel('True Values', fontsize=12)
axes[0, 0].set_ylabel('Predicted Values', fontsize=12)

axes[0, 1].scatter(y_test, y_pred_cnn, alpha=0.6, edgecolors='k', s=25)
axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red', lw=2)
axes[0, 1].set_title('1D CNN: Predicted vs. True', fontsize=14)
axes[0, 1].set_xlabel('True Values', fontsize=12)
axes[0, 1].set_ylabel('Predicted Values', fontsize=12)

# --- Residual Plots ---
residuals_mlp = y_test.values.flatten() - y_pred_mlp.flatten()
axes[1, 0].scatter(y_pred_mlp, residuals_mlp, alpha=0.6, edgecolors='k', s=25)
axes[1, 0].axhline(y=0, color='red', linestyle='--')
axes[1, 0].set_title('MLP: Residuals', fontsize=14)
axes[1, 0].set_xlabel('Predicted Values', fontsize=12)
axes[1, 0].set_ylabel('Residuals', fontsize=12)

residuals_cnn = y_test.values.flatten() - y_pred_cnn.flatten()
axes[1, 1].scatter(y_pred_cnn, residuals_cnn, alpha=0.6, edgecolors='k', s=25)
axes[1, 1].axhline(y=0, color='red', linestyle='--')
axes[1, 1].set_title('1D CNN: Residuals', fontsize=14)
axes[1, 1].set_xlabel('Predicted Values', fontsize=12)
axes[1, 1].set_ylabel('Residuals', fontsize=12)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("nn_baselines_comparison.png")
plt.show()
