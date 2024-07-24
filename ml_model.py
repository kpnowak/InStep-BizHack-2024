import os
import tempfile

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import time

import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from keras import layers, models, callbacks


mpl.rcParams['figure.figsize'] = (12, 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

#file = tf.keras.utils
final_dataframe = pd.read_csv('Large_Grocery_Store_Database_with_Integer_Discounts.csv')
print(final_dataframe.head())

# Preprocessing
label_encoder_product = LabelEncoder()
label_encoder_category = LabelEncoder()

final_dataframe['Product'] = label_encoder_product.fit_transform(final_dataframe['Product'])
final_dataframe['Category'] = label_encoder_category.fit_transform(final_dataframe['Category'])

# Convert date columns to numerical features (days since a fixed point in time, e.g., 2024-01-01)
reference_date = pd.Timestamp('2024-01-01')
final_dataframe['Manufactury Date'] = (pd.to_datetime(final_dataframe['Manufactury Date']) - reference_date).dt.days
final_dataframe['Expiry Date'] = (pd.to_datetime(final_dataframe['Expiry Date']) - reference_date).dt.days

# Log transform the target variable
final_dataframe['Log_Discount'] = np.log1p(final_dataframe['Discount'])

# Use a utility from sklearn to split and shuffle your dataset.
train_df, test_df = train_test_split(final_dataframe, test_size=0.2)
train_df, val_df = train_test_split(train_df, test_size=0.2)

"""
train_labels = np.array(train_df.pop('Discount'))
val_labels = np.array(val_df.pop('Discount'))
test_labels = np.array(test_df.pop('Discount'))

train_features = np.array(train_df)
val_features = np.array(val_df)
test_features = np.array(test_df)

print(train_features)

# Normalize the features
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
val_features = scaler.transform(val_features)
test_features = scaler.transform(test_features)
"""

# Linear Regression Model Data
train_labels_linear = np.array(train_df.pop('Discount'))
val_labels_linear = np.array(val_df.pop('Discount'))
test_labels_linear = np.array(test_df.pop('Discount'))

train_features_linear = np.array(train_df)
val_features_linear = np.array(val_df)
test_features_linear = np.array(test_df)

# Logarithmic Regression Model Data
train_labels_log = np.array(train_df.pop('Log_Discount'))
val_labels_log = np.array(val_df.pop('Log_Discount'))
test_labels_log = np.array(test_df.pop('Log_Discount'))

train_features_log = np.array(train_df)
val_features_log = np.array(val_df)
test_features_log = np.array(test_df)

# Normalize the features
scaler = StandardScaler()
train_features_linear = scaler.fit_transform(train_features_linear)
val_features_linear = scaler.transform(val_features_linear)
test_features_linear = scaler.transform(test_features_linear)

train_features_log = scaler.fit_transform(train_features_log)
val_features_log = scaler.transform(val_features_log)
test_features_log = scaler.transform(test_features_log)

# Define the linear regression model
model_linear = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(train_features_linear.shape[1],)),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(16, activation='relu'),
    layers.Dense(1)
])

model_linear.compile(optimizer='adam', loss='mean_absolute_error')

# Define the logarithmic regression model
model_log = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(train_features_log.shape[1],)),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(16, activation='relu'),
    layers.Dense(1)
])

model_log.compile(optimizer='adam', loss='mean_absolute_error')

#callback = callbacks.EarlyStopping(monitor='loss', patience=3)
callback = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the linear regression model
history_linear = model_linear.fit(
    train_features_linear, train_labels_linear,
    epochs=100,
    validation_data=(val_features_linear, val_labels_linear),
    callbacks=[callback]
)

# Train the logarithmic regression model
history_log = model_log.fit(
    train_features_log, train_labels_log,
    epochs=100,
    validation_data=(val_features_log, val_labels_log),
    callbacks=[callback]
)

# Save the models
model_linear.save('linear_regression_model.h5')
model_log.save('logarithmic_regression_model.h5')

# Evaluate the linear regression model
test_loss_linear = model_linear.evaluate(test_features_linear, test_labels_linear)
print(f"Linear Regression Test Loss: {test_loss_linear}")

# Evaluate the logarithmic regression model
test_loss_log = model_log.evaluate(test_features_log, test_labels_log)
print(f"Logarithmic Regression Test Loss: {test_loss_log}")

# Make predictions with the linear regression model
predictions_linear = model_linear.predict(test_features_linear)
print(predictions_linear[:5])

# Make predictions with the logarithmic regression model
log_predictions = model_log.predict(test_features_log)
predictions_log = np.expm1(log_predictions)  # Transform predictions back to original scale
print(predictions_log[:5])

# Visualize the training process for linear regression model
plt.figure(figsize=(10, 6))
plt.plot(history_linear.history['loss'], label='loss')
plt.plot(history_linear.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Error [Discount]')
plt.legend()
plt.grid(True)
plt.title('Linear Regression Model Training')
plt.savefig('linear_regression_training.png')
#plt.show()

# Visualize the training process for logarithmic regression model
plt.figure(figsize=(10, 6))
plt.plot(history_log.history['loss'], label='loss')
plt.plot(history_log.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Error [Log Discount]')
plt.legend()
plt.grid(True)
plt.title('Logarithmic Regression Model Training')
plt.savefig('logarithmic_regression_training.png')
#plt.show()

# Visualize prediction vs actual for linear regression model
plt.figure(figsize=(10, 6))
plt.scatter(test_labels_linear, predictions_linear, alpha=0.5)
plt.xlabel('Actual Discounts')
plt.ylabel('Predicted Discounts')
plt.plot([min(test_labels_linear), max(test_labels_linear)], [min(test_labels_linear), max(test_labels_linear)], color='red')
plt.legend()
plt.grid(True)
plt.title('Linear Regression Model Predictions')
plt.savefig('linear_regression_predictions.png')
#plt.show()

# Visualize prediction vs actual for logarithmic regression model
actual_discounts_log = np.expm1(test_labels_log)  # Transform actual labels back to original scale
plt.figure(figsize=(10, 6))
plt.scatter(actual_discounts_log, predictions_log, alpha=0.5)
plt.xlabel('Actual Discounts')
plt.ylabel('Predicted Discounts')
plt.plot([min(actual_discounts_log), max(actual_discounts_log)], [min(actual_discounts_log), max(actual_discounts_log)], color='red')
plt.legend()
plt.grid(True)
plt.title('Logarithmic Regression Model Predictions')
plt.savefig('logarithmic_regression_predictions.png')
#plt.show()