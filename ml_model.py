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

# Define the model
model = models.Sequential([
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(16, activation='relu'),
    layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_absolute_error')

callback = callbacks.EarlyStopping(monitor='loss', patience=3)
callback = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(
    train_features, train_labels,
    epochs=200,
    validation_data=(val_features, val_labels),
    callbacks=[callback]
)

# Evaluate the model
test_loss = model.evaluate(test_features, test_labels)
print(f"Test Loss: {test_loss}")

# Make predictions
predictions = model.predict(test_features)
print(predictions[:5])


# Optional: Visualize the training process
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Error [Discount]')
plt.legend()
plt.grid(True)
plt.show()

# Since ROC and AUC are not applicable for regression, we visualize prediction vs actual
plt.scatter(test_labels, predictions, alpha=0.5)
plt.xlabel('Actual Discounts')
plt.ylabel('Predicted Discounts')
plt.plot([min(test_labels), max(test_labels)], [min(test_labels), max(test_labels)], color='red')
plt.grid(True)
plt.show()