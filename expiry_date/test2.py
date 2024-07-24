import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from keras import models

# Load the dataset
final_dataframe = pd.read_csv('expiry_date/Large_Grocery_Store_Database_with_Integer_Discounts.csv')

# Load the trained models
model_linear = models.load_model('expiry_date/linear_regression_model.h5')
model_log = models.load_model('expiry_date/logarithmic_regression_model.h5')

# Input data
input_data = {
    'Product': 'Orange',
    'Category': 'Fruits',
    'Quantity': 45,
    'Manufactury Date': '2024-07-15',
    'Expiry Date': '2024-08-04',
    'Quantity Sold Last 30 Days': 42
}

# Preprocessing
label_encoder_product = LabelEncoder()
label_encoder_category = LabelEncoder()

# Fit the label encoders on the dataset
product_labels = final_dataframe['Product'].unique()
category_labels = final_dataframe['Category'].unique()
label_encoder_product.fit(product_labels)
label_encoder_category.fit(category_labels)

input_data['Product'] = label_encoder_product.transform([input_data['Product']])[0]
input_data['Category'] = label_encoder_category.transform([input_data['Category']])[0]

# Convert date columns to numerical features (days since a fixed point in time, e.g., 2024-01-01)
reference_date = pd.Timestamp('2024-01-01')
input_data['Manufactury Date'] = (pd.to_datetime(input_data['Manufactury Date']) - reference_date).days
input_data['Expiry Date'] = (pd.to_datetime(input_data['Expiry Date']) - reference_date).days

# Form the feature array
input_features = np.array([[input_data['Product'], input_data['Category'], input_data['Quantity'],
                            input_data['Manufactury Date'], input_data['Expiry Date'], input_data['Quantity Sold Last 30 Days']]])

# Normalize the features using the same scaler used for training
feature_columns = ['Product', 'Category', 'Quantity', 'Manufactury Date', 'Expiry Date', 'Quantity Sold Last 30 Days']
scaler = StandardScaler()
scaler.fit(final_dataframe[feature_columns])  # Fit scaler on the entire dataset

input_features_scaled = scaler.transform(input_features)

# Predict with the linear regression model
linear_prediction = model_linear.predict(input_features_scaled)
print(f"Linear Regression Model Prediction: {linear_prediction[0][0]}")

# Predict with the logarithmic regression model
log_prediction = model_log.predict(input_features_scaled)
log_prediction_original_scale = np.expm1(log_prediction)  # Transform prediction back to original scale
print(f"Logarithmic Regression Model Prediction: {log_prediction_original_scale[0][0]}")
