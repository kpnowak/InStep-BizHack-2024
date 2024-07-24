import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from keras import models

# Load the trained models
model_linear = models.load_model('expiry_date\\linear_regression_model.keras')
model_log = models.load_model('expiry_date\\logarithmic_regression_model.keras')

# Function to convert dates to numerical features
def convert_date_to_days(date_str):
    reference_date = pd.Timestamp('2024-01-01')
    return (pd.to_datetime(date_str) - reference_date).days

# Function to make predictions
def predict_discount(quantity, manufactury_date, expiry_date, quantity_sold_last_30_days):
    # Convert dates
    manufactury_date_days = convert_date_to_days(manufactury_date)
    expiry_date_days = convert_date_to_days(expiry_date)
    
    # Create feature array
    features = np.array([[quantity, manufactury_date_days, expiry_date_days, quantity_sold_last_30_days]])
    
    # Normalize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Predict with linear model
    linear_prediction = model_linear.predict(features_scaled)[0][0]
    
    # Predict with logarithmic model
    log_prediction = model_log.predict(features_scaled)[0][0]
    log_prediction_transformed = np.expm1(log_prediction)  # Transform back to original scale
    
    return linear_prediction, log_prediction_transformed

# Input feature values
quantity = 50000000000000000
manufactury_date = '2024-07-21'
expiry_date = '2024-09-19'
quantity_sold_last_30_days = 3

# Make predictions
linear_pred, log_pred = predict_discount(quantity, manufactury_date, expiry_date, quantity_sold_last_30_days)

print(f"Linear Model Prediction: {linear_pred}")
print(f"Logarithmic Model Prediction: {log_pred}")