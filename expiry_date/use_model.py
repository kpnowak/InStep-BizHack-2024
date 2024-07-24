import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
import keras

# Load the saved models
model_linear = keras.models.load_model('expiry_date\\linear_regression_model.h5')
model_log = keras.models.load_model('expiry_date\\logarithmic_regression_model.h5')

# Compile the models after loading
model_linear.compile(optimizer='adam', loss='mean_absolute_error')
model_log.compile(optimizer='adam', loss='mean_absolute_error')

# Preprocessing function
def preprocess_data(df, label_encoder_product, label_encoder_category, scaler):
    df['Product'] = label_encoder_product.transform(df['Product'])
    df['Category'] = label_encoder_category.transform(df['Category'])
    
    reference_date = pd.Timestamp('2024-01-01')
    df['Manufactury Date'] = (pd.to_datetime(df['Manufactury Date']) - reference_date).dt.days
    df['Expiry Date'] = (pd.to_datetime(df['Expiry Date']) - reference_date).dt.days
    
    features = df[['Product', 'Category', 'Quantity', 'Manufactury Date', 'Expiry Date', 'Quantity Sold Last 30 Days']]
    features = scaler.transform(features)
    
    return features

# Load and prepare the label encoders and scaler
def prepare_encoders_and_scaler(dataset_path):
    df = pd.read_csv(dataset_path)
    label_encoder_product = LabelEncoder()
    label_encoder_category = LabelEncoder()

    df['Product'] = label_encoder_product.fit_transform(df['Product'])
    df['Category'] = label_encoder_category.fit_transform(df['Category'])

    reference_date = pd.Timestamp('2024-01-01')
    df['Manufactury Date'] = (pd.to_datetime(df['Manufactury Date']) - reference_date).dt.days
    df['Expiry Date'] = (pd.to_datetime(df['Expiry Date']) - reference_date).dt.days

    features = df[['Product', 'Category', 'Quantity', 'Manufactury Date', 'Expiry Date', 'Quantity Sold Last 30 Days']]
    scaler = StandardScaler()
    scaler.fit(features)

    return label_encoder_product, label_encoder_category, scaler

# Predict function
def make_predictions(data, model, label_encoder_product, label_encoder_category, scaler, log_transform=False):
    features = preprocess_data(data, label_encoder_product, label_encoder_category, scaler)
    predictions = model.predict(features)
    
    if log_transform:
        predictions = np.expm1(predictions)  # Inverse of log1p
    
    return predictions

# Option 1: Provide data manually
def predict_by_hand(label_encoder_product, label_encoder_category, scaler):
    #product = input("Enter product: ")
    #category = input("Enter category: ")
    #quantity = int(input("Enter quantity: "))
    #manufactury_date = input("Enter manufactury date (YYYY-MM-DD): ")
    #expiry_date = input("Enter expiry date (YYYY-MM-DD): ")
    #quantity_sold_last_30_days = int(input("Enter quantity sold in last 30 days: "))

    product = "Cookies"
    category = "Snacks"
    quantity = int(15)
    manufactury_date = "2024-07-21"
    expiry_date = "2024-07-30"
    quantity_sold_last_30_days = int(35)

    data = {
        'Product': [product],
        'Category': [category],
        'Quantity': [quantity],
        'Manufactury Date': [manufactury_date],
        'Expiry Date': [expiry_date],
        'Quantity Sold Last 30 Days': [quantity_sold_last_30_days]
    }

    df = pd.DataFrame(data)

    predictions_linear = make_predictions(df, model_linear, label_encoder_product, label_encoder_category, scaler)
    predictions_log = make_predictions(df, model_log, label_encoder_product, label_encoder_category, scaler, log_transform=True)
    
    print("\nManual Data Predictions:")
    print(f"Linear Regression Model Predictions: {predictions_linear[0][0]:.2f}")
    print(f"Logarithmic Regression Model Predictions: {predictions_log[0][0]:.2f}")

# Option 2: Load data from a dataset
def predict_from_dataset(file_path, label_encoder_product, label_encoder_category, scaler):
    df = pd.read_csv(file_path)
    
    # Ensure the new data is also preprocessed
    df['Product'] = label_encoder_product.transform(df['Product'])
    df['Category'] = label_encoder_category.transform(df['Category'])
    reference_date = pd.Timestamp('2024-01-01')
    df['Manufactury Date'] = (pd.to_datetime(df['Manufactury Date']) - reference_date).dt.days
    df['Expiry Date'] = (pd.to_datetime(df['Expiry Date']) - reference_date).dt.days
    
    predictions_linear = make_predictions(df, model_linear, label_encoder_product, label_encoder_category, scaler)
    predictions_log = make_predictions(df, model_log, label_encoder_product, label_encoder_category, scaler, log_transform=True)
    
    print("\nDataset Predictions:")
    print("Linear Regression Model Predictions:", predictions_linear)
    print("Logarithmic Regression Model Predictions:", predictions_log)

# Load and prepare the label encoders and scaler using the original dataset
dataset_path = 'expiry_date\\Large_Grocery_Store_Database_with_Integer_Discounts.csv'
label_encoder_product, label_encoder_category, scaler = prepare_encoders_and_scaler(dataset_path)
    
# Option 1: Provide data manually
print("Manual Data Input")
predict_by_hand(label_encoder_product, label_encoder_category, scaler)
    
# Option 2: Load data from a dataset
#print("\nDataset Input")
#dataset_file_path = 'new_dataset.csv'  # Replace with your dataset file path
#predict_from_dataset(dataset_file_path, label_encoder_product, label_encoder_category, scaler)
