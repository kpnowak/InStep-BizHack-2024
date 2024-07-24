import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import numpy as np

# Load the CSV data
final_dataframe = pd.read_csv('expiry_date\\Large_Grocery_Store_Database_with_Integer_Discounts.csv')

# Convert Manufactury Date and Expiry Date to datetime
final_dataframe['Manufactury Date'] = pd.to_datetime(final_dataframe['Manufactury Date'])
final_dataframe['Expiry Date'] = pd.to_datetime(final_dataframe['Expiry Date'])

# Calculate expire_time
final_dataframe['expire_time'] = (final_dataframe['Expiry Date'] - final_dataframe['Manufactury Date']).dt.days

# Add a new column for product type
product_type_mapping = {
    'Apple': 0,
    'Fish': 1,
    'Yogurt': 2,
    'Milk': 3,
    'Bread': 4,
    'Orange': 5,
    'Carrot': 6,
    'Beef': 7,
    'Butter': 8,
    'Potato': 9,
    'Soda': 10,
    'Spinach': 11,
    'Broccoli': 12,
    'Chicken': 13,
    'Cookies': 14,
    'Banana': 15,
    'Juice': 16,
    'Chips': 17,
    'Shrimp': 18,
    'Tomato': 19
}
final_dataframe['product_type'] = final_dataframe['Product'].map(product_type_mapping)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(final_dataframe[['expire_time', 'Quantity', 'product_type']], final_dataframe['Discount'], test_size=0.2, random_state=42)

# Train a random forest regressor model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Function to forecast discount for a new product
def forecast_discount(expire_time, quantity, product_type):
    # Convert expire_time, quantity, and product_type to a pandas DataFrame
    new_data = pd.DataFrame({'expire_time': [expire_time], 'Quantity': [quantity], 'product_type': [product_type]})
    # Use the trained model to forecast the discount
    forecasted_discount = model.predict(new_data)
    return forecasted_discount

# Encode product types
product_type_input_mapping = {
    'Apple': 0,
    'Fish': 1,
    'Yogurt': 2,
    'Milk': 3,
    'Bread': 4,
    'Orange': 5,
    'Carrot': 6,
    'Beef': 7,
    'Butter': 8,
    'Potato': 9,
    'Soda': 10,
    'Spinach': 11,
    'Broccoli': 12,
    'Chicken': 13,
    'Cookies': 14,
    'Banana': 15,
    'Juice': 16,
    'Chips': 17,
    'Shrimp': 18,
    'Tomato': 19
}

# Example usage
new_expire_time = 8  # 8 days
new_quantity = 50  # 50 units
new_product_type = product_type_input_mapping['Soda']  # get the encoded value for 'Tomato'
forecasted_discount_tomato = forecast_discount(new_expire_time, new_quantity, new_product_type)
print(f'Forecasted discount for tomatoes: {forecasted_discount_tomato[0]:.2f}%')