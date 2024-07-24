import sys
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from datetime import datetime
import numpy as np
import json

def main(expire_time, quantity, product_type):
    # Load the CSV data
    final_dataframe = pd.read_csv('Large_Grocery_Store_Database_with_Integer_Discounts.csv')

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

    forecasted_discount = forecast_discount(expire_time, quantity, product_type)[0]
    print(f"Forecasted Discount: {forecasted_discount}")

    # Save the forecasted discount to a JSON file
    output_json = {
        'expire_time': expire_time,
        'quantity': quantity,
        'product_type': product_type,
        'forecasted_discount': forecasted_discount
    }
    with open('forecasted_discount.json', 'w') as json_file:
        json.dump(output_json, json_file)
    print("Forecasted discount saved to forecasted_discount.json")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python forecast.py <expire_time> <quantity> <product_type>")
        sys.exit(1)

    expire_time = int(sys.argv[1])
    quantity = int(sys.argv[2])
    product_type = int(sys.argv[3])

    main(expire_time, quantity, product_type)
