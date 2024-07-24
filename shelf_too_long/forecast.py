import sys
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from datetime import datetime
import numpy as np
import json

def main(expire_time, quantity, product_name):
    # Load the CSV data
    final_dataframe = pd.read_csv('Large_Grocery_Products_Dataset.csv')

    # Convert Manufacture Date to datetime
    final_dataframe['Manufacture date'] = pd.to_datetime(final_dataframe['Manufacture date'])

    # Calculate expire_time as the number of days since manufacture date
    final_dataframe['expire_time'] = (datetime.now() - final_dataframe['Manufacture date']).dt.days

    # Map product names to numerical values
    unique_products = final_dataframe['Product name'].unique()
    product_type_mapping = {name: idx for idx, name in enumerate(unique_products)}
    final_dataframe['product_type'] = final_dataframe['Product name'].map(product_type_mapping)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(final_dataframe[['expire_time', 'Quantity', 'product_type']], final_dataframe['Discount'], test_size=0.2, random_state=42)

    # Train a random forest regressor model
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    # Function to forecast discount for a new product
    def forecast_discount(expire_time, quantity, product_name):
        # Convert expire_time, quantity, and product_name to a pandas DataFrame
        new_data = pd.DataFrame({'expire_time': [expire_time], 'Quantity': [quantity], 'product_type': [product_type_mapping[product_name]]})
        # Use the trained model to forecast the discount
        forecasted_discount = model.predict(new_data)
        return forecasted_discount

    forecasted_discount = forecast_discount(expire_time, quantity, product_name)[0]
    print(f"Forecasted Discount: {forecasted_discount}")

    # Save the forecasted discount to a JSON file
    output_json = {
        'expire_time': expire_time,
        'quantity': quantity,
        'product_type': product_name,
        'forecasted_discount': forecasted_discount
    }
    with open('forecasted_discount.json', 'w') as json_file:
        json.dump(output_json, json_file)
    print("Forecasted discount saved to forecasted_discount.json")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python forecast.py <expire_time> <quantity> <product_name>")
        sys.exit(1)

    expire_time = int(sys.argv[1])
    quantity = int(sys.argv[2])
    product_name = sys.argv[3]

    main(expire_time, quantity, product_name)