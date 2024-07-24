import pandas as pd

# Load the CSV data
csv_data = """Product,Quantity,Manufactury Date,Expiry Date
Apple,93,2024-07-10,2024-07-25
Fish,90,2024-07-20,2024-07-24
Yogurt,13,2024-07-10,2024-07-22
Milk,50,2024-07-14,2024-07-24
Bread,57,2024-07-15,2024-07-18"""

# Read the CSV data into a DataFrame
df = pd.read_csv("shelf_too_long\\input.csv")

# Convert the DataFrame to JSON format
json_data = df.to_json(orient='records', date_format='iso')

# Save the JSON data to a file
with open('shelf_too_long\\data.json', 'w') as json_file:
    json_file.write(json_data)

print("CSV data has been converted and saved to data.json")
