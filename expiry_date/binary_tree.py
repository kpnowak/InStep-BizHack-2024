from datetime import datetime
import numpy as np

def calculate_discount(expiry_date, quantity_sold, quantity):
    today = datetime.today()
    days_to_expiry = (expiry_date - today).days

    if days_to_expiry <= 7:
        if quantity_sold < quantity / 2:
            return int(np.random.uniform(30, 50))  # High discount
        else:
            return int(np.random.uniform(10, 20))  # Medium discount
    elif days_to_expiry <= 30:
        if quantity_sold < quantity / 2:
            return int(np.random.uniform(20, 35))  # Medium discount
        else:
            return int(np.random.uniform(5, 15))  # Low discount
    else:
        if quantity_sold < quantity / 2:
            return int(np.random.uniform(0, 10))  # Very low discount
        else:
            return 0  # No discount
print()