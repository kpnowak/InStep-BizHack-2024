import json
import requests

# Load the JSON file
with open('data.json') as f:
    data = json.load(f)

# Check the forecasted discount
if data['forecasted_discount'] > 20:
    # Send to Place 1
    response_place1 = requests.post('http://localhost:3000/place1', json=data)
    # Check if the response is valid and contains JSON
    if response_place1.status_code == 200 and response_place1.headers.get('Content-Type') == 'application/json':
        response_json = response_place1.json()
        
        # Send the response JSON to Place 2
        response_place2 = requests.post('http://localhost:3000/place2', json=response_json)
        print('Response from Place 2:', response_place2.text)
    else:
        print('Invalid response from Place 1')
else:
    # Send to Place 2 directly
    response_place2 = requests.post('http://localhost:3000/place2', json=data)
    print('Response from Place 2:', response_place2.text)
