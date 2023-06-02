import requests
import csv
import pandas as pd

# API endpoint URL
url = 'https://api.opentheta.io/v1/items'
parameters = {'contractAddress': '0xe45610E578d4eb626121f55A61aB346A619B7d99', 'limit': '1000'}

# Fields to extract and write to CSV
fields = ['tokenId', 'type', 'valueTfuel', 'txn_hash', 'timestamp', 'creator']

# Send GET request to the API
response = requests.get(url, params=parameters)
data = response.json()

# Extract the 'items' list from the JSON response
items = data.get('items', [])

# Extract the specified fields from each item in the 'items' list
extracted_data = []
for item in items:
    extracted_data.append({
        "ID": item.get("ID"),
        "tokenId": item.get("tokenId"),
        # "name": item.get("name"),
        "imageUrl": item.get("imageUrl"),
        "listedPrice": item.get("listedPrice"),
        "total": item.get("total"),
        "offers": item.get("offers"),
        "maxListedPrice": item.get("maxListedPrice"),
        "rank": item.get("rank"),
        "auctionEnd": item.get("auctionEnd")
    })

# Create a DataFrame from the extracted data
df = pd.DataFrame(extracted_data)

# Write the DataFrame to a CSV file
df.to_csv('output.csv', index=False)