import requests

# Define the server URL
server_url = 'http://127.0.0.1:5000'  # Replace with the actual server URL

# Define the search query parameters
query = 'I am writing to report a lost item during my recent visit to the Airport. I have unfortunately misplaced a 1.5 meter long red wool scarf with tassels, and I would greatly appreciate your assistance in locating it.I am writing to report the loss of a passport from Canada during my recent visit to Schiphol Airport. Unfortunately, the passports were misplaced or left behind during my time at the airport. Considering the importance and sensitivity of these documents, I am reaching out to the appropriate authorities to help resolve this matter.'
topn = 8
similarity = 0.1


# Create the request URL
search_url = f"{server_url}/search"

# Create the request payload as a dictionary


payload = {
    'query': query,
    'topn': topn,
    'similarity': similarity
}

# Send the POST request with JSON data
response = requests.post(search_url, json=payload)

# Check the response status code
if response.status_code == 200:
    # Print the response content
    print(response.json())
else:
    print(f"Request failed with status code {response.status_code}")