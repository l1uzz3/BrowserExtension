import requests

endpoint = "http://127.0.0.1:5030/check_url"
payload = { "url": "https://www.example.com" }

r = requests.post(endpoint, json=payload)
print("Status code:", r.status_code)
print("Response text:", repr(r.text))
