import requests

resp = requests.post("http://localhost:5000", files={'file': open('test1.jpg', 'rb')})

print(resp.json())