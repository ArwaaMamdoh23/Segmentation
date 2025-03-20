import requests

url = "http://127.0.0.1:5000/predict"
files = {"file": open(r"D:\GitHub\Segmentation\Segmentation\data\images\0.tif", "rb")}
response = requests.post(url, files=files)

print(response.json())
print(response.status_code)  
print(response.text)  

