import requests
import zipfile

print("Downloading data.zip...")
url = "https://web.cs.dal.ca/~juanr/downloads/data.zip"
data = requests.get(url)

with open("data.zip", "wb") as file:
    file.write(data.content)
print("Download complete")

print("Extracting data.zip...")
zip = zipfile.ZipFile("data.zip", "r")
zip.extractall()
zip.close()
print("Extracting complete")
