import requests
import zipfile

print("Downloading data.zip...")
url = "https://www.dropbox.com/s/pf5r2bkobw65fwh/data.zip?dl=1"
data = requests.get(url)

with open("data.zip", "wb") as file:
    file.write(data.content)
print("Download complete")

print("Extracting data.zip...")
zip = zipfile.ZipFile("data.zip", "r")
zip.extractall()
zip.close()
print("Extracting complete")
