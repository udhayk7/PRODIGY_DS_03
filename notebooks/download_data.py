import requests
import os
import io
import zipfile

def download_dataset():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip"
    response = requests.get(url, verify=False)
    
    if response.status_code == 200:
        # Create a BytesIO object from the response content
        zip_file = io.BytesIO(response.content)
        
        # Extract the CSV file from the zip
        with zipfile.ZipFile(zip_file) as z:
            csv_name = [name for name in z.namelist() if name.endswith('full.csv')][0]
            os.makedirs('../data', exist_ok=True)
            with open('../data/bank-marketing.csv', 'wb') as f:
                f.write(z.read(csv_name))
        print("Dataset downloaded successfully!")
    else:
        print(f"Failed to download dataset. Status code: {response.status_code}")

if __name__ == "__main__":
    download_dataset()
