import os
import requests
import zipfile
import sys

DATASET_URL = "https://archive.ics.uci.edu/static/public/465/wesad+wearable+stress+and+affect+detection.zip"
ZIP_FILENAME = "WESAD.zip"
EXTRACT_DIR = "data"

def download_file(url, filename):
    print(f"Downloading WESAD Dataset from {url}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 * 1024 # 1 MB
    
    with open(filename, 'wb') as f:
        downloaded = 0
        for data in response.iter_content(block_size):
            f.write(data)
            downloaded += len(data)
            if total_size > 0:
                percent = (downloaded / total_size) * 100
                sys.stdout.write(f"\rDownloaded: {downloaded / (1024*1024):.2f} MB / {total_size / (1024*1024):.2f} MB ({percent:.2f}%)")
                sys.stdout.flush()
    print("\nDownload complete.")

def extract_zip(filename, extract_to):
    print(f"Extracting {filename} to {extract_to} directory...")
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction complete.")

if __name__ == "__main__":
    if not os.path.exists(EXTRACT_DIR):
        os.makedirs(EXTRACT_DIR)
        
    if not os.path.exists(ZIP_FILENAME):
        download_file(DATASET_URL, ZIP_FILENAME)
    else:
        print(f"{ZIP_FILENAME} already exists. Skipping download.")
        
    # Check if inside dataset exists
    inner_zip_path = os.path.join(EXTRACT_DIR, "WESAD.zip")
    if os.path.exists(ZIP_FILENAME) and not os.path.exists(inner_zip_path) and not os.path.exists(os.path.join(EXTRACT_DIR, "WESAD")):
        extract_zip(ZIP_FILENAME, EXTRACT_DIR)
        
    if os.path.exists(inner_zip_path):
        print("Extracting inner WESAD.zip...")
        extract_zip(inner_zip_path, EXTRACT_DIR)
        
    print("Dataset setup is ready. You can find the data in the 'data/WESAD' folder.")
