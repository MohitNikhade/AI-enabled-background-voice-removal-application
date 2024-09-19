import os
import requests
import tarfile
import logging
from tqdm import tqdm
from hashlib import md5
import time
import random

logging.basicConfig(filename="data_collection.log", level=logging.INFO, 
                    format="%(asctime)s:%(levelname)s:%(message)s")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
LIBRISPEECH_DIR = os.path.join(DATASETS_DIR, "LibriSpeech")
URBANSOUND8K_DIR = os.path.join(DATASETS_DIR, "UrbanSound8K")

LIBRISPEECH_URLS = [
    {
        "url": "http://www.openslr.org/resources/12/train-clean-100.tar.gz",
        "md5": "2f494334227864a8a8fec932999db9d8"
    },
    {
        "url": "http://www.openslr.org/resources/12/dev-clean.tar.gz",
        "md5": "42e2234ba48799c1f50f24a7926300a1"
    }
]

URBANSOUND8K_URL = {
    "url": "https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz?download=1",
    "md5": "8fd0635bf5bba613bbe69ec7e1077501"
}

os.makedirs(LIBRISPEECH_DIR, exist_ok=True)
os.makedirs(URBANSOUND8K_DIR, exist_ok=True)

def check_md5(file_path, expected_md5):
    hash_md5 = md5()
    try:
        with open(file_path, "rb") as file:
            for chunk in iter(lambda: file.read(4096), b""):
                hash_md5.update(chunk)
        file_md5 = hash_md5.hexdigest()
        return file_md5 == expected_md5
    except Exception as e:
        logging.error(f"Error computing MD5 for {file_path}: {str(e)}")
        return False

def download_file(url, destination, retries=3, sleep_time=5):
    file_name = url.split("/")[-1]
    logging.info(f"Starting download for: {file_name}")
    print(f"Downloading: {file_name}...")

    if os.path.exists(destination):
        logging.info(f"{destination} already exists, verifying file integrity...")
        print(f"{destination} already exists, skipping download.")
        return
    
    for attempt in range(retries):
        try:
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024  # 1 Kilobyte
            tqdm_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
            
            with open(destination, 'wb') as file:
                for data in response.iter_content(block_size):
                    tqdm_bar.update(len(data))
                    file.write(data)

            logging.info(f"Downloaded {destination}")
            print(f"Downloaded: {file_name}")
            break

        except Exception as e:
            logging.error(f"Error downloading {destination}, attempt {attempt + 1}/{retries}: {str(e)}")
            if attempt < retries - 1:
                sleep_duration = sleep_time + random.randint(1, 3)
                logging.info(f"Retrying in {sleep_duration} seconds...")
                print(f"Error downloading {file_name}, retrying in {sleep_duration} seconds...")
                time.sleep(sleep_duration)
            else:
                logging.critical(f"Failed to download {destination} after {retries} attempts.")
                print(f"Failed to download {file_name} after {retries} attempts.")

        finally:
            tqdm_bar.close()

def extract_tar_file(tar_path, extract_to):
    file_name = tar_path.split("/")[-1]
    logging.info(f"Extracting: {file_name}")
    print(f"Extracting: {file_name}...")

    if os.path.isdir(extract_to):
        logging.info(f"{extract_to} already exists, skipping extraction.")
        print(f"Skipping extraction, {extract_to} already exists.")
        return
    try:
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(path=extract_to)
        logging.info(f"Extracted {tar_path} to {extract_to}")
        print(f"Extracted: {file_name}")
    except Exception as e:
        logging.error(f"Error extracting {tar_path}: {str(e)}")
        print(f"Error extracting {file_name}: {str(e)}")

def download_and_verify(url, destination, expected_md5):
    file_name = url.split("/")[-1]
    file_path = os.path.join(destination, file_name)
    
    if os.path.exists(file_path):
        logging.info(f"{file_path} exists, verifying MD5 checksum...")
        print(f"{file_name} exists, verifying checksum...")

        if check_md5(file_path, expected_md5):
            logging.info(f"{file_path} passed integrity check, skipping download.")
            print(f"{file_name} passed integrity check, skipping download.")
            return file_path
        else:
            logging.warning(f"{file_path} failed MD5 check, re-downloading.")
            print(f"{file_name} failed checksum, re-downloading.")
            os.remove(file_path)
    
    download_file(url, file_path)
    
    if check_md5(file_path, expected_md5):
        logging.info(f"{file_path} passed MD5 verification.")
        print(f"{file_name} passed checksum verification.")
    else:
        logging.critical(f"{file_path} failed MD5 verification, corrupt file.")
        print(f"{file_name} failed checksum verification, file is corrupt.")
        raise Exception(f"MD5 checksum mismatch for {file_path}.")
    
    return file_path

for item in LIBRISPEECH_URLS:
    url = item["url"]
    expected_md5 = item["md5"]
    file_path = download_and_verify(url, LIBRISPEECH_DIR, expected_md5)
    extract_tar_file(file_path, LIBRISPEECH_DIR)

urban_sound_file_path = download_and_verify(URBANSOUND8K_URL["url"], URBANSOUND8K_DIR, URBANSOUND8K_URL["md5"])
extract_tar_file(urban_sound_file_path, URBANSOUND8K_DIR)

logging.info("Data collection and extraction completed successfully.")
print("Data collection and extraction completed successfully.")
