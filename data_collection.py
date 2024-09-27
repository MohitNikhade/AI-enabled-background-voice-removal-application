import os
import tarfile
import requests
import librosa
import soundfile as sf
import logging

# Setup logging
logging.basicConfig(filename='RealTime_Denoise_App/logs/data_collection.log', level=logging.INFO, 
                    format='%(asctime)s %(levelname)s:%(message)s')

# Directory setup
BASE_DIR = 'RealTime_Denoise_App/datasets'
LIBRISPEECH_DIR = os.path.join(BASE_DIR, 'LibriSpeech')
URBANSOUND_DIR = os.path.join(BASE_DIR, 'UrbanSound8K')

# URLs for datasets
LIBRISPEECH_URLS = [
    "http://www.openslr.org/resources/12/train-clean-100.tar.gz",
    "http://www.openslr.org/resources/12/dev-clean.tar.gz"
]

URBANSOUND8K_URL = "https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz"

# Create directories if they don't exist
def create_directories():
    os.makedirs(LIBRISPEECH_DIR, exist_ok=True)
    os.makedirs(URBANSOUND_DIR, exist_ok=True)
    os.makedirs('RealTime_Denoise_App/logs', exist_ok=True)

# Function to check if the dataset has been extracted
def is_extracted(directory, filename):
    extracted_flag_file = os.path.join(directory, f".{filename}.extracted")
    return os.path.exists(extracted_flag_file)

# Function to mark a dataset as extracted
def mark_as_extracted(directory, filename):
    extracted_flag_file = os.path.join(directory, f".{filename}.extracted")
    with open(extracted_flag_file, 'w') as f:
        f.write('')

# Function to check if all .flac files have corresponding .wav files
def check_flac_wav_conversion(directory):
    all_converted = True
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.flac'):
                flac_path = os.path.join(root, file)
                wav_path = flac_path.replace('.flac', '.wav')
                if not os.path.exists(wav_path):
                    all_converted = False
                    print(f"Missing WAV file for {flac_path}")
                    logging.warning(f"Missing WAV file for {flac_path}")
    return all_converted

# Function to delete all .flac files if corresponding .wav files exist
def delete_flac_files(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.flac'):
                flac_path = os.path.join(root, file)
                wav_path = flac_path.replace('.flac', '.wav')
                if os.path.exists(wav_path):
                    try:
                        os.remove(flac_path)
                        print(f"Deleted {flac_path}")
                        logging.info(f"Deleted {flac_path}")
                    except Exception as e:
                        print(f"Error deleting {flac_path}: {e}")
                        logging.error(f"Error deleting {flac_path}: {e}")

# Download and extract LibriSpeech dataset, convert .flac to .wav
def download_librispeech():
    for url in LIBRISPEECH_URLS:
        filename = url.split('/')[-1]
        filepath = os.path.join(LIBRISPEECH_DIR, filename)
        extract_dir = os.path.join(LIBRISPEECH_DIR, filename.replace('.tar.gz', ''))

        # Check if the dataset is already extracted
        if os.path.exists(extract_dir) and is_extracted(LIBRISPEECH_DIR, filename):
            print(f"{filename} already extracted. Checking for .flac to .wav conversions.")
            logging.info(f"{filename} already extracted. Checking for .flac to .wav conversions.")
            
            # Check if all .flac files have been converted to .wav
            if check_flac_wav_conversion(extract_dir):
                delete_flac_files(extract_dir)
            continue

        # Download the dataset
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            logging.info(f"Downloading {filename}...")
            response = requests.get(url, stream=True)
            with open(filepath, 'wb') as f:
                f.write(response.content)
            print(f"{filename} downloaded successfully.")
            logging.info(f"{filename} downloaded successfully.")
        
        # Extract and convert .flac to .wav
        print(f"Extracting {filename} and converting .flac files to .wav...")
        logging.info(f"Extracting {filename} and converting .flac files to .wav...")
        with tarfile.open(filepath, 'r:gz') as tar:
            for member in tar.getmembers():
                if member.name.endswith('.flac'):
                    tar.extract(member, LIBRISPEECH_DIR)
                    flac_path = os.path.join(LIBRISPEECH_DIR, member.name)
                    wav_path = flac_path.replace('.flac', '.wav')
                    
                    try:
                        # Load the FLAC file and save as WAV
                        audio, sr = librosa.load(flac_path, sr=16000)
                        sf.write(wav_path, audio, 16000)
                        print(f"Converted {flac_path} to {wav_path}")
                        logging.info(f"Converted {flac_path} to {wav_path}")
                        
                        # Remove the original FLAC file
                        os.remove(flac_path)
                        print(f"Removed {flac_path}")
                        logging.info(f"Removed {flac_path}")
                    except Exception as e:
                        print(f"Error converting {flac_path} to WAV: {e}")
                        logging.error(f"Error converting {flac_path} to WAV: {e}")
        
        mark_as_extracted(LIBRISPEECH_DIR, filename)
        print(f"{filename} extracted and processed successfully.")
        logging.info(f"{filename} extracted and processed successfully.")

# Download and extract UrbanSound8K dataset
def download_urbansound8k():
    filename = URBANSOUND8K_URL.split('/')[-1]
    filepath = os.path.join(URBANSOUND_DIR, filename)
    
    if not os.path.exists(filepath):
        print(f"Downloading UrbanSound8K dataset...")
        logging.info(f"Downloading UrbanSound8K dataset...")
        response = requests.get(URBANSOUND8K_URL, stream=True)
        with open(filepath, 'wb') as f:
            f.write(response.content)
        print(f"UrbanSound8K dataset downloaded successfully.")
        logging.info(f"UrbanSound8K dataset downloaded successfully.")
    
    if not is_extracted(URBANSOUND_DIR, filename):
        print(f"Extracting UrbanSound8K dataset...")
        logging.info(f"Extracting UrbanSound8K dataset...")
        with tarfile.open(filepath, 'r:gz') as tar:
            tar.extractall(URBANSOUND_DIR)
        mark_as_extracted(URBANSOUND_DIR, filename)
        print(f"UrbanSound8K dataset extracted successfully.")
        logging.info(f"UrbanSound8K dataset extracted successfully.")
    else:
        print("UrbanSound8K dataset already extracted. Skipping extraction.")
        logging.info("UrbanSound8K dataset already extracted. Skipping extraction.")

# Main function to coordinate the process
def main():
    create_directories()
    download_librispeech()
    download_urbansound8k()
    print("Data collection and processing completed successfully.")
    logging.info("Data collection and processing completed successfully.")

if __name__ == '__main__':
    main()
