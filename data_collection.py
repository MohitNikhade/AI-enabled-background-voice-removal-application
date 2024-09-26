import os
import tarfile
import requests
import zipfile
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
    os.makedirs('logs', exist_ok=True)

# Function to check if the dataset has been extracted
def is_extracted(directory, filename):
    extracted_flag_file = os.path.join(directory, f".{filename}.extracted")
    return os.path.exists(extracted_flag_file)

# Function to mark a dataset as extracted
def mark_as_extracted(directory, filename):
    extracted_flag_file = os.path.join(directory, f".{filename}.extracted")
    with open(extracted_flag_file, 'w') as f:
        f.write('')

def download_librispeech():
    for url in LIBRISPEECH_URLS:
        filename = url.split('/')[-1]
        filepath = os.path.join(LIBRISPEECH_DIR, filename)
        extract_dir = os.path.join(LIBRISPEECH_DIR, filename.replace('.tar.gz', ''))

        # Check if the dataset is already extracted
        if os.path.exists(extract_dir):
            logging.info(f"{filename} already extracted. Skipping extraction.")
            continue

        if not os.path.exists(filepath):
            logging.info(f"Downloading {filename}...")
            response = requests.get(url, stream=True)
            with open(filepath, 'wb') as f:
                f.write(response.content)
            logging.info(f"{filename} downloaded successfully.")
        
        # Extract files if not extracted already
        logging.info(f"Extracting {filename}...")
        try:
            with tarfile.open(filepath, 'r:gz') as tar:
                tar.extractall(LIBRISPEECH_DIR)
            logging.info(f"{filename} extracted successfully.")
        except PermissionError as e:
            logging.error(f"Permission denied while extracting {filename}: {e}")
        except Exception as e:
            logging.error(f"Error extracting {filename}: {e}")

# Function to download and extract UrbanSound8K dataset
def download_urbansound8k():
    filename = URBANSOUND8K_URL.split('/')[-1]
    filepath = os.path.join(URBANSOUND_DIR, filename)
    
    if not os.path.exists(filepath):
        logging.info(f"Downloading UrbanSound8K dataset...")
        response = requests.get(URBANSOUND8K_URL, stream=True)
        with open(filepath, 'wb') as f:
            f.write(response.content)
        logging.info(f"UrbanSound8K dataset downloaded successfully.")
    
    # Check if the dataset is already extracted
    if not is_extracted(URBANSOUND_DIR, filename):
        # Extract files
        logging.info(f"Extracting UrbanSound8K dataset...")
        with tarfile.open(filepath, 'r:gz') as tar:
            tar.extractall(URBANSOUND_DIR)
        mark_as_extracted(URBANSOUND_DIR, filename)
        logging.info(f"UrbanSound8K dataset extracted successfully.")
    else:
        logging.info("UrbanSound8K dataset already extracted. Skipping extraction.")

# Convert audio files to 16kHz WAV
def convert_to_16k_wav(input_dir, output_dir):
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.flac') or file.endswith('.wav'):
                input_path = os.path.join(root, file)
                # Ensure .wav output even for .flac files
                output_filename = file.replace('.flac', '.wav')
                output_path = os.path.join(output_dir, os.path.relpath(root, input_dir), output_filename)
                
                # Create the output directory if it doesn't exist
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                # Load and resample audio to 16kHz
                try:
                    audio, sr = librosa.load(input_path, sr=16000)
                    sf.write(output_path, audio, 16000)
                    logging.info(f"Successfully converted {input_path} to {output_path}")
                except Exception as e:
                    logging.error(f"Error processing {input_path}: {e}")

# Function to process LibriSpeech and UrbanSound8K datasets
def process_datasets():
    # Convert LibriSpeech to 16kHz WAV
    logging.info("Processing LibriSpeech dataset...")
    for dataset in ['train-clean-100', 'dev-clean']:
        input_dir = os.path.join(LIBRISPEECH_DIR, dataset)
        output_dir = os.path.join(LIBRISPEECH_DIR, dataset)
        convert_to_16k_wav(input_dir, output_dir)

    # Convert UrbanSound8K to 16kHz WAV
    logging.info("Processing UrbanSound8K dataset...")
    input_dir = os.path.join(URBANSOUND_DIR, 'UrbanSound8K', 'audio')
    output_dir = os.path.join(URBANSOUND_DIR, 'audio')
    convert_to_16k_wav(input_dir, output_dir)

# Main function to coordinate the process
def main():
    create_directories()
    download_librispeech()
    download_urbansound8k()
    process_datasets()
    logging.info("Data collection and processing completed successfully.")

if __name__ == '__main__':
    main()
