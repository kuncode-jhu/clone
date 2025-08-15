"""
Enhanced download script for brain-to-text project data.

Downloads both:
1. Main dataset files from Dryad (neural data, pretrained models, etc.)
2. Language model files from Google Drive (3gram and 5gram models)

First create the b2txt25 conda environment. Then in a Terminal, at this repository's
top-level directory (nejm-brain-to-text/), run:

conda activate b2txt25
pip install gdown  # If not already installed
python download_data_enhanced.py
"""

import sys
import os
import urllib.request
import json
import zipfile
import tarfile
import subprocess
import importlib.util


########################################################################################
#
# Configuration - UPDATE THESE WITH YOUR GOOGLE DRIVE FILE IDs
#
########################################################################################

# Google Drive file IDs - Updated with actual file IDs
GOOGLE_DRIVE_FILES = {
    "languageModel.tar.gz": {
        "file_id": "1BWq1u3MXp3gA4AnfKuz4a1NkfVgwNG3f",  # 3-gram language model
        "description": "3-gram language model"
    },
    "languageModel_5gram.tar.gz": {
        "file_id": "1KRJF8GaHfPzsPLPF7fn9BSd1lMfHWQEW",  # 5-gram language model
        "description": "5-gram language model"
    }
}


########################################################################################
#
# Helpers
#
########################################################################################

def display_progress_bar(block_num, block_size, total_size, message=""):
    """Display download progress bar"""
    bytes_downloaded_so_far = block_num * block_size
    MB_downloaded_so_far = bytes_downloaded_so_far / 1e6
    MB_total = total_size / 1e6
    sys.stdout.write(
        f"\r{message}\t\t{MB_downloaded_so_far:.1f} MB / {MB_total:.1f} MB"
    )
    sys.stdout.flush()


def check_gdown_installed():
    """Check if gdown is installed, install if not"""
    if importlib.util.find_spec("gdown") is None:
        print("gdown not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        print("gdown installed successfully!")


def download_from_google_drive(file_id, output_path, description):
    """Download a file from Google Drive using gdown"""
    try:
        import gdown
        print(f"\nDownloading {description} from Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)
        print(f"Successfully downloaded {description}")
        return True
    except Exception as e:
        print(f"Error downloading {description}: {e}")
        return False


def extract_if_needed(filepath, data_dir):
    """Extract tar.gz files if they exist"""
    if filepath.endswith('.tar.gz') and os.path.exists(filepath):
        print(f"Extracting {os.path.basename(filepath)}...")
        try:
            with tarfile.open(filepath, 'r:gz') as tar:
                tar.extractall(data_dir)
            print(f"Successfully extracted {os.path.basename(filepath)}")
        except Exception as e:
            print(f"Error extracting {os.path.basename(filepath)}: {e}")


########################################################################################
#
# Main function
#
########################################################################################

def main():
    """Download all required data files"""
    
    # Validate environment
    DATA_DIR = "data/"
    data_dirpath = os.path.abspath(DATA_DIR)
    assert os.getcwd().endswith(
        "nejm-brain-to-text"
    ), f"Please run the download command from the nejm-brain-to-text directory (instead of {os.getcwd()})"
    assert os.path.exists(
        data_dirpath
    ), "Cannot find the data directory to download into."

    print("=== Brain-to-Text Enhanced Data Downloader ===")
    print(f"Data directory: {data_dirpath}")
    
    # Check for gdown
    check_gdown_installed()
    
    # ================================================================================
    # Part 1: Download main dataset from Dryad
    # ================================================================================
    
    print("\n🔄 PHASE 1: Downloading main dataset from Dryad...")
    
    DRYAD_DOI = "10.5061/dryad.dncjsxm85"
    DRYAD_ROOT = "https://datadryad.org"
    urlified_doi = DRYAD_DOI.replace("/", "%2F")

    try:
        # Get file list from Dryad
        versions_url = f"{DRYAD_ROOT}/api/v2/datasets/doi:{urlified_doi}/versions"
        with urllib.request.urlopen(versions_url) as response:
            versions_info = json.loads(response.read().decode())

        files_url_path = versions_info["_embedded"]["stash:versions"][-1]["_links"]["stash:files"]["href"]
        files_url = f"{DRYAD_ROOT}{files_url_path}"
        with urllib.request.urlopen(files_url) as response:
            files_info = json.loads(response.read().decode())

        file_infos = files_info["_embedded"]["stash:files"]

        # Download each file from Dryad
        for file_info in file_infos:
            filename = file_info["path"]

            if filename == "README.md":
                continue

            download_to_filepath = os.path.join(data_dirpath, filename)
            
            # Skip if file already exists
            if os.path.exists(download_to_filepath):
                print(f"⏭️  {filename} already exists, skipping...")
                continue

            download_path = file_info["_links"]["stash:download"]["href"]
            download_url = f"{DRYAD_ROOT}{download_path}"

            print(f"\n📥 Downloading {filename}...")
            urllib.request.urlretrieve(
                download_url,
                download_to_filepath,
                reporthook=lambda *args: display_progress_bar(
                    *args, message=f"Downloading {filename}"
                ),
            )
            sys.stdout.write("\n")

            # Extract zip files
            if file_info["mimeType"] == "application/zip":
                print(f"📦 Extracting files from {filename}...")
                with zipfile.ZipFile(download_to_filepath, "r") as zf:
                    zf.extractall(data_dirpath)

    except Exception as e:
        print(f"❌ Error downloading from Dryad: {e}")
        print("You can continue with Google Drive downloads...")

    # ================================================================================
    # Part 2: Download language models from Google Drive  
    # ================================================================================
    
    print("\n🔄 PHASE 2: Downloading language models from Google Drive...")
    
    for filename, info in GOOGLE_DRIVE_FILES.items():
        file_path = os.path.join(data_dirpath, filename)
        
        # Skip if file already exists
        if os.path.exists(file_path):
            print(f"⏭️  {filename} already exists, skipping...")
            continue
            
        file_id = info["file_id"]
        description = info["description"]
        
        # Check if file ID is set
        if file_id.startswith("YOUR_") or len(file_id) < 10:
            print(f"⚠️  {filename}: Google Drive file ID not set. Please update the script with your file ID.")
            print(f"   Expected format: https://drive.google.com/file/d/FILE_ID/view")
            continue
            
        # Download from Google Drive
        success = download_from_google_drive(file_id, file_path, description)
        
        if success:
            # Extract if it's a tar.gz file
            extract_if_needed(file_path, data_dirpath)

    # ================================================================================
    # Summary
    # ================================================================================
    
    print("\n✅ Download process completed!")
    print(f"\n📁 Files in {data_dirpath}:")
    
    for item in sorted(os.listdir(data_dirpath)):
        item_path = os.path.join(data_dirpath, item)
        if os.path.isfile(item_path):
            size_mb = os.path.getsize(item_path) / 1e6
            print(f"   📄 {item} ({size_mb:.1f} MB)")
        elif os.path.isdir(item_path):
            print(f"   📁 {item}/")
    
    print("\n🎉 All data ready for brain-to-text experiments!")


if __name__ == "__main__":
    main()
