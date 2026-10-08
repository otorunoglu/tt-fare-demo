#!/usr/bin/env python3
"""
Setup script for Horse Kick Detection System
Downloads and configures necessary files for panns_inference
"""

import os
import sys
import shutil
import urllib.request
import subprocess

def setup_panns_data():
    """Download and set up the necessary files for panns_inference"""
    print("Setting up panns_inference data...")
    
    # Create local panns_data directory if it doesn't exist
    local_panns_dir = os.path.join(os.path.abspath(os.curdir), "panns_data")
    os.makedirs(local_panns_dir, exist_ok=True)
    
    # Define the files we need
    class_labels_file = os.path.join(local_panns_dir, "class_labels_indices.csv")
    model_file = os.path.join(local_panns_dir, "Cnn14_16k_mAP=0.438.pth")
    
    # Download class_labels_indices.csv if it doesn't exist
    if not os.path.exists(class_labels_file):
        print("Downloading class_labels_indices.csv...")
        labels_url = "https://raw.githubusercontent.com/qiuqiangkong/audioset_tagging_cnn/master/metadata/class_labels_indices.csv"
        try:
            urllib.request.urlretrieve(labels_url, class_labels_file)
            print("Downloaded successfully!")
        except Exception as e:
            print(f"Error downloading class_labels_indices.csv: {e}")
            print("Please download it manually from:")
            print(labels_url)
            print(f"and place it in: {local_panns_dir}")
    else:
        print("class_labels_indices.csv already exists.")
    
    # Check for model file
    if not os.path.exists(model_file):
        print("\nIMPORTANT: You need the PANNs model file:")
        print(f"Cnn14_mAP=0.438.pth should be placed in: {local_panns_dir}")
        print("Download it from: https://zenodo.org/records/3987831/files/Cnn14_16k_mAP=0.438.pth")
    else:
        print("PANNs model file already exists.")
    
    # Create user's panns_data directory and copy/link files there
    user_panns_dir = os.path.join(os.path.expanduser('~'), "panns_data")
    os.makedirs(user_panns_dir, exist_ok=True)
    
    user_labels_file = os.path.join(user_panns_dir, "class_labels_indices.csv")
    
    if not os.path.exists(user_labels_file):
        print(f"\nCopying class_labels_indices.csv to {user_panns_dir}")
        shutil.copy(class_labels_file, user_labels_file)
    
    # Check for required directories
    models_dir = os.path.join(os.path.abspath(os.curdir), "models")
    if not os.path.exists(models_dir):
        os.makedirs(models_dir, exist_ok=True)
        print(f"\nCreated models directory at: {models_dir}")
        print("Please place your model files in this directory:")
        print("- cae_2s_latent_dim_32.keras")
        print("- cae_2s_latent_dim_32.keras.json")
        print("- kick_detector_panns.keras")
    
    print("\nSetup complete!")

def run_main_script():
    """Run the main script"""
    main_script = os.path.join(os.path.abspath(os.curdir), "horse_kick_monitor.py")
    print(f"\nRunning {main_script}...")
    subprocess.run([sys.executable, main_script])

if __name__ == "__main__":
    setup_panns_data()