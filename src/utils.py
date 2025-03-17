# src/utils.py

import os

def ensure_directory_exists(directory: str):
    if not os.path.exists(directory):
        os.makedirs(directory)
