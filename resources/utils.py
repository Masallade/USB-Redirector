"""
utils.py
This module contains utility functions for logging, file handling, and other common tasks.
"""

import logging
import os
from datetime import datetime

def setup_logger(log_file: str = "app.log"):
    """
    Set up a logger to write logs to a file and console.
    
    :param log_file: The path to the log file.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def create_directory(path: str):
    """
    Create a directory if it does not exist.
    
    :param path: The path of the directory to create.
    """
    if not os.path.exists(path):
        os.makedirs(path)
        logging.info(f"Created directory: {path}")

def timestamp() -> str:
    """
    Generate a timestamp in the format YYYY-MM-DD_HH-MM-SS.
    
    :return: The timestamp as a string.
    """
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")