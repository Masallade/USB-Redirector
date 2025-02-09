"""
config_manager.py
This module handles loading and saving configuration settings from/to a JSON file.
"""

import json
import os
from typing import Dict, Any

class ConfigManager:
    def __init__(self, config_file: str = "config.json"):
        """
        Initialize the ConfigManager with a configuration file.
        
        :param config_file: The path to the configuration file.
        """
        self.config_file = config_file
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """
        Load the configuration from the file.
        
        :return: The configuration as a dictionary.
        """
        if os.path.exists(self.config_file):
            with open(self.config_file, "r") as file:
                return json.load(file)
        return {}

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key.
        
        :param key: The configuration key.
        :param default: The default value to return if the key is not found.
        :return: The configuration value or default.
        """
        return self.config.get(key, default)

    def set(self, key: str, value: Any):
        """
        Set a configuration value.
        
        :param key: The configuration key.
        :param value: The value to set.
        """
        self.config[key] = value
        self._save_config()

    def _save_config(self):
        """
        Save the configuration to the file.
        """
        with open(self.config_file, "w") as file:
            json.dump(self.config, file, indent=4)