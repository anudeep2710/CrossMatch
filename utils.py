"""
Utilities Module

This module provides common utility functions for the user matching system,
including configuration loading, logging setup, and serialization.
"""

import yaml
import logging
import logging.handlers
import os
import json
import pickle
import joblib
from typing import Dict, Any, List, Union, Optional


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dict: Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Expand environment variables in config strings
        config = expand_env_vars(config)
        
        return config
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        # Return default configuration
        return {}


def expand_env_vars(config: Any) -> Any:
    """
    Recursively expand environment variables in configuration.
    
    Args:
        config: Configuration value or dictionary
        
    Returns:
        Any: Configuration with environment variables expanded
    """
    if isinstance(config, dict):
        return {k: expand_env_vars(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [expand_env_vars(v) for v in config]
    elif isinstance(config, str):
        # Check if string contains environment variable reference
        if "${" in config and "}" in config:
            import re
            import os
            
            # Replace all ${VAR} with os.environ.get("VAR", "")
            def replace_env_var(match):
                var_name = match.group(1)
                return os.environ.get(var_name, "")
            
            return re.sub(r'\${([^}]+)}', replace_env_var, config)
        return config
    else:
        return config


def setup_logging(level: int = logging.INFO, log_format: str = None, log_file: str = None) -> None:
    """
    Set up logging configuration.
    
    Args:
        level: Logging level
        log_format: Format string for log messages
        log_file: Path to log file (if None, log to console only)
    """
    # Default format if not specified
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Root logger configuration
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        try:
            # Create directory for log file if it doesn't exist
            log_dir = os.path.dirname(log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            
            # Use rotating file handler to prevent huge log files
            file_handler = logging.handlers.RotatingFileHandler(
                log_file, maxBytes=10*1024*1024, backupCount=5
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            logging.error(f"Error setting up log file: {e}")


def save_to_json(data: Any, path: str) -> bool:
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        path: Path to JSON file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        
        return True
    except Exception as e:
        logging.error(f"Error saving to JSON: {e}")
        return False


def load_from_json(path: str) -> Optional[Any]:
    """
    Load data from JSON file.
    
    Args:
        path: Path to JSON file
        
    Returns:
        Any: Loaded data or None if error
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data
    except Exception as e:
        logging.error(f"Error loading from JSON: {e}")
        return None


def save_model(model: Any, path: str, use_joblib: bool = True) -> bool:
    """
    Save model to file using pickle or joblib.
    
    Args:
        model: Model to save
        path: Path to save model
        use_joblib: Use joblib instead of pickle for larger models
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        if use_joblib:
            joblib.dump(model, path)
        else:
            with open(path, 'wb') as f:
                pickle.dump(model, f)
        
        return True
    except Exception as e:
        logging.error(f"Error saving model: {e}")
        return False


def load_model(path: str, use_joblib: bool = True) -> Optional[Any]:
    """
    Load model from file using pickle or joblib.
    
    Args:
        path: Path to model file
        use_joblib: Use joblib instead of pickle for larger models
        
    Returns:
        Any: Loaded model or None if error
    """
    try:
        if use_joblib:
            model = joblib.load(path)
        else:
            with open(path, 'rb') as f:
                model = pickle.load(f)
        
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return None


def format_time_elapsed(seconds: float) -> str:
    """
    Format time elapsed in seconds to a human-readable string.
    
    Args:
        seconds: Time elapsed in seconds
        
    Returns:
        str: Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def create_directory_structure(config: Dict[str, Any]) -> bool:
    """
    Create directory structure based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create directories from config
        for _, dir_path in config.get("directories", {}).items():
            os.makedirs(dir_path, exist_ok=True)
            logging.info(f"Created directory: {dir_path}")
        
        return True
    except Exception as e:
        logging.error(f"Error creating directory structure: {e}")
        return False


if __name__ == "__main__":
    # Example usage
    config = load_config("config.yaml")
    
    # Set up logging
    setup_logging(
        level=getattr(logging, config["logging"]["level"]),
        log_format=config["logging"]["format"],
        log_file=config["logging"]["file"]
    )
    
    # Create directory structure
    create_directory_structure(config)
    
    logging.info("Utilities module test successful")
