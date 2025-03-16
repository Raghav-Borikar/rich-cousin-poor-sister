# src/utils/logger.py
import logging
import os
from datetime import datetime

def setup_logger(name, log_dir="logs"):
    """
    Set up a logger with console and file handlers
    
    Args:
        name: Logger name
        log_dir: Directory for log files
        
    Returns:
        Configured logger
    """
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Add file handler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(os.path.join(log_dir, f"{name}_{timestamp}.log"))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger
