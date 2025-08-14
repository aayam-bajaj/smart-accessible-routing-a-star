"""
Logging configuration for the Smart Accessible Routing System
"""
import logging
import os
from logging.handlers import RotatingFileHandler

def setup_logging(log_level=logging.INFO, log_file='smart_routing.log'):
    """
    Set up logging configuration for the application
    
    Args:
        log_level: The logging level (default: INFO)
        log_file: The log file path (default: smart_routing.log)
    """
    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(log_file) if os.path.dirname(log_file) else '.'
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Create a rotating file handler for better log management
    rotating_handler = RotatingFileHandler(
        log_file, 
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    
    # Set formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    rotating_handler.setFormatter(formatter)
    
    # Add handler to root logger
    logging.getLogger().addHandler(rotating_handler)
    
    # Set level for the rotating handler
    rotating_handler.setLevel(log_level)
    
    logging.info("Logging configuration set up successfully")

def get_logger(name):
    """
    Get a logger with the specified name
    
    Args:
        name: The name of the logger
        
    Returns:
        logging.Logger: Configured logger instance
    """
    return logging.getLogger(name)