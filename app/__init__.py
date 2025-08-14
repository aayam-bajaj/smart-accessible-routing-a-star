"""
Main application package for Smart Accessible Routing System
"""
import logging

# Set up logging
from .utils.logging_config import setup_logging, get_logger
setup_logging()

logger = get_logger(__name__)

logger.info("Smart Accessible Routing System package initialized")