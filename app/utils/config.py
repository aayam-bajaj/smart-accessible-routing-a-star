"""
Configuration management for the Smart Accessible Routing System
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Base configuration class"""
    
    # Flask configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key'
    FLASK_ENV = os.environ.get('FLASK_ENV') or 'development'
    FLASK_DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    # Database configuration
    DATABASE_URL = os.environ.get('DATABASE_URL') or 'sqlite:///smart_routing.db'
    
    # Map data configuration
    OSM_REGION = os.environ.get('OSM_REGION') or 'Kharghar, India'
    DEFAULT_LATITUDE = float(os.environ.get('DEFAULT_LATITUDE', 19.076090))
    DEFAULT_LONGITUDE = float(os.environ.get('DEFAULT_LONGITUDE', 73.076090))
    DEFAULT_ZOOM = int(os.environ.get('DEFAULT_ZOOM', 15))
    
    # Machine Learning configuration
    ML_MODEL_PATH = os.environ.get('ML_MODEL_PATH') or 'models/'
    HEURISTIC_MODEL_FILE = os.environ.get('HEURISTIC_MODEL_FILE') or 'heuristic_model.pkl'
    COST_MODEL_FILE = os.environ.get('COST_MODEL_FILE') or 'cost_models.pkl'
    
    # Logging configuration
    LOG_LEVEL = os.environ.get('LOG_LEVEL') or 'INFO'
    LOG_FILE = os.environ.get('LOG_FILE') or 'smart_routing.log'
    
    # API Keys
    OSM_API_KEY = os.environ.get('OSM_API_KEY') or None

class DevelopmentConfig(Config):
    """Development configuration"""
    FLASK_ENV = 'development'
    FLASK_DEBUG = True
    DATABASE_URL = os.environ.get('DEV_DATABASE_URL') or 'sqlite:///smart_routing_dev.db'

class TestingConfig(Config):
    """Testing configuration"""
    FLASK_ENV = 'testing'
    FLASK_DEBUG = False
    DATABASE_URL = os.environ.get('TEST_DATABASE_URL') or 'sqlite:///smart_routing_test.db'
    TESTING = True

class ProductionConfig(Config):
    """Production configuration"""
    FLASK_ENV = 'production'
    FLASK_DEBUG = False
    DATABASE_URL = os.environ.get('DATABASE_URL') or 'sqlite:///smart_routing.db'

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}

def get_config():
    """
    Get the appropriate configuration based on environment
    
    Returns:
        Config: Configuration object
    """
    config_name = os.environ.get('FLASK_ENV') or 'default'
    return config.get(config_name, config['default'])

def init_app(app):
    """
    Initialize application with configuration
    
    Args:
        app: Flask application instance
    """
    config_class = get_config()
    app.config.from_object(config_class)
    
    # Ensure instance folder exists
    os.makedirs(app.instance_path, exist_ok=True)