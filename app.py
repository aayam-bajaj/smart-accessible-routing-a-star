"""
Main application entry point for Smart Accessible Routing System
"""
import os
import logging
from flask import Flask
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
from app.utils.logging_config import setup_logging
setup_logging()

logger = logging.getLogger(__name__)

def create_app():
    """Create and configure the Flask application"""
    app = Flask(__name__)
    
    # Initialize configuration
    from app.utils.config import init_app
    init_app(app)
    
    # Initialize database
    from app import init_db
    init_db(app)
    
    # Register blueprints
    from app.routes.main import main_bp
    from app.routes.auth import auth_bp
    from app.routes.api import api_bp
    
    app.register_blueprint(main_bp)
    app.register_blueprint(auth_bp, url_prefix='/auth')
    app.register_blueprint(api_bp, url_prefix='/api')
    
    # Add health check endpoint
    @app.route('/health')
    def health_check():
        """Basic health check endpoint"""
        return {
            'status': 'healthy',
            'service': 'Smart Accessible Routing System',
            'version': '1.0.0'
        }
    
    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        """Handle 404 errors"""
        return {'error': 'Not found'}, 404
    
    @app.errorhandler(500)
    def internal_error(error):
        """Handle 500 errors"""
        logger.error(f"Internal server error: {error}")
        return {'error': 'Internal server error'}, 500
    
    logger.info("Flask application created successfully")
    return app

def init_db(app):
    """Initialize database with the Flask app"""
    from flask_sqlalchemy import SQLAlchemy
    from flask_migrate import Migrate
    
    # Initialize database
    db = SQLAlchemy(app)
    migrate = Migrate(app, db)
    
    # Store db and migrate in app extensions
    app.extensions['sqlalchemy'] = db
    app.extensions['migrate'] = migrate
    
    # Create tables
    with app.app_context():
        db.create_all()
    
    logger.info("Database initialized successfully")

# Create app instance
app = create_app()

if __name__ == '__main__':
    # Get configuration from environment
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    host = os.environ.get('FLASK_HOST', '127.0.0.1')
    port = int(os.environ.get('FLASK_PORT', 5000))
    
    logger.info(f"Starting Flask application on {host}:{port}")
    
    # Run the application
    app.run(
        host=host,
        port=port,
        debug=debug
    )