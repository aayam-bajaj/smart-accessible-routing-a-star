#!/usr/bin/env python3
"""
Database initialization script for Smart Accessible Routing System
"""
import sys
import os

# Add the parent directory to the path so we can import app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

def init_database():
    """Initialize the database"""
    try:
        from app import create_app
        
        # Create the Flask app
        app = create_app()
        
        # Initialize the database
        with app.app_context():
            from flask_sqlalchemy import SQLAlchemy
            db = SQLAlchemy(app)
            db.create_all()
        
        print("Database initialized successfully!")
        return True
        
    except Exception as e:
        print(f"Error initializing database: {e}")
        return False

if __name__ == '__main__':
    print("Initializing database...")
    success = init_database()
    if success:
        print("Database initialization completed.")
    else:
        print("Database initialization failed.")
        sys.exit(1)