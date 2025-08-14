"""
Test cases for the Smart Accessible Routing System
"""
import unittest
import sys
import os

# Add the parent directory to the path so we can import app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestAppSetup(unittest.TestCase):
    """Test basic application setup"""
    
    def test_import_app(self):
        """Test that we can import the main application"""
        try:
            from app import create_app
            self.assertTrue(True)
        except ImportError:
            self.fail("Failed to import app module")
    
    def test_create_app(self):
        """Test that we can create the Flask application"""
        try:
            from app import create_app
            app = create_app()
            self.assertIsNotNone(app)
        except Exception as e:
            self.fail(f"Failed to create app: {e}")
    
    def test_app_has_routes(self):
        """Test that the app has routes registered"""
        try:
            from app import create_app
            app = create_app()
            
            # Check that we have some routes registered
            self.assertGreater(len(app.url_map._rules), 0)
        except Exception as e:
            self.fail(f"Failed to check app routes: {e}")

if __name__ == '__main__':
    unittest.main()