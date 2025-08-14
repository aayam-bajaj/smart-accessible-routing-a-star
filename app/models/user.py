"""
User models for Smart Accessible Routing System
"""
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text, ForeignKey
from sqlalchemy.orm import relationship
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import json

# Import db from app package
try:
    from app import db
except ImportError:
    # For testing purposes, create a dummy db object
    from flask_sqlalchemy import SQLAlchemy
    db = SQLAlchemy()

class User(db.Model):
    """
    User model for the Smart Accessible Routing System
    """
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    username = Column(String(80), unique=True, nullable=False)
    email = Column(String(120), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    
    # User profile information
    first_name = Column(String(50), nullable=True)
    last_name = Column(String(50), nullable=True)
    date_of_birth = Column(DateTime, nullable=True)
    
    # Mobility information
    mobility_aid_type = Column(String(50), nullable=True)  # wheelchair, walker, cane, none
    max_slope_degrees = Column(Float, nullable=True)  # Maximum slope user can handle
    min_path_width = Column(Float, nullable=True)  # Minimum path width in meters
    avoid_stairs = Column(Boolean, default=True)  # Whether to avoid stairs
    
    # Preferences for route optimization
    distance_weight = Column(Float, default=0.3)  # Weight for distance optimization
    energy_efficiency_weight = Column(Float, default=0.3)  # Weight for energy efficiency
    comfort_weight = Column(Float, default=0.4)  # Weight for comfort
    
    # Surface preferences (stored as JSON)
    surface_preferences = Column(Text, default='{}')
    
    # Account information
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    obstacle_reports = relationship('ObstacleReport', backref='user')
    route_feedbacks = relationship('RouteFeedback', backref='user')
    
    def set_password(self, password):
        """
        Set the user's password using a hash
        
        Args:
            password: Plain text password
        """
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """
        Check if the provided password matches the hash
        
        Args:
            password: Plain text password to check
            
        Returns:
            bool: True if password matches, False otherwise
        """
        return check_password_hash(self.password_hash, password)
    
    def get_surface_preferences(self):
        """
        Get surface preferences as dictionary
        
        Returns:
            dict: Surface preferences dictionary
        """
        try:
            return json.loads(self.surface_preferences)
        except (json.JSONDecodeError, TypeError):
            return {}
    
    def set_surface_preferences(self, preferences):
        """
        Set surface preferences from dictionary
        
        Args:
            preferences: Dictionary of surface preferences
        """
        self.surface_preferences = json.dumps(preferences)
    
    def get_full_name(self):
        """
        Get the user's full name
        
        Returns:
            str: Full name (first + last)
        """
        names = [self.first_name, self.last_name]
        return ' '.join(filter(None, names))
    
    def get_accessibility_profile(self):
        """
        Get the user's accessibility profile as a dictionary
        
        Returns:
            dict: Accessibility profile
        """
        return {
            'mobility_aid_type': self.mobility_aid_type,
            'max_slope_degrees': self.max_slope_degrees,
            'min_path_width': self.min_path_width,
            'avoid_stairs': self.avoid_stairs,
            'surface_preferences': self.get_surface_preferences()
        }
    
    def get_route_preferences(self):
        """
        Get the user's route optimization preferences
        
        Returns:
            dict: Route optimization preferences
        """
        return {
            'distance_weight': self.distance_weight,
            'energy_efficiency_weight': self.energy_efficiency_weight,
            'comfort_weight': self.comfort_weight
        }
    
    def __repr__(self):
        return f'<User {self.username}>'

class UserSession(db.Model):
    """
    User session model for tracking user sessions
    """
    __tablename__ = 'user_sessions'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    session_token = Column(String(255), unique=True, nullable=False)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)
    is_active = Column(Boolean, default=True)
    
    # Relationship
    user = relationship('User', backref='sessions')
    
    def __repr__(self):
        return f'<UserSession {self.session_token}>'

# Helper functions for user operations
def create_user(username, email, password, **kwargs):
    """
    Create a new user
    
    Args:
        username: Username
        email: Email address
        password: Plain text password
        **kwargs: Additional user attributes
        
    Returns:
        User: New user instance
    """
    user = User(
        username=username,
        email=email,
        **kwargs
    )
    user.set_password(password)
    return user

def get_user_by_username(username):
    """
    Get a user by username
    
    Args:
        username: Username to search for
        
    Returns:
        User: User with matching username or None
    """
    return User.query.filter_by(username=username).first()

def get_user_by_email(email):
    """
    Get a user by email
    
    Args:
        email: Email to search for
        
    Returns:
        User: User with matching email or None
    """
    return User.query.filter_by(email=email).first()

def get_user_by_id(user_id):
    """
    Get a user by ID
    
    Args:
        user_id: User ID
        
    Returns:
        User: User with matching ID or None
    """
    return User.query.get(user_id)

def authenticate_user(username, password):
    """
    Authenticate a user
    
    Args:
        username: Username
        password: Plain text password
        
    Returns:
        User: Authenticated user or None
    """
    user = get_user_by_username(username)
    if user and user.check_password(password):
        return user
    return None

def update_user_profile(user_id, **kwargs):
    """
    Update a user's profile
    
    Args:
        user_id: User ID
        **kwargs: Profile attributes to update
        
    Returns:
        bool: True if successful, False otherwise
    """
    user = get_user_by_id(user_id)
    if user:
        for key, value in kwargs.items():
            setattr(user, key, value)
        return True
    return False