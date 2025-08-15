"""
User Management and Authentication API Endpoints
===============================================

Provides comprehensive user authentication, registration, profile management,
and accessibility preference management endpoints with JWT token handling
and security measures.
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import hashlib
import secrets
import jwt
from flask import request, jsonify, current_app
from functools import wraps
import re
import sqlite3
import os
import bcrypt
from dataclasses import asdict

from ..models.user_profile import (
    UserProfile, MobilityAid, AccessibilityConstraints, 
    UserPreferences, create_user_template
)
from .validation import validate_request_data, handle_api_errors
from .schemas import USER_MANAGEMENT_SCHEMAS


class UserManager:
    """Handles user data persistence and management."""
    
    def __init__(self, db_path: str = "users.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize user database tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT UNIQUE NOT NULL,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT TRUE,
                    email_verified BOOLEAN DEFAULT FALSE,
                    last_login TIMESTAMP,
                    login_count INTEGER DEFAULT 0,
                    failed_login_attempts INTEGER DEFAULT 0,
                    locked_until TIMESTAMP,
                    password_reset_token TEXT,
                    password_reset_expires TIMESTAMP,
                    email_verification_token TEXT
                )
            """)
            
            # User profiles table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id INTEGER PRIMARY KEY,
                    mobility_aid TEXT,
                    max_walking_distance REAL,
                    max_slope_percent REAL,
                    requires_elevator BOOLEAN,
                    requires_ramp BOOLEAN,
                    avoids_stairs BOOLEAN,
                    needs_rest_areas BOOLEAN,
                    walking_speed_factor REAL,
                    surface_preferences TEXT,
                    weather_considerations TEXT,
                    time_preferences TEXT,
                    route_complexity_preference TEXT,
                    learning_enabled BOOLEAN DEFAULT TRUE,
                    data_sharing_consent BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """)
            
            # User sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    session_token TEXT UNIQUE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NOT NULL,
                    is_active BOOLEAN DEFAULT TRUE,
                    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    ip_address TEXT,
                    user_agent TEXT,
                    device_info TEXT,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """)
            
            conn.commit()
    
    def create_user(self, email: str, username: str, password: str) -> Dict[str, Any]:
        """Create a new user account."""
        # Generate salt and hash password
        salt = bcrypt.gensalt()
        password_hash = bcrypt.hashpw(password.encode('utf-8'), salt)
        
        # Generate email verification token
        verification_token = secrets.token_urlsafe(32)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            try:
                cursor.execute("""
                    INSERT INTO users (email, username, password_hash, salt, 
                                     email_verification_token)
                    VALUES (?, ?, ?, ?, ?)
                """, (email, username, password_hash, salt, verification_token))
                
                user_id = cursor.lastrowid
                
                # Create default profile
                cursor.execute("""
                    INSERT INTO user_profiles (user_id, mobility_aid, 
                                             max_walking_distance, max_slope_percent,
                                             walking_speed_factor)
                    VALUES (?, ?, ?, ?, ?)
                """, (user_id, "none", 1000.0, 8.0, 1.0))
                
                conn.commit()
                
                return {
                    "user_id": user_id,
                    "email": email,
                    "username": username,
                    "verification_token": verification_token,
                    "created_at": datetime.now().isoformat()
                }
                
            except sqlite3.IntegrityError as e:
                if "email" in str(e):
                    raise ValueError("Email already registered")
                elif "username" in str(e):
                    raise ValueError("Username already taken")
                else:
                    raise ValueError("User creation failed")
    
    def authenticate_user(self, login: str, password: str, ip_address: str = None) -> Dict[str, Any]:
        """Authenticate user credentials and create session."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Find user by email or username
            cursor.execute("""
                SELECT id, email, username, password_hash, salt, is_active, 
                       failed_login_attempts, locked_until, email_verified
                FROM users 
                WHERE email = ? OR username = ?
            """, (login, login))
            
            user_data = cursor.fetchone()
            
            if not user_data:
                raise ValueError("Invalid credentials")
            
            user_id, email, username, stored_hash, salt, is_active, failed_attempts, locked_until, email_verified = user_data
            
            # Check if account is locked
            if locked_until and datetime.fromisoformat(locked_until) > datetime.now():
                raise ValueError("Account is temporarily locked")
            
            # Check if account is active
            if not is_active:
                raise ValueError("Account is disabled")
            
            # Verify password
            if not bcrypt.checkpw(password.encode('utf-8'), stored_hash):
                # Increment failed attempts
                failed_attempts += 1
                lock_time = None
                
                if failed_attempts >= 5:
                    lock_time = datetime.now() + timedelta(minutes=30)
                
                cursor.execute("""
                    UPDATE users 
                    SET failed_login_attempts = ?, locked_until = ?
                    WHERE id = ?
                """, (failed_attempts, lock_time.isoformat() if lock_time else None, user_id))
                
                conn.commit()
                raise ValueError("Invalid credentials")
            
            # Reset failed attempts on successful login
            cursor.execute("""
                UPDATE users 
                SET failed_login_attempts = 0, locked_until = NULL,
                    last_login = ?, login_count = login_count + 1
                WHERE id = ?
            """, (datetime.now().isoformat(), user_id))
            
            conn.commit()
            
            return {
                "user_id": user_id,
                "email": email,
                "username": username,
                "email_verified": bool(email_verified),
                "is_active": bool(is_active)
            }
    
    def get_user_profile(self, user_id: int) -> Optional[UserProfile]:
        """Retrieve user profile."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT u.email, u.username, u.created_at,
                       p.mobility_aid, p.max_walking_distance, p.max_slope_percent,
                       p.requires_elevator, p.requires_ramp, p.avoids_stairs,
                       p.needs_rest_areas, p.walking_speed_factor, p.surface_preferences,
                       p.weather_considerations, p.time_preferences,
                       p.route_complexity_preference, p.learning_enabled,
                       p.data_sharing_consent
                FROM users u
                LEFT JOIN user_profiles p ON u.id = p.user_id
                WHERE u.id = ?
            """, (user_id,))
            
            data = cursor.fetchone()
            
            if not data:
                return None
            
            # Parse the data into UserProfile
            mobility_aid = MobilityAid(data[3]) if data[3] else MobilityAid.NONE
            
            constraints = AccessibilityConstraints(
                max_walking_distance=data[4] or 1000.0,
                max_slope_percent=data[5] or 8.0,
                requires_elevator=bool(data[6]),
                requires_ramp=bool(data[7]),
                avoids_stairs=bool(data[8]),
                needs_rest_areas=bool(data[9])
            )
            
            preferences = UserPreferences(
                walking_speed_factor=data[10] or 1.0,
                surface_preferences=data[11].split(',') if data[11] else [],
                weather_considerations=data[12].split(',') if data[12] else [],
                time_preferences=data[13].split(',') if data[13] else [],
                route_complexity_preference=data[14] or 'balanced'
            )
            
            return UserProfile(
                user_id=str(user_id),
                mobility_aid=mobility_aid,
                constraints=constraints,
                preferences=preferences,
                learning_enabled=bool(data[15]),
                data_sharing_consent=bool(data[16]),
                route_history=[],
                learned_preferences={}
            )


# Initialize user manager
user_manager = UserManager()


def generate_jwt_token(user_id: int, email: str, username: str) -> str:
    """Generate JWT token for authenticated user."""
    payload = {
        'user_id': user_id,
        'email': email,
        'username': username,
        'iat': datetime.utcnow(),
        'exp': datetime.utcnow() + timedelta(days=7)
    }
    
    secret_key = current_app.config.get('JWT_SECRET_KEY', 'default-secret-key')
    return jwt.encode(payload, secret_key, algorithm='HS256')


def verify_jwt_token(token: str) -> Optional[Dict[str, Any]]:
    """Verify JWT token and return payload."""
    try:
        secret_key = current_app.config.get('JWT_SECRET_KEY', 'default-secret-key')
        payload = jwt.decode(token, secret_key, algorithms=['HS256'])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


def require_auth(f):
    """Decorator to require authentication for endpoint."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({"error": "Missing or invalid authorization header"}), 401
        
        token = auth_header.split(' ')[1]
        payload = verify_jwt_token(token)
        
        if not payload:
            return jsonify({"error": "Invalid or expired token"}), 401
        
        # Add user info to request context
        request.current_user = payload
        return f(*args, **kwargs)
    
    return decorated_function


# API Endpoints

@handle_api_errors
@validate_request_data(USER_MANAGEMENT_SCHEMAS['register_user'])
def register_user():
    """Register a new user account."""
    data = request.json
    
    # Validate email format
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(email_pattern, data['email']):
        return jsonify({"error": "Invalid email format"}), 400
    
    # Validate username format
    if not re.match(r'^[a-zA-Z0-9_]{3,20}$', data['username']):
        return jsonify({
            "error": "Username must be 3-20 characters, alphanumeric and underscores only"
        }), 400
    
    # Validate password strength
    password = data['password']
    if len(password) < 8:
        return jsonify({"error": "Password must be at least 8 characters"}), 400
    
    if not re.search(r'[A-Z]', password) or not re.search(r'[a-z]', password) or not re.search(r'\d', password):
        return jsonify({
            "error": "Password must contain uppercase, lowercase, and numeric characters"
        }), 400
    
    try:
        user_data = user_manager.create_user(
            email=data['email'],
            username=data['username'],
            password=password
        )
        
        # Generate JWT token
        token = generate_jwt_token(
            user_data['user_id'],
            user_data['email'],
            user_data['username']
        )
        
        return jsonify({
            "message": "User registered successfully",
            "user": {
                "id": user_data['user_id'],
                "email": user_data['email'],
                "username": user_data['username'],
                "created_at": user_data['created_at']
            },
            "token": token,
            "verification_required": True
        }), 201
        
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@handle_api_errors
@validate_request_data(USER_MANAGEMENT_SCHEMAS['login_user'])
def login_user():
    """Authenticate user and create session."""
    data = request.json
    
    try:
        user_data = user_manager.authenticate_user(
            login=data['login'],
            password=data['password'],
            ip_address=request.remote_addr
        )
        
        # Generate JWT token
        token = generate_jwt_token(
            user_data['user_id'],
            user_data['email'],
            user_data['username']
        )
        
        # Get user profile
        profile = user_manager.get_user_profile(user_data['user_id'])
        
        return jsonify({
            "message": "Login successful",
            "user": {
                "id": user_data['user_id'],
                "email": user_data['email'],
                "username": user_data['username'],
                "email_verified": user_data['email_verified']
            },
            "token": token,
            "profile": asdict(profile) if profile else None
        }), 200
        
    except ValueError as e:
        return jsonify({"error": str(e)}), 401


@handle_api_errors
@require_auth
def get_current_user():
    """Get current authenticated user information."""
    user_id = request.current_user['user_id']
    profile = user_manager.get_user_profile(user_id)
    
    return jsonify({
        "user": {
            "id": user_id,
            "email": request.current_user['email'],
            "username": request.current_user['username']
        },
        "profile": asdict(profile) if profile else None
    }), 200


@handle_api_errors
@require_auth
@validate_request_data(USER_MANAGEMENT_SCHEMAS['update_profile'])
def update_user_profile():
    """Update user accessibility profile and preferences."""
    data = request.json
    user_id = request.current_user['user_id']
    
    # Get current profile
    current_profile = user_manager.get_user_profile(user_id)
    if not current_profile:
        return jsonify({"error": "User profile not found"}), 404
    
    # Update profile data
    try:
        with sqlite3.connect(user_manager.db_path) as conn:
            cursor = conn.cursor()
            
            # Build update query dynamically based on provided fields
            update_fields = []
            values = []
            
            if 'mobility_aid' in data:
                update_fields.append("mobility_aid = ?")
                values.append(data['mobility_aid'])
            
            if 'constraints' in data:
                constraints = data['constraints']
                for field in ['max_walking_distance', 'max_slope_percent', 'requires_elevator',
                             'requires_ramp', 'avoids_stairs', 'needs_rest_areas']:
                    if field in constraints:
                        update_fields.append(f"{field} = ?")
                        values.append(constraints[field])
            
            if 'preferences' in data:
                prefs = data['preferences']
                if 'walking_speed_factor' in prefs:
                    update_fields.append("walking_speed_factor = ?")
                    values.append(prefs['walking_speed_factor'])
                
                for field in ['surface_preferences', 'weather_considerations',
                             'time_preferences']:
                    if field in prefs:
                        update_fields.append(f"{field} = ?")
                        values.append(','.join(prefs[field]))
                
                if 'route_complexity_preference' in prefs:
                    update_fields.append("route_complexity_preference = ?")
                    values.append(prefs['route_complexity_preference'])
            
            if 'learning_enabled' in data:
                update_fields.append("learning_enabled = ?")
                values.append(data['learning_enabled'])
            
            if 'data_sharing_consent' in data:
                update_fields.append("data_sharing_consent = ?")
                values.append(data['data_sharing_consent'])
            
            if update_fields:
                update_fields.append("updated_at = ?")
                values.append(datetime.now().isoformat())
                values.append(user_id)
                
                query = f"""
                    UPDATE user_profiles 
                    SET {', '.join(update_fields)}
                    WHERE user_id = ?
                """
                
                cursor.execute(query, values)
                conn.commit()
        
        # Return updated profile
        updated_profile = user_manager.get_user_profile(user_id)
        
        return jsonify({
            "message": "Profile updated successfully",
            "profile": asdict(updated_profile) if updated_profile else None
        }), 200
        
    except Exception as e:
        return jsonify({"error": f"Profile update failed: {str(e)}"}), 500


@handle_api_errors
@require_auth
@validate_request_data(USER_MANAGEMENT_SCHEMAS['change_password'])
def change_password():
    """Change user password."""
    data = request.json
    user_id = request.current_user['user_id']
    
    # Validate new password strength
    new_password = data['new_password']
    if len(new_password) < 8:
        return jsonify({"error": "Password must be at least 8 characters"}), 400
    
    if not re.search(r'[A-Z]', new_password) or not re.search(r'[a-z]', new_password) or not re.search(r'\d', new_password):
        return jsonify({
            "error": "Password must contain uppercase, lowercase, and numeric characters"
        }), 400
    
    try:
        with sqlite3.connect(user_manager.db_path) as conn:
            cursor = conn.cursor()
            
            # Verify current password
            cursor.execute("SELECT password_hash FROM users WHERE id = ?", (user_id,))
            result = cursor.fetchone()
            
            if not result or not bcrypt.checkpw(data['current_password'].encode('utf-8'), result[0]):
                return jsonify({"error": "Current password is incorrect"}), 400
            
            # Hash new password
            salt = bcrypt.gensalt()
            new_hash = bcrypt.hashpw(new_password.encode('utf-8'), salt)
            
            # Update password
            cursor.execute("""
                UPDATE users 
                SET password_hash = ?, salt = ?, updated_at = ?
                WHERE id = ?
            """, (new_hash, salt, datetime.now().isoformat(), user_id))
            
            conn.commit()
        
        return jsonify({"message": "Password changed successfully"}), 200
        
    except Exception as e:
        return jsonify({"error": f"Password change failed: {str(e)}"}), 500


@handle_api_errors
@require_auth
def logout_user():
    """Logout user and invalidate session."""
    # In a full implementation, you would invalidate the JWT token
    # For now, we'll just return a success message as JWT tokens
    # are stateless and expire automatically
    
    return jsonify({"message": "Logout successful"}), 200


@handle_api_errors
@require_auth
def delete_user_account():
    """Delete user account and all associated data."""
    user_id = request.current_user['user_id']
    
    try:
        with sqlite3.connect(user_manager.db_path) as conn:
            cursor = conn.cursor()
            
            # Delete user profile
            cursor.execute("DELETE FROM user_profiles WHERE user_id = ?", (user_id,))
            
            # Delete user sessions
            cursor.execute("DELETE FROM user_sessions WHERE user_id = ?", (user_id,))
            
            # Delete user account
            cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
            
            conn.commit()
        
        return jsonify({"message": "Account deleted successfully"}), 200
        
    except Exception as e:
        return jsonify({"error": f"Account deletion failed: {str(e)}"}), 500


# Export endpoint functions for Flask app registration
USER_AUTH_ENDPOINTS = {
    'register': register_user,
    'login': login_user,
    'current_user': get_current_user,
    'update_profile': update_user_profile,
    'change_password': change_password,
    'logout': logout_user,
    'delete_account': delete_user_account
}
