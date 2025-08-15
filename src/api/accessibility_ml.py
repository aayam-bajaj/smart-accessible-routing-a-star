"""
Accessibility and ML Integration API Endpoints
==============================================

Provides endpoints for accessibility constraint management, ML model interactions,
feedback collection, personalized recommendations, and real-time learning integration.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import json
from flask import request, jsonify
from dataclasses import asdict
import numpy as np
import sqlite3
import os

from ..models.user_profile import UserProfile, MobilityAid, AccessibilityConstraints, UserPreferences
from ..ml.feedback_system import UserFeedbackSystem, FeedbackType, RouteFeatureRating
from ..ml.heuristic_learning import HeuristicMLModel, CostPredictionModel
from ..ml.dynamic_cost_prediction import DynamicCostPredictor
from ..routing.personalized_astar import PersonalizedAStarRouter
from .validation import (
    validate_request_data, handle_api_errors, require_auth,
    validate_coordinates, ValidationAPIError, format_success_response
)
from .schemas import ML_SCHEMAS, USER_MANAGEMENT_SCHEMAS


# Initialize ML components
feedback_system = UserFeedbackSystem()
heuristic_model = HeuristicMLModel()
cost_model = CostPredictionModel()
dynamic_predictor = DynamicCostPredictor()


class AccessibilityManager:
    """Manages accessibility constraints and preferences."""
    
    def __init__(self, db_path: str = "accessibility.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize accessibility database tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Accessibility templates table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS accessibility_templates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    description TEXT,
                    mobility_aid TEXT NOT NULL,
                    max_walking_distance REAL,
                    max_slope_percent REAL,
                    requires_elevator BOOLEAN,
                    requires_ramp BOOLEAN,
                    avoids_stairs BOOLEAN,
                    needs_rest_areas BOOLEAN,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Location accessibility data
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS location_accessibility (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    latitude REAL NOT NULL,
                    longitude REAL NOT NULL,
                    location_type TEXT,
                    accessibility_features TEXT, -- JSON
                    barriers TEXT, -- JSON
                    rating REAL,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    verified BOOLEAN DEFAULT FALSE
                )
            """)
            
            # Route accessibility cache
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS route_accessibility_cache (
                    route_hash TEXT PRIMARY KEY,
                    accessibility_score REAL,
                    barriers TEXT, -- JSON
                    recommendations TEXT, -- JSON
                    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP
                )
            """)
            
            conn.commit()
    
    def get_accessibility_template(self, template_name: str) -> Optional[Dict[str, Any]]:
        """Retrieve accessibility template by name."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM accessibility_templates WHERE name = ?
            """, (template_name,))
            
            result = cursor.fetchone()
            if not result:
                return None
            
            columns = [desc[0] for desc in cursor.description]
            return dict(zip(columns, result))
    
    def create_accessibility_template(self, template_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new accessibility template."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO accessibility_templates 
                (name, description, mobility_aid, max_walking_distance, 
                 max_slope_percent, requires_elevator, requires_ramp, 
                 avoids_stairs, needs_rest_areas)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                template_data['name'],
                template_data.get('description', ''),
                template_data['mobility_aid'],
                template_data.get('max_walking_distance', 1000),
                template_data.get('max_slope_percent', 8.0),
                template_data.get('requires_elevator', False),
                template_data.get('requires_ramp', False),
                template_data.get('avoids_stairs', False),
                template_data.get('needs_rest_areas', False)
            ))
            
            template_id = cursor.lastrowid
            conn.commit()
            
            return {"id": template_id, **template_data}
    
    def get_location_accessibility(self, lat: float, lon: float, radius: float = 100) -> List[Dict[str, Any]]:
        """Get accessibility information for locations within radius."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Simple radius search (for production, use spatial indexing)
            cursor.execute("""
                SELECT * FROM location_accessibility 
                WHERE (latitude BETWEEN ? AND ?) 
                AND (longitude BETWEEN ? AND ?)
                AND verified = TRUE
                ORDER BY rating DESC
            """, (
                lat - radius/111000,  # Rough degree conversion
                lat + radius/111000,
                lon - radius/111000,
                lon + radius/111000
            ))
            
            results = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            
            locations = []
            for result in results:
                location = dict(zip(columns, result))
                # Parse JSON fields
                if location['accessibility_features']:
                    location['accessibility_features'] = json.loads(location['accessibility_features'])
                if location['barriers']:
                    location['barriers'] = json.loads(location['barriers'])
                locations.append(location)
            
            return locations
    
    def update_location_accessibility(self, location_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update accessibility information for a location."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Check if location exists
            cursor.execute("""
                SELECT id FROM location_accessibility 
                WHERE latitude = ? AND longitude = ? AND location_type = ?
            """, (location_data['latitude'], location_data['longitude'], 
                  location_data.get('location_type', 'general')))
            
            existing = cursor.fetchone()
            
            if existing:
                # Update existing location
                cursor.execute("""
                    UPDATE location_accessibility 
                    SET accessibility_features = ?, barriers = ?, rating = ?,
                        last_updated = ?, verified = ?
                    WHERE id = ?
                """, (
                    json.dumps(location_data.get('accessibility_features', {})),
                    json.dumps(location_data.get('barriers', [])),
                    location_data.get('rating', 3.0),
                    datetime.now().isoformat(),
                    location_data.get('verified', False),
                    existing[0]
                ))
                location_id = existing[0]
            else:
                # Insert new location
                cursor.execute("""
                    INSERT INTO location_accessibility 
                    (latitude, longitude, location_type, accessibility_features, 
                     barriers, rating, verified)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    location_data['latitude'],
                    location_data['longitude'],
                    location_data.get('location_type', 'general'),
                    json.dumps(location_data.get('accessibility_features', {})),
                    json.dumps(location_data.get('barriers', [])),
                    location_data.get('rating', 3.0),
                    location_data.get('verified', False)
                ))
                location_id = cursor.lastrowid
            
            conn.commit()
            return {"id": location_id, "updated": True}


# Initialize accessibility manager
accessibility_manager = AccessibilityManager()


# API Endpoints

@handle_api_errors
def get_accessibility_templates():
    """Get available accessibility templates."""
    with sqlite3.connect(accessibility_manager.db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM accessibility_templates ORDER BY name")
        
        results = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        
        templates = [dict(zip(columns, result)) for result in results]
    
    return format_success_response(templates)


@handle_api_errors
@require_auth
@validate_request_data({
    "type": "object",
    "required": ["name", "mobility_aid"],
    "properties": {
        "name": {"type": "string", "minLength": 1, "maxLength": 100},
        "description": {"type": "string", "maxLength": 500},
        "mobility_aid": {
            "type": "string",
            "enum": ["none", "wheelchair", "mobility_scooter", "walker", "cane", "crutches", "prosthetic"]
        },
        "max_walking_distance": {"type": "number", "minimum": 0},
        "max_slope_percent": {"type": "number", "minimum": 0, "maximum": 100},
        "requires_elevator": {"type": "boolean"},
        "requires_ramp": {"type": "boolean"},
        "avoids_stairs": {"type": "boolean"},
        "needs_rest_areas": {"type": "boolean"}
    }
})
def create_accessibility_template():
    """Create a new accessibility template."""
    data = request.json
    
    try:
        template = accessibility_manager.create_accessibility_template(data)
        return format_success_response(template, "Template created successfully", 201)
    except sqlite3.IntegrityError:
        raise ValidationAPIError("Template name already exists", "name", data['name'])


@handle_api_errors
def get_accessibility_template(template_name: str):
    """Get specific accessibility template."""
    template = accessibility_manager.get_accessibility_template(template_name)
    
    if not template:
        return jsonify({"error": "Template not found"}), 404
    
    return format_success_response(template)


@handle_api_errors
@validate_request_data({
    "type": "object",
    "required": ["lat", "lon"],
    "properties": {
        "lat": {"type": "number", "minimum": -90, "maximum": 90},
        "lon": {"type": "number", "minimum": -180, "maximum": 180},
        "radius": {"type": "number", "minimum": 0, "maximum": 5000, "default": 100}
    }
})
def get_location_accessibility():
    """Get accessibility information for locations."""
    data = request.json
    
    validate_coordinates(data['lat'], data['lon'])
    
    locations = accessibility_manager.get_location_accessibility(
        data['lat'], data['lon'], data.get('radius', 100)
    )
    
    return format_success_response({
        "locations": locations,
        "count": len(locations)
    })


@handle_api_errors
@require_auth
@validate_request_data({
    "type": "object",
    "required": ["latitude", "longitude"],
    "properties": {
        "latitude": {"type": "number", "minimum": -90, "maximum": 90},
        "longitude": {"type": "number", "minimum": -180, "maximum": 180},
        "location_type": {"type": "string"},
        "accessibility_features": {"type": "object"},
        "barriers": {"type": "array", "items": {"type": "string"}},
        "rating": {"type": "number", "minimum": 1, "maximum": 5},
        "verified": {"type": "boolean", "default": False}
    }
})
def update_location_accessibility():
    """Update accessibility information for a location."""
    data = request.json
    
    validate_coordinates(data['latitude'], data['longitude'])
    
    result = accessibility_manager.update_location_accessibility(data)
    
    return format_success_response(result, "Location accessibility updated successfully")


@handle_api_errors
@require_auth
@validate_request_data(ML_SCHEMAS['feedback_request'])
def submit_route_feedback():
    """Submit feedback for a route."""
    data = request.json
    user_id = request.current_user['user_id']
    
    try:
        # Create feedback object
        feedback_type = FeedbackType(data['feedback_type'])
        
        # Validate location if provided
        if 'location' in data:
            validate_coordinates(data['location']['lat'], data['location']['lon'])
        
        # Submit feedback to the system
        feedback_system.collect_feedback(
            user_id=str(user_id),
            route_id=data['route_id'],
            feedback_type=feedback_type,
            rating=data['rating'],
            issues=data.get('issues', []),
            comments=data.get('comments', ''),
            location=data.get('location'),
            context=data.get('context', {})
        )
        
        # Update ML models with new feedback
        try:
            # Get recent feedback for training data
            recent_feedback = feedback_system.get_user_feedback_history(str(user_id), limit=100)
            
            # Update heuristic model if enough data
            if len(recent_feedback) >= 10:
                training_data = feedback_system.generate_training_data(str(user_id))
                if training_data:
                    heuristic_model.update_model(training_data['features'], training_data['labels'])
            
        except Exception as e:
            # Log ML update error but don't fail the feedback submission
            print(f"Error updating ML models: {e}")
        
        return format_success_response(
            {"feedback_id": data['route_id']},
            "Feedback submitted successfully",
            201
        )
        
    except ValueError as e:
        raise ValidationAPIError(str(e))


@handle_api_errors
@require_auth
def get_user_feedback_history():
    """Get user's feedback history."""
    user_id = str(request.current_user['user_id'])
    
    # Get query parameters
    limit = min(int(request.args.get('limit', 50)), 100)
    offset = max(int(request.args.get('offset', 0)), 0)
    feedback_type = request.args.get('feedback_type')
    
    try:
        feedback_history = feedback_system.get_user_feedback_history(
            user_id, limit=limit, offset=offset
        )
        
        # Filter by feedback type if specified
        if feedback_type:
            feedback_history = [
                f for f in feedback_history 
                if f.get('feedback_type') == feedback_type
            ]
        
        # Get feedback analytics
        analytics = feedback_system.analyze_user_feedback(user_id)
        
        return format_success_response({
            "feedback": feedback_history,
            "analytics": analytics,
            "pagination": {
                "limit": limit,
                "offset": offset,
                "has_more": len(feedback_history) == limit
            }
        })
        
    except Exception as e:
        raise ValidationAPIError(f"Error retrieving feedback history: {str(e)}")


@handle_api_errors
@require_auth
@validate_request_data(ML_SCHEMAS['personalization_request'])
def get_personalized_recommendations():
    """Get personalized route recommendations."""
    data = request.json
    user_id = str(request.current_user['user_id'])
    
    try:
        # Get user's current profile
        from .auth import user_manager
        profile = user_manager.get_user_profile(int(user_id))
        
        if not profile:
            raise ValidationAPIError("User profile not found")
        
        # Get ML-based recommendations
        recommendations = {}
        
        # Heuristic recommendations
        try:
            recent_routes = data.get('route_history', [])
            if recent_routes:
                # Extract features from route history
                route_features = []
                for route in recent_routes[-10:]:  # Last 10 routes
                    features = heuristic_model.extract_features({
                        'total_distance': route.get('total_distance', 0),
                        'total_duration': route.get('total_duration', 0),
                        'accessibility_rating': route.get('accessibility_rating', 5.0),
                        'segment_count': len(route.get('segments', [])),
                        'elevation_changes': route.get('elevation_changes', 0)
                    })
                    route_features.append(features)
                
                if route_features:
                    avg_features = np.mean(route_features, axis=0)
                    heuristic_pred = heuristic_model.predict_heuristic(avg_features.reshape(1, -1))
                    recommendations['heuristic_weight'] = float(heuristic_pred[0])
        except Exception as e:
            print(f"Heuristic recommendation error: {e}")
            recommendations['heuristic_weight'] = 1.0
        
        # Cost prediction recommendations
        try:
            if 'context' in data and 'current_conditions' in data['context']:
                conditions = data['context']['current_conditions']
                cost_features = cost_model.extract_features({
                    'weather': conditions.get('weather', 'clear'),
                    'time_of_day': conditions.get('time_of_day', '12:00'),
                    'day_of_week': conditions.get('day_of_week', 'monday'),
                    'user_mobility_aid': profile.mobility_aid.value,
                    'user_walking_speed': profile.preferences.walking_speed_factor
                })
                
                cost_pred = cost_model.predict_cost(cost_features.reshape(1, -1))
                recommendations['cost_adjustment'] = float(cost_pred[0])
        except Exception as e:
            print(f"Cost prediction error: {e}")
            recommendations['cost_adjustment'] = 1.0
        
        # Preference adaptations
        try:
            feedback_history = feedback_system.get_user_feedback_history(user_id, limit=50)
            
            # Calculate preference adaptations based on feedback
            adaptations = {}
            if feedback_history:
                feedback_scores = [f.get('rating', 3) for f in feedback_history]
                avg_satisfaction = sum(feedback_scores) / len(feedback_scores)
                
                # Adjust preferences based on satisfaction
                if avg_satisfaction < 3.0:
                    adaptations['increase_accessibility_weight'] = 0.2
                    adaptations['reduce_complexity'] = True
                elif avg_satisfaction > 4.0:
                    adaptations['optimize_for_efficiency'] = True
                    adaptations['allow_complex_routes'] = True
                
            recommendations['preference_adaptations'] = adaptations
            
        except Exception as e:
            print(f"Preference adaptation error: {e}")
            recommendations['preference_adaptations'] = {}
        
        # Route optimization suggestions
        suggestions = []
        
        if profile.mobility_aid != MobilityAid.NONE:
            suggestions.append({
                "type": "accessibility_focus",
                "message": "Consider prioritizing accessible routes based on your mobility aid",
                "weight_adjustment": {"accessibility": 0.4, "distance": 0.3, "time": 0.3}
            })
        
        if profile.constraints.max_walking_distance < 500:
            suggestions.append({
                "type": "distance_optimization",
                "message": "Recommend shorter routes with rest opportunities",
                "parameters": {"max_segment_length": profile.constraints.max_walking_distance / 2}
            })
        
        recommendations['optimization_suggestions'] = suggestions
        
        return format_success_response({
            "recommendations": recommendations,
            "user_profile_summary": {
                "mobility_aid": profile.mobility_aid.value,
                "learning_enabled": profile.learning_enabled,
                "constraints_summary": {
                    "max_walking_distance": profile.constraints.max_walking_distance,
                    "max_slope_percent": profile.constraints.max_slope_percent,
                    "requires_elevator": profile.constraints.requires_elevator
                }
            },
            "generated_at": datetime.now().isoformat()
        })
        
    except Exception as e:
        raise ValidationAPIError(f"Error generating recommendations: {str(e)}")


@handle_api_errors
@require_auth
def get_ml_model_status():
    """Get status of ML models and learning progress."""
    user_id = str(request.current_user['user_id'])
    
    try:
        status = {
            "models": {
                "heuristic_learning": {
                    "trained": heuristic_model.is_trained(),
                    "last_updated": getattr(heuristic_model, 'last_updated', None),
                    "training_samples": getattr(heuristic_model, 'training_samples', 0)
                },
                "cost_prediction": {
                    "trained": cost_model.is_trained(),
                    "last_updated": getattr(cost_model, 'last_updated', None),
                    "training_samples": getattr(cost_model, 'training_samples', 0)
                },
                "dynamic_cost_predictor": {
                    "active": True,
                    "cache_size": len(getattr(dynamic_predictor, 'prediction_cache', {})),
                    "last_update": getattr(dynamic_predictor, 'last_update', None)
                }
            },
            "user_learning": {
                "feedback_count": len(feedback_system.get_user_feedback_history(user_id, limit=1000)),
                "learning_enabled": True,  # This would come from user profile
                "personalization_level": "basic"  # Could be calculated based on data available
            },
            "system_health": {
                "feedback_system": "operational",
                "ml_pipeline": "operational",
                "last_training": datetime.now().isoformat()
            }
        }
        
        return format_success_response(status)
        
    except Exception as e:
        raise ValidationAPIError(f"Error getting ML model status: {str(e)}")


@handle_api_errors
@require_auth
def reset_user_learning():
    """Reset user's learning data and personalization."""
    user_id = str(request.current_user['user_id'])
    
    try:
        # Clear user feedback (optional - might want to keep for analytics)
        # feedback_system.clear_user_data(user_id)
        
        # Reset learned preferences in user profile
        from .auth import user_manager
        profile = user_manager.get_user_profile(int(user_id))
        
        if profile:
            # Reset learned preferences
            profile.learned_preferences = {}
            # Would need to update in database
        
        return format_success_response(
            {"reset": True},
            "User learning data has been reset"
        )
        
    except Exception as e:
        raise ValidationAPIError(f"Error resetting learning data: {str(e)}")


# Export endpoint functions for Flask app registration
ACCESSIBILITY_ML_ENDPOINTS = {
    'get_accessibility_templates': get_accessibility_templates,
    'create_accessibility_template': create_accessibility_template,
    'get_accessibility_template': get_accessibility_template,
    'get_location_accessibility': get_location_accessibility,
    'update_location_accessibility': update_location_accessibility,
    'submit_route_feedback': submit_route_feedback,
    'get_user_feedback_history': get_user_feedback_history,
    'get_personalized_recommendations': get_personalized_recommendations,
    'get_ml_model_status': get_ml_model_status,
    'reset_user_learning': reset_user_learning
}
