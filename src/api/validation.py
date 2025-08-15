"""
API Validation and Error Handling
=================================

Provides comprehensive request validation, error handling decorators,
and response formatting utilities for the API endpoints.
"""

import json
import traceback
from typing import Dict, Any, Optional, Callable
from functools import wraps
from flask import request, jsonify, current_app
from jsonschema import validate, ValidationError
import logging

# Set up logging
logger = logging.getLogger(__name__)


class APIError(Exception):
    """Base exception for API-related errors."""
    
    def __init__(self, message: str, status_code: int = 400, details: Optional[Dict] = None):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class ValidationAPIError(APIError):
    """Exception for validation-related errors."""
    
    def __init__(self, message: str, field: str = None, value: Any = None):
        details = {}
        if field:
            details['field'] = field
        if value is not None:
            details['value'] = value
        super().__init__(message, 400, details)


class AuthenticationError(APIError):
    """Exception for authentication-related errors."""
    
    def __init__(self, message: str = "Authentication required"):
        super().__init__(message, 401)


class AuthorizationError(APIError):
    """Exception for authorization-related errors."""
    
    def __init__(self, message: str = "Insufficient permissions"):
        super().__init__(message, 403)


class NotFoundError(APIError):
    """Exception for resource not found errors."""
    
    def __init__(self, resource: str, identifier: str = None):
        message = f"{resource} not found"
        if identifier:
            message += f": {identifier}"
        super().__init__(message, 404)


class RateLimitError(APIError):
    """Exception for rate limit exceeded errors."""
    
    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(message, 429)


class ServerError(APIError):
    """Exception for internal server errors."""
    
    def __init__(self, message: str = "Internal server error"):
        super().__init__(message, 500)


def validate_request_data(schema: Dict[str, Any]):
    """
    Decorator to validate incoming JSON request data against a schema.
    
    Args:
        schema: JSON schema to validate against
    """
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not request.is_json:
                return jsonify({
                    "error": "Content-Type must be application/json",
                    "error_type": "validation_error"
                }), 400
            
            try:
                data = request.get_json()
                if data is None:
                    return jsonify({
                        "error": "Invalid JSON data",
                        "error_type": "validation_error"
                    }), 400
                
                # Validate against schema
                validate(instance=data, schema=schema)
                
            except ValidationError as e:
                logger.warning(f"Validation error: {e.message}")
                return jsonify({
                    "error": f"Validation error: {e.message}",
                    "error_type": "validation_error",
                    "field_path": list(e.path) if e.path else None,
                    "invalid_value": e.instance if hasattr(e, 'instance') else None
                }), 400
                
            except json.JSONDecodeError as e:
                logger.warning(f"JSON decode error: {e}")
                return jsonify({
                    "error": "Invalid JSON format",
                    "error_type": "json_error",
                    "details": str(e)
                }), 400
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator


def validate_query_params(param_schema: Dict[str, Dict[str, Any]]):
    """
    Decorator to validate query parameters.
    
    Args:
        param_schema: Dictionary mapping parameter names to validation rules
                     Example: {"page": {"type": "integer", "minimum": 1}}
    """
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def decorated_function(*args, **kwargs):
            errors = []
            
            for param_name, rules in param_schema.items():
                value = request.args.get(param_name)
                
                # Check if required parameter is missing
                if rules.get('required', False) and value is None:
                    errors.append(f"Missing required parameter: {param_name}")
                    continue
                
                if value is None:
                    continue
                
                # Type validation
                param_type = rules.get('type', 'string')
                try:
                    if param_type == 'integer':
                        value = int(value)
                        if 'minimum' in rules and value < rules['minimum']:
                            errors.append(f"{param_name} must be at least {rules['minimum']}")
                        if 'maximum' in rules and value > rules['maximum']:
                            errors.append(f"{param_name} must be at most {rules['maximum']}")
                    elif param_type == 'float':
                        value = float(value)
                        if 'minimum' in rules and value < rules['minimum']:
                            errors.append(f"{param_name} must be at least {rules['minimum']}")
                        if 'maximum' in rules and value > rules['maximum']:
                            errors.append(f"{param_name} must be at most {rules['maximum']}")
                    elif param_type == 'boolean':
                        value = value.lower() in ('true', '1', 'yes', 'on')
                    elif param_type == 'string':
                        if 'enum' in rules and value not in rules['enum']:
                            errors.append(f"{param_name} must be one of: {', '.join(rules['enum'])}")
                        if 'min_length' in rules and len(value) < rules['min_length']:
                            errors.append(f"{param_name} must be at least {rules['min_length']} characters")
                        if 'max_length' in rules and len(value) > rules['max_length']:
                            errors.append(f"{param_name} must be at most {rules['max_length']} characters")
                    
                    # Store validated value
                    request.validated_params = getattr(request, 'validated_params', {})
                    request.validated_params[param_name] = value
                    
                except (ValueError, TypeError):
                    errors.append(f"{param_name} must be a valid {param_type}")
            
            if errors:
                return jsonify({
                    "error": "Parameter validation failed",
                    "error_type": "validation_error",
                    "details": errors
                }), 400
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator


def handle_api_errors(f: Callable) -> Callable:
    """
    Decorator to handle API exceptions and format error responses consistently.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
            
        except APIError as e:
            logger.warning(f"API error in {f.__name__}: {e.message}")
            response = {
                "error": e.message,
                "error_type": type(e).__name__.lower().replace('error', '_error'),
                "status_code": e.status_code
            }
            if e.details:
                response["details"] = e.details
            return jsonify(response), e.status_code
            
        except ValidationError as e:
            logger.warning(f"Validation error in {f.__name__}: {e.message}")
            return jsonify({
                "error": f"Validation error: {e.message}",
                "error_type": "validation_error",
                "field_path": list(e.path) if e.path else None
            }), 400
            
        except ValueError as e:
            logger.warning(f"Value error in {f.__name__}: {str(e)}")
            return jsonify({
                "error": str(e),
                "error_type": "value_error"
            }), 400
            
        except TypeError as e:
            logger.warning(f"Type error in {f.__name__}: {str(e)}")
            return jsonify({
                "error": "Invalid data type provided",
                "error_type": "type_error",
                "details": str(e)
            }), 400
            
        except KeyError as e:
            logger.warning(f"Key error in {f.__name__}: {str(e)}")
            return jsonify({
                "error": f"Missing required field: {str(e)}",
                "error_type": "missing_field_error"
            }), 400
            
        except Exception as e:
            logger.error(f"Unexpected error in {f.__name__}: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Don't expose internal errors in production
            if current_app.config.get('DEBUG', False):
                return jsonify({
                    "error": f"Internal server error: {str(e)}",
                    "error_type": "server_error",
                    "traceback": traceback.format_exc()
                }), 500
            else:
                return jsonify({
                    "error": "Internal server error",
                    "error_type": "server_error"
                }), 500
    
    return decorated_function


def validate_coordinates(lat: float, lon: float) -> bool:
    """
    Validate latitude and longitude coordinates.
    
    Args:
        lat: Latitude value
        lon: Longitude value
        
    Returns:
        True if coordinates are valid
        
    Raises:
        ValidationAPIError: If coordinates are invalid
    """
    if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
        raise ValidationAPIError("Coordinates must be numeric values")
    
    if not (-90 <= lat <= 90):
        raise ValidationAPIError("Latitude must be between -90 and 90", "lat", lat)
    
    if not (-180 <= lon <= 180):
        raise ValidationAPIError("Longitude must be between -180 and 180", "lon", lon)
    
    return True


def validate_distance(distance: float, max_distance: float = 100000) -> bool:
    """
    Validate distance values.
    
    Args:
        distance: Distance value to validate
        max_distance: Maximum allowed distance
        
    Returns:
        True if distance is valid
        
    Raises:
        ValidationAPIError: If distance is invalid
    """
    if not isinstance(distance, (int, float)):
        raise ValidationAPIError("Distance must be numeric", "distance", distance)
    
    if distance < 0:
        raise ValidationAPIError("Distance cannot be negative", "distance", distance)
    
    if distance > max_distance:
        raise ValidationAPIError(f"Distance exceeds maximum allowed: {max_distance}m", "distance", distance)
    
    return True


def validate_user_profile_data(profile_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and sanitize user profile data.
    
    Args:
        profile_data: Raw profile data to validate
        
    Returns:
        Validated and sanitized profile data
        
    Raises:
        ValidationAPIError: If profile data is invalid
    """
    validated = {}
    
    # Validate mobility aid
    if 'mobility_aid' in profile_data:
        valid_aids = ["none", "wheelchair", "mobility_scooter", "walker", "cane", "crutches", "prosthetic"]
        if profile_data['mobility_aid'] not in valid_aids:
            raise ValidationAPIError("Invalid mobility aid", "mobility_aid", profile_data['mobility_aid'])
        validated['mobility_aid'] = profile_data['mobility_aid']
    
    # Validate constraints
    if 'constraints' in profile_data:
        constraints = profile_data['constraints']
        validated_constraints = {}
        
        if 'max_walking_distance' in constraints:
            distance = constraints['max_walking_distance']
            if not isinstance(distance, (int, float)) or distance < 0:
                raise ValidationAPIError("Invalid walking distance", "max_walking_distance", distance)
            validated_constraints['max_walking_distance'] = float(distance)
        
        if 'max_slope_percent' in constraints:
            slope = constraints['max_slope_percent']
            if not isinstance(slope, (int, float)) or not (0 <= slope <= 100):
                raise ValidationAPIError("Slope must be between 0-100%", "max_slope_percent", slope)
            validated_constraints['max_slope_percent'] = float(slope)
        
        # Boolean constraints
        for field in ['requires_elevator', 'requires_ramp', 'avoids_stairs', 'needs_rest_areas']:
            if field in constraints:
                validated_constraints[field] = bool(constraints[field])
        
        validated['constraints'] = validated_constraints
    
    # Validate preferences
    if 'preferences' in profile_data:
        preferences = profile_data['preferences']
        validated_preferences = {}
        
        if 'walking_speed_factor' in preferences:
            speed = preferences['walking_speed_factor']
            if not isinstance(speed, (int, float)) or not (0.1 <= speed <= 3.0):
                raise ValidationAPIError("Speed factor must be between 0.1-3.0", "walking_speed_factor", speed)
            validated_preferences['walking_speed_factor'] = float(speed)
        
        # Array preferences
        for field in ['surface_preferences', 'weather_considerations', 'time_preferences']:
            if field in preferences:
                if not isinstance(preferences[field], list):
                    raise ValidationAPIError(f"{field} must be an array", field, preferences[field])
                validated_preferences[field] = preferences[field]
        
        if 'route_complexity_preference' in preferences:
            complexity = preferences['route_complexity_preference']
            if complexity not in ['simple', 'balanced', 'complex']:
                raise ValidationAPIError("Invalid complexity preference", "route_complexity_preference", complexity)
            validated_preferences['route_complexity_preference'] = complexity
        
        validated['preferences'] = validated_preferences
    
    # Boolean settings
    for field in ['learning_enabled', 'data_sharing_consent']:
        if field in profile_data:
            validated[field] = bool(profile_data[field])
    
    return validated


def format_success_response(data: Any, message: str = None, status_code: int = 200) -> tuple:
    """
    Format a successful API response.
    
    Args:
        data: Response data
        message: Optional success message
        status_code: HTTP status code
        
    Returns:
        Tuple of (response_dict, status_code)
    """
    response = {
        "success": True,
        "data": data
    }
    
    if message:
        response["message"] = message
    
    return jsonify(response), status_code


def format_error_response(error_message: str, error_type: str = "error", 
                         details: Dict = None, status_code: int = 400) -> tuple:
    """
    Format an error API response.
    
    Args:
        error_message: Error message
        error_type: Type of error
        details: Additional error details
        status_code: HTTP status code
        
    Returns:
        Tuple of (response_dict, status_code)
    """
    response = {
        "success": False,
        "error": error_message,
        "error_type": error_type
    }
    
    if details:
        response["details"] = details
    
    return jsonify(response), status_code


# Common query parameter schemas
PAGINATION_PARAMS = {
    "page": {"type": "integer", "minimum": 1, "default": 1},
    "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 20}
}

LOCATION_PARAMS = {
    "lat": {"type": "float", "required": True, "minimum": -90, "maximum": 90},
    "lon": {"type": "float", "required": True, "minimum": -180, "maximum": 180},
    "radius": {"type": "float", "minimum": 0, "maximum": 50000, "default": 1000}
}

SORTING_PARAMS = {
    "sort_by": {"type": "string", "enum": ["created_at", "updated_at", "distance", "rating"]},
    "sort_order": {"type": "string", "enum": ["asc", "desc"], "default": "desc"}
}
