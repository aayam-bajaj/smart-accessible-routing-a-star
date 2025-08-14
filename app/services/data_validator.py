"""
Data validation service for Smart Accessible Routing System
"""
import logging
import re
from typing import Dict, List, Tuple, Any
from datetime import datetime

# Get logger
logger = logging.getLogger(__name__)

class DataValidator:
    """Service for validating data in the Smart Accessible Routing System"""
    
    def __init__(self):
        """Initialize the data validator"""
        pass
    
    def validate_map_node_data(self, node_data: Dict) -> Tuple[bool, List[str]]:
        """
        Validate map node data
        
        Args:
            node_data: Dictionary containing node data
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_errors)
        """
        errors = []
        
        try:
            # Check required fields
            required_fields = ['osm_id', 'latitude', 'longitude']
            for field in required_fields:
                if field not in node_data or node_data[field] is None:
                    errors.append(f"Missing required field: {field}")
            
            # Validate OSM ID
            if 'osm_id' in node_data:
                if not isinstance(node_data['osm_id'], int) or node_data['osm_id'] <= 0:
                    errors.append("OSM ID must be a positive integer")
            
            # Validate latitude
            if 'latitude' in node_data:
                try:
                    lat = float(node_data['latitude'])
                    if lat < -90 or lat > 90:
                        errors.append("Latitude must be between -90 and 90")
                except (ValueError, TypeError):
                    errors.append("Latitude must be a valid number")
            
            # Validate longitude
            if 'longitude' in node_data:
                try:
                    lng = float(node_data['longitude'])
                    if lng < -180 or lng > 180:
                        errors.append("Longitude must be between -180 and 180")
                except (ValueError, TypeError):
                    errors.append("Longitude must be a valid number")
            
            # Validate elevation if provided
            if 'elevation' in node_data and node_data['elevation'] is not None:
                try:
                    elevation = float(node_data['elevation'])
                    if elevation < -1000 or elevation > 10000:
                        errors.append("Elevation must be between -1000 and 10000 meters")
                except (ValueError, TypeError):
                    errors.append("Elevation must be a valid number")
            
            # Validate boolean fields
            boolean_fields = ['has_ramp', 'has_elevator', 'has_rest_area', 'has_accessible_toilet']
            for field in boolean_fields:
                if field in node_data and node_data[field] is not None:
                    if not isinstance(node_data[field], bool):
                        errors.append(f"{field} must be a boolean value")
            
            is_valid = len(errors) == 0
            return is_valid, errors
            
        except Exception as e:
            logger.error(f"Error validating map node data: {e}")
            return False, [f"Error validating map node data: {e}"]
    
    def validate_map_edge_data(self, edge_data: Dict) -> Tuple[bool, List[str]]:
        """
        Validate map edge data
        
        Args:
            edge_data: Dictionary containing edge data
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_errors)
        """
        errors = []
        
        try:
            # Check required fields
            required_fields = ['osm_id', 'start_node_id', 'end_node_id', 'length_meters']
            for field in required_fields:
                if field not in edge_data or edge_data[field] is None:
                    errors.append(f"Missing required field: {field}")
            
            # Validate OSM ID
            if 'osm_id' in edge_data:
                if not isinstance(edge_data['osm_id'], int) or edge_data['osm_id'] <= 0:
                    errors.append("OSM ID must be a positive integer")
            
            # Validate node IDs
            node_fields = ['start_node_id', 'end_node_id']
            for field in node_fields:
                if field in edge_data:
                    if not isinstance(edge_data[field], int) or edge_data[field] <= 0:
                        errors.append(f"{field} must be a positive integer")
            
            # Validate length
            if 'length_meters' in edge_data:
                try:
                    length = float(edge_data['length_meters'])
                    if length <= 0:
                        errors.append("Length must be a positive number")
                    elif length > 100000:
                        errors.append("Length must be less than 100km")
                except (ValueError, TypeError):
                    errors.append("Length must be a valid number")
            
            # Validate surface type if provided
            if 'surface_type' in edge_data and edge_data['surface_type'] is not None:
                valid_surfaces = ['asphalt', 'concrete', 'paved', 'gravel', 'grass', 'dirt', 'unknown']
                if edge_data['surface_type'] not in valid_surfaces:
                    errors.append(f"Surface type must be one of: {', '.join(valid_surfaces)}")
            
            # Validate width if provided
            if 'width_meters' in edge_data and edge_data['width_meters'] is not None:
                try:
                    width = float(edge_data['width_meters'])
                    if width <= 0:
                        errors.append("Width must be a positive number")
                    elif width > 100:
                        errors.append("Width must be less than 100 meters")
                except (ValueError, TypeError):
                    errors.append("Width must be a valid number")
            
            # Validate slope if provided
            slope_fields = ['avg_slope_degrees', 'max_slope_degrees']
            for field in slope_fields:
                if field in edge_data and edge_data[field] is not None:
                    try:
                        slope = float(edge_data[field])
                        if slope < 0 or slope > 90:
                            errors.append(f"{field} must be between 0 and 90 degrees")
                    except (ValueError, TypeError):
                        errors.append(f"{field} must be a valid number")
            
            # Validate boolean fields
            boolean_fields = [
                'wheelchair_accessible', 'has_steps', 'has_kerb', 'has_barrier', 'is_blocked'
            ]
            for field in boolean_fields:
                if field in edge_data and edge_data[field] is not None:
                    if not isinstance(edge_data[field], bool):
                        errors.append(f"{field} must be a boolean value")
            
            # Validate energy cost if provided
            if 'energy_cost' in edge_data and edge_data['energy_cost'] is not None:
                try:
                    energy_cost = float(edge_data['energy_cost'])
                    if energy_cost < 0:
                        errors.append("Energy cost must be a non-negative number")
                except (ValueError, TypeError):
                    errors.append("Energy cost must be a valid number")
            
            # Validate comfort score if provided
            if 'comfort_score' in edge_data and edge_data['comfort_score'] is not None:
                try:
                    comfort_score = float(edge_data['comfort_score'])
                    if comfort_score < 0 or comfort_score > 1:
                        errors.append("Comfort score must be between 0 and 1")
                except (ValueError, TypeError):
                    errors.append("Comfort score must be a valid number")
            
            is_valid = len(errors) == 0
            return is_valid, errors
            
        except Exception as e:
            logger.error(f"Error validating map edge data: {e}")
            return False, [f"Error validating map edge data: {e}"]
    
    def validate_user_data(self, user_data: Dict) -> Tuple[bool, List[str]]:
        """
        Validate user data
        
        Args:
            user_data: Dictionary containing user data
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_errors)
        """
        errors = []
        
        try:
            # Check required fields
            required_fields = ['username', 'email', 'password']
            for field in required_fields:
                if field not in user_data or not user_data[field]:
                    errors.append(f"Missing required field: {field}")
            
            # Validate username
            if 'username' in user_data and user_data['username']:
                username = user_data['username']
                if len(username) < 3:
                    errors.append("Username must be at least 3 characters long")
                elif len(username) > 80:
                    errors.append("Username must be less than 80 characters long")
                elif not re.match(r'^[a-zA-Z0-9_]+$', username):
                    errors.append("Username can only contain letters, numbers, and underscores")
            
            # Validate email
            if 'email' in user_data and user_data['email']:
                email = user_data['email']
                if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
                    errors.append("Email must be a valid email address")
            
            # Validate password
            if 'password' in user_data and user_data['password']:
                password = user_data['password']
                if len(password) < 6:
                    errors.append("Password must be at least 6 characters long")
            
            # Validate mobility aid type if provided
            if 'mobility_aid_type' in user_data and user_data['mobility_aid_type'] is not None:
                valid_aid_types = ['wheelchair', 'walker', 'cane', 'none']
                if user_data['mobility_aid_type'] not in valid_aid_types:
                    errors.append(f"Mobility aid type must be one of: {', '.join(valid_aid_types)}")
            
            # Validate slope degrees if provided
            if 'max_slope_degrees' in user_data and user_data['max_slope_degrees'] is not None:
                try:
                    slope = float(user_data['max_slope_degrees'])
                    if slope < 0 or slope > 45:
                        errors.append("Maximum slope must be between 0 and 45 degrees")
                except (ValueError, TypeError):
                    errors.append("Maximum slope must be a valid number")
            
            # Validate path width if provided
            if 'min_path_width' in user_data and user_data['min_path_width'] is not None:
                try:
                    width = float(user_data['min_path_width'])
                    if width < 0 or width > 10:
                        errors.append("Minimum path width must be between 0 and 10 meters")
                except (ValueError, TypeError):
                    errors.append("Minimum path width must be a valid number")
            
            # Validate boolean fields
            boolean_fields = ['avoid_stairs']
            for field in boolean_fields:
                if field in user_data and user_data[field] is not None:
                    if not isinstance(user_data[field], bool):
                        errors.append(f"{field} must be a boolean value")
            
            # Validate weight fields
            weight_fields = ['distance_weight', 'energy_efficiency_weight', 'comfort_weight']
            total_weight = 0
            for field in weight_fields:
                if field in user_data and user_data[field] is not None:
                    try:
                        weight = float(user_data[field])
                        if weight < 0 or weight > 1:
                            errors.append(f"{field} must be between 0 and 1")
                        total_weight += weight
                    except (ValueError, TypeError):
                        errors.append(f"{field} must be a valid number")
            
            # Check that weights sum to approximately 1
            if total_weight > 0 and abs(total_weight - 1.0) > 0.01:
                errors.append("Route preference weights must sum to 1.0")
            
            is_valid = len(errors) == 0
            return is_valid, errors
            
        except Exception as e:
            logger.error(f"Error validating user data: {e}")
            return False, [f"Error validating user data: {e}"]
    
    def validate_obstacle_report_data(self, report_data: Dict) -> Tuple[bool, List[str]]:
        """
        Validate obstacle report data
        
        Args:
            report_data: Dictionary containing report data
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_errors)
        """
        errors = []
        
        try:
            # Check required fields
            required_fields = ['user_id', 'latitude', 'longitude', 'obstacle_type']
            for field in required_fields:
                if field not in report_data or report_data[field] is None:
                    errors.append(f"Missing required field: {field}")
            
            # Validate user ID
            if 'user_id' in report_data:
                if not isinstance(report_data['user_id'], int) or report_data['user_id'] <= 0:
                    errors.append("User ID must be a positive integer")
            
            # Validate latitude
            if 'latitude' in report_data:
                try:
                    lat = float(report_data['latitude'])
                    if lat < -90 or lat > 90:
                        errors.append("Latitude must be between -90 and 90")
                except (ValueError, TypeError):
                    errors.append("Latitude must be a valid number")
            
            # Validate longitude
            if 'longitude' in report_data:
                try:
                    lng = float(report_data['longitude'])
                    if lng < -180 or lng > 180:
                        errors.append("Longitude must be between -180 and 180")
                except (ValueError, TypeError):
                    errors.append("Longitude must be a valid number")
            
            # Validate obstacle type
            if 'obstacle_type' in report_data and report_data['obstacle_type']:
                valid_types = ['blocked', 'damaged', 'construction', 'missing_curb_ramp', 'other']
                if report_data['obstacle_type'] not in valid_types:
                    errors.append(f"Obstacle type must be one of: {', '.join(valid_types)}")
            
            # Validate severity if provided
            if 'severity' in report_data and report_data['severity'] is not None:
                valid_severities = ['low', 'medium', 'high', 'critical']
                if report_data['severity'] not in valid_severities:
                    errors.append(f"Severity must be one of: {', '.join(valid_severities)}")
            
            is_valid = len(errors) == 0
            return is_valid, errors
            
        except Exception as e:
            logger.error(f"Error validating obstacle report data: {e}")
            return False, [f"Error validating obstacle report data: {e}"]
    
    def validate_route_feedback_data(self, feedback_data: Dict) -> Tuple[bool, List[str]]:
        """
        Validate route feedback data
        
        Args:
            feedback_data: Dictionary containing feedback data
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_errors)
        """
        errors = []
        
        try:
            # Check required fields
            required_fields = [
                'user_id', 'start_lat', 'start_lng', 'end_lat', 'end_lng',
                'accessibility_score', 'comfort_score', 'accuracy_score', 'overall_score'
            ]
            for field in required_fields:
                if field not in feedback_data or feedback_data[field] is None:
                    errors.append(f"Missing required field: {field}")
            
            # Validate user ID
            if 'user_id' in feedback_data:
                if not isinstance(feedback_data['user_id'], int) or feedback_data['user_id'] <= 0:
                    errors.append("User ID must be a positive integer")
            
            # Validate coordinates
            coord_fields = ['start_lat', 'end_lat']
            for field in coord_fields:
                if field in feedback_data:
                    try:
                        lat = float(feedback_data[field])
                        if lat < -90 or lat > 90:
                            errors.append(f"{field} must be between -90 and 90")
                    except (ValueError, TypeError):
                        errors.append(f"{field} must be a valid number")
            
            coord_fields = ['start_lng', 'end_lng']
            for field in coord_fields:
                if field in feedback_data:
                    try:
                        lng = float(feedback_data[field])
                        if lng < -180 or lng > 180:
                            errors.append(f"{field} must be between -180 and 180")
                    except (ValueError, TypeError):
                        errors.append(f"{field} must be a valid number")
            
            # Validate scores (1-5 scale)
            score_fields = ['accessibility_score', 'comfort_score', 'accuracy_score', 'overall_score']
            for field in score_fields:
                if field in feedback_data:
                    try:
                        score = int(feedback_data[field])
                        if score < 1 or score > 5:
                            errors.append(f"{field} must be between 1 and 5")
                    except (ValueError, TypeError):
                        errors.append(f"{field} must be a valid integer between 1 and 5")
            
            is_valid = len(errors) == 0
            return is_valid, errors
            
        except Exception as e:
            logger.error(f"Error validating route feedback data: {e}")
            return False, [f"Error validating route feedback data: {e}"]

# Example usage
if __name__ == "__main__":
    # Create data validator instance
    validator = DataValidator()
    
    # Example node data
    node_data = {
        'osm_id': 123456789,
        'latitude': 19.0760,
        'longitude': 73.0760,
        'elevation': 10.5,
        'has_ramp': True,
        'has_elevator': False
    }
    
    # Validate node data
    is_valid, errors = validator.validate_map_node_data(node_data)
    if is_valid:
        print("Map node data is valid")
    else:
        print("Map node data validation errors:")
        for error in errors:
            print(f"  - {error}")
    
    # Example edge data
    edge_data = {
        'osm_id': 987654321,
        'start_node_id': 1,
        'end_node_id': 2,
        'length_meters': 100.5,
        'surface_type': 'asphalt',
        'width_meters': 2.5,
        'max_slope_degrees': 2.1,
        'wheelchair_accessible': True,
        'has_steps': False,
        'energy_cost': 1.2,
        'comfort_score': 0.9
    }
    
    # Validate edge data
    is_valid, errors = validator.validate_map_edge_data(edge_data)
    if is_valid:
        print("Map edge data is valid")
    else:
        print("Map edge data validation errors:")
        for error in errors:
            print(f"  - {error}")