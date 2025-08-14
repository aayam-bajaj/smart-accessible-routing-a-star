"""
Data preprocessing service for Smart Accessible Routing System
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from datetime import datetime
import json

# Import other services
try:
    from app.services.data_validator import DataValidator
except ImportError:
    # For testing purposes, create a dummy validator
    class DataValidator:
        def validate_map_node_data(self, data):
            return True, []
        def validate_map_edge_data(self, data):
            return True, []

# Get logger
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Service for preprocessing data in the Smart Accessible Routing System"""
    
    def __init__(self):
        """Initialize the data preprocessor"""
        self.validator = DataValidator()
    
    def preprocess_map_data(self, nodes_data: List[Dict], edges_data: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Preprocess map data (nodes and edges)
        
        Args:
            nodes_data: List of node data dictionaries
            edges_data: List of edge data dictionaries
            
        Returns:
            Tuple[List[Dict], List[Dict]]: (processed_nodes, processed_edges)
        """
        try:
            logger.info(f"Preprocessing map data: {len(nodes_data)} nodes, {len(edges_data)} edges")
            
            # Process nodes
            processed_nodes = []
            node_errors = []
            for i, node_data in enumerate(nodes_data):
                try:
                    # Validate node data
                    is_valid, errors = self.validator.validate_map_node_data(node_data)
                    if not is_valid:
                        node_errors.extend([f"Node {i}: {error}" for error in errors])
                        continue
                    
                    # Add timestamp if not present
                    if 'created_at' not in node_data:
                        node_data['created_at'] = datetime.utcnow().isoformat()
                    
                    processed_nodes.append(node_data)
                except Exception as e:
                    node_errors.append(f"Node {i}: Error processing node data: {e}")
            
            # Process edges
            processed_edges = []
            edge_errors = []
            for i, edge_data in enumerate(edges_data):
                try:
                    # Validate edge data
                    is_valid, errors = self.validator.validate_map_edge_data(edge_data)
                    if not is_valid:
                        edge_errors.extend([f"Edge {i}: {error}" for error in errors])
                        continue
                    
                    # Add timestamp if not present
                    if 'created_at' not in edge_data:
                        edge_data['created_at'] = datetime.utcnow().isoformat()
                    
                    # Calculate energy cost if not present
                    if 'energy_cost' not in edge_data or edge_data['energy_cost'] is None:
                        edge_data['energy_cost'] = self._calculate_energy_cost(edge_data)
                    
                    # Calculate comfort score if not present
                    if 'comfort_score' not in edge_data or edge_data['comfort_score'] is None:
                        edge_data['comfort_score'] = self._calculate_comfort_score(edge_data)
                    
                    processed_edges.append(edge_data)
                except Exception as e:
                    edge_errors.append(f"Edge {i}: Error processing edge data: {e}")
            
            # Log any errors
            if node_errors:
                logger.warning(f"Node processing errors: {len(node_errors)} errors")
                for error in node_errors:
                    logger.warning(f"  - {error}")
            
            if edge_errors:
                logger.warning(f"Edge processing errors: {len(edge_errors)} errors")
                for error in edge_errors:
                    logger.warning(f"  - {error}")
            
            logger.info(f"Preprocessed map data: {len(processed_nodes)} nodes, {len(processed_edges)} edges")
            return processed_nodes, processed_edges
            
        except Exception as e:
            logger.error(f"Error preprocessing map data: {e}")
            return [], []
    
    def _calculate_energy_cost(self, edge_data: Dict) -> float:
        """
        Calculate energy cost for an edge based on its attributes
        
        Args:
            edge_data: Edge data dictionary
            
        Returns:
            float: Energy cost (relative, 1.0 is baseline)
        """
        try:
            energy_cost = 1.0
            
            # Adjust for length (longer paths require more energy)
            if 'length_meters' in edge_data:
                length = edge_data['length_meters']
                # Normalize to 100m as baseline
                energy_cost *= (length / 100.0)
            
            # Adjust for slope (steeper slopes require more energy)
            if 'max_slope_degrees' in edge_data:
                slope = edge_data['max_slope_degrees']
                # Increase energy cost for slopes > 2 degrees
                if slope > 2:
                    energy_cost *= (1.0 + (slope - 2) / 10.0)
            
            # Adjust for surface type
            surface_multipliers = {
                'asphalt': 1.0,
                'concrete': 1.0,
                'paved': 1.1,
                'gravel': 1.3,
                'grass': 1.5,
                'dirt': 1.7,
                'unknown': 1.2
            }
            if 'surface_type' in edge_data:
                surface = edge_data['surface_type']
                energy_cost *= surface_multipliers.get(surface, 1.2)
            
            # Adjust for stairs (significant energy cost)
            if edge_data.get('has_steps', False):
                energy_cost *= 2.0
            
            return max(0.1, energy_cost)
            
        except Exception as e:
            logger.warning(f"Error calculating energy cost: {e}")
            return 1.0
    
    def _calculate_comfort_score(self, edge_data: Dict) -> float:
        """
        Calculate comfort score for an edge based on its attributes
        
        Args:
            edge_data: Edge data dictionary
            
        Returns:
            float: Comfort score (0.0-1.0, 1.0 is most comfortable)
        """
        try:
            comfort_score = 1.0
            
            # Adjust for surface type (smoother surfaces are more comfortable)
            surface_scores = {
                'asphalt': 1.0,
                'concrete': 0.9,
                'paved': 0.8,
                'gravel': 0.6,
                'grass': 0.5,
                'dirt': 0.4,
                'unknown': 0.7
            }
            if 'surface_type' in edge_data:
                surface = edge_data['surface_type']
                comfort_score *= surface_scores.get(surface, 0.7)
            
            # Adjust for width (wider paths are more comfortable)
            if 'width_meters' in edge_data:
                width = edge_data['width_meters']
                if width is not None:
                    # Normalize to 2m as comfortable width
                    comfort_score *= min(1.0, width / 2.0)
            
            # Adjust for slope (flatter is more comfortable)
            if 'max_slope_degrees' in edge_data:
                slope = edge_data['max_slope_degrees']
                if slope is not None:
                    # Comfort decreases significantly for slopes > 3 degrees
                    if slope > 3:
                        comfort_score *= max(0.1, 1.0 - (slope - 3) / 15.0)
            
            # Adjust for stairs (less comfortable)
            if edge_data.get('has_steps', False):
                comfort_score *= 0.3
            
            # Adjust for barriers
            if edge_data.get('has_barrier', False):
                comfort_score *= 0.5
            
            return max(0.0, min(1.0, comfort_score))
            
        except Exception as e:
            logger.warning(f"Error calculating comfort score: {e}")
            return 1.0
    
    def preprocess_user_data(self, user_data: Dict) -> Tuple[Dict, List[str]]:
        """
        Preprocess user data
        
        Args:
            user_data: User data dictionary
            
        Returns:
            Tuple[Dict, List[str]]: (processed_user_data, list_of_errors)
        """
        try:
            logger.info(f"Preprocessing user data for user: {user_data.get('username', 'unknown')}")
            
            # Validate user data
            is_valid, errors = self.validator.validate_user_data(user_data)
            if not is_valid:
                return user_data, errors
            
            # Add timestamp if not present
            if 'created_at' not in user_data:
                user_data['created_at'] = datetime.utcnow().isoformat()
            
            # Set default values for missing fields
            if 'mobility_aid_type' not in user_data or user_data['mobility_aid_type'] is None:
                user_data['mobility_aid_type'] = 'none'
            
            if 'max_slope_degrees' not in user_data or user_data['max_slope_degrees'] is None:
                user_data['max_slope_degrees'] = 8.0  # Default max slope
            
            if 'min_path_width' not in user_data or user_data['min_path_width'] is None:
                user_data['min_path_width'] = 0.8  # Default min width
            
            if 'avoid_stairs' not in user_data or user_data['avoid_stairs'] is None:
                user_data['avoid_stairs'] = True  # Default to avoiding stairs
            
            # Normalize weight fields to sum to 1.0
            weight_fields = ['distance_weight', 'energy_efficiency_weight', 'comfort_weight']
            total_weight = sum(user_data.get(field, 0) for field in weight_fields)
            if total_weight > 0:
                for field in weight_fields:
                    user_data[field] = user_data[field] / total_weight
            
            # Set default weights if all are zero
            if total_weight == 0:
                user_data['distance_weight'] = 0.3
                user_data['energy_efficiency_weight'] = 0.3
                user_data['comfort_weight'] = 0.4
            
            logger.info("User data preprocessed successfully")
            return user_data, []
            
        except Exception as e:
            logger.error(f"Error preprocessing user data: {e}")
            return user_data, [f"Error preprocessing user data: {e}"]
    
    def preprocess_obstacle_report_data(self, report_data: Dict) -> Tuple[Dict, List[str]]:
        """
        Preprocess obstacle report data
        
        Args:
            report_data: Obstacle report data dictionary
            
        Returns:
            Tuple[Dict, List[str]]: (processed_report_data, list_of_errors)
        """
        try:
            logger.info("Preprocessing obstacle report data")
            
            # Validate report data
            is_valid, errors = self.validator.validate_obstacle_report_data(report_data)
            if not is_valid:
                return report_data, errors
            
            # Add timestamp if not present
            if 'created_at' not in report_data:
                report_data['created_at'] = datetime.utcnow().isoformat()
            
            # Set default severity if not present
            if 'severity' not in report_data or report_data['severity'] is None:
                report_data['severity'] = 'medium'
            
            # Set default status if not present
            if 'status' not in report_data or report_data['status'] is None:
                report_data['status'] = 'reported'
            
            logger.info("Obstacle report data preprocessed successfully")
            return report_data, []
            
        except Exception as e:
            logger.error(f"Error preprocessing obstacle report data: {e}")
            return report_data, [f"Error preprocessing obstacle report data: {e}"]
    
    def preprocess_route_feedback_data(self, feedback_data: Dict) -> Tuple[Dict, List[str]]:
        """
        Preprocess route feedback data
        
        Args:
            feedback_data: Route feedback data dictionary
            
        Returns:
            Tuple[Dict, List[str]]: (processed_feedback_data, list_of_errors)
        """
        try:
            logger.info("Preprocessing route feedback data")
            
            # Validate feedback data
            is_valid, errors = self.validator.validate_route_feedback_data(feedback_data)
            if not is_valid:
                return feedback_data, errors
            
            # Add timestamp if not present
            if 'created_at' not in feedback_data:
                feedback_data['created_at'] = datetime.utcnow().isoformat()
            
            # Calculate overall score if not present
            if 'overall_score' not in feedback_data or feedback_data['overall_score'] is None:
                scores = [
                    feedback_data.get('accessibility_score', 0),
                    feedback_data.get('comfort_score', 0),
                    feedback_data.get('accuracy_score', 0)
                ]
                valid_scores = [s for s in scores if s is not None and s > 0]
                if valid_scores:
                    feedback_data['overall_score'] = sum(valid_scores) / len(valid_scores)
            
            logger.info("Route feedback data preprocessed successfully")
            return feedback_data, []
            
        except Exception as e:
            logger.error(f"Error preprocessing route feedback data: {e}")
            return feedback_data, [f"Error preprocessing route feedback data: {e}"]
    
    def clean_and_deduplicate_nodes(self, nodes_data: List[Dict]) -> List[Dict]:
        """
        Clean and deduplicate node data
        
        Args:
            nodes_data: List of node data dictionaries
            
        Returns:
            List[Dict]: Cleaned and deduplicated node data
        """
        try:
            logger.info(f"Cleaning and deduplicating {len(nodes_data)} nodes")
            
            # Convert to DataFrame for easier processing
            df = pd.DataFrame(nodes_data)
            
            # Remove duplicates based on OSM ID
            if 'osm_id' in df.columns:
                df = df.drop_duplicates(subset=['osm_id'], keep='first')
            
            # Remove nodes with invalid coordinates
            if 'latitude' in df.columns and 'longitude' in df.columns:
                df = df[
                    (df['latitude'] >= -90) & (df['latitude'] <= 90) &
                    (df['longitude'] >= -180) & (df['longitude'] <= 180)
                ]
            
            # Convert back to list of dictionaries
            cleaned_nodes = df.to_dict('records')
            
            logger.info(f"Cleaned and deduplicated nodes: {len(cleaned_nodes)} remaining")
            return cleaned_nodes
            
        except Exception as e:
            logger.error(f"Error cleaning and deduplicating nodes: {e}")
            return nodes_data
    
    def clean_and_deduplicate_edges(self, edges_data: List[Dict]) -> List[Dict]:
        """
        Clean and deduplicate edge data
        
        Args:
            edges_data: List of edge data dictionaries
            
        Returns:
            List[Dict]: Cleaned and deduplicated edge data
        """
        try:
            logger.info(f"Cleaning and deduplicating {len(edges_data)} edges")
            
            # Convert to DataFrame for easier processing
            df = pd.DataFrame(edges_data)
            
            # Remove duplicates based on OSM ID
            if 'osm_id' in df.columns:
                df = df.drop_duplicates(subset=['osm_id'], keep='first')
            
            # Remove edges with invalid node references
            if 'start_node_id' in df.columns and 'end_node_id' in df.columns:
                df = df[df['start_node_id'] != df['end_node_id']]
            
            # Remove edges with invalid lengths
            if 'length_meters' in df.columns:
                df = df[df['length_meters'] > 0]
            
            # Convert back to list of dictionaries
            cleaned_edges = df.to_dict('records')
            
            logger.info(f"Cleaned and deduplicated edges: {len(cleaned_edges)} remaining")
            return cleaned_edges
            
        except Exception as e:
            logger.error(f"Error cleaning and deduplicating edges: {e}")
            return edges_data

# Example usage
if __name__ == "__main__":
    # Create data preprocessor instance
    preprocessor = DataPreprocessor()
    
    # Example node data
    nodes_data = [
        {
            'osm_id': 123456789,
            'latitude': 19.0760,
            'longitude': 73.0760,
            'elevation': 10.5,
            'has_ramp': True,
            'has_elevator': False
        },
        {
            'osm_id': 987654321,
            'latitude': 19.0770,
            'longitude': 73.0770,
            'elevation': 12.0,
            'has_ramp': False,
            'has_elevator': True
        }
    ]
    
    # Example edge data
    edges_data = [
        {
            'osm_id': 111111111,
            'start_node_id': 1,
            'end_node_id': 2,
            'length_meters': 100.5,
            'surface_type': 'asphalt',
            'width_meters': 2.5,
            'max_slope_degrees': 2.1,
            'wheelchair_accessible': True,
            'has_steps': False
        }
    ]
    
    # Preprocess map data
    processed_nodes, processed_edges = preprocessor.preprocess_map_data(nodes_data, edges_data)
    print(f"Processed {len(processed_nodes)} nodes and {len(processed_edges)} edges")
    
    # Example user data
    user_data = {
        'username': 'testuser',
        'email': 'test@example.com',
        'password': 'password123',
        'mobility_aid_type': 'wheelchair',
        'max_slope_degrees': 5.0,
        'min_path_width': 0.8,
        'avoid_stairs': True
    }
    
    # Preprocess user data
    processed_user, errors = preprocessor.preprocess_user_data(user_data)
    if errors:
        print("User data validation errors:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("User data preprocessed successfully")