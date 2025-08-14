"""
Sample data generation service for Smart Accessible Routing System
"""
import logging
import random
import math
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import json

# Get logger
logger = logging.getLogger(__name__)

class SampleDataGenerator:
    """Service for generating sample data for testing and development"""
    
    def __init__(self):
        """Initialize the sample data generator"""
        self.node_id_counter = 1000000  # Start with high IDs to avoid conflicts
        self.edge_id_counter = 2000000
        self.user_id_counter = 3000000
        self.report_id_counter = 4000000
        self.feedback_id_counter = 5000000
    
    def generate_sample_nodes(self, count: int = 10, center_lat: float = 19.0760, 
                              center_lng: float = 73.0760, radius_km: float = 2.0) -> List[Dict]:
        """
        Generate sample map nodes
        
        Args:
            count: Number of nodes to generate
            center_lat: Center latitude
            center_lng: Center longitude
            radius_km: Radius in kilometers
            
        Returns:
            List[Dict]: List of node data dictionaries
        """
        try:
            logger.info(f"Generating {count} sample nodes")
            nodes = []
            
            for i in range(count):
                # Generate random coordinates within radius
                lat, lng = self._generate_random_coordinates(center_lat, center_lng, radius_km)
                
                # Generate node data
                node_data = {
                    'osm_id': self.node_id_counter,
                    'latitude': lat,
                    'longitude': lng,
                    'elevation': random.uniform(0, 50),  # Elevation 0-50 meters
                    'has_ramp': random.random() < 0.3,  # 30% have ramps
                    'has_elevator': random.random() < 0.1,  # 10% have elevators
                    'has_rest_area': random.random() < 0.2,  # 20% have rest areas
                    'has_accessible_toilet': random.random() < 0.15,  # 15% have accessible toilets
                    'created_at': datetime.utcnow().isoformat()
                }
                
                nodes.append(node_data)
                self.node_id_counter += 1
            
            logger.info(f"Generated {len(nodes)} sample nodes")
            return nodes
            
        except Exception as e:
            logger.error(f"Error generating sample nodes: {e}")
            return []
    
    def generate_sample_edges(self, nodes: List[Dict], count: int = 15) -> List[Dict]:
        """
        Generate sample map edges connecting nodes
        
        Args:
            nodes: List of node data dictionaries
            count: Number of edges to generate
            
        Returns:
            List[Dict]: List of edge data dictionaries
        """
        try:
            logger.info(f"Generating {count} sample edges")
            edges = []
            
            if len(nodes) < 2:
                logger.warning("Not enough nodes to generate edges")
                return edges
            
            # Create connections between random nodes
            for i in range(count):
                # Select two random nodes
                node1 = random.choice(nodes)
                node2 = random.choice(nodes)
                
                # Ensure nodes are different
                attempts = 0
                while node1['osm_id'] == node2['osm_id'] and attempts < 10:
                    node2 = random.choice(nodes)
                    attempts += 1
                
                if node1['osm_id'] == node2['osm_id']:
                    continue  # Skip if we can't find different nodes
                
                # Calculate distance between nodes (simplified)
                distance = self._calculate_distance(
                    node1['latitude'], node1['longitude'],
                    node2['latitude'], node2['longitude']
                )
                
                # Generate edge data
                edge_data = {
                    'osm_id': self.edge_id_counter,
                    'start_node_id': node1['osm_id'],
                    'end_node_id': node2['osm_id'],
                    'highway_type': random.choice(['footway', 'path', 'pedestrian', 'residential']),
                    'name': f"Sample Road {self.edge_id_counter}",
                    'length_meters': distance,
                    'surface_type': random.choice(['asphalt', 'concrete', 'paved', 'gravel', 'grass']),
                    'smoothness': random.choice(['excellent', 'good', 'average', 'bad']),
                    'width_meters': random.uniform(0.5, 5.0),
                    'avg_slope_degrees': random.uniform(0, 10),
                    'max_slope_degrees': random.uniform(0, 15),
                    'wheelchair_accessible': random.random() < 0.8,
                    'has_steps': random.random() < 0.1,
                    'has_kerb': random.random() < 0.3,
                    'has_barrier': random.random() < 0.05,
                    'energy_cost': random.uniform(0.5, 3.0),
                    'comfort_score': random.uniform(0.3, 1.0),
                    'is_blocked': random.random() < 0.05,
                    'blockage_reason': random.choice(['construction', 'accident', 'maintenance', None]),
                    'created_at': datetime.utcnow().isoformat()
                }
                
                # Add barrier details occasionally
                if edge_data['has_barrier']:
                    edge_data['barrier_details'] = json.dumps({
                        'type': random.choice(['gate', 'bollard', 'fence', 'wall']),
                        'height': random.uniform(0.5, 2.0)
                    })
                else:
                    edge_data['barrier_details'] = '{}'
                
                edges.append(edge_data)
                self.edge_id_counter += 1
            
            logger.info(f"Generated {len(edges)} sample edges")
            return edges
            
        except Exception as e:
            logger.error(f"Error generating sample edges: {e}")
            return []
    
    def generate_sample_users(self, count: int = 5) -> List[Dict]:
        """
        Generate sample user data
        
        Args:
            count: Number of users to generate
            
        Returns:
            List[Dict]: List of user data dictionaries
        """
        try:
            logger.info(f"Generating {count} sample users")
            users = []
            
            # Sample usernames and emails
            usernames = ['alice', 'bob', 'charlie', 'diana', 'eve', 'frank', 'grace', 'henry']
            domains = ['example.com', 'test.org', 'sample.net']
            
            for i in range(min(count, len(usernames))):
                username = usernames[i]
                email = f"{username}@{random.choice(domains)}"
                
                # Generate user data
                user_data = {
                    'id': self.user_id_counter,
                    'username': username,
                    'email': email,
                    'password': 'password123',  # This will be hashed later
                    'first_name': username.capitalize(),
                    'last_name': f"User{self.user_id_counter}",
                    'mobility_aid_type': random.choice(['wheelchair', 'walker', 'cane', 'none']),
                    'max_slope_degrees': random.uniform(3.0, 10.0),
                    'min_path_width': random.uniform(0.7, 1.5),
                    'avoid_stairs': random.random() < 0.8,
                    'distance_weight': random.uniform(0.2, 0.5),
                    'energy_efficiency_weight': random.uniform(0.2, 0.5),
                    'comfort_weight': random.uniform(0.2, 0.5),
                    'created_at': datetime.utcnow().isoformat()
                }
                
                # Normalize weights to sum to 1.0
                total_weight = (user_data['distance_weight'] + 
                              user_data['energy_efficiency_weight'] + 
                              user_data['comfort_weight'])
                
                user_data['distance_weight'] /= total_weight
                user_data['energy_efficiency_weight'] /= total_weight
                user_data['comfort_weight'] /= total_weight
                
                # Add surface preferences
                surfaces = ['asphalt', 'concrete', 'paved', 'gravel', 'grass']
                preferences = {}
                for surface in surfaces:
                    preferences[surface] = random.uniform(0.5, 1.0)
                user_data['surface_preferences'] = json.dumps(preferences)
                
                users.append(user_data)
                self.user_id_counter += 1
            
            logger.info(f"Generated {len(users)} sample users")
            return users
            
        except Exception as e:
            logger.error(f"Error generating sample users: {e}")
            return []
    
    def generate_sample_obstacle_reports(self, users: List[Dict], nodes: List[Dict], count: int = 8) -> List[Dict]:
        """
        Generate sample obstacle reports
        
        Args:
            users: List of user data dictionaries
            nodes: List of node data dictionaries
            count: Number of reports to generate
            
        Returns:
            List[Dict]: List of obstacle report data dictionaries
        """
        try:
            logger.info(f"Generating {count} sample obstacle reports")
            reports = []
            
            if not users or not nodes:
                logger.warning("No users or nodes provided for obstacle reports")
                return reports
            
            obstacle_types = ['blocked', 'damaged', 'construction', 'missing_curb_ramp', 'other']
            severities = ['low', 'medium', 'high', 'critical']
            statuses = ['reported', 'verified', 'resolved', 'false_alarm']
            
            for i in range(count):
                # Select random user and node
                user = random.choice(users)
                node = random.choice(nodes)
                
                # Generate report data
                report_data = {
                    'id': self.report_id_counter,
                    'user_id': user['id'],
                    'latitude': node['latitude'] + random.uniform(-0.001, 0.001),
                    'longitude': node['longitude'] + random.uniform(-0.001, 0.001),
                    'obstacle_type': random.choice(obstacle_types),
                    'description': f"Sample obstacle report {self.report_id_counter}",
                    'severity': random.choice(severities),
                    'status': random.choice(statuses),
                    'created_at': datetime.utcnow().isoformat()
                }
                
                reports.append(report_data)
                self.report_id_counter += 1
            
            logger.info(f"Generated {len(reports)} sample obstacle reports")
            return reports
            
        except Exception as e:
            logger.error(f"Error generating sample obstacle reports: {e}")
            return []
    
    def generate_sample_route_feedback(self, users: List[Dict], count: int = 10) -> List[Dict]:
        """
        Generate sample route feedback
        
        Args:
            users: List of user data dictionaries
            count: Number of feedback entries to generate
            
        Returns:
            List[Dict]: List of route feedback data dictionaries
        """
        try:
            logger.info(f"Generating {count} sample route feedback entries")
            feedbacks = []
            
            if not users:
                logger.warning("No users provided for route feedback")
                return feedbacks
            
            for i in range(count):
                # Select random user
                user = random.choice(users)
                
                # Generate random start/end points
                start_lat = random.uniform(19.0, 19.2)
                start_lng = random.uniform(73.0, 73.2)
                end_lat = random.uniform(19.0, 19.2)
                end_lng = random.uniform(73.0, 73.2)
                
                # Generate feedback data
                feedback_data = {
                    'id': self.feedback_id_counter,
                    'user_id': user['id'],
                    'start_lat': start_lat,
                    'start_lng': start_lng,
                    'end_lat': end_lat,
                    'end_lng': end_lng,
                    'accessibility_score': random.randint(1, 5),
                    'comfort_score': random.randint(1, 5),
                    'accuracy_score': random.randint(1, 5),
                    'overall_score': random.randint(1, 5),
                    'comments': f"Sample feedback {self.feedback_id_counter}",
                    'created_at': datetime.utcnow().isoformat()
                }
                
                # Add some issues occasionally
                if random.random() < 0.3:
                    issues = ['steep_slope', 'narrow_path', 'construction', 'missing_curb_ramp']
                    selected_issues = random.sample(issues, random.randint(1, 3))
                    feedback_data['issues_encountered'] = json.dumps(selected_issues)
                else:
                    feedback_data['issues_encountered'] = '[]'
                
                feedbacks.append(feedback_data)
                self.feedback_id_counter += 1
            
            logger.info(f"Generated {len(feedbacks)} sample route feedback entries")
            return feedbacks
            
        except Exception as e:
            logger.error(f"Error generating sample route feedback: {e}")
            return []
    
    def generate_complete_sample_dataset(self, node_count: int = 15, edge_count: int = 20, 
                                         user_count: int = 5, report_count: int = 8, 
                                         feedback_count: int = 10) -> Dict:
        """
        Generate a complete sample dataset with all data types
        
        Args:
            node_count: Number of nodes to generate
            edge_count: Number of edges to generate
            user_count: Number of users to generate
            report_count: Number of obstacle reports to generate
            feedback_count: Number of route feedback entries to generate
            
        Returns:
            Dict: Complete dataset with all data types
        """
        try:
            logger.info("Generating complete sample dataset")
            
            # Generate nodes
            nodes = self.generate_sample_nodes(node_count)
            
            # Generate edges
            edges = self.generate_sample_edges(nodes, edge_count)
            
            # Generate users
            users = self.generate_sample_users(user_count)
            
            # Generate obstacle reports
            reports = self.generate_sample_obstacle_reports(users, nodes, report_count)
            
            # Generate route feedback
            feedbacks = self.generate_sample_route_feedback(users, feedback_count)
            
            dataset = {
                'nodes': nodes,
                'edges': edges,
                'users': users,
                'obstacle_reports': reports,
                'route_feedbacks': feedbacks
            }
            
            logger.info("Complete sample dataset generated successfully")
            return dataset
            
        except Exception as e:
            logger.error(f"Error generating complete sample dataset: {e}")
            return {}
    
    def _generate_random_coordinates(self, center_lat: float, center_lng: float, radius_km: float) -> Tuple[float, float]:
        """
        Generate random coordinates within a radius of a center point
        
        Args:
            center_lat: Center latitude
            center_lng: Center longitude
            radius_km: Radius in kilometers
            
        Returns:
            Tuple[float, float]: (latitude, longitude)
        """
        # Convert radius from km to degrees (approximate)
        radius_deg = radius_km / 111.0
        
        # Generate random angle and distance
        angle = random.uniform(0, 2 * math.pi)
        distance = random.uniform(0, radius_deg)
        
        # Calculate new coordinates
        lat = center_lat + distance * math.cos(angle)
        lng = center_lng + distance * math.sin(angle) / math.cos(math.radians(center_lat))
        
        return lat, lng
    
    def _calculate_distance(self, lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        """
        Calculate distance between two points in meters (simplified)
        
        Args:
            lat1: Latitude of point 1
            lng1: Longitude of point 1
            lat2: Latitude of point 2
            lng2: Longitude of point 2
            
        Returns:
            float: Distance in meters
        """
        # Simplified distance calculation
        # In a real implementation, we would use Haversine formula or similar
        lat_diff = abs(lat1 - lat2) * 111000  # Convert to meters
        lng_diff = abs(lng1 - lng2) * 111000 * math.cos(math.radians((lat1 + lat2) / 2))
        distance = math.sqrt(lat_diff**2 + lng_diff**2)
        return distance

# Example usage
if __name__ == "__main__":
    # Create sample data generator
    generator = SampleDataGenerator()
    
    # Generate complete sample dataset
    dataset = generator.generate_complete_sample_dataset()
    
    # Print summary
    print("Sample Dataset Summary:")
    print(f"  Nodes: {len(dataset.get('nodes', []))}")
    print(f"  Edges: {len(dataset.get('edges', []))}")
    print(f"  Users: {len(dataset.get('users', []))}")
    print(f"  Obstacle Reports: {len(dataset.get('obstacle_reports', []))}")
    print(f"  Route Feedback: {len(dataset.get('route_feedbacks', []))}")