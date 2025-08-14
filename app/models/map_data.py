"""
Map data models for Smart Accessible Routing System
"""
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Column, Integer, BigInteger, Float, String, Boolean, DateTime, Text, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
import json

# Import db from app package
try:
    from app import db
except ImportError:
    # For testing purposes, create a dummy db object
    from flask_sqlalchemy import SQLAlchemy
    db = SQLAlchemy()

class MapNode(db.Model):
    """
    Represents a node (intersection or point) in the map graph
    """
    __tablename__ = 'map_nodes'
    
    id = Column(Integer, primary_key=True)
    osm_id = Column(BigInteger, unique=True, nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    elevation = Column(Float, nullable=True)
    
    # Accessibility attributes
    has_ramp = Column(Boolean, default=False)
    has_elevator = Column(Boolean, default=False)
    has_rest_area = Column(Boolean, default=False)
    has_accessible_toilet = Column(Boolean, default=False)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship to edges
    outgoing_edges = relationship("MapEdge", foreign_keys="MapEdge.start_node_id")
    incoming_edges = relationship("MapEdge", foreign_keys="MapEdge.end_node_id")
    
    def __repr__(self):
        return f'<MapNode {self.osm_id}>'

class MapEdge(db.Model):
    """
    Represents an edge (road/path segment) in the map graph with accessibility attributes
    """
    __tablename__ = 'map_edges'
    
    id = Column(Integer, primary_key=True)
    osm_id = Column(BigInteger, unique=True, nullable=False)
    start_node_id = Column(Integer, ForeignKey('map_nodes.id'), nullable=False)
    end_node_id = Column(Integer, ForeignKey('map_nodes.id'), nullable=False)
    
    # Basic attributes
    highway_type = Column(String(50), nullable=True)  # footway, residential, etc.
    name = Column(String(200), nullable=True)
    length_meters = Column(Float, nullable=False)
    
    # Surface and quality attributes
    surface_type = Column(String(50), nullable=True)  # asphalt, concrete, gravel, etc.
    smoothness = Column(String(50), nullable=True)  # excellent, good, bad, etc.
    width_meters = Column(Float, nullable=True)
    
    # Slope and elevation
    avg_slope_degrees = Column(Float, nullable=True)
    max_slope_degrees = Column(Float, nullable=True)
    
    # Accessibility attributes
    wheelchair_accessible = Column(Boolean, default=True)
    has_steps = Column(Boolean, default=False)
    has_kerb = Column(Boolean, default=False)
    has_barrier = Column(Boolean, default=False)
    
    # Barrier details (stored as JSON)
    barrier_details = Column(Text, default='{}')
    
    # Energy and comfort metrics
    energy_cost = Column(Float, default=1.0)  # Relative energy cost
    comfort_score = Column(Float, default=1.0)  # Comfort score (0-1)
    
    # Real-time attributes
    is_blocked = Column(Boolean, default=False)
    blockage_reason = Column(String(200), nullable=True)
    last_verified = Column(DateTime, default=datetime.utcnow)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    start_node = relationship('MapNode', foreign_keys=[start_node_id])
    end_node = relationship('MapNode', foreign_keys=[end_node_id])
    
    def get_barrier_details(self):
        """Get barrier details as dictionary"""
        try:
            return json.loads(self.barrier_details)
        except (json.JSONDecodeError, TypeError):
            return {}
    
    def set_barrier_details(self, details):
        """Set barrier details from dictionary"""
        self.barrier_details = json.dumps(details)
    
    def get_accessibility_score(self, user_profile=None):
        """
        Calculate accessibility score for a specific user profile
        Returns a score between 0 (inaccessible) and 1 (fully accessible)
        """
        score = 1.0
        
        # If no user profile, return basic accessibility score
        if user_profile is None:
            return score
        
        # Check width requirements
        if user_profile.get('min_path_width') and self.width_meters:
            if self.width_meters < user_profile['min_path_width']:
                score *= 0.5
        
        # Check slope requirements
        if user_profile.get('max_slope_degrees') and self.max_slope_degrees:
            if self.max_slope_degrees > user_profile['max_slope_degrees']:
                score *= 0.3
        
        # Check for steps
        if self.has_steps and user_profile.get('avoid_stairs', True):
            score *= 0.1
        
        # Check for barriers
        if self.has_barrier:
            score *= 0.2
        
        # Check wheelchair accessibility
        if user_profile.get('mobility_aid_type') == 'wheelchair' and not self.wheelchair_accessible:
            score *= 0.1
        
        # Check if blocked
        if self.is_blocked:
            score *= 0.0
        
        return max(0.0, score)
    
    def __repr__(self):
        return f'<MapEdge {self.osm_id}: {self.start_node_id} -> {self.end_node_id}>'

class ObstacleReport(db.Model):
    """
    User-reported obstacles and accessibility issues
    """
    __tablename__ = 'obstacle_reports'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    edge_id = Column(Integer, ForeignKey('map_edges.id'), nullable=True)
    node_id = Column(Integer, ForeignKey('map_nodes.id'), nullable=True)
    
    # Location
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    
    # Obstacle details
    obstacle_type = Column(String(50), nullable=False)  # blocked, damaged, construction, etc.
    description = Column(Text, nullable=True)
    severity = Column(String(20), default='medium')  # low, medium, high, critical
    
    # Status
    status = Column(String(20), default='reported')  # reported, verified, resolved, false_alarm
    verified_by = Column(Integer, ForeignKey('users.id'), nullable=True)
    verified_at = Column(DateTime, nullable=True)
    
    # Media
    photo_url = Column(String(500), nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    # Note: User model will be defined in user.py
    # user = relationship('User', foreign_keys=[user_id])
    # edge = relationship('MapEdge')
    # node = relationship('MapNode')
    
    def __repr__(self):
        return f'<ObstacleReport {self.obstacle_type} at ({self.latitude}, {self.longitude})>'

class RouteFeedback(db.Model):
    """
    User feedback on completed routes
    """
    __tablename__ = 'route_feedbacks'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    
    # Route details
    start_lat = Column(Float, nullable=False)
    start_lng = Column(Float, nullable=False)
    end_lat = Column(Float, nullable=False)
    end_lng = Column(Float, nullable=False)
    
    # Feedback scores (1-5 scale)
    accessibility_score = Column(Integer, nullable=False)
    comfort_score = Column(Integer, nullable=False)
    accuracy_score = Column(Integer, nullable=False)
    overall_score = Column(Integer, nullable=False)
    
    # Additional feedback
    comments = Column(Text, nullable=True)
    issues_encountered = Column(Text, nullable=True)  # JSON string of issues
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    # Note: User model will be defined in user.py
    # user = relationship('User')
    
    def get_issues_encountered(self):
        """Get issues encountered as list"""
        try:
            return json.loads(self.issues_encountered)
        except (json.JSONDecodeError, TypeError):
            return []
    
    def set_issues_encountered(self, issues):
        """Set issues encountered from list"""
        self.issues_encountered = json.dumps(issues)
    
    def __repr__(self):
        return f'<RouteFeedback {self.overall_score}/5 by {self.user_id}>'

# Helper functions for map data operations
def create_map_node(osm_id, latitude, longitude, **kwargs):
    """
    Create a new map node
    
    Args:
        osm_id: OpenStreetMap ID
        latitude: Latitude coordinate
        longitude: Longitude coordinate
        **kwargs: Additional attributes
        
    Returns:
        MapNode: New map node instance
    """
    node = MapNode(
        osm_id=osm_id,
        latitude=latitude,
        longitude=longitude,
        **kwargs
    )
    return node

def create_map_edge(osm_id, start_node_id, end_node_id, length_meters, **kwargs):
    """
    Create a new map edge
    
    Args:
        osm_id: OpenStreetMap ID
        start_node_id: Starting node ID
        end_node_id: Ending node ID
        length_meters: Length in meters
        **kwargs: Additional attributes
        
    Returns:
        MapEdge: New map edge instance
    """
    edge = MapEdge(
        osm_id=osm_id,
        start_node_id=start_node_id,
        end_node_id=end_node_id,
        length_meters=length_meters,
        **kwargs
    )
    return edge

def get_node_by_osm_id(osm_id):
    """
    Get a node by its OpenStreetMap ID
    
    Args:
        osm_id: OpenStreetMap ID
        
    Returns:
        MapNode: Node with matching OSM ID or None
    """
    return MapNode.query.filter_by(osm_id=osm_id).first()

def get_edge_by_osm_id(osm_id):
    """
    Get an edge by its OpenStreetMap ID
    
    Args:
        osm_id: OpenStreetMap ID
        
    Returns:
        MapEdge: Edge with matching OSM ID or None
    """
    return MapEdge.query.filter_by(osm_id=osm_id).first()