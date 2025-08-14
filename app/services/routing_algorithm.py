"""
Core routing algorithm implementation for Smart Accessible Routing System
"""
import heapq
import math
import logging
from typing import Dict, List, Tuple, Optional, Set, Union, Any
from dataclasses import dataclass, field
from collections import defaultdict
try:
    from app.models.user_profile import UserProfile, MobilityAidType
except ImportError:
    # For standalone testing
    UserProfile = None
    MobilityAidType = None

# Get logger
logger = logging.getLogger(__name__)

@dataclass
class RouteSegment:
    """Enhanced segment of a route with comprehensive accessibility and metadata"""
    start_node_id: int
    end_node_id: int
    edge_id: int
    length_meters: float
    accessibility_score: float
    energy_cost: float
    comfort_score: float
    surface_type: str
    slope_degrees: float
    has_barriers: bool
    estimated_time_minutes: float
    
    # Enhanced metadata
    start_coordinates: Tuple[float, float] = (0.0, 0.0)  # (lat, lng)
    end_coordinates: Tuple[float, float] = (0.0, 0.0)    # (lat, lng)
    width_meters: float = 2.0
    surface_quality: float = 1.0  # 0-1 score
    safety_rating: float = 3.0    # 1-5 scale
    lighting_quality: float = 3.0 # 1-5 scale
    pedestrian_traffic: str = "medium"  # none, low, medium, high
    
    # Accessibility details
    wheelchair_accessible: bool = True
    has_steps: bool = False
    has_ramps: bool = False
    has_handrails: bool = False
    has_tactile_paving: bool = False
    kerb_height_cm: float = 0.0
    
    # Environmental factors
    noise_level: float = 2.0      # 1-5 scale
    air_quality: float = 3.0      # 1-5 scale
    shade_coverage: float = 0.5   # 0-1 coverage
    weather_protection: bool = False
    
    # Dynamic conditions
    is_blocked: bool = False
    blockage_reason: str = ""
    construction_level: str = "none"  # none, minor, major
    crowding_level: str = "normal"   # empty, normal, busy, crowded
    
    # Instructions and guidance
    turn_instruction: str = "continue"  # continue, turn_left, turn_right, etc.
    landmark_description: str = ""
    audio_guidance: str = ""
    navigation_difficulty: float = 1.0  # 1-5 scale
    
    # Performance metrics
    segment_score: float = 1.0    # Overall segment quality score
    confidence_level: float = 0.8 # Confidence in route data accuracy
    last_updated: Optional[str] = None  # ISO timestamp
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert segment to dictionary for JSON serialization"""
        return {
            'start_node_id': self.start_node_id,
            'end_node_id': self.end_node_id,
            'edge_id': self.edge_id,
            'length_meters': self.length_meters,
            'estimated_time_minutes': self.estimated_time_minutes,
            'accessibility_score': self.accessibility_score,
            'energy_cost': self.energy_cost,
            'comfort_score': self.comfort_score,
            'segment_score': self.segment_score,
            'start_coordinates': self.start_coordinates,
            'end_coordinates': self.end_coordinates,
            'surface_type': self.surface_type,
            'surface_quality': self.surface_quality,
            'width_meters': self.width_meters,
            'slope_degrees': self.slope_degrees,
            'safety_rating': self.safety_rating,
            'lighting_quality': self.lighting_quality,
            'pedestrian_traffic': self.pedestrian_traffic,
            'wheelchair_accessible': self.wheelchair_accessible,
            'has_steps': self.has_steps,
            'has_barriers': self.has_barriers,
            'has_ramps': self.has_ramps,
            'has_handrails': self.has_handrails,
            'has_tactile_paving': self.has_tactile_paving,
            'kerb_height_cm': self.kerb_height_cm,
            'noise_level': self.noise_level,
            'air_quality': self.air_quality,
            'shade_coverage': self.shade_coverage,
            'weather_protection': self.weather_protection,
            'is_blocked': self.is_blocked,
            'blockage_reason': self.blockage_reason,
            'construction_level': self.construction_level,
            'crowding_level': self.crowding_level,
            'turn_instruction': self.turn_instruction,
            'landmark_description': self.landmark_description,
            'audio_guidance': self.audio_guidance,
            'navigation_difficulty': self.navigation_difficulty,
            'confidence_level': self.confidence_level,
            'last_updated': self.last_updated
        }

@dataclass 
class AlternativeRoute:
    """Represents an alternative route option"""
    route_id: str
    optimization_type: str  # shortest, safest, most_accessible, etc.
    segments: List[RouteSegment]
    total_distance: float
    total_time: float
    total_energy_cost: float
    accessibility_score: float
    route_score: float
    description: str
    pros: List[str] = field(default_factory=list)
    cons: List[str] = field(default_factory=list)
    
@dataclass
class RouteResult:
    """Comprehensive route result with multi-criteria metrics and alternatives"""
    # Primary route
    segments: List[RouteSegment]
    total_distance: float
    total_time: float
    total_energy_cost: float
    average_comfort: float
    accessibility_score: float
    route_score: float  # Overall weighted score
    
    # Enhanced metrics
    route_id: str = ""
    route_type: str = "personalized"  # basic, multi_criteria, personalized
    optimization_criteria: str = "balanced"
    
    # Detailed breakdowns
    total_ascent: float = 0.0        # Total uphill meters
    total_descent: float = 0.0       # Total downhill meters
    max_slope: float = 0.0           # Steepest slope encountered
    avg_slope: float = 0.0           # Average slope
    surface_breakdown: Dict[str, float] = field(default_factory=dict)  # surface_type -> distance
    
    # Safety and accessibility metrics
    safety_score: float = 3.0        # 1-5 scale
    barrier_count: int = 0           # Number of barriers
    step_count: int = 0              # Number of steps
    crossing_count: int = 0          # Number of road crossings
    rest_area_count: int = 0         # Number of rest areas available
    
    # Environmental factors
    noise_exposure: float = 2.0      # Average noise level
    air_quality_avg: float = 3.0     # Average air quality
    shade_percentage: float = 0.5    # Percentage of route with shade
    weather_protected_percentage: float = 0.2  # Percentage under cover
    
    # Navigation complexity
    turn_count: int = 0              # Number of turns required
    navigation_complexity: float = 1.0  # 1-5 scale
    landmark_count: int = 0          # Number of identifiable landmarks
    
    # Alternative routes
    alternative_routes: List[AlternativeRoute] = field(default_factory=list)
    
    # Route warnings and recommendations
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    accessibility_notes: List[str] = field(default_factory=list)
    
    # Metadata
    calculation_time_ms: float = 0.0
    algorithm_used: str = "a_star"
    user_profile_id: Optional[int] = None
    confidence_level: float = 0.8    # Overall confidence in route
    generated_at: Optional[str] = None  # ISO timestamp
    expires_at: Optional[str] = None    # ISO timestamp for dynamic data
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert route result to dictionary for JSON serialization"""
        return {
            'route_id': self.route_id,
            'route_type': self.route_type,
            'optimization_criteria': self.optimization_criteria,
            'segments': [segment.to_dict() for segment in self.segments],
            'total_distance': self.total_distance,
            'total_time': self.total_time,
            'total_energy_cost': self.total_energy_cost,
            'average_comfort': self.average_comfort,
            'accessibility_score': self.accessibility_score,
            'route_score': self.route_score,
            'safety_score': self.safety_score,
            'total_ascent': self.total_ascent,
            'total_descent': self.total_descent,
            'max_slope': self.max_slope,
            'avg_slope': self.avg_slope,
            'surface_breakdown': self.surface_breakdown,
            'barrier_count': self.barrier_count,
            'step_count': self.step_count,
            'crossing_count': self.crossing_count,
            'rest_area_count': self.rest_area_count,
            'noise_exposure': self.noise_exposure,
            'air_quality_avg': self.air_quality_avg,
            'shade_percentage': self.shade_percentage,
            'weather_protected_percentage': self.weather_protected_percentage,
            'turn_count': self.turn_count,
            'navigation_complexity': self.navigation_complexity,
            'landmark_count': self.landmark_count,
            'alternative_routes': [{
                'route_id': alt.route_id,
                'optimization_type': alt.optimization_type,
                'total_distance': alt.total_distance,
                'total_time': alt.total_time,
                'accessibility_score': alt.accessibility_score,
                'route_score': alt.route_score,
                'description': alt.description,
                'pros': alt.pros,
                'cons': alt.cons
            } for alt in self.alternative_routes],
            'warnings': self.warnings,
            'recommendations': self.recommendations,
            'accessibility_notes': self.accessibility_notes,
            'calculation_time_ms': self.calculation_time_ms,
            'algorithm_used': self.algorithm_used,
            'user_profile_id': self.user_profile_id,
            'confidence_level': self.confidence_level,
            'generated_at': self.generated_at,
            'expires_at': self.expires_at
        }
    
    def add_warning(self, message: str):
        """Add a warning message to the route"""
        if message not in self.warnings:
            self.warnings.append(message)
    
    def add_recommendation(self, message: str):
        """Add a recommendation message to the route"""
        if message not in self.recommendations:
            self.recommendations.append(message)
    
    def add_accessibility_note(self, message: str):
        """Add an accessibility note to the route"""
        if message not in self.accessibility_notes:
            self.accessibility_notes.append(message)
    
    def calculate_detailed_metrics(self):
        """Calculate detailed route metrics from segments"""
        if not self.segments:
            return
        
        # Calculate elevation metrics
        total_ascent = 0.0
        total_descent = 0.0
        slopes = []
        
        for segment in self.segments:
            slope = segment.slope_degrees
            slopes.append(slope)
            
            # Approximate elevation change
            elevation_change = segment.length_meters * math.sin(math.radians(slope))
            if elevation_change > 0:
                total_ascent += elevation_change
            else:
                total_descent += abs(elevation_change)
        
        self.total_ascent = total_ascent
        self.total_descent = total_descent
        self.max_slope = max(slopes) if slopes else 0.0
        self.avg_slope = sum(slopes) / len(slopes) if slopes else 0.0
        
        # Calculate surface breakdown
        surface_breakdown = defaultdict(float)
        for segment in self.segments:
            surface_breakdown[segment.surface_type] += segment.length_meters
        self.surface_breakdown = dict(surface_breakdown)
        
        # Count features
        self.barrier_count = sum(1 for seg in self.segments if seg.has_barriers)
        self.step_count = sum(1 for seg in self.segments if seg.has_steps)
        self.rest_area_count = sum(1 for seg in self.segments if seg.landmark_description and 'rest' in seg.landmark_description.lower())
        
        # Calculate averages
        if self.segments:
            self.noise_exposure = sum(seg.noise_level for seg in self.segments) / len(self.segments)
            self.air_quality_avg = sum(seg.air_quality for seg in self.segments) / len(self.segments)
            
            # Weighted by distance
            total_distance = sum(seg.length_meters for seg in self.segments)
            if total_distance > 0:
                self.shade_percentage = sum(seg.shade_coverage * seg.length_meters for seg in self.segments) / total_distance
                self.weather_protected_percentage = sum((1.0 if seg.weather_protection else 0.0) * seg.length_meters for seg in self.segments) / total_distance
        
        # Count navigation elements
        self.turn_count = sum(1 for seg in self.segments if 'turn' in seg.turn_instruction)
        self.landmark_count = sum(1 for seg in self.segments if seg.landmark_description)
        
        # Calculate navigation complexity
        complexity_scores = [seg.navigation_difficulty for seg in self.segments]
        self.navigation_complexity = max(complexity_scores) if complexity_scores else 1.0

class AStarRouter:
    """
    Basic A* algorithm implementation for route finding
    """
    
    def __init__(self, graph_data: Dict):
        """
        Initialize router with graph data
        
        Args:
            graph_data: Dictionary containing nodes and edges data
        """
        self.nodes = graph_data.get('nodes', {})
        self.edges = graph_data.get('edges', {})
        self.node_positions = {}  # Cache for node positions
        
        # Build node positions cache
        for node_id, node_data in self.nodes.items():
            self.node_positions[node_id] = (
                node_data.get('latitude', 0),
                node_data.get('longitude', 0)
            )
        
        logger.info("AStarRouter initialized")
    
    def calculate_heuristic(self, node1_id: int, node2_id: int) -> float:
        """
        Calculate heuristic distance between two nodes using Euclidean distance
        
        Args:
            node1_id: ID of first node
            node2_id: ID of second node
            
        Returns:
            float: Heuristic distance in meters
        """
        try:
            # Get positions from cache
            pos1 = self.node_positions.get(node1_id, (0, 0))
            pos2 = self.node_positions.get(node2_id, (0, 0))
            
            # Calculate Euclidean distance
            lat_diff = pos1[0] - pos2[0]
            lng_diff = pos1[1] - pos2[1]
            
            # Convert to meters (approximate)
            lat_diff_meters = lat_diff * 111000  # Rough conversion
            lng_diff_meters = lng_diff * 111000 * math.cos(math.radians((pos1[0] + pos2[0]) / 2))
            
            distance_meters = math.sqrt(lat_diff_meters**2 + lng_diff_meters**2)
            return distance_meters
            
        except Exception as e:
            logger.warning(f"Error calculating heuristic: {e}")
            return 0.0
    
    def get_neighbors(self, node_id: int) -> List[Tuple[int, Dict]]:
        """
        Get neighbors of a node with edge data
        
        Args:
            node_id: ID of node to get neighbors for
            
        Returns:
            List[Tuple[int, Dict]]: List of (neighbor_id, edge_data) tuples
        """
        neighbors = []
        
        # Find all edges connected to this node
        for edge_id, edge_data in self.edges.items():
            if edge_data.get('start_node_id') == node_id:
                neighbor_id = edge_data.get('end_node_id')
                neighbors.append((neighbor_id, edge_data))
            elif edge_data.get('end_node_id') == node_id:
                neighbor_id = edge_data.get('start_node_id')
                neighbors.append((neighbor_id, edge_data))
        
        return neighbors
    
    def find_route(self, start_node_id: int, end_node_id: int) -> Optional[RouteResult]:
        """
        Find optimal route using A* algorithm
        
        Args:
            start_node_id: ID of starting node
            end_node_id: ID of destination node
            
        Returns:
            RouteResult: Route result or None if no route found
        """
        try:
            logger.info(f"Finding route from node {start_node_id} to node {end_node_id}")
            
            # Initialize algorithm data structures
            open_set = []
            heapq.heappush(open_set, (0, start_node_id))
            
            came_from = {}
            g_score = defaultdict(lambda: float('inf'))
            f_score = defaultdict(lambda: float('inf'))
            visited = set()
            
            g_score[start_node_id] = 0
            f_score[start_node_id] = self.calculate_heuristic(start_node_id, end_node_id)
            
            while open_set:
                # Get node with lowest f_score
                current_f, current_node = heapq.heappop(open_set)
                
                # Check if we've reached the destination
                if current_node == end_node_id:
                    logger.info("Route found, reconstructing path")
                    return self._reconstruct_path(start_node_id, end_node_id, came_from)
                
                visited.add(current_node)
                
                # Explore neighbors
                for neighbor_id, edge_data in self.get_neighbors(current_node):
                    if neighbor_id in visited:
                        continue
                    
                    # Calculate tentative g_score
                    tentative_g = g_score[current_node] + edge_data.get('length_meters', 0)
                    
                    # If this path to neighbor is better than previous one
                    if tentative_g < g_score[neighbor_id]:
                        came_from[neighbor_id] = (current_node, edge_data)
                        g_score[neighbor_id] = tentative_g
                        f_score[neighbor_id] = tentative_g + self.calculate_heuristic(neighbor_id, end_node_id)
                        heapq.heappush(open_set, (f_score[neighbor_id], neighbor_id))
            
            logger.warning("No route found")
            return None
            
        except Exception as e:
            logger.error(f"Error finding route: {e}")
            return None
    
    def _reconstruct_path(self, start_node_id: int, end_node_id: int, 
                          came_from: Dict) -> RouteResult:
        """
        Reconstruct path from came_from dictionary
        
        Args:
            start_node_id: ID of starting node
            end_node_id: ID of destination node
            came_from: Dictionary mapping nodes to (previous_node, edge_data)
            
        Returns:
            RouteResult: Complete route result
        """
        try:
            # Reconstruct path
            path = []
            current = end_node_id
            
            while current != start_node_id:
                if current not in came_from:
                    logger.error("Path reconstruction failed: missing node in came_from")
                    return None
                path.append(current)
                current = came_from[current][0]  # Get previous node
            
            path.append(start_node_id)
            path.reverse()
            
            # Build route segments
            segments = []
            total_distance = 0
            total_energy_cost = 0
            total_comfort = 0
            total_time = 0
            
            for i in range(len(path) - 1):
                current_node = path[i]
                next_node = path[i + 1]
                
                # Get edge data
                edge_data = came_from[next_node][1] if next_node in came_from else {}
                
                # Create route segment
                segment = RouteSegment(
                    start_node_id=current_node,
                    end_node_id=next_node,
                    edge_id=edge_data.get('osm_id', 0),
                    length_meters=edge_data.get('length_meters', 0),
                    accessibility_score=edge_data.get('accessibility_score', 1.0),
                    energy_cost=edge_data.get('energy_cost', 1.0),
                    comfort_score=edge_data.get('comfort_score', 1.0),
                    surface_type=edge_data.get('surface_type', 'unknown'),
                    slope_degrees=edge_data.get('max_slope_degrees', 0),
                    has_barriers=edge_data.get('has_barrier', False),
                    estimated_time_minutes=edge_data.get('length_meters', 0) / 84  # 1.4 m/s = 84 m/min
                )
                
                segments.append(segment)
                total_distance += segment.length_meters
                total_energy_cost += segment.energy_cost
                total_comfort += segment.comfort_score
                total_time += segment.estimated_time_minutes
            
            # Calculate overall metrics
            average_comfort = total_comfort / len(segments) if segments else 0
            accessibility_score = min(seg.accessibility_score for seg in segments) if segments else 0
            
            # Calculate overall route score (simple distance-based for basic A*)
            route_score = 1.0 / (1.0 + total_distance / 1000)  # Normalize by km
            
            route_result = RouteResult(
                segments=segments,
                total_distance=total_distance,
                total_time=total_time,
                total_energy_cost=total_energy_cost,
                average_comfort=average_comfort,
                accessibility_score=accessibility_score,
                route_score=route_score
            )
            
            logger.info(f"Path reconstructed: {len(segments)} segments, {total_distance:.2f} meters")
            return route_result
            
        except Exception as e:
            logger.error(f"Error reconstructing path: {e}")
            return None

class MultiCriteriaAStarRouter(AStarRouter):
    """
    Enhanced A* algorithm with multi-criteria optimization
    """
    
    def __init__(self, graph_data: Dict, user_profile: Dict = None):
        """
        Initialize enhanced router with user profile
        
        Args:
            graph_data: Dictionary containing nodes and edges data
            user_profile: User's mobility profile and preferences
        """
        super().__init__(graph_data)
        self.user_profile = user_profile or {}
        logger.info("MultiCriteriaAStarRouter initialized")
    
    def calculate_weighted_cost(self, edge_data: Dict) -> float:
        """
        Calculate multi-criteria weighted cost for an edge with sophisticated normalization
        
        Args:
            edge_data: Edge data dictionary
            
        Returns:
            float: Weighted cost
        """
        try:
            # Extract base metrics
            distance = edge_data.get('length_meters', 100)
            energy_cost = edge_data.get('energy_cost', 1.0)
            comfort_score = edge_data.get('comfort_score', 1.0)
            accessibility_score = edge_data.get('accessibility_score', 1.0)
            slope = edge_data.get('max_slope_degrees', 0)
            surface_quality = self._get_surface_quality_score(edge_data.get('surface_type', 'unknown'))
            width = edge_data.get('width_meters', 2.0)
            
            # Get user preferences with defaults
            weights = {
                'distance': self.user_profile.get('distance_weight', 0.25),
                'energy': self.user_profile.get('energy_efficiency_weight', 0.25),
                'comfort': self.user_profile.get('comfort_weight', 0.25),
                'accessibility': self.user_profile.get('accessibility_weight', 0.15),
                'surface': self.user_profile.get('surface_weight', 0.05),
                'slope': self.user_profile.get('slope_weight', 0.05)
            }
            
            # Normalize weights to sum to 1.0
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v / total_weight for k, v in weights.items()}
            
            # Advanced normalization using statistical methods
            normalized_costs = {
                'distance': self._normalize_distance(distance),
                'energy': self._normalize_energy(energy_cost),
                'comfort': self._normalize_comfort(comfort_score),
                'accessibility': self._normalize_accessibility(accessibility_score),
                'surface': 1.0 - surface_quality,  # Invert for cost
                'slope': self._normalize_slope(slope)
            }
            
            # Apply user-specific penalties
            penalties = self._calculate_user_penalties(edge_data)
            
            # Calculate weighted sum
            weighted_cost = sum(
                weights[factor] * normalized_costs[factor] 
                for factor in weights.keys()
            )
            
            # Apply penalties
            weighted_cost *= (1.0 + penalties)
            
            return max(0.01, min(10.0, weighted_cost))  # Bounded cost
            
        except Exception as e:
            logger.warning(f"Error calculating weighted cost: {e}")
            return 1.0
    
    def _get_surface_quality_score(self, surface_type: str) -> float:
        """
        Get quality score for surface type (0-1, higher is better)
        """
        surface_scores = {
            'asphalt': 1.0,
            'concrete': 0.95,
            'paved': 0.9,
            'paving_stones': 0.85,
            'compacted': 0.7,
            'fine_gravel': 0.6,
            'gravel': 0.4,
            'unpaved': 0.3,
            'dirt': 0.2,
            'grass': 0.15,
            'sand': 0.1,
            'unknown': 0.5
        }
        return surface_scores.get(surface_type, 0.5)
    
    def _normalize_distance(self, distance: float) -> float:
        """
        Normalize distance using logarithmic scaling
        """
        # Use logarithmic scaling for distance to handle wide range
        log_distance = math.log(max(1, distance))
        max_log_distance = math.log(2000)  # 2km segments considered very long
        return min(1.0, log_distance / max_log_distance)
    
    def _normalize_energy(self, energy: float) -> float:
        """
        Normalize energy cost using sigmoid function
        """
        # Sigmoid normalization for energy (0-1)
        return 2 / (1 + math.exp(-energy + 2)) - 1
    
    def _normalize_comfort(self, comfort: float) -> float:
        """
        Normalize comfort score (invert since lower cost is better)
        """
        return 1.0 - max(0.0, min(1.0, comfort))
    
    def _normalize_accessibility(self, accessibility: float) -> float:
        """
        Normalize accessibility score (invert since lower cost is better)
        """
        return 1.0 - max(0.0, min(1.0, accessibility))
    
    def _normalize_slope(self, slope: float) -> float:
        """
        Normalize slope with exponential penalty for steep slopes
        """
        max_comfortable_slope = 2.0  # degrees
        if slope <= max_comfortable_slope:
            return slope / max_comfortable_slope
        else:
            # Exponential penalty for slopes above comfortable level
            return 1.0 + (slope - max_comfortable_slope) / 10.0
    
    def _calculate_user_penalties(self, edge_data: Dict) -> float:
        """
        Calculate additional penalties based on user-specific constraints
        """
        penalties = 0.0
        
        try:
            # Mobility aid specific penalties
            mobility_aid = self.user_profile.get('mobility_aid_type', 'none')
            
            if mobility_aid == 'wheelchair':
                # Higher penalties for wheelchairs
                if edge_data.get('has_kerb', False):
                    penalties += 0.5
                if edge_data.get('width_meters', 2.0) < 1.2:
                    penalties += 0.8
                if edge_data.get('surface_type') in ['gravel', 'dirt', 'grass']:
                    penalties += 0.6
            
            elif mobility_aid in ['cane', 'walker']:
                # Moderate penalties for walking aids
                if edge_data.get('has_steps', False):
                    penalties += 0.4
                if edge_data.get('max_slope_degrees', 0) > 3.0:
                    penalties += 0.3
            
            # Weather-related penalties
            weather_condition = self.user_profile.get('weather_condition', 'clear')
            if weather_condition == 'rain' and edge_data.get('surface_type') in ['dirt', 'grass']:
                penalties += 0.3
            elif weather_condition == 'snow' and not edge_data.get('cleared_of_snow', True):
                penalties += 0.5
            
            # Time-of-day penalties
            time_preference = self.user_profile.get('time_preference', 'day')
            if time_preference == 'night' and not edge_data.get('well_lit', True):
                penalties += 0.2
            
            return penalties
            
        except Exception as e:
            logger.warning(f"Error calculating user penalties: {e}")
            return 0.0
    
    def is_edge_accessible(self, edge_data: Dict) -> bool:
        """
        Check if an edge is accessible for the user
        
        Args:
            edge_data: Edge data dictionary
            
        Returns:
            bool: True if edge is accessible, False otherwise
        """
        try:
            # Check width requirements
            if self.user_profile.get('min_path_width'):
                edge_width = edge_data.get('width_meters', float('inf'))
                if edge_width < self.user_profile['min_path_width']:
                    return False
            
            # Check slope requirements
            if self.user_profile.get('max_slope_degrees'):
                edge_slope = edge_data.get('max_slope_degrees', 0)
                if edge_slope > self.user_profile['max_slope_degrees']:
                    return False
            
            # Check for steps
            if self.user_profile.get('avoid_stairs', True) and edge_data.get('has_steps', False):
                return False
            
            # Check wheelchair accessibility
            if self.user_profile.get('mobility_aid_type') == 'wheelchair':
                if not edge_data.get('wheelchair_accessible', True):
                    return False
            
            # Check if blocked
            if edge_data.get('is_blocked', False):
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Error checking edge accessibility: {e}")
            return True  # Default to accessible if check fails
    
    def find_route(self, start_node_id: int, end_node_id: int) -> Optional[RouteResult]:
        """
        Find optimal route using enhanced A* algorithm with multi-criteria optimization
        
        Args:
            start_node_id: ID of starting node
            end_node_id: ID of destination node
            
        Returns:
            RouteResult: Route result or None if no route found
        """
        try:
            logger.info(f"Finding enhanced route from node {start_node_id} to node {end_node_id}")
            
            # Initialize algorithm data structures
            open_set = []
            heapq.heappush(open_set, (0, start_node_id))
            
            came_from = {}
            g_score = defaultdict(lambda: float('inf'))
            f_score = defaultdict(lambda: float('inf'))
            visited = set()
            
            g_score[start_node_id] = 0
            f_score[start_node_id] = self.calculate_heuristic(start_node_id, end_node_id)
            
            while open_set:
                # Get node with lowest f_score
                current_f, current_node = heapq.heappop(open_set)
                
                # Check if we've reached the destination
                if current_node == end_node_id:
                    logger.info("Enhanced route found, reconstructing path")
                    return self._reconstruct_path(start_node_id, end_node_id, came_from)
                
                visited.add(current_node)
                
                # Explore neighbors
                for neighbor_id, edge_data in self.get_neighbors(current_node):
                    if neighbor_id in visited:
                        continue
                    
                    # Check accessibility
                    if not self.is_edge_accessible(edge_data):
                        continue
                    
                    # Calculate weighted cost
                    edge_cost = self.calculate_weighted_cost(edge_data)
                    tentative_g = g_score[current_node] + edge_cost
                    
                    # If this path to neighbor is better than previous one
                    if tentative_g < g_score[neighbor_id]:
                        came_from[neighbor_id] = (current_node, edge_data)
                        g_score[neighbor_id] = tentative_g
                        f_score[neighbor_id] = tentative_g + self.calculate_heuristic(neighbor_id, end_node_id)
                        heapq.heappush(open_set, (f_score[neighbor_id], neighbor_id))
            
            logger.warning("No enhanced route found")
            return None
            
        except Exception as e:
            logger.error(f"Error finding enhanced route: {e}")
            return None

class PersonalizedAStarRouter(MultiCriteriaAStarRouter):
    """
    Advanced A* router with full UserProfile integration and learning capabilities
    """
    
    def __init__(self, graph_data: Dict, user_profile: Union[UserProfile, Dict]):
        """
        Initialize personalized router with UserProfile object
        
        Args:
            graph_data: Dictionary containing nodes and edges data
            user_profile: UserProfile object or dictionary
        """
        # Handle both UserProfile objects and dictionaries
        if UserProfile and isinstance(user_profile, UserProfile):
            profile_dict = user_profile.get_routing_config()
            self.user_profile_obj = user_profile
        else:
            profile_dict = user_profile or {}
            self.user_profile_obj = None
        
        super().__init__(graph_data, profile_dict)
        logger.info("PersonalizedAStarRouter initialized")
    
    def is_edge_accessible(self, edge_data: Dict) -> bool:
        """
        Enhanced accessibility checking with comprehensive user profile constraints
        
        Args:
            edge_data: Edge data dictionary
            
        Returns:
            bool: True if edge is accessible, False otherwise
        """
        try:
            if not self.user_profile_obj:
                return super().is_edge_accessible(edge_data)
            
            constraints = self.user_profile_obj.accessibility_constraints
            mobility_aid = self.user_profile_obj.mobility_aid_type.value
            
            # Basic width and slope checks
            edge_width = edge_data.get('width_meters', float('inf'))
            if edge_width < constraints.min_path_width:
                return False
            
            edge_slope = edge_data.get('max_slope_degrees', 0)
            if edge_slope > constraints.max_slope_degrees:
                return False
            
            # Stairs and escalators
            if constraints.avoid_stairs and edge_data.get('has_steps', False):
                return False
            
            if constraints.avoid_escalators and edge_data.get('has_escalator', False):
                return False
            
            # Surface conditions
            if constraints.avoid_uneven_surfaces:
                surface_type = edge_data.get('surface_type', 'unknown')
                if surface_type in ['gravel', 'dirt', 'grass', 'cobblestone']:
                    return False
            
            # Tactile guidance for visually impaired
            if constraints.require_tactile_guidance:
                if not edge_data.get('has_tactile_paving', False):
                    return False
            
            # Construction and barriers
            if constraints.avoid_construction and edge_data.get('under_construction', False):
                return False
            
            if edge_data.get('has_barrier', False) and not edge_data.get('barrier_removable', False):
                return False
            
            # Mobility aid specific checks
            if mobility_aid in ['wheelchair_manual', 'wheelchair_electric']:
                if not edge_data.get('wheelchair_accessible', True):
                    return False
                # Require wider paths for wheelchairs
                if edge_width < 1.2:
                    return False
            
            elif mobility_aid == 'mobility_scooter':
                if edge_width < 1.0:
                    return False
                # Check for weight restrictions
                if edge_data.get('weight_limit_kg', 1000) < 150:
                    return False
            
            # Weather-related checks
            weather_constraints = self.user_profile_obj.weather_constraints
            current_weather = self.user_profile_obj.current_weather_condition
            
            if weather_constraints.avoid_rain_exposure and current_weather == 'rain':
                if not edge_data.get('covered', False):
                    return False
            
            if weather_constraints.avoid_snow_ice and current_weather in ['snow', 'ice']:
                if not edge_data.get('cleared_of_snow', True):
                    return False
            
            # Time-based constraints
            current_time = self.user_profile_obj.current_time_of_day
            if self.user_profile_obj.require_well_lit_paths and current_time == 'night':
                if not edge_data.get('well_lit', False):
                    return False
            
            if self.user_profile_obj.avoid_isolated_areas:
                if edge_data.get('pedestrian_traffic', 'low') == 'none':
                    return False
            
            # Segment length constraints (for users who need frequent rests)
            if constraints.require_rest_areas:
                segment_length = edge_data.get('length_meters', 0)
                if segment_length > constraints.max_segment_length:
                    # Check if there are rest areas along the segment
                    if not edge_data.get('has_rest_areas', False):
                        return False
            
            # Crosswalk signal requirements
            if constraints.require_crosswalk_signals:
                if edge_data.get('crosses_road', False) and not edge_data.get('has_crossing_signal', False):
                    return False
            
            # Check if temporarily blocked
            if edge_data.get('is_blocked', False):
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Error in enhanced accessibility check: {e}")
            return super().is_edge_accessible(edge_data)
    
    def calculate_weighted_cost(self, edge_data: Dict) -> float:
        """
        Enhanced cost calculation with personalized preferences and learning
        
        Args:
            edge_data: Edge data dictionary
            
        Returns:
            float: Weighted cost
        """
        try:
            # Start with base cost calculation
            base_cost = super().calculate_weighted_cost(edge_data)
            
            if not self.user_profile_obj:
                return base_cost
            
            # Apply learned preferences
            learned_adjustments = 0.0
            learned_prefs = self.user_profile_obj.learned_preferences
            
            # Surface preference learning
            surface_type = edge_data.get('surface_type', 'unknown')
            if f'prefer_{surface_type}' in learned_prefs:
                learned_adjustments -= learned_prefs[f'prefer_{surface_type}'] * 0.1
            if f'avoid_{surface_type}' in learned_prefs:
                learned_adjustments += learned_prefs[f'avoid_{surface_type}'] * 0.2
            
            # Slope preference learning
            slope = edge_data.get('max_slope_degrees', 0)
            if slope <= 1.0 and learned_prefs.get('prefer_low_slope', 0) > 0.5:
                learned_adjustments -= 0.1
            elif slope > 3.0 and learned_prefs.get('avoid_steep_slope', 0) > 0.5:
                learned_adjustments += 0.2
            
            # Width preference learning
            width = edge_data.get('width_meters', 2.0)
            if width >= 2.0 and learned_prefs.get('prefer_wide_paths', 0) > 0.5:
                learned_adjustments -= 0.05
            
            # Energy level adjustments
            energy_level = self.user_profile_obj.current_energy_level
            if energy_level < 0.5:
                # When tired, prefer easier routes
                if slope > 2.0:
                    learned_adjustments += 0.3
                if edge_data.get('energy_cost', 1.0) > 1.5:
                    learned_adjustments += 0.2
            
            # Carrying load adjustments
            if self.user_profile_obj.is_carrying_load:
                learned_adjustments += slope * 0.1  # Extra penalty for slope when carrying load
                if surface_type in ['gravel', 'dirt', 'grass']:
                    learned_adjustments += 0.15
            
            # Companion considerations
            companion = self.user_profile_obj.companion_type
            if companion == 'guide':
                # With a guide, navigation complexity is less important
                learned_adjustments -= 0.1
            elif companion == 'caregiver':
                # With caregiver, prioritize safety and comfort
                if edge_data.get('safety_rating', 3) < 3:
                    learned_adjustments += 0.2
            
            # Time-of-day adjustments
            current_time = self.user_profile_obj.current_time_of_day
            if current_time == 'night':
                if not edge_data.get('well_lit', False):
                    learned_adjustments += 0.3
                if edge_data.get('pedestrian_traffic', 'medium') == 'low':
                    learned_adjustments += 0.2
            
            # Apply fatigue factor
            fatigue_factor = self.user_profile_obj.fatigue_factor
            if fatigue_factor > 1.0:
                energy_penalty = (fatigue_factor - 1.0) * edge_data.get('energy_cost', 1.0) * 0.1
                learned_adjustments += energy_penalty
            
            # Combine base cost with learned adjustments
            final_cost = base_cost * (1.0 + learned_adjustments)
            
            return max(0.01, min(10.0, final_cost))
            
        except Exception as e:
            logger.warning(f"Error in enhanced cost calculation: {e}")
            return super().calculate_weighted_cost(edge_data)
    
    def calculate_estimated_time(self, edge_data: Dict) -> float:
        """
        Calculate estimated travel time based on user profile
        
        Args:
            edge_data: Edge data dictionary
            
        Returns:
            float: Estimated time in minutes
        """
        try:
            if not self.user_profile_obj:
                return edge_data.get('length_meters', 0) / 84  # Default 1.4 m/s
            
            distance = edge_data.get('length_meters', 0)
            base_speed = self.user_profile_obj.walking_speed_ms
            
            # Adjust speed based on surface type
            surface_type = edge_data.get('surface_type', 'asphalt')
            surface_multiplier = {
                'asphalt': 1.0,
                'concrete': 1.0,
                'paved': 1.0,
                'paving_stones': 0.95,
                'compacted': 0.9,
                'gravel': 0.7,
                'dirt': 0.6,
                'grass': 0.5,
                'sand': 0.4
            }.get(surface_type, 0.8)
            
            # Adjust for slope
            slope = edge_data.get('max_slope_degrees', 0)
            slope_multiplier = max(0.3, 1.0 - (slope * 0.1))
            
            # Adjust for energy level
            energy_multiplier = max(0.5, self.user_profile_obj.current_energy_level)
            
            # Adjust for fatigue factor
            fatigue_multiplier = 1.0 / self.user_profile_obj.fatigue_factor
            
            # Calculate adjusted speed
            adjusted_speed = (base_speed * surface_multiplier * 
                            slope_multiplier * energy_multiplier * fatigue_multiplier)
            
            # Convert to minutes
            time_minutes = distance / (adjusted_speed * 60)
            
            return max(0.1, time_minutes)
            
        except Exception as e:
            logger.warning(f"Error calculating estimated time: {e}")
            return edge_data.get('length_meters', 0) / 84
    
    def record_route_experience(self, route_result: RouteResult, feedback: Dict[str, Any]):
        """
        Record route experience for learning
        
        Args:
            route_result: The route that was taken
            feedback: User feedback about the route
        """
        try:
            if not self.user_profile_obj:
                return
            
            # Update user profile with feedback
            self.user_profile_obj.update_from_feedback(feedback)
            
            # Log the experience
            logger.info(f"Route experience recorded for user {self.user_profile_obj.user_id}")
            
        except Exception as e:
            logger.error(f"Error recording route experience: {e}")

# Example usage
if __name__ == "__main__":
    # Sample graph data
    sample_graph = {
        'nodes': {
            1: {'osm_id': 1000001, 'latitude': 19.0760, 'longitude': 73.0760},
            2: {'osm_id': 1000002, 'latitude': 19.0770, 'longitude': 73.0770},
            3: {'osm_id': 1000003, 'latitude': 19.0780, 'longitude': 73.0780}
        },
        'edges': {
            1: {
                'osm_id': 2000001,
                'start_node_id': 1,
                'end_node_id': 2,
                'length_meters': 150.0,
                'energy_cost': 1.2,
                'comfort_score': 0.8,
                'accessibility_score': 0.9,
                'surface_type': 'asphalt',
                'max_slope_degrees': 1.5,
                'has_barrier': False,
                'has_steps': False,
                'wheelchair_accessible': True,
                'width_meters': 2.5
            },
            2: {
                'osm_id': 2000002,
                'start_node_id': 2,
                'end_node_id': 3,
                'length_meters': 200.0,
                'energy_cost': 1.5,
                'comfort_score': 0.7,
                'accessibility_score': 0.8,
                'surface_type': 'concrete',
                'max_slope_degrees': 2.0,
                'has_barrier': False,
                'has_steps': False,
                'wheelchair_accessible': True,
                'width_meters': 2.0
            }
        }
    }
    
    # Sample user profile
    user_profile = {
        'distance_weight': 0.3,
        'energy_efficiency_weight': 0.3,
        'comfort_weight': 0.4,
        'mobility_aid_type': 'wheelchair',
        'max_slope_degrees': 5.0,
        'min_path_width': 0.8,
        'avoid_stairs': True
    }
    
    # Create router and find route
    router = MultiCriteriaAStarRouter(sample_graph, user_profile)
    route = router.find_route(1, 3)
    
    if route:
        print(f"Route found: {len(route.segments)} segments")
        print(f"Total distance: {route.total_distance:.2f} meters")
        print(f"Accessibility score: {route.accessibility_score:.2f}")
    else:
        print("No route found")