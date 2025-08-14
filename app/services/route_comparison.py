"""
Route comparison and ranking system for multiple route options
"""
import uuid
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import math
from collections import defaultdict

# Import routing components
try:
    from app.services.routing_algorithm import (
        PersonalizedAStarRouter,
        MultiCriteriaAStarRouter,
        RouteResult,
        RouteSegment,
        AlternativeRoute
    )
    from app.services.advanced_routing_algorithms import (
        BidirectionalAStarRouter,
        AnytimeAStarRouter,
        MultiDestinationRouter,
        create_advanced_router
    )
    from app.models.user_profile import UserProfile, RoutingPriority
except ImportError:
    # For standalone testing
    PersonalizedAStarRouter = None
    MultiCriteriaAStarRouter = None
    RouteResult = None
    RouteSegment = None
    AlternativeRoute = None
    BidirectionalAStarRouter = None
    AnytimeAStarRouter = None
    MultiDestinationRouter = None
    UserProfile = None
    RoutingPriority = None

logger = logging.getLogger(__name__)

class OptimizationType(Enum):
    """Types of route optimization"""
    SHORTEST_DISTANCE = "shortest_distance"
    FASTEST_TIME = "fastest_time"
    LEAST_ENERGY = "least_energy"
    MOST_ACCESSIBLE = "most_accessible"
    HIGHEST_COMFORT = "highest_comfort"
    SAFEST_ROUTE = "safest_route"
    BALANCED = "balanced"
    SCENIC = "scenic"

@dataclass
class RouteScore:
    """Detailed scoring breakdown for a route"""
    overall_score: float = 0.0
    distance_score: float = 0.0
    time_score: float = 0.0
    accessibility_score: float = 0.0
    comfort_score: float = 0.0
    safety_score: float = 0.0
    energy_score: float = 0.0
    scenic_score: float = 0.0
    normalized_scores: Dict[str, float] = field(default_factory=dict)

@dataclass
class RouteComparison:
    """Comparison result between multiple routes"""
    route_id: str
    routes: List[RouteResult]
    rankings: List[Tuple[int, float, str]]  # (rank, score, route_id)
    comparison_matrix: Dict[str, Dict[str, float]] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    user_profile_id: Optional[int] = None
    generated_at: Optional[str] = None

class RouteRankingSystem:
    """System for generating and ranking multiple route options"""
    
    def __init__(self, graph_data: Dict, user_profile: Union[UserProfile, Dict]):
        """Initialize ranking system"""
        self.graph_data = graph_data
        self.user_profile = user_profile
        self.routers = {}
        self.scoring_weights = self._get_scoring_weights()
        logger.info("RouteRankingSystem initialized")
    
    def _get_scoring_weights(self) -> Dict[str, float]:
        """Get scoring weights from user profile"""
        if isinstance(self.user_profile, dict):
            return {
                'distance': self.user_profile.get('distance_weight', 0.25),
                'time': self.user_profile.get('time_weight', 0.15),
                'accessibility': self.user_profile.get('accessibility_weight', 0.25),
                'comfort': self.user_profile.get('comfort_weight', 0.20),
                'safety': self.user_profile.get('safety_weight', 0.10),
                'energy': self.user_profile.get('energy_efficiency_weight', 0.05),
                'scenic': self.user_profile.get('scenic_weight', 0.0)
            }
        elif hasattr(self.user_profile, 'route_preferences'):
            prefs = self.user_profile.route_preferences
            return {
                'distance': prefs.distance_weight,
                'time': prefs.time_weight,
                'accessibility': prefs.accessibility_weight,
                'comfort': prefs.comfort_weight,
                'safety': prefs.safety_weight,
                'energy': prefs.energy_efficiency_weight,
                'scenic': prefs.scenic_weight
            }
        else:
            # Default weights
            return {
                'distance': 0.25, 'time': 0.15, 'accessibility': 0.25,
                'comfort': 0.20, 'safety': 0.10, 'energy': 0.05, 'scenic': 0.0
            }
    
    def generate_route_alternatives(self, start_node_id: int, end_node_id: int,
                                  num_alternatives: int = 5,
                                  optimization_types: Optional[List[OptimizationType]] = None) -> List[RouteResult]:
        """
        Generate multiple route alternatives with different optimization criteria
        
        Args:
            start_node_id: Starting node ID
            end_node_id: Destination node ID
            num_alternatives: Number of alternative routes to generate
            optimization_types: Specific optimization types to use
            
        Returns:
            List of alternative route results
        """
        try:
            logger.info(f"Generating {num_alternatives} route alternatives")
            
            if optimization_types is None:
                optimization_types = [
                    OptimizationType.BALANCED,
                    OptimizationType.SHORTEST_DISTANCE,
                    OptimizationType.MOST_ACCESSIBLE,
                    OptimizationType.HIGHEST_COMFORT,
                    OptimizationType.SAFEST_ROUTE
                ][:num_alternatives]
            
            routes = []
            
            for opt_type in optimization_types:
                route = self._generate_optimized_route(start_node_id, end_node_id, opt_type)
                if route:
                    routes.append(route)
                    
                # Stop if we have enough routes
                if len(routes) >= num_alternatives:
                    break
            
            # Remove duplicate routes (same path)
            unique_routes = self._remove_duplicate_routes(routes)
            
            logger.info(f"Generated {len(unique_routes)} unique route alternatives")
            return unique_routes
            
        except Exception as e:
            logger.error(f"Error generating route alternatives: {e}")
            return []
    
    def _generate_optimized_route(self, start_node_id: int, end_node_id: int,
                                opt_type: OptimizationType) -> Optional[RouteResult]:
        """Generate a route optimized for specific criteria"""
        try:
            # Create optimized user profile for this route type
            optimized_profile = self._create_optimized_profile(opt_type)
            
            # Select appropriate routing algorithm
            router_key = f"{opt_type.value}"
            if router_key not in self.routers:
                # Choose router based on optimization type
                if opt_type == OptimizationType.FASTEST_TIME:
                    self.routers[router_key] = BidirectionalAStarRouter(self.graph_data, optimized_profile)
                elif opt_type == OptimizationType.SCENIC:
                    self.routers[router_key] = AnytimeAStarRouter(self.graph_data, optimized_profile)
                else:
                    self.routers[router_key] = PersonalizedAStarRouter(self.graph_data, optimized_profile)
            
            router = self.routers[router_key]
            
            # Generate route
            if isinstance(router, AnytimeAStarRouter):
                route = router.find_route(start_node_id, end_node_id, max_time_seconds=5.0)
            else:
                route = router.find_route(start_node_id, end_node_id)
            
            if route:
                # Add optimization metadata
                route.route_id = f"{opt_type.value}_{uuid.uuid4().hex[:8]}"
                route.optimization_criteria = opt_type.value
                route.algorithm_used = f"{route.algorithm_used}_{opt_type.value}"
                
                # Calculate detailed metrics
                route.calculate_detailed_metrics()
                
                # Add route description
                self._add_route_description(route, opt_type)
            
            return route
            
        except Exception as e:
            logger.error(f"Error generating {opt_type.value} optimized route: {e}")
            return None
    
    def _create_optimized_profile(self, opt_type: OptimizationType) -> Dict[str, Any]:
        """Create user profile optimized for specific criteria"""
        base_profile = self.user_profile if isinstance(self.user_profile, dict) else {}
        
        optimized_weights = {
            OptimizationType.SHORTEST_DISTANCE: {
                'distance_weight': 0.8, 'time_weight': 0.1, 'accessibility_weight': 0.05,
                'comfort_weight': 0.03, 'safety_weight': 0.01, 'energy_efficiency_weight': 0.01
            },
            OptimizationType.FASTEST_TIME: {
                'distance_weight': 0.3, 'time_weight': 0.6, 'accessibility_weight': 0.05,
                'comfort_weight': 0.03, 'safety_weight': 0.01, 'energy_efficiency_weight': 0.01
            },
            OptimizationType.MOST_ACCESSIBLE: {
                'distance_weight': 0.1, 'time_weight': 0.1, 'accessibility_weight': 0.6,
                'comfort_weight': 0.15, 'safety_weight': 0.04, 'energy_efficiency_weight': 0.01
            },
            OptimizationType.HIGHEST_COMFORT: {
                'distance_weight': 0.15, 'time_weight': 0.1, 'accessibility_weight': 0.2,
                'comfort_weight': 0.45, 'safety_weight': 0.08, 'energy_efficiency_weight': 0.02
            },
            OptimizationType.SAFEST_ROUTE: {
                'distance_weight': 0.2, 'time_weight': 0.1, 'accessibility_weight': 0.15,
                'comfort_weight': 0.15, 'safety_weight': 0.35, 'energy_efficiency_weight': 0.05
            },
            OptimizationType.LEAST_ENERGY: {
                'distance_weight': 0.25, 'time_weight': 0.1, 'accessibility_weight': 0.15,
                'comfort_weight': 0.1, 'safety_weight': 0.05, 'energy_efficiency_weight': 0.35
            },
            OptimizationType.SCENIC: {
                'distance_weight': 0.1, 'time_weight': 0.05, 'accessibility_weight': 0.15,
                'comfort_weight': 0.25, 'safety_weight': 0.15, 'scenic_weight': 0.3
            },
            OptimizationType.BALANCED: base_profile  # Use base profile
        }
        
        if opt_type in optimized_weights:
            profile = base_profile.copy()
            profile.update(optimized_weights[opt_type])
            return profile
        
        return base_profile
    
    def _add_route_description(self, route: RouteResult, opt_type: OptimizationType):
        """Add human-readable description to route"""
        descriptions = {
            OptimizationType.SHORTEST_DISTANCE: "Optimized for shortest distance",
            OptimizationType.FASTEST_TIME: "Optimized for fastest travel time", 
            OptimizationType.MOST_ACCESSIBLE: "Optimized for maximum accessibility",
            OptimizationType.HIGHEST_COMFORT: "Optimized for highest comfort",
            OptimizationType.SAFEST_ROUTE: "Optimized for safety and security",
            OptimizationType.LEAST_ENERGY: "Optimized for minimum energy expenditure",
            OptimizationType.SCENIC: "Optimized for scenic and pleasant journey",
            OptimizationType.BALANCED: "Balanced optimization across all criteria"
        }
        
        if hasattr(route, 'add_recommendation'):
            route.add_recommendation(descriptions.get(opt_type, "Optimized route"))
            
            # Add specific notes based on route characteristics
            if route.accessibility_score >= 0.9:
                route.add_accessibility_note("Highly accessible route")
            elif route.accessibility_score < 0.5:
                route.add_accessibility_note("Limited accessibility - check constraints")
            
            if route.total_distance < 500:
                route.add_recommendation("Short distance - suitable for quick trips")
            elif route.total_distance > 2000:
                route.add_recommendation("Long distance - consider rest stops")
    
    def _remove_duplicate_routes(self, routes: List[RouteResult]) -> List[RouteResult]:
        """Remove routes with identical or very similar paths"""
        if not routes:
            return []
        
        unique_routes = [routes[0]]
        
        for route in routes[1:]:
            is_duplicate = False
            
            for existing_route in unique_routes:
                if self._routes_are_similar(route, existing_route):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_routes.append(route)
        
        return unique_routes
    
    def _routes_are_similar(self, route1: RouteResult, route2: RouteResult,
                          similarity_threshold: float = 0.8) -> bool:
        """Check if two routes are similar based on segments used"""
        if not route1.segments or not route2.segments:
            return False
        
        # Extract edge IDs from both routes
        edges1 = set(seg.edge_id for seg in route1.segments)
        edges2 = set(seg.edge_id for seg in route2.segments)
        
        # Calculate Jaccard similarity
        intersection = len(edges1 & edges2)
        union = len(edges1 | edges2)
        
        if union == 0:
            return True  # Both routes have no segments
        
        jaccard_similarity = intersection / union
        return jaccard_similarity >= similarity_threshold
    
    def compare_routes(self, routes: List[RouteResult]) -> RouteComparison:
        """
        Compare multiple routes and generate rankings
        
        Args:
            routes: List of routes to compare
            
        Returns:
            RouteComparison with rankings and analysis
        """
        try:
            logger.info(f"Comparing {len(routes)} routes")
            
            if not routes:
                return RouteComparison(
                    route_id=str(uuid.uuid4()),
                    routes=[],
                    rankings=[],
                    recommendations=["No routes to compare"]
                )
            
            # Calculate scores for all routes
            route_scores = []
            for route in routes:
                score = self._calculate_route_score(route, routes)
                route_scores.append(score)
            
            # Create rankings
            rankings = []
            for i, (route, score) in enumerate(zip(routes, route_scores)):
                rankings.append((i, score.overall_score, route.route_id))
            
            # Sort by score (higher is better)
            rankings.sort(key=lambda x: x[1], reverse=True)
            
            # Add rank positions
            final_rankings = []
            for rank, (route_index, score, route_id) in enumerate(rankings, 1):
                final_rankings.append((rank, score, route_id))
            
            # Build comparison matrix
            comparison_matrix = self._build_comparison_matrix(routes, route_scores)
            
            # Generate recommendations
            recommendations = self._generate_comparison_recommendations(routes, route_scores, final_rankings)
            
            comparison = RouteComparison(
                route_id=str(uuid.uuid4()),
                routes=routes,
                rankings=final_rankings,
                comparison_matrix=comparison_matrix,
                recommendations=recommendations,
                user_profile_id=getattr(self.user_profile, 'user_id', None)
            )
            
            logger.info(f"Route comparison completed, best route: {final_rankings[0][2]}")
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing routes: {e}")
            return RouteComparison(
                route_id=str(uuid.uuid4()),
                routes=routes,
                rankings=[],
                recommendations=["Error occurred during comparison"]
            )
    
    def _calculate_route_score(self, route: RouteResult, all_routes: List[RouteResult]) -> RouteScore:
        """Calculate comprehensive score for a route"""
        try:
            # Normalize metrics against all routes
            distances = [r.total_distance for r in all_routes]
            times = [r.total_time for r in all_routes]
            accessibilities = [r.accessibility_score for r in all_routes]
            comforts = [r.average_comfort for r in all_routes]
            energies = [r.total_energy_cost for r in all_routes]
            
            # Calculate normalized scores (0-1, higher is better)
            distance_score = 1.0 - self._normalize_value(route.total_distance, distances)
            time_score = 1.0 - self._normalize_value(route.total_time, times)
            accessibility_score = self._normalize_value(route.accessibility_score, accessibilities)
            comfort_score = self._normalize_value(route.average_comfort, comforts)
            energy_score = 1.0 - self._normalize_value(route.total_energy_cost, energies)
            
            # Calculate safety score based on route characteristics
            safety_score = self._calculate_safety_score(route)
            
            # Calculate scenic score (if applicable)
            scenic_score = self._calculate_scenic_score(route)
            
            # Create RouteScore object
            route_score = RouteScore(
                distance_score=distance_score,
                time_score=time_score,
                accessibility_score=accessibility_score,
                comfort_score=comfort_score,
                energy_score=energy_score,
                safety_score=safety_score,
                scenic_score=scenic_score,
                normalized_scores={
                    'distance': distance_score,
                    'time': time_score,
                    'accessibility': accessibility_score,
                    'comfort': comfort_score,
                    'energy': energy_score,
                    'safety': safety_score,
                    'scenic': scenic_score
                }
            )
            
            # Calculate weighted overall score
            weights = self.scoring_weights
            overall_score = (
                weights['distance'] * distance_score +
                weights['time'] * time_score +
                weights['accessibility'] * accessibility_score +
                weights['comfort'] * comfort_score +
                weights['energy'] * energy_score +
                weights['safety'] * safety_score +
                weights['scenic'] * scenic_score
            )
            
            route_score.overall_score = overall_score
            
            return route_score
            
        except Exception as e:
            logger.error(f"Error calculating route score: {e}")
            return RouteScore()
    
    def _normalize_value(self, value: float, all_values: List[float]) -> float:
        """Normalize a value against a list of values (0-1 scale)"""
        if not all_values or min(all_values) == max(all_values):
            return 1.0
        
        min_val = min(all_values)
        max_val = max(all_values)
        
        return (value - min_val) / (max_val - min_val)
    
    def _calculate_safety_score(self, route: RouteResult) -> float:
        """Calculate safety score based on route characteristics"""
        try:
            safety_factors = []
            
            for segment in route.segments:
                segment_safety = 0.5  # Base safety score
                
                # Adjust based on segment characteristics
                if hasattr(segment, 'lighting_quality'):
                    segment_safety += (segment.lighting_quality - 2.5) * 0.1
                
                if hasattr(segment, 'pedestrian_traffic'):
                    traffic_scores = {'none': 0.2, 'low': 0.4, 'medium': 0.8, 'high': 0.9}
                    segment_safety += traffic_scores.get(segment.pedestrian_traffic, 0.5)
                
                if hasattr(segment, 'safety_rating'):
                    segment_safety = segment.safety_rating / 5.0
                
                safety_factors.append(max(0.0, min(1.0, segment_safety)))
            
            return sum(safety_factors) / len(safety_factors) if safety_factors else 0.5
            
        except Exception as e:
            logger.warning(f"Error calculating safety score: {e}")
            return 0.5
    
    def _calculate_scenic_score(self, route: RouteResult) -> float:
        """Calculate scenic score based on route characteristics"""
        try:
            scenic_factors = []
            
            for segment in route.segments:
                segment_scenic = 0.5  # Base scenic score
                
                # Adjust based on segment characteristics
                if hasattr(segment, 'air_quality'):
                    segment_scenic += (segment.air_quality - 2.5) * 0.1
                
                if hasattr(segment, 'noise_level'):
                    segment_scenic -= (segment.noise_level - 2.5) * 0.1
                
                if hasattr(segment, 'shade_coverage'):
                    segment_scenic += segment.shade_coverage * 0.2
                
                scenic_factors.append(max(0.0, min(1.0, segment_scenic)))
            
            return sum(scenic_factors) / len(scenic_factors) if scenic_factors else 0.5
            
        except Exception as e:
            logger.warning(f"Error calculating scenic score: {e}")
            return 0.5
    
    def _build_comparison_matrix(self, routes: List[RouteResult], 
                               route_scores: List[RouteScore]) -> Dict[str, Dict[str, float]]:
        """Build comparison matrix between routes"""
        matrix = {}
        
        criteria = ['distance', 'time', 'accessibility', 'comfort', 'safety', 'energy', 'scenic']
        
        for i, route in enumerate(routes):
            route_id = route.route_id
            matrix[route_id] = {}
            
            for criterion in criteria:
                score = getattr(route_scores[i], f'{criterion}_score', 0.0)
                matrix[route_id][criterion] = score
            
            matrix[route_id]['overall'] = route_scores[i].overall_score
        
        return matrix
    
    def _generate_comparison_recommendations(self, routes: List[RouteResult], 
                                          route_scores: List[RouteScore],
                                          rankings: List[Tuple[int, float, str]]) -> List[str]:
        """Generate recommendations based on route comparison"""
        recommendations = []
        
        if not routes or not rankings:
            return ["No routes available for comparison"]
        
        best_route_id = rankings[0][2]
        best_route = next((r for r in routes if r.route_id == best_route_id), None)
        
        if best_route:
            recommendations.append(f"Recommended route: {best_route.optimization_criteria}")
            
            # Add specific recommendations based on route characteristics
            if best_route.accessibility_score >= 0.9:
                recommendations.append("Best route has excellent accessibility")
            elif best_route.accessibility_score < 0.6:
                recommendations.append("Consider accessibility constraints for recommended route")
            
            if best_route.total_time < 10:
                recommendations.append("Quick route - suitable for short trips")
            elif best_route.total_time > 30:
                recommendations.append("Long route - plan for adequate time")
            
            # Compare with other routes
            if len(routes) > 1:
                second_best_score = rankings[1][1]
                score_difference = rankings[0][1] - second_best_score
                
                if score_difference < 0.1:
                    recommendations.append("Multiple routes have similar quality - consider personal preferences")
                else:
                    recommendations.append("Clear winner - significantly better than alternatives")
        
        # Add general recommendations
        if len(routes) >= 3:
            recommendations.append("Multiple route options available - choose based on priorities")
        
        return recommendations
    
    def find_best_routes_for_criteria(self, start_node_id: int, end_node_id: int) -> Dict[str, RouteResult]:
        """
        Find the best route for each optimization criteria
        
        Args:
            start_node_id: Starting node ID
            end_node_id: Destination node ID
            
        Returns:
            Dictionary mapping criteria to best routes
        """
        try:
            logger.info("Finding best routes for each criteria")
            
            criteria_routes = {}
            
            for opt_type in OptimizationType:
                route = self._generate_optimized_route(start_node_id, end_node_id, opt_type)
                if route:
                    criteria_routes[opt_type.value] = route
            
            return criteria_routes
            
        except Exception as e:
            logger.error(f"Error finding best routes for criteria: {e}")
            return {}


def create_route_ranking_system(graph_data: Dict, user_profile: Union[UserProfile, Dict]) -> RouteRankingSystem:
    """
    Factory function to create route ranking system
    
    Args:
        graph_data: Graph data dictionary
        user_profile: User profile for personalization
        
    Returns:
        RouteRankingSystem instance
    """
    try:
        return RouteRankingSystem(graph_data, user_profile)
    except Exception as e:
        logger.error(f"Error creating route ranking system: {e}")
        return None


# Example usage
if __name__ == "__main__":
    # Sample test data
    sample_graph = {
        'nodes': {
            1: {'osm_id': 1000001, 'latitude': 19.0760, 'longitude': 73.0760},
            2: {'osm_id': 1000002, 'latitude': 19.0770, 'longitude': 73.0770},
            3: {'osm_id': 1000003, 'latitude': 19.0780, 'longitude': 73.0780}
        },
        'edges': {
            1: {'osm_id': 2000001, 'start_node_id': 1, 'end_node_id': 2, 'length_meters': 150.0},
            2: {'osm_id': 2000002, 'start_node_id': 2, 'end_node_id': 3, 'length_meters': 200.0}
        }
    }
    
    user_profile = {'distance_weight': 0.4, 'accessibility_weight': 0.4, 'comfort_weight': 0.2}
    
    print("Testing Route Comparison System")
    
    # Create ranking system
    ranking_system = create_route_ranking_system(sample_graph, user_profile)
    
    if ranking_system:
        # Generate alternative routes
        alternatives = ranking_system.generate_route_alternatives(1, 3, num_alternatives=3)
        print(f"Generated {len(alternatives)} alternative routes")
        
        # Compare routes
        if alternatives:
            comparison = ranking_system.compare_routes(alternatives)
            print(f"Best route: {comparison.rankings[0][2] if comparison.rankings else 'None'}")
        
    print("Route comparison test completed")
