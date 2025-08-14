"""
Advanced A* algorithm variants for enhanced routing capabilities
"""
import heapq
import math
import time
import logging
from typing import Dict, List, Tuple, Optional, Set, Union, Any
from dataclasses import dataclass, field
from collections import defaultdict
from threading import Thread, Event
import uuid

# Import base routing components
try:
    from app.services.routing_algorithm import (
        PersonalizedAStarRouter,
        MultiCriteriaAStarRouter,
        RouteResult,
        RouteSegment,
        AlternativeRoute
    )
    from app.models.user_profile import UserProfile
except ImportError:
    # For standalone testing
    PersonalizedAStarRouter = None
    MultiCriteriaAStarRouter = None
    RouteResult = None
    RouteSegment = None
    AlternativeRoute = None
    UserProfile = None

logger = logging.getLogger(__name__)

@dataclass
class SearchState:
    """Represents the state of an A* search"""
    node_id: int
    g_score: float
    h_score: float
    parent: Optional['SearchState'] = None
    
    @property
    def f_score(self) -> float:
        return self.g_score + self.h_score
    
    def __lt__(self, other):
        return self.f_score < other.f_score


class BidirectionalAStarRouter(PersonalizedAStarRouter):
    """
    Bidirectional A* algorithm that searches from both start and goal simultaneously
    Often faster for long-distance routing
    """
    
    def __init__(self, graph_data: Dict, user_profile: Union[UserProfile, Dict]):
        """Initialize bidirectional A* router"""
        super().__init__(graph_data, user_profile)
        logger.info("BidirectionalAStarRouter initialized")
    
    def find_route(self, start_node_id: int, end_node_id: int) -> Optional[RouteResult]:
        """
        Find route using bidirectional A* search
        
        Args:
            start_node_id: Starting node ID
            end_node_id: Destination node ID
            
        Returns:
            RouteResult or None if no route found
        """
        try:
            logger.info(f"Finding bidirectional route from {start_node_id} to {end_node_id}")
            
            if start_node_id == end_node_id:
                return self._create_empty_route(start_node_id)
            
            # Initialize forward and backward searches
            forward_open = []
            backward_open = []
            
            forward_closed = set()
            backward_closed = set()
            
            forward_g_scores = defaultdict(lambda: float('inf'))
            backward_g_scores = defaultdict(lambda: float('inf'))
            
            forward_came_from = {}
            backward_came_from = {}
            
            # Initialize start states
            forward_g_scores[start_node_id] = 0
            backward_g_scores[end_node_id] = 0
            
            heapq.heappush(forward_open, (
                self.calculate_heuristic(start_node_id, end_node_id),
                start_node_id, 'forward'
            ))
            heapq.heappush(backward_open, (
                self.calculate_heuristic(end_node_id, start_node_id),
                end_node_id, 'backward'
            ))
            
            best_path_cost = float('inf')
            meeting_node = None
            
            while forward_open or backward_open:
                # Alternate between forward and backward searches
                if forward_open and (not backward_open or len(forward_closed) <= len(backward_closed)):
                    # Forward search step
                    current_f, current_node, direction = heapq.heappop(forward_open)
                    
                    if current_node in forward_closed:
                        continue
                    
                    forward_closed.add(current_node)
                    
                    # Check if we've met the backward search
                    if current_node in backward_closed:
                        path_cost = forward_g_scores[current_node] + backward_g_scores[current_node]
                        if path_cost < best_path_cost:
                            best_path_cost = path_cost
                            meeting_node = current_node
                    
                    # Expand forward neighbors
                    for neighbor_id, edge_data in self.get_neighbors(current_node):
                        if neighbor_id in forward_closed or not self.is_edge_accessible(edge_data):
                            continue
                        
                        edge_cost = self.calculate_weighted_cost(edge_data)
                        tentative_g = forward_g_scores[current_node] + edge_cost
                        
                        if tentative_g < forward_g_scores[neighbor_id]:
                            forward_came_from[neighbor_id] = (current_node, edge_data)
                            forward_g_scores[neighbor_id] = tentative_g
                            h_score = self.calculate_heuristic(neighbor_id, end_node_id)
                            heapq.heappush(forward_open, (tentative_g + h_score, neighbor_id, 'forward'))
                
                elif backward_open:
                    # Backward search step
                    current_f, current_node, direction = heapq.heappop(backward_open)
                    
                    if current_node in backward_closed:
                        continue
                    
                    backward_closed.add(current_node)
                    
                    # Check if we've met the forward search
                    if current_node in forward_closed:
                        path_cost = forward_g_scores[current_node] + backward_g_scores[current_node]
                        if path_cost < best_path_cost:
                            best_path_cost = path_cost
                            meeting_node = current_node
                    
                    # Expand backward neighbors (reverse direction)
                    for neighbor_id, edge_data in self.get_neighbors(current_node):
                        if neighbor_id in backward_closed or not self.is_edge_accessible(edge_data):
                            continue
                        
                        edge_cost = self.calculate_weighted_cost(edge_data)
                        tentative_g = backward_g_scores[current_node] + edge_cost
                        
                        if tentative_g < backward_g_scores[neighbor_id]:
                            backward_came_from[neighbor_id] = (current_node, edge_data)
                            backward_g_scores[neighbor_id] = tentative_g
                            h_score = self.calculate_heuristic(neighbor_id, start_node_id)
                            heapq.heappush(backward_open, (tentative_g + h_score, neighbor_id, 'backward'))
                
                # Terminate if we found a meeting point
                if meeting_node is not None:
                    logger.info(f"Bidirectional search met at node {meeting_node}")
                    return self._reconstruct_bidirectional_path(
                        start_node_id, end_node_id, meeting_node,
                        forward_came_from, backward_came_from
                    )
            
            logger.warning("No bidirectional route found")
            return None
            
        except Exception as e:
            logger.error(f"Error in bidirectional A* search: {e}")
            return None
    
    def _reconstruct_bidirectional_path(self, start_node_id: int, end_node_id: int, 
                                      meeting_node: int, forward_came_from: Dict, 
                                      backward_came_from: Dict) -> RouteResult:
        """Reconstruct path from bidirectional search results"""
        try:
            # Reconstruct forward path (start to meeting point)
            forward_path = []
            current = meeting_node
            
            while current != start_node_id:
                if current not in forward_came_from:
                    break
                forward_path.append(current)
                current = forward_came_from[current][0]
            
            forward_path.append(start_node_id)
            forward_path.reverse()
            
            # Reconstruct backward path (meeting point to end)
            backward_path = []
            current = meeting_node
            
            while current != end_node_id:
                if current not in backward_came_from:
                    break
                backward_path.append(current)
                current = backward_came_from[current][0]
            
            backward_path.append(end_node_id)
            
            # Combine paths
            full_path = forward_path[:-1] + backward_path  # Remove duplicate meeting node
            
            # Build segments
            segments = []
            for i in range(len(full_path) - 1):
                current_node = full_path[i]
                next_node = full_path[i + 1]
                
                # Find edge data
                edge_data = None
                if i < len(forward_path) - 1:
                    # Forward path
                    if next_node in forward_came_from:
                        edge_data = forward_came_from[next_node][1]
                else:
                    # Backward path - need to find edge in reverse
                    for edge_id, edge_info in self.edges.items():
                        if ((edge_info.get('start_node_id') == current_node and 
                             edge_info.get('end_node_id') == next_node) or
                            (edge_info.get('start_node_id') == next_node and 
                             edge_info.get('end_node_id') == current_node)):
                            edge_data = edge_info
                            break
                
                if edge_data:
                    segment = self._create_route_segment(current_node, next_node, edge_data)
                    segments.append(segment)
            
            return self._create_route_result(segments)
            
        except Exception as e:
            logger.error(f"Error reconstructing bidirectional path: {e}")
            return None
    
    def _create_empty_route(self, node_id: int) -> RouteResult:
        """Create empty route for same start/end node"""
        return RouteResult(
            segments=[],
            total_distance=0.0,
            total_time=0.0,
            total_energy_cost=0.0,
            average_comfort=1.0,
            accessibility_score=1.0,
            route_score=1.0,
            algorithm_used="bidirectional_astar"
        )


class AnytimeAStarRouter(PersonalizedAStarRouter):
    """
    Anytime A* algorithm that can return improving solutions over time
    Useful for time-constrained routing scenarios
    """
    
    def __init__(self, graph_data: Dict, user_profile: Union[UserProfile, Dict]):
        """Initialize anytime A* router"""
        super().__init__(graph_data, user_profile)
        self.stop_search = Event()
        self.current_best_route = None
        self.search_thread = None
        logger.info("AnytimeAStarRouter initialized")
    
    def find_route(self, start_node_id: int, end_node_id: int, 
                   max_time_seconds: float = 10.0,
                   improvement_threshold: float = 0.01) -> Optional[RouteResult]:
        """
        Find route using anytime A* with time limit
        
        Args:
            start_node_id: Starting node ID
            end_node_id: Destination node ID
            max_time_seconds: Maximum search time
            improvement_threshold: Minimum improvement to continue search
            
        Returns:
            Best route found within time limit
        """
        try:
            logger.info(f"Starting anytime A* search with {max_time_seconds}s limit")
            
            self.stop_search.clear()
            self.current_best_route = None
            
            # Start search in separate thread
            self.search_thread = Thread(
                target=self._anytime_search,
                args=(start_node_id, end_node_id, improvement_threshold)
            )
            self.search_thread.start()
            
            # Wait for completion or timeout
            self.search_thread.join(timeout=max_time_seconds)
            
            # Signal stop if still running
            self.stop_search.set()
            
            if self.search_thread.is_alive():
                logger.info("Anytime search timed out, returning best solution found")
            else:
                logger.info("Anytime search completed")
            
            return self.current_best_route
            
        except Exception as e:
            logger.error(f"Error in anytime A* search: {e}")
            return None
    
    def _anytime_search(self, start_node_id: int, end_node_id: int, improvement_threshold: float):
        """Internal anytime search implementation"""
        try:
            weights = [2.0, 1.5, 1.2, 1.0]  # Decreasing weight factors for multiple iterations
            
            for weight in weights:
                if self.stop_search.is_set():
                    break
                
                route = self._weighted_a_star(start_node_id, end_node_id, weight)
                
                if route and (self.current_best_route is None or 
                             route.route_score > self.current_best_route.route_score + improvement_threshold):
                    self.current_best_route = route
                    self.current_best_route.algorithm_used = f"anytime_astar_w{weight}"
                    logger.debug(f"Found improved route with weight {weight}, score: {route.route_score}")
                
                if weight == 1.0:  # Optimal solution found
                    break
                    
        except Exception as e:
            logger.error(f"Error in anytime search: {e}")
    
    def _weighted_a_star(self, start_node_id: int, end_node_id: int, weight: float) -> Optional[RouteResult]:
        """Weighted A* search with inflated heuristic"""
        try:
            open_set = []
            heapq.heappush(open_set, (0, start_node_id))
            
            came_from = {}
            g_score = defaultdict(lambda: float('inf'))
            visited = set()
            
            g_score[start_node_id] = 0
            
            while open_set and not self.stop_search.is_set():
                current_f, current_node = heapq.heappop(open_set)
                
                if current_node == end_node_id:
                    return self._reconstruct_path(start_node_id, end_node_id, came_from)
                
                if current_node in visited:
                    continue
                
                visited.add(current_node)
                
                for neighbor_id, edge_data in self.get_neighbors(current_node):
                    if neighbor_id in visited or not self.is_edge_accessible(edge_data):
                        continue
                    
                    edge_cost = self.calculate_weighted_cost(edge_data)
                    tentative_g = g_score[current_node] + edge_cost
                    
                    if tentative_g < g_score[neighbor_id]:
                        came_from[neighbor_id] = (current_node, edge_data)
                        g_score[neighbor_id] = tentative_g
                        h_score = self.calculate_heuristic(neighbor_id, end_node_id) * weight
                        f_score = tentative_g + h_score
                        heapq.heappush(open_set, (f_score, neighbor_id))
            
            return None
            
        except Exception as e:
            logger.error(f"Error in weighted A*: {e}")
            return None


class MultiDestinationRouter(PersonalizedAStarRouter):
    """
    Multi-destination routing for visiting multiple waypoints
    Solves traveling salesman-like problems with accessibility constraints
    """
    
    def __init__(self, graph_data: Dict, user_profile: Union[UserProfile, Dict]):
        """Initialize multi-destination router"""
        super().__init__(graph_data, user_profile)
        self.waypoint_routes = {}  # Cache routes between waypoints
        logger.info("MultiDestinationRouter initialized")
    
    def find_multi_destination_route(self, start_node_id: int, destination_nodes: List[int],
                                   return_to_start: bool = False,
                                   optimize_order: bool = True) -> Optional[RouteResult]:
        """
        Find route visiting multiple destinations
        
        Args:
            start_node_id: Starting node ID
            destination_nodes: List of destination node IDs to visit
            return_to_start: Whether to return to start node at end
            optimize_order: Whether to optimize visit order
            
        Returns:
            Combined route visiting all destinations
        """
        try:
            logger.info(f"Finding multi-destination route to {len(destination_nodes)} locations")
            
            if not destination_nodes:
                return self._create_empty_route(start_node_id)
            
            # Build distance matrix between all points
            all_nodes = [start_node_id] + destination_nodes
            if return_to_start:
                all_nodes.append(start_node_id)
            
            distance_matrix = self._build_distance_matrix(all_nodes)
            
            if optimize_order:
                # Find optimal order using nearest neighbor heuristic
                visit_order = self._optimize_visit_order(start_node_id, destination_nodes, 
                                                       distance_matrix, return_to_start)
            else:
                visit_order = [start_node_id] + destination_nodes
                if return_to_start:
                    visit_order.append(start_node_id)
            
            # Build complete route
            complete_segments = []
            total_distance = 0.0
            total_time = 0.0
            total_energy_cost = 0.0
            
            for i in range(len(visit_order) - 1):
                from_node = visit_order[i]
                to_node = visit_order[i + 1]
                
                if from_node == to_node:
                    continue
                
                # Get cached route or compute new one
                route_key = (from_node, to_node)
                if route_key not in self.waypoint_routes:
                    segment_route = super().find_route(from_node, to_node)
                    self.waypoint_routes[route_key] = segment_route
                
                segment_route = self.waypoint_routes[route_key]
                if segment_route:
                    complete_segments.extend(segment_route.segments)
                    total_distance += segment_route.total_distance
                    total_time += segment_route.total_time
                    total_energy_cost += segment_route.total_energy_cost
                else:
                    logger.warning(f"Could not find route from {from_node} to {to_node}")
                    return None
            
            # Calculate combined metrics
            average_comfort = (sum(seg.comfort_score for seg in complete_segments) / 
                             len(complete_segments)) if complete_segments else 0
            accessibility_score = (min(seg.accessibility_score for seg in complete_segments) 
                                 if complete_segments else 0)
            
            route_result = RouteResult(
                segments=complete_segments,
                total_distance=total_distance,
                total_time=total_time,
                total_energy_cost=total_energy_cost,
                average_comfort=average_comfort,
                accessibility_score=accessibility_score,
                route_score=1.0 / (1.0 + total_distance / 1000),
                algorithm_used="multi_destination_astar",
                route_id=str(uuid.uuid4()),
                optimization_criteria=f"multi_dest_{len(destination_nodes)}_waypoints"
            )
            
            # Add route metadata
            route_result.add_accessibility_note(f"Route visits {len(destination_nodes)} destinations")
            if optimize_order:
                route_result.add_recommendation("Visit order optimized for efficiency")
            
            return route_result
            
        except Exception as e:
            logger.error(f"Error in multi-destination routing: {e}")
            return None
    
    def _build_distance_matrix(self, nodes: List[int]) -> Dict[Tuple[int, int], float]:
        """Build distance matrix between all node pairs"""
        matrix = {}
        
        for i, from_node in enumerate(nodes):
            for j, to_node in enumerate(nodes):
                if i == j:
                    matrix[(from_node, to_node)] = 0.0
                elif (from_node, to_node) not in matrix:
                    # Calculate distance or use cached route
                    route_key = (from_node, to_node)
                    if route_key in self.waypoint_routes and self.waypoint_routes[route_key]:
                        distance = self.waypoint_routes[route_key].total_distance
                    else:
                        # Use heuristic distance for initial estimation
                        distance = self.calculate_heuristic(from_node, to_node)
                    
                    matrix[(from_node, to_node)] = distance
                    matrix[(to_node, from_node)] = distance  # Assume symmetric
        
        return matrix
    
    def _optimize_visit_order(self, start_node: int, destinations: List[int],
                            distance_matrix: Dict[Tuple[int, int], float],
                            return_to_start: bool) -> List[int]:
        """Optimize visit order using nearest neighbor heuristic"""
        if not destinations:
            return [start_node]
        
        unvisited = destinations.copy()
        route = [start_node]
        current = start_node
        
        while unvisited:
            # Find nearest unvisited destination
            nearest = min(unvisited, key=lambda dest: distance_matrix.get((current, dest), float('inf')))
            unvisited.remove(nearest)
            route.append(nearest)
            current = nearest
        
        if return_to_start:
            route.append(start_node)
        
        return route
    
    def find_optimal_delivery_route(self, depot_node: int, delivery_nodes: List[int],
                                  max_route_distance: float = float('inf'),
                                  max_route_time: float = float('inf')) -> List[RouteResult]:
        """
        Find optimal delivery routes with capacity constraints
        
        Args:
            depot_node: Starting depot node
            delivery_nodes: List of delivery destination nodes
            max_route_distance: Maximum distance per route
            max_route_time: Maximum time per route
            
        Returns:
            List of routes, each starting and ending at depot
        """
        try:
            if not delivery_nodes:
                return []
            
            routes = []
            remaining_deliveries = delivery_nodes.copy()
            
            while remaining_deliveries:
                # Start new route from depot
                current_route_nodes = []
                current_distance = 0.0
                current_time = 0.0
                last_node = depot_node
                
                # Add deliveries to current route within constraints
                while remaining_deliveries:
                    # Find best next delivery considering constraints
                    best_delivery = None
                    best_cost = float('inf')
                    
                    for delivery_node in remaining_deliveries:
                        # Estimate cost to add this delivery
                        to_delivery = self.calculate_heuristic(last_node, delivery_node)
                        from_delivery = self.calculate_heuristic(delivery_node, depot_node)
                        additional_distance = to_delivery + from_delivery
                        additional_time = additional_distance / (self.user_profile_obj.walking_speed_ms * 60 if self.user_profile_obj else 84)
                        
                        if (current_distance + additional_distance <= max_route_distance and
                            current_time + additional_time <= max_route_time):
                            if additional_distance < best_cost:
                                best_delivery = delivery_node
                                best_cost = additional_distance
                    
                    if best_delivery is None:
                        break  # No more deliveries fit in current route
                    
                    current_route_nodes.append(best_delivery)
                    remaining_deliveries.remove(best_delivery)
                    
                    # Update current totals
                    current_distance += self.calculate_heuristic(last_node, best_delivery)
                    current_time += current_distance / (self.user_profile_obj.walking_speed_ms * 60 if self.user_profile_obj else 84)
                    last_node = best_delivery
                
                # Create route for current batch
                if current_route_nodes:
                    route = self.find_multi_destination_route(
                        depot_node, current_route_nodes, return_to_start=True
                    )
                    if route:
                        route.route_id = f"delivery_route_{len(routes) + 1}"
                        route.add_recommendation(f"Delivers to {len(current_route_nodes)} locations")
                        routes.append(route)
            
            logger.info(f"Created {len(routes)} delivery routes for {len(delivery_nodes)} deliveries")
            return routes
            
        except Exception as e:
            logger.error(f"Error in delivery route optimization: {e}")
            return []


class HierarchicalAStarRouter(PersonalizedAStarRouter):
    """
    Hierarchical A* for very large graphs using route hierarchies
    Pre-computes routes between major nodes for faster long-distance routing
    """
    
    def __init__(self, graph_data: Dict, user_profile: Union[UserProfile, Dict]):
        """Initialize hierarchical A* router"""
        super().__init__(graph_data, user_profile)
        self.major_nodes = set()  # High-level nodes for hierarchy
        self.hierarchy_routes = {}  # Pre-computed routes between major nodes
        self.node_to_major = {}  # Mapping from nodes to nearest major nodes
        logger.info("HierarchicalAStarRouter initialized")
    
    def build_hierarchy(self, major_node_ids: List[int]):
        """
        Build routing hierarchy with specified major nodes
        
        Args:
            major_node_ids: List of important nodes to use as hierarchy backbone
        """
        try:
            logger.info(f"Building hierarchy with {len(major_node_ids)} major nodes")
            
            self.major_nodes = set(major_node_ids)
            
            # Map each regular node to nearest major node
            for node_id in self.nodes:
                if node_id not in self.major_nodes:
                    nearest_major = min(
                        self.major_nodes,
                        key=lambda major: self.calculate_heuristic(node_id, major)
                    )
                    self.node_to_major[node_id] = nearest_major
            
            # Pre-compute routes between major nodes
            for i, from_major in enumerate(major_node_ids):
                for j, to_major in enumerate(major_node_ids):
                    if i < j:  # Avoid duplicate computation
                        route = super().find_route(from_major, to_major)
                        if route:
                            self.hierarchy_routes[(from_major, to_major)] = route
                            self.hierarchy_routes[(to_major, from_major)] = route
            
            logger.info(f"Pre-computed {len(self.hierarchy_routes)} hierarchy routes")
            
        except Exception as e:
            logger.error(f"Error building hierarchy: {e}")
    
    def find_route(self, start_node_id: int, end_node_id: int) -> Optional[RouteResult]:
        """
        Find route using hierarchical approach
        
        Args:
            start_node_id: Starting node ID
            end_node_id: Destination node ID
            
        Returns:
            Route result using hierarchy when beneficial
        """
        try:
            # For short distances, use regular A*
            direct_distance = self.calculate_heuristic(start_node_id, end_node_id)
            if direct_distance < 1000:  # Less than 1km, use direct routing
                return super().find_route(start_node_id, end_node_id)
            
            # Use hierarchical routing for longer distances
            start_major = (start_node_id if start_node_id in self.major_nodes 
                          else self.node_to_major.get(start_node_id))
            end_major = (end_node_id if end_node_id in self.major_nodes 
                        else self.node_to_major.get(end_node_id))
            
            if not start_major or not end_major:
                # Fall back to regular A* if hierarchy not available
                return super().find_route(start_node_id, end_node_id)
            
            route_segments = []
            
            # Route from start to start_major (if different)
            if start_node_id != start_major:
                start_route = super().find_route(start_node_id, start_major)
                if start_route:
                    route_segments.extend(start_route.segments)
                else:
                    return None
            
            # Route between major nodes using pre-computed hierarchy
            if start_major != end_major:
                hierarchy_key = (start_major, end_major)
                if hierarchy_key in self.hierarchy_routes:
                    major_route = self.hierarchy_routes[hierarchy_key]
                    route_segments.extend(major_route.segments)
                else:
                    # Pre-computed route not available, compute directly
                    major_route = super().find_route(start_major, end_major)
                    if major_route:
                        route_segments.extend(major_route.segments)
                    else:
                        return None
            
            # Route from end_major to end (if different)
            if end_major != end_node_id:
                end_route = super().find_route(end_major, end_node_id)
                if end_route:
                    route_segments.extend(end_route.segments)
                else:
                    return None
            
            # Combine all segments into final route
            if route_segments:
                return self._create_route_result(route_segments, "hierarchical_astar")
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error in hierarchical routing: {e}")
            return super().find_route(start_node_id, end_node_id)
    
    def _create_route_result(self, segments: List[RouteSegment], algorithm: str = "hierarchical_astar") -> RouteResult:
        """Create RouteResult from segments list"""
        if not segments:
            return None
        
        total_distance = sum(seg.length_meters for seg in segments)
        total_time = sum(seg.estimated_time_minutes for seg in segments)
        total_energy_cost = sum(seg.energy_cost for seg in segments)
        average_comfort = sum(seg.comfort_score for seg in segments) / len(segments)
        accessibility_score = min(seg.accessibility_score for seg in segments)
        
        route_result = RouteResult(
            segments=segments,
            total_distance=total_distance,
            total_time=total_time,
            total_energy_cost=total_energy_cost,
            average_comfort=average_comfort,
            accessibility_score=accessibility_score,
            route_score=1.0 / (1.0 + total_distance / 1000),
            algorithm_used=algorithm,
            route_id=str(uuid.uuid4())
        )
        
        return route_result


# Factory function for creating advanced routers
def create_advanced_router(router_type: str, graph_data: Dict, 
                          user_profile: Union[UserProfile, Dict],
                          **kwargs) -> Optional[PersonalizedAStarRouter]:
    """
    Factory function to create advanced routing algorithms
    
    Args:
        router_type: Type of router ("bidirectional", "anytime", "multi_destination", "hierarchical")
        graph_data: Graph data dictionary
        user_profile: User profile
        **kwargs: Additional router-specific parameters
        
    Returns:
        Appropriate router instance
    """
    router_classes = {
        'bidirectional': BidirectionalAStarRouter,
        'anytime': AnytimeAStarRouter,
        'multi_destination': MultiDestinationRouter,
        'hierarchical': HierarchicalAStarRouter
    }
    
    router_class = router_classes.get(router_type.lower())
    if not router_class:
        logger.error(f"Unknown router type: {router_type}")
        return None
    
    try:
        router = router_class(graph_data, user_profile)
        
        # Apply router-specific configurations
        if router_type.lower() == 'hierarchical' and 'major_nodes' in kwargs:
            router.build_hierarchy(kwargs['major_nodes'])
        
        return router
        
    except Exception as e:
        logger.error(f"Error creating {router_type} router: {e}")
        return None


# Example usage and testing
if __name__ == "__main__":
    # Sample test data
    sample_graph = {
        'nodes': {
            1: {'osm_id': 1000001, 'latitude': 19.0760, 'longitude': 73.0760},
            2: {'osm_id': 1000002, 'latitude': 19.0770, 'longitude': 73.0770},
            3: {'osm_id': 1000003, 'latitude': 19.0780, 'longitude': 73.0780},
            4: {'osm_id': 1000004, 'latitude': 19.0785, 'longitude': 73.0785},
            5: {'osm_id': 1000005, 'latitude': 19.0790, 'longitude': 73.0790}
        },
        'edges': {
            1: {'osm_id': 2000001, 'start_node_id': 1, 'end_node_id': 2, 'length_meters': 150.0},
            2: {'osm_id': 2000002, 'start_node_id': 2, 'end_node_id': 3, 'length_meters': 200.0},
            3: {'osm_id': 2000003, 'start_node_id': 3, 'end_node_id': 4, 'length_meters': 180.0},
            4: {'osm_id': 2000004, 'start_node_id': 4, 'end_node_id': 5, 'length_meters': 160.0}
        }
    }
    
    user_profile = {'distance_weight': 0.5, 'comfort_weight': 0.5}
    
    print("Testing Advanced Routing Algorithms")
    
    # Test bidirectional A*
    bidirectional_router = create_advanced_router('bidirectional', sample_graph, user_profile)
    if bidirectional_router:
        route = bidirectional_router.find_route(1, 5)
        print(f"Bidirectional A* route: {len(route.segments) if route else 0} segments")
    
    # Test multi-destination routing
    multi_router = create_advanced_router('multi_destination', sample_graph, user_profile)
    if multi_router:
        route = multi_router.find_multi_destination_route(1, [3, 4, 5])
        print(f"Multi-destination route: {len(route.segments) if route else 0} segments")
    
    print("Advanced routing algorithms test completed")
