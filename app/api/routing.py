"""
Core Routing API Endpoints
Smart Accessible Routing System
"""
import logging
import time
import uuid
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from flask import Blueprint, request, jsonify, current_app
from functools import wraps
import traceback

# Import routing components
try:
    from app.services.routing_algorithm import (
        PersonalizedAStarRouter, MultiCriteriaAStarRouter, RouteResult, RouteSegment
    )
    from app.services.advanced_routing_algorithms import (
        create_advanced_router, BidirectionalAStarRouter, AnytimeAStarRouter
    )
    from app.services.route_comparison import RouteComparator, RouteRankingSystem
    from app.models.user_profile import UserProfile, MobilityAidType, RoutingPriority
    from app.models.map_data import Graph, Node, Edge
    from app.ml.heuristic_models import HeuristicLearningModel
    from app.ml.dynamic_cost_prediction import EnsembleCostPredictor
    from app.ml.model_persistence import ModelManager
except ImportError:
    # For standalone testing
    PersonalizedAStarRouter = None
    MultiCriteriaAStarRouter = None
    RouteResult = None
    RouteSegment = None
    create_advanced_router = None
    BidirectionalAStarRouter = None
    AnytimeAStarRouter = None
    RouteComparator = None
    RouteRankingSystem = None
    UserProfile = None
    MobilityAidType = None
    RoutingPriority = None
    Graph = None
    Node = None
    Edge = None
    HeuristicLearningModel = None
    EnsembleCostPredictor = None
    ModelManager = None

logger = logging.getLogger(__name__)

# Create Blueprint
routing_bp = Blueprint('routing', __name__, url_prefix='/api/v1/routes')

# ============================================================================
# REQUEST/RESPONSE DATA STRUCTURES
# ============================================================================

@dataclass
class RouteRequest:
    """Route calculation request"""
    start: Dict[str, float]  # {"latitude": float, "longitude": float}
    end: Dict[str, float]    # {"latitude": float, "longitude": float}
    user_profile: Optional[Dict[str, Any]] = None
    routing_options: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate coordinates"""
        self._validate_coordinate(self.start, "start")
        self._validate_coordinate(self.end, "end")
    
    def _validate_coordinate(self, coord: Dict[str, float], name: str):
        """Validate coordinate format"""
        if not isinstance(coord, dict):
            raise ValueError(f"{name} must be a dictionary")
        
        required_keys = {"latitude", "longitude"}
        if not required_keys.issubset(coord.keys()):
            raise ValueError(f"{name} must contain 'latitude' and 'longitude'")
        
        lat, lng = coord["latitude"], coord["longitude"]
        if not (-90 <= lat <= 90):
            raise ValueError(f"{name} latitude must be between -90 and 90")
        if not (-180 <= lng <= 180):
            raise ValueError(f"{name} longitude must be between -180 and 180")

@dataclass
class RouteResponse:
    """Route calculation response"""
    status: str
    route: Optional[Dict[str, Any]] = None
    alternatives: List[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    warnings: List[str] = None
    request_id: str = None
    
    def __post_init__(self):
        """Initialize default values"""
        if self.alternatives is None:
            self.alternatives = []
        if self.warnings is None:
            self.warnings = []
        if self.request_id is None:
            self.request_id = str(uuid.uuid4())

@dataclass 
class MultiCriteriaRequest:
    """Multi-criteria routing request"""
    start: Dict[str, float]
    end: Dict[str, float]
    criteria: Dict[str, float]  # weights for different criteria
    user_profile: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate request"""
        RouteRequest(self.start, self.end)  # Validate coordinates
        self._validate_criteria()
    
    def _validate_criteria(self):
        """Validate criteria weights"""
        if not isinstance(self.criteria, dict):
            raise ValueError("criteria must be a dictionary")
        
        for key, weight in self.criteria.items():
            if not isinstance(weight, (int, float)) or not (0 <= weight <= 1):
                raise ValueError(f"criteria weight '{key}' must be between 0 and 1")
        
        # Ensure weights sum to approximately 1
        total_weight = sum(self.criteria.values())
        if abs(total_weight - 1.0) > 0.1:
            raise ValueError("criteria weights should sum to approximately 1.0")

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def validate_json_request(required_fields: List[str] = None):
    """Decorator to validate JSON request data"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            try:
                if not request.is_json:
                    return jsonify({
                        "status": "error",
                        "error": "invalid_content_type",
                        "message": "Content-Type must be application/json"
                    }), 400
                
                data = request.get_json()
                if data is None:
                    return jsonify({
                        "status": "error",
                        "error": "invalid_json",
                        "message": "Invalid JSON in request body"
                    }), 400
                
                # Check required fields
                if required_fields:
                    missing_fields = [field for field in required_fields if field not in data]
                    if missing_fields:
                        return jsonify({
                            "status": "error",
                            "error": "missing_fields",
                            "message": f"Missing required fields: {', '.join(missing_fields)}"
                        }), 400
                
                return f(data, *args, **kwargs)
                
            except Exception as e:
                logger.error(f"Request validation error: {e}")
                return jsonify({
                    "status": "error",
                    "error": "validation_error",
                    "message": str(e)
                }), 400
        
        return decorated_function
    return decorator

def handle_routing_errors(f):
    """Decorator to handle routing calculation errors"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            start_time = time.time()
            result = f(*args, **kwargs)
            
            # Add timing information
            if isinstance(result, tuple) and len(result) >= 2:
                response_data, status_code = result[0], result[1]
                if isinstance(response_data, dict) and "metadata" in response_data:
                    response_data["metadata"]["api_processing_time"] = time.time() - start_time
            
            return result
            
        except ValueError as e:
            logger.warning(f"Validation error in {f.__name__}: {e}")
            return jsonify({
                "status": "error",
                "error": "validation_error", 
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }), 400
            
        except Exception as e:
            logger.error(f"Unexpected error in {f.__name__}: {e}")
            logger.error(traceback.format_exc())
            return jsonify({
                "status": "error",
                "error": "internal_error",
                "message": "An internal error occurred while processing your request",
                "timestamp": datetime.now().isoformat()
            }), 500
    
    return decorated_function

def create_user_profile_from_request(profile_data: Dict[str, Any]) -> UserProfile:
    """Create UserProfile object from request data"""
    try:
        if not profile_data or not UserProfile:
            return None
        
        # Map request data to UserProfile
        mobility_aid = MobilityAidType.NONE
        if "mobility_aid" in profile_data:
            aid_mapping = {
                "wheelchair": MobilityAidType.WHEELCHAIR,
                "walker": MobilityAidType.WALKER, 
                "cane": MobilityAidType.CANE,
                "scooter": MobilityAidType.MOBILITY_SCOOTER,
                "none": MobilityAidType.NONE
            }
            mobility_aid = aid_mapping.get(profile_data["mobility_aid"], MobilityAidType.NONE)
        
        # Create UserProfile with available data
        user_profile = UserProfile(
            user_id=profile_data.get("user_id", "anonymous"),
            mobility_aid=mobility_aid,
            walking_speed=profile_data.get("walking_speed", 1.2),
            routing_priority=RoutingPriority.BALANCED,
            accessibility_constraints=set(profile_data.get("accessibility_needs", [])),
            environmental_preferences=profile_data.get("preferences", {})
        )
        
        return user_profile
        
    except Exception as e:
        logger.error(f"Error creating user profile: {e}")
        return None

def convert_route_result_to_dict(route_result: RouteResult) -> Dict[str, Any]:
    """Convert RouteResult to dictionary for API response"""
    try:
        if not route_result:
            return None
        
        return {
            "id": route_result.route_id,
            "path": [
                {
                    "latitude": segment.end_node.latitude,
                    "longitude": segment.end_node.longitude,
                    "elevation": getattr(segment.end_node, 'elevation', 0.0)
                }
                for segment in route_result.segments
            ],
            "segments": [
                {
                    "start": {"latitude": seg.start_node.latitude, "longitude": seg.start_node.longitude},
                    "end": {"latitude": seg.end_node.latitude, "longitude": seg.end_node.longitude},
                    "distance": seg.distance,
                    "time": seg.estimated_time,
                    "cost": seg.cost,
                    "accessibility_score": seg.accessibility_score,
                    "surface_type": getattr(seg, 'surface_type', 'unknown'),
                    "warnings": seg.warnings or []
                }
                for seg in route_result.segments
            ],
            "total_distance": route_result.total_distance,
            "total_time": route_result.total_time,
            "total_cost": route_result.total_cost,
            "accessibility_score": route_result.accessibility_score,
            "warnings": route_result.warnings or [],
            "recommendations": route_result.recommendations or [],
            "metadata": {
                "algorithm_used": getattr(route_result, 'algorithm_used', 'unknown'),
                "calculation_time": getattr(route_result, 'calculation_time', 0.0),
                "nodes_explored": getattr(route_result, 'nodes_explored', 0)
            }
        }
        
    except Exception as e:
        logger.error(f"Error converting route result: {e}")
        return None

# ============================================================================
# CORE ROUTING ENDPOINTS
# ============================================================================

@routing_bp.route('/calculate', methods=['POST'])
@validate_json_request(required_fields=['start', 'end'])
@handle_routing_errors
def calculate_route(data):
    """
    Calculate optimal route between two points
    
    POST /api/v1/routes/calculate
    """
    try:
        # Parse request
        route_request = RouteRequest(
            start=data['start'],
            end=data['end'],
            user_profile=data.get('user_profile'),
            routing_options=data.get('routing_options', {})
        )
        
        logger.info(f"Route calculation request: {route_request.start} -> {route_request.end}")
        
        # Create user profile
        user_profile = create_user_profile_from_request(route_request.user_profile)
        
        # Determine algorithm
        algorithm = route_request.routing_options.get('algorithm', 'astar')
        
        # Initialize router based on algorithm
        if algorithm == 'personalized' and user_profile and PersonalizedAStarRouter:
            router = PersonalizedAStarRouter(
                graph=current_app.config.get('GRAPH_INSTANCE'),
                user_profile=user_profile
            )
        elif algorithm in ['bidirectional', 'anytime'] and create_advanced_router:
            router = create_advanced_router(
                algorithm_type=algorithm,
                graph=current_app.config.get('GRAPH_INSTANCE'),
                user_profile=user_profile
            )
        elif MultiCriteriaAStarRouter:
            router = MultiCriteriaAStarRouter(
                graph=current_app.config.get('GRAPH_INSTANCE')
            )
        else:
            # Fallback to mock router for testing
            return _create_mock_route_response(route_request)
        
        # Convert coordinates to nodes (simplified)
        start_node = Node(
            node_id=f"start_{uuid.uuid4()}",
            latitude=route_request.start['latitude'],
            longitude=route_request.start['longitude']
        )
        
        end_node = Node(
            node_id=f"end_{uuid.uuid4()}",
            latitude=route_request.end['latitude'],
            longitude=route_request.end['longitude']
        )
        
        # Calculate route
        start_time = time.time()
        route_result = router.find_route(start_node, end_node)
        calculation_time = time.time() - start_time
        
        if not route_result:
            return jsonify({
                "status": "error",
                "error": "no_route_found",
                "message": "No accessible route found between the specified points",
                "request_id": str(uuid.uuid4())
            }), 404
        
        # Generate alternatives if requested
        alternatives = []
        if route_request.routing_options.get('include_alternatives', False):
            max_alternatives = route_request.routing_options.get('max_alternatives', 2)
            alternatives = _generate_alternative_routes(
                router, start_node, end_node, route_result, max_alternatives
            )
        
        # Create response
        route_dict = convert_route_result_to_dict(route_result)
        if route_dict:
            route_dict['metadata']['calculation_time'] = calculation_time
        
        response = RouteResponse(
            status="success",
            route=route_dict,
            alternatives=[convert_route_result_to_dict(alt) for alt in alternatives],
            metadata={
                "algorithm_used": algorithm,
                "calculation_time": calculation_time,
                "total_alternatives": len(alternatives),
                "user_profile_applied": user_profile is not None
            }
        )
        
        return jsonify(asdict(response)), 200
        
    except Exception as e:
        logger.error(f"Error in calculate_route: {e}")
        raise

@routing_bp.route('/multicriteria', methods=['POST'])
@validate_json_request(required_fields=['start', 'end', 'criteria'])
@handle_routing_errors
def calculate_multicriteria_route(data):
    """
    Calculate route optimized for multiple criteria
    
    POST /api/v1/routes/multicriteria
    """
    try:
        # Parse and validate request
        request_obj = MultiCriteriaRequest(
            start=data['start'],
            end=data['end'],
            criteria=data['criteria'],
            user_profile=data.get('user_profile')
        )
        
        logger.info(f"Multi-criteria route request with criteria: {request_obj.criteria}")
        
        # Create user profile
        user_profile = create_user_profile_from_request(request_obj.user_profile)
        
        # Initialize multi-criteria router
        if MultiCriteriaAStarRouter:
            router = MultiCriteriaAStarRouter(
                graph=current_app.config.get('GRAPH_INSTANCE')
            )
            
            # Set criteria weights
            router.set_criteria_weights(request_obj.criteria)
        else:
            # Fallback to mock response
            return _create_mock_multicriteria_response(request_obj)
        
        # Convert coordinates to nodes
        start_node = Node(
            node_id=f"start_{uuid.uuid4()}",
            latitude=request_obj.start['latitude'],
            longitude=request_obj.start['longitude']
        )
        
        end_node = Node(
            node_id=f"end_{uuid.uuid4()}",
            latitude=request_obj.end['latitude'],
            longitude=request_obj.end['longitude']
        )
        
        # Calculate route
        start_time = time.time()
        route_result = router.find_route(start_node, end_node)
        calculation_time = time.time() - start_time
        
        if not route_result:
            return jsonify({
                "status": "error",
                "error": "no_route_found",
                "message": "No route satisfying the specified criteria was found",
                "request_id": str(uuid.uuid4())
            }), 404
        
        # Create response with criteria analysis
        route_dict = convert_route_result_to_dict(route_result)
        if route_dict:
            route_dict['metadata']['calculation_time'] = calculation_time
            route_dict['metadata']['criteria_weights'] = request_obj.criteria
            route_dict['criteria_scores'] = {
                'distance_score': _calculate_distance_score(route_result),
                'time_score': _calculate_time_score(route_result),
                'accessibility_score': route_result.accessibility_score,
                'safety_score': _calculate_safety_score(route_result),
                'comfort_score': _calculate_comfort_score(route_result)
            }
        
        response = RouteResponse(
            status="success",
            route=route_dict,
            metadata={
                "algorithm_used": "multicriteria_astar",
                "calculation_time": calculation_time,
                "criteria_applied": request_obj.criteria,
                "optimization_method": "weighted_sum"
            }
        )
        
        return jsonify(asdict(response)), 200
        
    except Exception as e:
        logger.error(f"Error in calculate_multicriteria_route: {e}")
        raise

@routing_bp.route('/personalized', methods=['POST'])
@validate_json_request(required_fields=['start', 'end'])
@handle_routing_errors
def calculate_personalized_route(data):
    """
    Calculate personalized route based on user history and ML models
    
    POST /api/v1/routes/personalized
    """
    try:
        # This endpoint requires authentication
        user_id = _get_user_id_from_request()
        if not user_id:
            return jsonify({
                "status": "error",
                "error": "authentication_required",
                "message": "This endpoint requires user authentication"
            }), 401
        
        # Parse request
        route_request = RouteRequest(
            start=data['start'],
            end=data['end'],
            user_profile=data.get('user_profile'),
            routing_options=data.get('routing_options', {})
        )
        
        logger.info(f"Personalized route request for user {user_id}")
        
        # Load user profile (in production, from database)
        user_profile = _load_user_profile(user_id)
        if not user_profile and route_request.user_profile:
            user_profile = create_user_profile_from_request(route_request.user_profile)
        
        # Initialize personalized router with ML integration
        if PersonalizedAStarRouter and ModelManager:
            model_manager = ModelManager()
            
            # Load ML models
            heuristic_model = model_manager.load_model("heuristic")
            cost_model = model_manager.load_model("cost_prediction")
            
            router = PersonalizedAStarRouter(
                graph=current_app.config.get('GRAPH_INSTANCE'),
                user_profile=user_profile,
                heuristic_model=heuristic_model,
                cost_model=cost_model
            )
        else:
            # Fallback to mock response
            return _create_mock_personalized_response(route_request, user_id)
        
        # Convert coordinates to nodes
        start_node = Node(
            node_id=f"start_{uuid.uuid4()}",
            latitude=route_request.start['latitude'],
            longitude=route_request.start['longitude']
        )
        
        end_node = Node(
            node_id=f"end_{uuid.uuid4()}",
            latitude=route_request.end['latitude'],
            longitude=route_request.end['longitude']
        )
        
        # Calculate personalized route
        start_time = time.time()
        route_result = router.find_route(start_node, end_node)
        calculation_time = time.time() - start_time
        
        if not route_result:
            return jsonify({
                "status": "error",
                "error": "no_route_found", 
                "message": "No personalized route found matching your preferences",
                "request_id": str(uuid.uuid4())
            }), 404
        
        # Create response with personalization insights
        route_dict = convert_route_result_to_dict(route_result)
        if route_dict:
            route_dict['metadata']['calculation_time'] = calculation_time
            route_dict['personalization'] = {
                'user_id': user_id,
                'profile_applied': user_profile.mobility_aid.value if user_profile else 'none',
                'ml_models_used': ['heuristic', 'cost_prediction'],
                'confidence_score': _calculate_personalization_confidence(route_result, user_profile),
                'learning_opportunities': _identify_learning_opportunities(route_result)
            }
        
        response = RouteResponse(
            status="success",
            route=route_dict,
            metadata={
                "algorithm_used": "personalized_astar",
                "calculation_time": calculation_time,
                "personalization_applied": True,
                "ml_enhancement": True
            }
        )
        
        return jsonify(asdict(response)), 200
        
    except Exception as e:
        logger.error(f"Error in calculate_personalized_route: {e}")
        raise

@routing_bp.route('/compare', methods=['POST'])
@validate_json_request(required_fields=['routes'])
@handle_routing_errors  
def compare_routes(data):
    """
    Compare multiple routing options
    
    POST /api/v1/routes/compare
    """
    try:
        routes_data = data['routes']
        comparison_criteria = data.get('criteria', ['distance', 'time', 'accessibility'])
        
        if not isinstance(routes_data, list) or len(routes_data) < 2:
            return jsonify({
                "status": "error",
                "error": "insufficient_routes",
                "message": "At least 2 routes are required for comparison"
            }), 400
        
        logger.info(f"Route comparison request for {len(routes_data)} routes")
        
        # Process routes (in production, these would be route IDs to fetch)
        route_results = []
        for i, route_data in enumerate(routes_data):
            # Mock route result for comparison
            mock_route = _create_mock_route_for_comparison(route_data, i)
            route_results.append(mock_route)
        
        # Perform comparison
        if RouteComparator:
            comparator = RouteComparator()
            comparison_result = comparator.compare_routes(route_results, comparison_criteria)
        else:
            comparison_result = _create_mock_comparison_result(route_results, comparison_criteria)
        
        response = {
            "status": "success",
            "comparison": comparison_result,
            "metadata": {
                "routes_compared": len(routes_data),
                "criteria_used": comparison_criteria,
                "timestamp": datetime.now().isoformat()
            }
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error in compare_routes: {e}")
        raise

@routing_bp.route('/<route_id>/update', methods=['GET'])
@handle_routing_errors
def get_route_updates(route_id):
    """
    Get real-time updates for an active route
    
    GET /api/v1/routes/{route_id}/update
    """
    try:
        # Validate route ID
        if not route_id or len(route_id) < 10:
            return jsonify({
                "status": "error",
                "error": "invalid_route_id",
                "message": "Invalid route identifier"
            }), 400
        
        logger.info(f"Route update request for route {route_id}")
        
        # In production, fetch route from database and check for updates
        # For now, return mock updates
        updates = _get_mock_route_updates(route_id)
        
        response = {
            "status": "success",
            "route_id": route_id,
            "updates": updates,
            "last_updated": datetime.now().isoformat(),
            "next_update_in": 30  # seconds
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error in get_route_updates: {e}")
        raise

# ============================================================================
# HELPER FUNCTIONS FOR MOCK RESPONSES (DEVELOPMENT/TESTING)
# ============================================================================

def _create_mock_route_response(route_request: RouteRequest) -> Tuple[Dict[str, Any], int]:
    """Create mock route response for testing"""
    mock_route = {
        "id": str(uuid.uuid4()),
        "path": [
            {"latitude": route_request.start["latitude"], "longitude": route_request.start["longitude"]},
            {"latitude": (route_request.start["latitude"] + route_request.end["latitude"]) / 2,
             "longitude": (route_request.start["longitude"] + route_request.end["longitude"]) / 2},
            {"latitude": route_request.end["latitude"], "longitude": route_request.end["longitude"]}
        ],
        "total_distance": 1500.0,
        "total_time": 1200.0,
        "accessibility_score": 0.85,
        "warnings": ["Minor elevation change ahead"],
        "metadata": {
            "algorithm_used": "mock_astar",
            "calculation_time": 0.15,
            "nodes_explored": 245
        }
    }
    
    response = RouteResponse(
        status="success",
        route=mock_route,
        metadata={"algorithm_used": "mock", "test_mode": True}
    )
    
    return jsonify(asdict(response)), 200

def _create_mock_multicriteria_response(request_obj: MultiCriteriaRequest) -> Tuple[Dict[str, Any], int]:
    """Create mock multi-criteria response"""
    mock_route = {
        "id": str(uuid.uuid4()),
        "path": [
            {"latitude": request_obj.start["latitude"], "longitude": request_obj.start["longitude"]},
            {"latitude": request_obj.end["latitude"], "longitude": request_obj.end["longitude"]}
        ],
        "total_distance": 1200.0,
        "total_time": 900.0,
        "accessibility_score": 0.9,
        "criteria_scores": {
            "distance_score": 0.8,
            "time_score": 0.85,
            "accessibility_score": 0.9,
            "safety_score": 0.75,
            "comfort_score": 0.8
        },
        "metadata": {
            "algorithm_used": "mock_multicriteria",
            "criteria_weights": request_obj.criteria
        }
    }
    
    response = RouteResponse(status="success", route=mock_route)
    return jsonify(asdict(response)), 200

def _create_mock_personalized_response(route_request: RouteRequest, user_id: str) -> Tuple[Dict[str, Any], int]:
    """Create mock personalized response"""
    mock_route = {
        "id": str(uuid.uuid4()),
        "path": [
            {"latitude": route_request.start["latitude"], "longitude": route_request.start["longitude"]},
            {"latitude": route_request.end["latitude"], "longitude": route_request.end["longitude"]}
        ],
        "total_distance": 1300.0,
        "total_time": 1000.0,
        "accessibility_score": 0.95,
        "personalization": {
            "user_id": user_id,
            "profile_applied": "wheelchair",
            "ml_models_used": ["mock_heuristic", "mock_cost"],
            "confidence_score": 0.87
        },
        "metadata": {
            "algorithm_used": "mock_personalized",
            "personalization_applied": True
        }
    }
    
    response = RouteResponse(status="success", route=mock_route)
    return jsonify(asdict(response)), 200

def _generate_alternative_routes(router, start_node, end_node, primary_route, max_alternatives: int) -> List:
    """Generate alternative routes (simplified)"""
    alternatives = []
    try:
        for i in range(min(max_alternatives, 2)):  # Limit for testing
            # Create slight variation of primary route
            alt_route = _create_alternative_route_variation(primary_route, i)
            if alt_route:
                alternatives.append(alt_route)
    except Exception as e:
        logger.error(f"Error generating alternatives: {e}")
    
    return alternatives

def _create_alternative_route_variation(primary_route, variation_index: int):
    """Create a variation of the primary route"""
    # Mock alternative with slightly different metrics
    if not primary_route:
        return None
    
    variation_factor = 1.0 + (variation_index * 0.1)  # 10% variation per alternative
    
    # Create mock alternative (simplified)
    return type('MockRoute', (), {
        'route_id': f"alt_{uuid.uuid4()}",
        'segments': primary_route.segments,
        'total_distance': primary_route.total_distance * variation_factor,
        'total_time': primary_route.total_time * variation_factor,
        'accessibility_score': max(0.1, primary_route.accessibility_score - (variation_index * 0.1)),
        'warnings': [],
        'recommendations': []
    })()

def _get_user_id_from_request() -> Optional[str]:
    """Extract user ID from request (authentication)"""
    # In production, this would extract from JWT token
    auth_header = request.headers.get('Authorization')
    if auth_header and auth_header.startswith('Bearer '):
        return "mock_user_123"  # Mock user ID
    return None

def _load_user_profile(user_id: str) -> Optional[UserProfile]:
    """Load user profile from database"""
    # Mock user profile for testing
    if UserProfile and MobilityAidType and RoutingPriority:
        return UserProfile(
            user_id=user_id,
            mobility_aid=MobilityAidType.WHEELCHAIR,
            walking_speed=0.8,
            routing_priority=RoutingPriority.ACCESSIBILITY,
            accessibility_constraints={"wheelchair_accessible", "avoid_stairs"},
            environmental_preferences={"prefer_shade": True}
        )
    return None

def _calculate_distance_score(route_result) -> float:
    """Calculate distance optimization score"""
    # Mock implementation
    return min(1.0, 2000.0 / max(route_result.total_distance, 100.0))

def _calculate_time_score(route_result) -> float:
    """Calculate time optimization score"""
    # Mock implementation  
    return min(1.0, 1800.0 / max(route_result.total_time, 60.0))

def _calculate_safety_score(route_result) -> float:
    """Calculate safety score"""
    # Mock implementation
    return 0.8 + (0.2 * route_result.accessibility_score)

def _calculate_comfort_score(route_result) -> float:
    """Calculate comfort score"""
    # Mock implementation
    return 0.75 + (0.25 * route_result.accessibility_score)

def _calculate_personalization_confidence(route_result, user_profile) -> float:
    """Calculate confidence in personalization"""
    if not user_profile:
        return 0.0
    
    # Mock calculation based on profile completeness and route characteristics
    base_confidence = 0.7
    profile_bonus = 0.2 if user_profile.accessibility_constraints else 0.0
    route_bonus = 0.1 if route_result.accessibility_score > 0.8 else 0.0
    
    return min(1.0, base_confidence + profile_bonus + route_bonus)

def _identify_learning_opportunities(route_result) -> List[str]:
    """Identify opportunities for ML learning"""
    opportunities = []
    
    if route_result.accessibility_score < 0.8:
        opportunities.append("accessibility_optimization")
    
    if len(route_result.warnings or []) > 0:
        opportunities.append("hazard_prediction")
    
    opportunities.append("route_preference_learning")
    
    return opportunities

def _create_mock_route_for_comparison(route_data: Dict, index: int):
    """Create mock route object for comparison"""
    return type('MockRoute', (), {
        'route_id': f"route_{index}",
        'total_distance': route_data.get('distance', 1000 + index * 200),
        'total_time': route_data.get('time', 800 + index * 150),
        'accessibility_score': route_data.get('accessibility', 0.8 - index * 0.05),
        'segments': []
    })()

def _create_mock_comparison_result(routes: List, criteria: List[str]) -> Dict[str, Any]:
    """Create mock route comparison result"""
    return {
        "routes": [
            {
                "route_id": route.route_id,
                "scores": {
                    criterion: round(0.8 - i * 0.1, 2) for i, criterion in enumerate(criteria)
                },
                "rank": i + 1,
                "recommended": i == 0
            }
            for i, route in enumerate(routes)
        ],
        "best_route": routes[0].route_id if routes else None,
        "criteria_weights": {criterion: 1.0 / len(criteria) for criterion in criteria},
        "summary": f"Compared {len(routes)} routes using {len(criteria)} criteria"
    }

def _get_mock_route_updates(route_id: str) -> List[Dict[str, Any]]:
    """Get mock real-time route updates"""
    return [
        {
            "type": "traffic_update",
            "message": "Light traffic ahead, no delays expected",
            "severity": "info",
            "location": {"latitude": 40.7500, "longitude": -73.9800},
            "timestamp": datetime.now().isoformat()
        },
        {
            "type": "accessibility_alert", 
            "message": "Elevator at next station is working normally",
            "severity": "info",
            "location": {"latitude": 40.7520, "longitude": -73.9820},
            "timestamp": (datetime.now() - timedelta(minutes=2)).isoformat()
        }
    ]

# Example usage and testing
if __name__ == "__main__":
    print("Core Routing API Endpoints")
    print("=" * 40)
    print("Available endpoints:")
    print("  POST /api/v1/routes/calculate - Basic route calculation")
    print("  POST /api/v1/routes/multicriteria - Multi-criteria optimization")
    print("  POST /api/v1/routes/personalized - Personalized routing")
    print("  POST /api/v1/routes/compare - Route comparison")
    print("  GET  /api/v1/routes/{id}/update - Real-time route updates")
    print("\nAll endpoints include comprehensive error handling and validation.")
