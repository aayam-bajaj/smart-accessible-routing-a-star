"""
User profile models for personalized accessible routing
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import json
import logging

logger = logging.getLogger(__name__)

class MobilityAidType(Enum):
    """Types of mobility aids"""
    NONE = "none"
    WHEELCHAIR_MANUAL = "wheelchair_manual"
    WHEELCHAIR_ELECTRIC = "wheelchair_electric"
    SCOOTER = "mobility_scooter"
    WALKER = "walker"
    ROLLATOR = "rollator"
    CANE = "cane"
    CRUTCHES = "crutches"
    PROSTHETICS = "prosthetics"
    GUIDE_DOG = "guide_dog"

class RoutingPriority(Enum):
    """Routing optimization priorities"""
    SHORTEST_DISTANCE = "shortest_distance"
    LOWEST_ENERGY = "lowest_energy"
    HIGHEST_COMFORT = "highest_comfort"
    BEST_ACCESSIBILITY = "best_accessibility"
    SAFEST_ROUTE = "safest_route"
    SCENIC_ROUTE = "scenic_route"
    AVOID_CROWDS = "avoid_crowds"

@dataclass
class AccessibilityConstraints:
    """Accessibility constraints for routing"""
    max_slope_degrees: float = 5.0
    min_path_width: float = 0.8  # meters
    avoid_stairs: bool = True
    avoid_escalators: bool = False
    require_handrails: bool = False
    require_rest_areas: bool = False
    max_segment_length: float = 500.0  # meters before requiring rest
    avoid_uneven_surfaces: bool = False
    require_tactile_guidance: bool = False
    avoid_construction: bool = True
    require_elevators: bool = False
    avoid_busy_roads: bool = False
    require_crosswalk_signals: bool = False
    
class WeatherConstraints:
    """Weather-related routing constraints"""
    def __init__(self):
        self.avoid_rain_exposure: bool = False
        self.require_covered_paths: bool = False
        self.avoid_snow_ice: bool = True
        self.require_heated_paths: bool = False
        self.avoid_extreme_temperatures: bool = False
        self.wind_sensitivity: float = 0.5  # 0-1 scale

@dataclass 
class RoutePreferences:
    """User preferences for route characteristics"""
    distance_weight: float = 0.25
    energy_efficiency_weight: float = 0.25
    comfort_weight: float = 0.25
    accessibility_weight: float = 0.15
    surface_weight: float = 0.05
    slope_weight: float = 0.05
    safety_weight: float = 0.1
    time_weight: float = 0.1
    scenic_weight: float = 0.0
    
    def normalize_weights(self):
        """Normalize weights to sum to 1.0"""
        total = (self.distance_weight + self.energy_efficiency_weight + 
                self.comfort_weight + self.accessibility_weight + 
                self.surface_weight + self.slope_weight + 
                self.safety_weight + self.time_weight + self.scenic_weight)
        
        if total > 0:
            self.distance_weight /= total
            self.energy_efficiency_weight /= total
            self.comfort_weight /= total
            self.accessibility_weight /= total
            self.surface_weight /= total
            self.slope_weight /= total
            self.safety_weight /= total
            self.time_weight /= total
            self.scenic_weight /= total

@dataclass
class UserProfile:
    """Comprehensive user profile for personalized routing"""
    user_id: int
    name: str = ""
    
    # Mobility characteristics
    mobility_aid_type: MobilityAidType = MobilityAidType.NONE
    mobility_aid_details: Dict[str, Any] = field(default_factory=dict)
    walking_speed_ms: float = 1.4  # meters per second
    energy_efficiency: float = 1.0  # multiplier for energy calculations
    fatigue_factor: float = 1.0  # how quickly user gets tired
    
    # Physical constraints
    accessibility_constraints: AccessibilityConstraints = field(default_factory=AccessibilityConstraints)
    weather_constraints: WeatherConstraints = field(default_factory=WeatherConstraints)
    
    # Route preferences
    route_preferences: RoutePreferences = field(default_factory=RoutePreferences)
    primary_priority: RoutingPriority = RoutingPriority.BEST_ACCESSIBILITY
    secondary_priority: Optional[RoutingPriority] = None
    
    # Time-based preferences
    preferred_travel_times: List[str] = field(default_factory=lambda: ["day"])
    avoid_rush_hours: bool = False
    
    # Safety preferences
    require_well_lit_paths: bool = False
    avoid_isolated_areas: bool = False
    emergency_contact_integration: bool = False
    
    # Experience and learning
    route_history: List[Dict] = field(default_factory=list)
    feedback_history: List[Dict] = field(default_factory=list)
    learned_preferences: Dict[str, float] = field(default_factory=dict)
    
    # Context-aware settings
    current_weather_condition: str = "clear"
    current_energy_level: float = 1.0  # 0-1 scale
    current_time_of_day: str = "day"
    is_carrying_load: bool = False
    companion_type: Optional[str] = None  # "guide", "caregiver", "friend", etc.
    
    def __post_init__(self):
        """Post-initialization processing"""
        self.route_preferences.normalize_weights()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary for storage/transmission"""
        result = {
            'user_id': self.user_id,
            'name': self.name,
            'mobility_aid_type': self.mobility_aid_type.value if self.mobility_aid_type else None,
            'mobility_aid_details': self.mobility_aid_details,
            'walking_speed_ms': self.walking_speed_ms,
            'energy_efficiency': self.energy_efficiency,
            'fatigue_factor': self.fatigue_factor,
            'accessibility_constraints': self.accessibility_constraints.__dict__,
            'weather_constraints': self.weather_constraints.__dict__,
            'route_preferences': self.route_preferences.__dict__,
            'primary_priority': self.primary_priority.value if self.primary_priority else None,
            'secondary_priority': self.secondary_priority.value if self.secondary_priority else None,
            'preferred_travel_times': self.preferred_travel_times,
            'avoid_rush_hours': self.avoid_rush_hours,
            'require_well_lit_paths': self.require_well_lit_paths,
            'avoid_isolated_areas': self.avoid_isolated_areas,
            'emergency_contact_integration': self.emergency_contact_integration,
            'route_history': self.route_history,
            'feedback_history': self.feedback_history,
            'learned_preferences': self.learned_preferences,
            'current_weather_condition': self.current_weather_condition,
            'current_energy_level': self.current_energy_level,
            'current_time_of_day': self.current_time_of_day,
            'is_carrying_load': self.is_carrying_load,
            'companion_type': self.companion_type
        }
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserProfile':
        """Create profile from dictionary"""
        profile = cls(user_id=data['user_id'])
        
        # Basic attributes
        profile.name = data.get('name', '')
        profile.walking_speed_ms = data.get('walking_speed_ms', 1.4)
        profile.energy_efficiency = data.get('energy_efficiency', 1.0)
        profile.fatigue_factor = data.get('fatigue_factor', 1.0)
        
        # Enum attributes
        if data.get('mobility_aid_type'):
            profile.mobility_aid_type = MobilityAidType(data['mobility_aid_type'])
        if data.get('primary_priority'):
            profile.primary_priority = RoutingPriority(data['primary_priority'])
        if data.get('secondary_priority'):
            profile.secondary_priority = RoutingPriority(data['secondary_priority'])
        
        # Complex attributes
        profile.mobility_aid_details = data.get('mobility_aid_details', {})
        profile.preferred_travel_times = data.get('preferred_travel_times', ['day'])
        profile.route_history = data.get('route_history', [])
        profile.feedback_history = data.get('feedback_history', [])
        profile.learned_preferences = data.get('learned_preferences', {})
        
        # Boolean attributes
        profile.avoid_rush_hours = data.get('avoid_rush_hours', False)
        profile.require_well_lit_paths = data.get('require_well_lit_paths', False)
        profile.avoid_isolated_areas = data.get('avoid_isolated_areas', False)
        profile.emergency_contact_integration = data.get('emergency_contact_integration', False)
        profile.is_carrying_load = data.get('is_carrying_load', False)
        
        # Context attributes
        profile.current_weather_condition = data.get('current_weather_condition', 'clear')
        profile.current_energy_level = data.get('current_energy_level', 1.0)
        profile.current_time_of_day = data.get('current_time_of_day', 'day')
        profile.companion_type = data.get('companion_type')
        
        # Nested objects
        if 'accessibility_constraints' in data:
            constraints_data = data['accessibility_constraints']
            profile.accessibility_constraints = AccessibilityConstraints(**constraints_data)
        
        if 'weather_constraints' in data:
            weather_data = data['weather_constraints']
            profile.weather_constraints = WeatherConstraints()
            for key, value in weather_data.items():
                setattr(profile.weather_constraints, key, value)
        
        if 'route_preferences' in data:
            prefs_data = data['route_preferences']
            profile.route_preferences = RoutePreferences(**prefs_data)
            profile.route_preferences.normalize_weights()
        
        return profile
    
    def update_from_feedback(self, feedback: Dict[str, Any]):
        """Update profile based on route feedback"""
        try:
            # Add to feedback history
            self.feedback_history.append(feedback)
            
            # Learn from feedback patterns
            route_type = feedback.get('route_characteristics', {})
            satisfaction = feedback.get('overall_satisfaction', 3)  # 1-5 scale
            
            # Adjust preferences based on satisfaction
            if satisfaction >= 4:  # Positive feedback
                if route_type.get('low_slope', False):
                    self.learned_preferences['prefer_low_slope'] = min(1.0, 
                        self.learned_preferences.get('prefer_low_slope', 0.5) + 0.1)
                
                if route_type.get('smooth_surface', False):
                    self.learned_preferences['prefer_smooth_surface'] = min(1.0,
                        self.learned_preferences.get('prefer_smooth_surface', 0.5) + 0.1)
                    
            elif satisfaction <= 2:  # Negative feedback
                issues = feedback.get('reported_issues', [])
                for issue in issues:
                    if issue == 'too_steep':
                        # Reduce max slope tolerance
                        self.accessibility_constraints.max_slope_degrees *= 0.9
                    elif issue == 'surface_too_rough':
                        self.learned_preferences['avoid_rough_surface'] = min(1.0,
                            self.learned_preferences.get('avoid_rough_surface', 0.5) + 0.2)
            
            # Limit history size
            if len(self.feedback_history) > 100:
                self.feedback_history = self.feedback_history[-50:]
                
        except Exception as e:
            logger.error(f"Error updating profile from feedback: {e}")
    
    def get_routing_config(self) -> Dict[str, Any]:
        """Get configuration dictionary for routing algorithm"""
        return {
            'distance_weight': self.route_preferences.distance_weight,
            'energy_efficiency_weight': self.route_preferences.energy_efficiency_weight,
            'comfort_weight': self.route_preferences.comfort_weight,
            'accessibility_weight': self.route_preferences.accessibility_weight,
            'surface_weight': self.route_preferences.surface_weight,
            'slope_weight': self.route_preferences.slope_weight,
            'safety_weight': self.route_preferences.safety_weight,
            'time_weight': self.route_preferences.time_weight,
            'scenic_weight': self.route_preferences.scenic_weight,
            'mobility_aid_type': self.mobility_aid_type.value,
            'max_slope_degrees': self.accessibility_constraints.max_slope_degrees,
            'min_path_width': self.accessibility_constraints.min_path_width,
            'avoid_stairs': self.accessibility_constraints.avoid_stairs,
            'walking_speed_ms': self.walking_speed_ms,
            'weather_condition': self.current_weather_condition,
            'time_preference': self.current_time_of_day,
            'energy_level': self.current_energy_level,
            'learned_preferences': self.learned_preferences
        }

# Pre-defined profile templates
class ProfileTemplates:
    """Pre-defined user profile templates for common scenarios"""
    
    @staticmethod
    def wheelchair_user() -> UserProfile:
        """Profile template for wheelchair users"""
        profile = UserProfile(user_id=0)
        profile.mobility_aid_type = MobilityAidType.WHEELCHAIR_MANUAL
        profile.accessibility_constraints.max_slope_degrees = 2.0
        profile.accessibility_constraints.min_path_width = 1.2
        profile.accessibility_constraints.avoid_stairs = True
        profile.accessibility_constraints.require_elevators = True
        profile.route_preferences.accessibility_weight = 0.4
        profile.route_preferences.surface_weight = 0.15
        profile.route_preferences.normalize_weights()
        return profile
    
    @staticmethod
    def elderly_walker() -> UserProfile:
        """Profile template for elderly users with walkers"""
        profile = UserProfile(user_id=0)
        profile.mobility_aid_type = MobilityAidType.WALKER
        profile.walking_speed_ms = 0.8
        profile.fatigue_factor = 1.5
        profile.accessibility_constraints.max_slope_degrees = 3.0
        profile.accessibility_constraints.require_rest_areas = True
        profile.accessibility_constraints.max_segment_length = 200.0
        profile.route_preferences.comfort_weight = 0.35
        profile.route_preferences.energy_efficiency_weight = 0.35
        profile.route_preferences.normalize_weights()
        return profile
    
    @staticmethod
    def visually_impaired() -> UserProfile:
        """Profile template for visually impaired users"""
        profile = UserProfile(user_id=0)
        profile.mobility_aid_type = MobilityAidType.GUIDE_DOG
        profile.accessibility_constraints.require_tactile_guidance = True
        profile.accessibility_constraints.avoid_construction = True
        profile.accessibility_constraints.require_crosswalk_signals = True
        profile.require_well_lit_paths = False  # Light not relevant
        profile.route_preferences.safety_weight = 0.3
        profile.route_preferences.accessibility_weight = 0.3
        profile.route_preferences.normalize_weights()
        return profile
    
    @staticmethod
    def mobility_scooter() -> UserProfile:
        """Profile template for mobility scooter users"""
        profile = UserProfile(user_id=0)
        profile.mobility_aid_type = MobilityAidType.SCOOTER
        profile.walking_speed_ms = 2.5
        profile.accessibility_constraints.max_slope_degrees = 4.0
        profile.accessibility_constraints.min_path_width = 1.0
        profile.accessibility_constraints.avoid_stairs = True
        profile.route_preferences.distance_weight = 0.2
        profile.route_preferences.accessibility_weight = 0.3
        profile.route_preferences.normalize_weights()
        return profile

def create_profile_from_template(template_name: str, user_id: int) -> Optional[UserProfile]:
    """Create a user profile from a template"""
    templates = {
        'wheelchair': ProfileTemplates.wheelchair_user,
        'elderly_walker': ProfileTemplates.elderly_walker,
        'visually_impaired': ProfileTemplates.visually_impaired,
        'mobility_scooter': ProfileTemplates.mobility_scooter
    }
    
    if template_name in templates:
        profile = templates[template_name]()
        profile.user_id = user_id
        return profile
    else:
        logger.warning(f"Unknown profile template: {template_name}")
        return None
