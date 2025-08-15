"""
Machine Learning models for heuristic learning and route optimization
"""
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import joblib
import os
from datetime import datetime, timedelta
import uuid

# Import routing components
try:
    from app.services.routing_algorithm import RouteResult, RouteSegment
    from app.models.user_profile import UserProfile
except ImportError:
    # For standalone testing
    RouteResult = None
    RouteSegment = None
    UserProfile = None

logger = logging.getLogger(__name__)

@dataclass
class FeatureConfig:
    """Configuration for feature extraction"""
    include_spatial: bool = True
    include_temporal: bool = True
    include_user_profile: bool = True
    include_historical: bool = True
    spatial_aggregation: str = 'mean'  # mean, sum, max, min
    temporal_window_hours: int = 24

@dataclass
class ModelMetrics:
    """Metrics for model performance evaluation"""
    mse: float = 0.0
    rmse: float = 0.0
    mae: float = 0.0
    r2: float = 0.0
    cross_val_scores: List[float] = field(default_factory=list)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    training_time: float = 0.0
    prediction_time: float = 0.0

@dataclass
class TrainingData:
    """Container for training data and metadata"""
    features: pd.DataFrame
    targets: pd.Series
    feature_names: List[str]
    target_name: str
    data_source: str
    collection_date: str
    user_count: int
    route_count: int

class HeuristicFeatureExtractor:
    """Feature extraction for heuristic learning"""
    
    def __init__(self, feature_config: FeatureConfig = None):
        """Initialize feature extractor"""
        self.config = feature_config or FeatureConfig()
        self.scaler = StandardScaler()
        self.encoders = {}
        self.feature_names = []
        logger.info("HeuristicFeatureExtractor initialized")
    
    def extract_route_features(self, start_node: Dict, end_node: Dict, 
                             route_history: List[Dict] = None,
                             user_profile: UserProfile = None,
                             context: Dict = None) -> np.ndarray:
        """
        Extract features for heuristic prediction
        
        Args:
            start_node: Starting node data
            end_node: Ending node data
            route_history: Historical route data
            user_profile: User profile information
            context: Additional context (time, weather, etc.)
            
        Returns:
            Feature vector as numpy array
        """
        try:
            features = []
            feature_names = []
            
            # Spatial features
            if self.config.include_spatial:
                spatial_features, spatial_names = self._extract_spatial_features(
                    start_node, end_node)
                features.extend(spatial_features)
                feature_names.extend(spatial_names)
            
            # Temporal features
            if self.config.include_temporal and context:
                temporal_features, temporal_names = self._extract_temporal_features(context)
                features.extend(temporal_features)
                feature_names.extend(temporal_names)
            
            # User profile features
            if self.config.include_user_profile and user_profile:
                profile_features, profile_names = self._extract_user_profile_features(user_profile)
                features.extend(profile_features)
                feature_names.extend(profile_names)
            
            # Historical features
            if self.config.include_historical and route_history:
                historical_features, historical_names = self._extract_historical_features(
                    start_node, end_node, route_history)
                features.extend(historical_features)
                feature_names.extend(historical_names)
            
            self.feature_names = feature_names
            return np.array(features)
            
        except Exception as e:
            logger.error(f"Error extracting route features: {e}")
            return np.array([])
    
    def _extract_spatial_features(self, start_node: Dict, end_node: Dict) -> Tuple[List[float], List[str]]:
        """Extract spatial features between nodes"""
        features = []
        feature_names = []
        
        try:
            # Basic distance features
            euclidean_distance = self._calculate_euclidean_distance(start_node, end_node)
            manhattan_distance = self._calculate_manhattan_distance(start_node, end_node)
            
            features.extend([euclidean_distance, manhattan_distance])
            feature_names.extend(['euclidean_distance', 'manhattan_distance'])
            
            # Coordinate differences
            lat_diff = end_node.get('latitude', 0) - start_node.get('latitude', 0)
            lng_diff = end_node.get('longitude', 0) - start_node.get('longitude', 0)
            
            features.extend([lat_diff, lng_diff])
            feature_names.extend(['latitude_diff', 'longitude_diff'])
            
            # Bearing (direction)
            bearing = self._calculate_bearing(start_node, end_node)
            features.append(bearing)
            feature_names.append('bearing')
            
            # Node characteristics
            start_elevation = start_node.get('elevation', 0)
            end_elevation = end_node.get('elevation', 0)
            elevation_diff = end_elevation - start_elevation
            
            features.extend([start_elevation, end_elevation, elevation_diff])
            feature_names.extend(['start_elevation', 'end_elevation', 'elevation_diff'])
            
            return features, feature_names
            
        except Exception as e:
            logger.error(f"Error extracting spatial features: {e}")
            return [], []
    
    def _extract_temporal_features(self, context: Dict) -> Tuple[List[float], List[str]]:
        """Extract temporal features from context"""
        features = []
        feature_names = []
        
        try:
            current_time = context.get('timestamp', datetime.now())
            
            # Time-based features
            hour = current_time.hour
            day_of_week = current_time.weekday()
            month = current_time.month
            
            # Cyclic encoding for time features
            hour_sin = np.sin(2 * np.pi * hour / 24)
            hour_cos = np.cos(2 * np.pi * hour / 24)
            dow_sin = np.sin(2 * np.pi * day_of_week / 7)
            dow_cos = np.cos(2 * np.pi * day_of_week / 7)
            
            features.extend([hour_sin, hour_cos, dow_sin, dow_cos, month])
            feature_names.extend(['hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos', 'month'])
            
            # Weather features
            weather = context.get('weather', {})
            temperature = weather.get('temperature', 20)  # Default 20°C
            humidity = weather.get('humidity', 50)  # Default 50%
            precipitation = weather.get('precipitation', 0)
            
            features.extend([temperature, humidity, precipitation])
            feature_names.extend(['temperature', 'humidity', 'precipitation'])
            
            return features, feature_names
            
        except Exception as e:
            logger.error(f"Error extracting temporal features: {e}")
            return [], []
    
    def _extract_user_profile_features(self, user_profile: UserProfile) -> Tuple[List[float], List[str]]:
        """Extract features from user profile"""
        features = []
        feature_names = []
        
        try:
            # Mobility characteristics
            walking_speed = user_profile.walking_speed_ms
            energy_efficiency = user_profile.energy_efficiency
            fatigue_factor = user_profile.fatigue_factor
            
            features.extend([walking_speed, energy_efficiency, fatigue_factor])
            feature_names.extend(['walking_speed', 'energy_efficiency', 'fatigue_factor'])
            
            # Accessibility constraints
            constraints = user_profile.accessibility_constraints
            max_slope = constraints.max_slope_degrees
            min_width = constraints.min_path_width
            max_segment_length = constraints.max_segment_length
            
            features.extend([max_slope, min_width, max_segment_length])
            feature_names.extend(['max_slope', 'min_path_width', 'max_segment_length'])
            
            # Route preferences (weights)
            prefs = user_profile.route_preferences
            distance_weight = prefs.distance_weight
            comfort_weight = prefs.comfort_weight
            accessibility_weight = prefs.accessibility_weight
            
            features.extend([distance_weight, comfort_weight, accessibility_weight])
            feature_names.extend(['distance_weight', 'comfort_weight', 'accessibility_weight'])
            
            # Mobility aid type (one-hot encoded)
            mobility_aid = user_profile.mobility_aid_type.value
            mobility_features = self._encode_mobility_aid(mobility_aid)
            features.extend(mobility_features)
            feature_names.extend(['wheelchair', 'walker', 'cane', 'scooter', 'guide_dog', 'none'])
            
            return features, feature_names
            
        except Exception as e:
            logger.error(f"Error extracting user profile features: {e}")
            return [], []
    
    def _extract_historical_features(self, start_node: Dict, end_node: Dict, 
                                   route_history: List[Dict]) -> Tuple[List[float], List[str]]:
        """Extract features from historical route data"""
        features = []
        feature_names = []
        
        try:
            if not route_history:
                return [], []
            
            # Filter relevant historical routes
            relevant_routes = self._filter_relevant_routes(start_node, end_node, route_history)
            
            if not relevant_routes:
                return [], []
            
            # Aggregate historical metrics
            distances = [route.get('actual_distance', 0) for route in relevant_routes]
            times = [route.get('actual_time', 0) for route in relevant_routes]
            satisfaction_scores = [route.get('satisfaction', 3) for route in relevant_routes]
            
            # Statistical features
            avg_distance = np.mean(distances) if distances else 0
            std_distance = np.std(distances) if distances else 0
            avg_time = np.mean(times) if times else 0
            std_time = np.std(times) if times else 0
            avg_satisfaction = np.mean(satisfaction_scores) if satisfaction_scores else 3
            
            features.extend([avg_distance, std_distance, avg_time, std_time, avg_satisfaction])
            feature_names.extend(['avg_hist_distance', 'std_hist_distance', 'avg_hist_time', 
                                'std_hist_time', 'avg_hist_satisfaction'])
            
            # Route frequency
            route_count = len(relevant_routes)
            features.append(route_count)
            feature_names.append('historical_route_count')
            
            return features, feature_names
            
        except Exception as e:
            logger.error(f"Error extracting historical features: {e}")
            return [], []
    
    def _calculate_euclidean_distance(self, start_node: Dict, end_node: Dict) -> float:
        """Calculate Euclidean distance between nodes"""
        try:
            lat1, lng1 = start_node.get('latitude', 0), start_node.get('longitude', 0)
            lat2, lng2 = end_node.get('latitude', 0), end_node.get('longitude', 0)
            
            # Convert to meters (approximate)
            lat_diff_meters = (lat2 - lat1) * 111000
            lng_diff_meters = (lng2 - lng1) * 111000 * np.cos(np.radians((lat1 + lat2) / 2))
            
            return np.sqrt(lat_diff_meters**2 + lng_diff_meters**2)
            
        except Exception:
            return 0.0
    
    def _calculate_manhattan_distance(self, start_node: Dict, end_node: Dict) -> float:
        """Calculate Manhattan distance between nodes"""
        try:
            lat1, lng1 = start_node.get('latitude', 0), start_node.get('longitude', 0)
            lat2, lng2 = end_node.get('latitude', 0), end_node.get('longitude', 0)
            
            # Convert to meters (approximate)
            lat_diff_meters = abs(lat2 - lat1) * 111000
            lng_diff_meters = abs(lng2 - lng1) * 111000 * np.cos(np.radians((lat1 + lat2) / 2))
            
            return lat_diff_meters + lng_diff_meters
            
        except Exception:
            return 0.0
    
    def _calculate_bearing(self, start_node: Dict, end_node: Dict) -> float:
        """Calculate bearing between nodes"""
        try:
            lat1 = np.radians(start_node.get('latitude', 0))
            lat2 = np.radians(end_node.get('latitude', 0))
            lng_diff = np.radians(end_node.get('longitude', 0) - start_node.get('longitude', 0))
            
            y = np.sin(lng_diff) * np.cos(lat2)
            x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_diff)
            
            bearing = np.degrees(np.arctan2(y, x))
            return (bearing + 360) % 360  # Normalize to 0-360
            
        except Exception:
            return 0.0
    
    def _encode_mobility_aid(self, mobility_aid: str) -> List[float]:
        """One-hot encode mobility aid type"""
        aids = ['wheelchair_manual', 'wheelchair_electric', 'walker', 'rollator', 'cane', 'mobility_scooter', 'guide_dog', 'none']
        encoded = [1.0 if mobility_aid == aid else 0.0 for aid in aids]
        return encoded[:6]  # Limit to 6 categories
    
    def _filter_relevant_routes(self, start_node: Dict, end_node: Dict, 
                               route_history: List[Dict], radius_km: float = 1.0) -> List[Dict]:
        """Filter historical routes relevant to current start/end points"""
        try:
            relevant_routes = []
            
            for route in route_history:
                route_start = route.get('start_node', {})
                route_end = route.get('end_node', {})
                
                # Check if route start/end are within radius
                start_distance = self._calculate_euclidean_distance(start_node, route_start)
                end_distance = self._calculate_euclidean_distance(end_node, route_end)
                
                if start_distance <= radius_km * 1000 and end_distance <= radius_km * 1000:
                    relevant_routes.append(route)
            
            return relevant_routes
            
        except Exception as e:
            logger.error(f"Error filtering relevant routes: {e}")
            return []


class HeuristicLearningModel:
    """Random Forest model for learning improved heuristics"""
    
    def __init__(self, model_config: Dict = None):
        """Initialize heuristic learning model"""
        self.config = model_config or {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42
        }
        
        self.model = RandomForestRegressor(**self.config)
        self.feature_extractor = HeuristicFeatureExtractor()
        self.is_trained = False
        self.metrics = ModelMetrics()
        self.model_id = str(uuid.uuid4())
        self.created_at = datetime.now()
        
        logger.info(f"HeuristicLearningModel initialized with ID: {self.model_id}")
    
    def train(self, training_data: TrainingData) -> ModelMetrics:
        """
        Train the heuristic learning model
        
        Args:
            training_data: Training data container
            
        Returns:
            Model performance metrics
        """
        try:
            start_time = datetime.now()
            logger.info(f"Training heuristic model on {len(training_data.features)} samples")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                training_data.features, training_data.targets,
                test_size=0.2, random_state=42
            )
            
            # Fit the model
            self.model.fit(X_train, y_train)
            self.is_trained = True
            
            # Make predictions
            y_pred_train = self.model.predict(X_train)
            y_pred_test = self.model.predict(X_test)
            
            # Calculate metrics
            self.metrics = self._calculate_metrics(y_train, y_pred_train, y_test, y_pred_test, X_train)
            self.metrics.training_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Model training completed. R² score: {self.metrics.r2:.4f}")
            return self.metrics
            
        except Exception as e:
            logger.error(f"Error training heuristic model: {e}")
            return ModelMetrics()
    
    def predict_heuristic(self, start_node: Dict, end_node: Dict, 
                         user_profile: UserProfile = None, context: Dict = None,
                         route_history: List[Dict] = None) -> float:
        """
        Predict improved heuristic value
        
        Args:
            start_node: Starting node data
            end_node: Ending node data
            user_profile: User profile for personalization
            context: Current context (time, weather, etc.)
            route_history: Historical route data
            
        Returns:
            Predicted heuristic value
        """
        try:
            if not self.is_trained:
                logger.warning("Model not trained, using default heuristic")
                return self.feature_extractor._calculate_euclidean_distance(start_node, end_node)
            
            # Extract features
            features = self.feature_extractor.extract_route_features(
                start_node, end_node, route_history, user_profile, context
            )
            
            if len(features) == 0:
                return self.feature_extractor._calculate_euclidean_distance(start_node, end_node)
            
            # Predict heuristic adjustment
            prediction_start = datetime.now()
            predicted_value = self.model.predict([features])[0]
            self.metrics.prediction_time = (datetime.now() - prediction_start).total_seconds()
            
            # Ensure positive heuristic
            return max(0.1, predicted_value)
            
        except Exception as e:
            logger.error(f"Error predicting heuristic: {e}")
            return self.feature_extractor._calculate_euclidean_distance(start_node, end_node)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained model"""
        try:
            if not self.is_trained:
                return {}
            
            importance_dict = {}
            if hasattr(self.model, 'feature_importances_'):
                for i, importance in enumerate(self.model.feature_importances_):
                    if i < len(self.feature_extractor.feature_names):
                        feature_name = self.feature_extractor.feature_names[i]
                        importance_dict[feature_name] = float(importance)
            
            return importance_dict
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return {}
    
    def _calculate_metrics(self, y_train: np.ndarray, y_pred_train: np.ndarray,
                          y_test: np.ndarray, y_pred_test: np.ndarray,
                          X_train: np.ndarray) -> ModelMetrics:
        """Calculate comprehensive model metrics"""
        try:
            metrics = ModelMetrics()
            
            # Test set metrics
            metrics.mse = mean_squared_error(y_test, y_pred_test)
            metrics.rmse = np.sqrt(metrics.mse)
            metrics.mae = mean_absolute_error(y_test, y_pred_test)
            metrics.r2 = r2_score(y_test, y_pred_test)
            
            # Cross-validation
            cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='r2')
            metrics.cross_val_scores = cv_scores.tolist()
            
            # Feature importance
            metrics.feature_importance = self.get_feature_importance()
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return ModelMetrics()


class CostPredictionModel:
    """Gradient Boosting model for dynamic cost function prediction"""
    
    def __init__(self, model_config: Dict = None):
        """Initialize cost prediction model"""
        self.config = model_config or {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6,
            'min_samples_split': 10,
            'min_samples_leaf': 4,
            'random_state': 42
        }
        
        self.model = GradientBoostingRegressor(**self.config)
        self.feature_extractor = HeuristicFeatureExtractor()
        self.is_trained = False
        self.metrics = ModelMetrics()
        self.model_id = str(uuid.uuid4())
        self.created_at = datetime.now()
        
        logger.info(f"CostPredictionModel initialized with ID: {self.model_id}")
    
    def train(self, training_data: TrainingData) -> ModelMetrics:
        """Train the cost prediction model"""
        try:
            start_time = datetime.now()
            logger.info(f"Training cost prediction model on {len(training_data.features)} samples")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                training_data.features, training_data.targets,
                test_size=0.2, random_state=42
            )
            
            # Fit the model
            self.model.fit(X_train, y_train)
            self.is_trained = True
            
            # Make predictions
            y_pred_train = self.model.predict(X_train)
            y_pred_test = self.model.predict(X_test)
            
            # Calculate metrics
            self.metrics = self._calculate_metrics(y_train, y_pred_train, y_test, y_pred_test, X_train)
            self.metrics.training_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Cost model training completed. R² score: {self.metrics.r2:.4f}")
            return self.metrics
            
        except Exception as e:
            logger.error(f"Error training cost prediction model: {e}")
            return ModelMetrics()
    
    def predict_cost(self, edge_data: Dict, user_profile: UserProfile = None,
                    context: Dict = None) -> float:
        """
        Predict dynamic cost for an edge
        
        Args:
            edge_data: Edge/segment data
            user_profile: User profile for personalization
            context: Current context information
            
        Returns:
            Predicted cost value
        """
        try:
            if not self.is_trained:
                logger.warning("Cost model not trained, using default cost")
                return edge_data.get('length_meters', 100) / 1000  # Default: distance in km
            
            # Extract edge features
            features = self._extract_edge_features(edge_data, user_profile, context)
            
            if len(features) == 0:
                return edge_data.get('length_meters', 100) / 1000
            
            # Predict cost
            prediction_start = datetime.now()
            predicted_cost = self.model.predict([features])[0]
            self.metrics.prediction_time = (datetime.now() - prediction_start).total_seconds()
            
            # Ensure positive cost
            return max(0.01, predicted_cost)
            
        except Exception as e:
            logger.error(f"Error predicting cost: {e}")
            return edge_data.get('length_meters', 100) / 1000
    
    def _extract_edge_features(self, edge_data: Dict, user_profile: UserProfile = None,
                              context: Dict = None) -> np.ndarray:
        """Extract features for edge cost prediction"""
        try:
            features = []
            
            # Basic edge features
            length = edge_data.get('length_meters', 100)
            width = edge_data.get('width_meters', 2.0)
            slope = edge_data.get('max_slope_degrees', 0)
            
            features.extend([length, width, slope])
            
            # Surface and accessibility
            surface_quality = self._get_surface_quality(edge_data.get('surface_type', 'unknown'))
            accessibility_score = edge_data.get('accessibility_score', 1.0)
            comfort_score = edge_data.get('comfort_score', 1.0)
            
            features.extend([surface_quality, accessibility_score, comfort_score])
            
            # Barriers and obstacles
            has_steps = 1.0 if edge_data.get('has_steps', False) else 0.0
            has_barriers = 1.0 if edge_data.get('has_barriers', False) else 0.0
            is_blocked = 1.0 if edge_data.get('is_blocked', False) else 0.0
            
            features.extend([has_steps, has_barriers, is_blocked])
            
            # User profile features
            if user_profile:
                profile_features, _ = self.feature_extractor._extract_user_profile_features(user_profile)
                features.extend(profile_features[:6])  # Limit features
            else:
                features.extend([1.4, 1.0, 1.0, 5.0, 0.8, 500.0])  # Default values
            
            # Context features
            if context:
                temporal_features, _ = self.feature_extractor._extract_temporal_features(context)
                features.extend(temporal_features[:5])  # Limit features
            else:
                features.extend([0.0, 1.0, 0.0, 1.0, 6.0])  # Default values
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"Error extracting edge features: {e}")
            return np.array([])
    
    def _get_surface_quality(self, surface_type: str) -> float:
        """Get numeric quality score for surface type"""
        surface_scores = {
            'asphalt': 1.0, 'concrete': 0.95, 'paved': 0.9, 'paving_stones': 0.85,
            'compacted': 0.7, 'fine_gravel': 0.6, 'gravel': 0.4, 'unpaved': 0.3,
            'dirt': 0.2, 'grass': 0.15, 'sand': 0.1, 'unknown': 0.5
        }
        return surface_scores.get(surface_type, 0.5)
    
    def _calculate_metrics(self, y_train: np.ndarray, y_pred_train: np.ndarray,
                          y_test: np.ndarray, y_pred_test: np.ndarray,
                          X_train: np.ndarray) -> ModelMetrics:
        """Calculate comprehensive model metrics"""
        try:
            metrics = ModelMetrics()
            
            # Test set metrics
            metrics.mse = mean_squared_error(y_test, y_pred_test)
            metrics.rmse = np.sqrt(metrics.mse)
            metrics.mae = mean_absolute_error(y_test, y_pred_test)
            metrics.r2 = r2_score(y_test, y_pred_test)
            
            # Cross-validation
            cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='r2')
            metrics.cross_val_scores = cv_scores.tolist()
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return ModelMetrics()


def create_sample_training_data(num_samples: int = 1000) -> Tuple[TrainingData, TrainingData]:
    """
    Create sample training data for heuristic and cost models
    
    Args:
        num_samples: Number of samples to generate
        
    Returns:
        Tuple of (heuristic_data, cost_data)
    """
    try:
        logger.info(f"Creating {num_samples} sample training data points")
        
        # Generate synthetic data
        np.random.seed(42)
        
        # Heuristic training data
        heuristic_features = []
        heuristic_targets = []
        
        # Cost training data
        cost_features = []
        cost_targets = []
        
        for i in range(num_samples):
            # Random coordinates (Mumbai region)
            lat1 = 19.0760 + np.random.normal(0, 0.05)
            lng1 = 72.8777 + np.random.normal(0, 0.05)
            lat2 = 19.0760 + np.random.normal(0, 0.05)
            lng2 = 72.8777 + np.random.normal(0, 0.05)
            
            # Calculate actual distance
            euclidean = np.sqrt((lat2-lat1)**2 + (lng2-lng1)**2) * 111000
            
            # Heuristic features: [euclidean, manhattan, lat_diff, lng_diff, bearing, ...]
            heuristic_sample = [
                euclidean,
                abs(lat2-lat1)*111000 + abs(lng2-lng1)*111000,
                lat2-lat1, lng2-lng1,
                np.random.uniform(0, 360),  # bearing
                np.random.uniform(0, 100),  # elevation_diff
                np.random.uniform(0, 1),    # hour_sin
                np.random.uniform(-1, 1),   # hour_cos
                np.random.uniform(15, 35),  # temperature
                np.random.uniform(1.0, 2.0) # walking_speed
            ]
            
            # True heuristic (with some noise)
            true_heuristic = euclidean * (0.9 + np.random.normal(0, 0.1))
            
            heuristic_features.append(heuristic_sample)
            heuristic_targets.append(max(0.1, true_heuristic))
            
            # Cost features: [length, width, slope, surface_quality, ...]
            length = np.random.uniform(10, 500)
            cost_sample = [
                length,
                np.random.uniform(0.8, 3.0),  # width
                np.random.uniform(0, 8.0),    # slope
                np.random.uniform(0.1, 1.0),  # surface_quality
                np.random.uniform(0.1, 1.0),  # accessibility_score
                np.random.uniform(0.1, 1.0),  # comfort_score
                np.random.randint(0, 2),      # has_steps
                np.random.randint(0, 2),      # has_barriers
                0,                            # is_blocked
                np.random.uniform(1.0, 2.0)   # walking_speed
            ]
            
            # True cost (distance-based with adjustments)
            base_cost = length / 1000
            surface_penalty = (1.0 - cost_sample[3]) * 0.5
            slope_penalty = cost_sample[2] * 0.1
            true_cost = base_cost * (1 + surface_penalty + slope_penalty)
            
            cost_features.append(cost_sample)
            cost_targets.append(max(0.01, true_cost))
        
        # Create DataFrames
        heuristic_df = pd.DataFrame(heuristic_features)
        heuristic_series = pd.Series(heuristic_targets)
        
        cost_df = pd.DataFrame(cost_features)
        cost_series = pd.Series(cost_targets)
        
        # Create training data containers
        heuristic_training_data = TrainingData(
            features=heuristic_df,
            targets=heuristic_series,
            feature_names=[f'feature_{i}' for i in range(heuristic_df.shape[1])],
            target_name='heuristic_value',
            data_source='synthetic',
            collection_date=datetime.now().isoformat(),
            user_count=100,
            route_count=num_samples
        )
        
        cost_training_data = TrainingData(
            features=cost_df,
            targets=cost_series,
            feature_names=[f'feature_{i}' for i in range(cost_df.shape[1])],
            target_name='cost_value',
            data_source='synthetic',
            collection_date=datetime.now().isoformat(),
            user_count=100,
            route_count=num_samples
        )
        
        logger.info("Sample training data created successfully")
        return heuristic_training_data, cost_training_data
        
    except Exception as e:
        logger.error(f"Error creating sample training data: {e}")
        return None, None


# Example usage and testing
if __name__ == "__main__":
    print("Testing ML Heuristic Models")
    
    # Create sample training data
    heuristic_data, cost_data = create_sample_training_data(500)
    
    if heuristic_data and cost_data:
        # Test heuristic model
        heuristic_model = HeuristicLearningModel()
        heuristic_metrics = heuristic_model.train(heuristic_data)
        print(f"Heuristic model R² score: {heuristic_metrics.r2:.4f}")
        
        # Test cost model
        cost_model = CostPredictionModel()
        cost_metrics = cost_model.train(cost_data)
        print(f"Cost model R² score: {cost_metrics.r2:.4f}")
        
        # Test predictions
        sample_start = {'latitude': 19.0760, 'longitude': 72.8777, 'elevation': 10}
        sample_end = {'latitude': 19.0800, 'longitude': 72.8800, 'elevation': 15}
        
        heuristic_pred = heuristic_model.predict_heuristic(sample_start, sample_end)
        print(f"Predicted heuristic: {heuristic_pred:.2f}")
        
        sample_edge = {'length_meters': 200, 'width_meters': 2.0, 'max_slope_degrees': 2.5,
                      'surface_type': 'asphalt', 'accessibility_score': 0.9}
        cost_pred = cost_model.predict_cost(sample_edge)
        print(f"Predicted cost: {cost_pred:.4f}")
    
    print("ML heuristic models test completed")
