"""
Dynamic cost function prediction for real-time route optimization
"""
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os
from datetime import datetime, timedelta
import uuid
import threading
import time
from collections import defaultdict, deque

# Import other ML components
try:
    from app.ml.heuristic_models import FeatureConfig, ModelMetrics, TrainingData, HeuristicFeatureExtractor
    from app.models.user_profile import UserProfile
    from app.services.routing_algorithm import RouteResult, RouteSegment
except ImportError:
    # For standalone testing
    FeatureConfig = None
    ModelMetrics = None
    TrainingData = None
    HeuristicFeatureExtractor = None
    UserProfile = None
    RouteResult = None
    RouteSegment = None

logger = logging.getLogger(__name__)

@dataclass
class CostPredictionConfig:
    """Configuration for dynamic cost prediction"""
    update_frequency_seconds: int = 300  # 5 minutes
    model_types: List[str] = field(default_factory=lambda: ['random_forest', 'extra_trees', 'neural_network'])
    ensemble_weights: Dict[str, float] = field(default_factory=lambda: {'random_forest': 0.4, 'extra_trees': 0.4, 'neural_network': 0.2})
    real_time_features: bool = True
    traffic_integration: bool = True
    weather_integration: bool = True
    max_prediction_cache_size: int = 10000
    prediction_confidence_threshold: float = 0.8

@dataclass
class RealTimeContext:
    """Real-time context for cost prediction"""
    timestamp: datetime
    traffic_density: float = 0.5  # 0.0 to 1.0
    average_speed: float = 1.4  # m/s
    congestion_level: str = 'moderate'  # low, moderate, high
    weather_conditions: Dict = field(default_factory=dict)
    user_density: int = 0  # Number of users in area
    construction_alerts: List[Dict] = field(default_factory=list)
    accessibility_incidents: List[Dict] = field(default_factory=list)

@dataclass
class CostPrediction:
    """Result of cost prediction"""
    predicted_cost: float
    confidence: float
    model_contributions: Dict[str, float] = field(default_factory=dict)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    real_time_adjustments: Dict[str, float] = field(default_factory=dict)
    prediction_timestamp: datetime = field(default_factory=datetime.now)

class DynamicFeatureExtractor:
    """Enhanced feature extractor for dynamic cost prediction"""
    
    def __init__(self, config: CostPredictionConfig = None):
        """Initialize dynamic feature extractor"""
        self.config = config or CostPredictionConfig()
        self.base_extractor = HeuristicFeatureExtractor() if HeuristicFeatureExtractor else None
        self.scaler = StandardScaler()
        self.feature_history = defaultdict(deque)
        self.max_history_size = 100
        
        logger.info("DynamicFeatureExtractor initialized")
    
    def extract_dynamic_features(self, edge_data: Dict, user_profile: UserProfile = None,
                                real_time_context: RealTimeContext = None,
                                historical_data: List[Dict] = None) -> np.ndarray:
        """
        Extract comprehensive features for dynamic cost prediction
        
        Args:
            edge_data: Edge/segment information
            user_profile: User profile for personalization
            real_time_context: Current real-time conditions
            historical_data: Historical performance data
            
        Returns:
            Feature vector as numpy array
        """
        try:
            features = []
            
            # Basic edge features
            basic_features = self._extract_basic_edge_features(edge_data)
            features.extend(basic_features)
            
            # User profile features
            if user_profile:
                profile_features = self._extract_user_features(user_profile)
                features.extend(profile_features)
            else:
                features.extend([1.4, 1.0, 1.0, 5.0, 0.8, 500.0])  # Default values
            
            # Real-time context features
            if real_time_context and self.config.real_time_features:
                context_features = self._extract_real_time_features(real_time_context)
                features.extend(context_features)
            else:
                features.extend([0.5, 1.4, 0.0, 20.0, 50.0, 0.0])  # Default values
            
            # Historical performance features
            if historical_data:
                historical_features = self._extract_historical_features(edge_data, historical_data)
                features.extend(historical_features)
            else:
                features.extend([0.0, 0.0, 0.0, 0.0, 1.0])  # Default values
            
            # Time-based features
            time_features = self._extract_time_features(real_time_context)
            features.extend(time_features)
            
            # Weather-based features
            if self.config.weather_integration and real_time_context:
                weather_features = self._extract_weather_features(real_time_context.weather_conditions)
                features.extend(weather_features)
            else:
                features.extend([20.0, 50.0, 0.0, 0.0])  # Default values
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"Error extracting dynamic features: {e}")
            return np.array([])
    
    def _extract_basic_edge_features(self, edge_data: Dict) -> List[float]:
        """Extract basic edge characteristics"""
        try:
            features = []
            
            # Physical properties
            length = edge_data.get('length_meters', 100)
            width = edge_data.get('width_meters', 2.0)
            slope = edge_data.get('max_slope_degrees', 0)
            
            features.extend([length, width, slope])
            
            # Surface and quality
            surface_quality = self._get_surface_quality(edge_data.get('surface_type', 'unknown'))
            accessibility_score = edge_data.get('accessibility_score', 1.0)
            comfort_score = edge_data.get('comfort_score', 1.0)
            lighting_score = edge_data.get('lighting_score', 0.5)
            
            features.extend([surface_quality, accessibility_score, comfort_score, lighting_score])
            
            # Obstacles and barriers
            has_steps = 1.0 if edge_data.get('has_steps', False) else 0.0
            has_barriers = 1.0 if edge_data.get('has_barriers', False) else 0.0
            has_curb_cuts = 1.0 if edge_data.get('has_curb_cuts', True) else 0.0
            is_blocked = 1.0 if edge_data.get('is_blocked', False) else 0.0
            
            features.extend([has_steps, has_barriers, has_curb_cuts, is_blocked])
            
            # Safety and security
            safety_score = edge_data.get('safety_score', 0.5)
            security_score = edge_data.get('security_score', 0.5)
            foot_traffic = edge_data.get('typical_foot_traffic', 0.3)
            
            features.extend([safety_score, security_score, foot_traffic])
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting basic edge features: {e}")
            return [100, 2.0, 0, 0.5, 1.0, 1.0, 0.5, 0, 0, 1, 0, 0.5, 0.5, 0.3]
    
    def _extract_user_features(self, user_profile: UserProfile) -> List[float]:
        """Extract user-specific features"""
        try:
            features = []
            
            # Basic mobility characteristics
            walking_speed = user_profile.walking_speed_ms
            energy_efficiency = user_profile.energy_efficiency
            fatigue_factor = user_profile.fatigue_factor
            
            features.extend([walking_speed, energy_efficiency, fatigue_factor])
            
            # Accessibility constraints
            constraints = user_profile.accessibility_constraints
            max_slope = constraints.max_slope_degrees
            min_width = constraints.min_path_width
            max_segment_length = constraints.max_segment_length
            
            features.extend([max_slope, min_width, max_segment_length])
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting user features: {e}")
            return [1.4, 1.0, 1.0, 5.0, 0.8, 500.0]
    
    def _extract_real_time_features(self, context: RealTimeContext) -> List[float]:
        """Extract real-time context features"""
        try:
            features = []
            
            # Traffic and movement
            traffic_density = context.traffic_density
            average_speed = context.average_speed
            congestion_numeric = self._encode_congestion_level(context.congestion_level)
            
            features.extend([traffic_density, average_speed, congestion_numeric])
            
            # User density and crowding
            user_density = min(context.user_density / 100.0, 1.0)  # Normalize to 0-1
            features.append(user_density)
            
            # Incidents and disruptions
            construction_impact = len(context.construction_alerts) * 0.1  # Each alert adds 0.1
            accessibility_impact = len(context.accessibility_incidents) * 0.2  # Each incident adds 0.2
            
            features.extend([construction_impact, accessibility_impact])
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting real-time features: {e}")
            return [0.5, 1.4, 0.0, 0.0, 0.0, 0.0]
    
    def _extract_historical_features(self, edge_data: Dict, historical_data: List[Dict]) -> List[float]:
        """Extract features from historical performance"""
        try:
            features = []
            
            if not historical_data:
                return [0.0, 0.0, 0.0, 0.0, 1.0]
            
            # Filter relevant historical data
            edge_id = edge_data.get('id', '')
            relevant_data = [item for item in historical_data if item.get('edge_id') == edge_id]
            
            if not relevant_data:
                return [0.0, 0.0, 0.0, 0.0, 1.0]
            
            # Calculate historical performance metrics
            actual_costs = [item.get('actual_cost', 0) for item in relevant_data]
            predicted_costs = [item.get('predicted_cost', 0) for item in relevant_data]
            completion_times = [item.get('completion_time', 0) for item in relevant_data]
            satisfaction_scores = [item.get('user_satisfaction', 3) for item in relevant_data]
            
            # Statistical features
            avg_actual_cost = np.mean(actual_costs) if actual_costs else 0
            cost_variance = np.var(actual_costs) if actual_costs else 0
            avg_completion_time = np.mean(completion_times) if completion_times else 0
            avg_satisfaction = np.mean(satisfaction_scores) if satisfaction_scores else 3
            
            # Prediction accuracy
            accuracy = 1.0
            if actual_costs and predicted_costs:
                mse = mean_squared_error(actual_costs, predicted_costs)
                accuracy = max(0.1, 1.0 - min(mse, 1.0))
            
            features.extend([avg_actual_cost, cost_variance, avg_completion_time, avg_satisfaction, accuracy])
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting historical features: {e}")
            return [0.0, 0.0, 0.0, 0.0, 1.0]
    
    def _extract_time_features(self, context: RealTimeContext = None) -> List[float]:
        """Extract time-based features"""
        try:
            features = []
            
            current_time = context.timestamp if context else datetime.now()
            
            # Hour of day (cyclic encoding)
            hour = current_time.hour
            hour_sin = np.sin(2 * np.pi * hour / 24)
            hour_cos = np.cos(2 * np.pi * hour / 24)
            
            # Day of week (cyclic encoding)
            day_of_week = current_time.weekday()
            dow_sin = np.sin(2 * np.pi * day_of_week / 7)
            dow_cos = np.cos(2 * np.pi * day_of_week / 7)
            
            # Time categories
            is_rush_hour = 1.0 if hour in [7, 8, 9, 17, 18, 19] else 0.0
            is_weekend = 1.0 if day_of_week >= 5 else 0.0
            is_night = 1.0 if hour < 6 or hour > 22 else 0.0
            
            features.extend([hour_sin, hour_cos, dow_sin, dow_cos, is_rush_hour, is_weekend, is_night])
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting time features: {e}")
            return [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    
    def _extract_weather_features(self, weather_conditions: Dict) -> List[float]:
        """Extract weather-based features"""
        try:
            features = []
            
            if not weather_conditions:
                return [20.0, 50.0, 0.0, 0.0]
            
            # Temperature
            temperature = weather_conditions.get('temperature_celsius', 20.0)
            
            # Humidity
            humidity = weather_conditions.get('humidity_percent', 50.0)
            
            # Precipitation
            precipitation = weather_conditions.get('precipitation_mm', 0.0)
            
            # Wind speed
            wind_speed = weather_conditions.get('wind_speed_ms', 0.0)
            
            features.extend([temperature, humidity, precipitation, wind_speed])
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting weather features: {e}")
            return [20.0, 50.0, 0.0, 0.0]
    
    def _get_surface_quality(self, surface_type: str) -> float:
        """Get numeric quality score for surface type"""
        surface_scores = {
            'asphalt': 1.0, 'concrete': 0.95, 'paved': 0.9, 'paving_stones': 0.85,
            'compacted': 0.7, 'fine_gravel': 0.6, 'gravel': 0.4, 'unpaved': 0.3,
            'dirt': 0.2, 'grass': 0.15, 'sand': 0.1, 'unknown': 0.5
        }
        return surface_scores.get(surface_type, 0.5)
    
    def _encode_congestion_level(self, congestion_level: str) -> float:
        """Encode congestion level as numeric value"""
        levels = {'low': 0.2, 'moderate': 0.5, 'high': 0.8, 'severe': 1.0}
        return levels.get(congestion_level, 0.5)


class EnsembleCostPredictor:
    """Ensemble model for dynamic cost prediction"""
    
    def __init__(self, config: CostPredictionConfig = None):
        """Initialize ensemble cost predictor"""
        self.config = config or CostPredictionConfig()
        self.models = {}
        self.feature_extractor = DynamicFeatureExtractor(config)
        self.is_trained = False
        self.prediction_cache = {}
        self.cache_lock = threading.Lock()
        self.last_update = datetime.now()
        
        # Initialize models
        self._initialize_models()
        
        logger.info(f"EnsembleCostPredictor initialized with models: {list(self.models.keys())}")
    
    def _initialize_models(self):
        """Initialize individual prediction models"""
        try:
            if 'random_forest' in self.config.model_types:
                self.models['random_forest'] = RandomForestRegressor(
                    n_estimators=100, max_depth=10, min_samples_split=5,
                    min_samples_leaf=2, random_state=42
                )
            
            if 'extra_trees' in self.config.model_types:
                self.models['extra_trees'] = ExtraTreesRegressor(
                    n_estimators=100, max_depth=8, min_samples_split=10,
                    min_samples_leaf=4, random_state=42
                )
            
            if 'neural_network' in self.config.model_types:
                self.models['neural_network'] = MLPRegressor(
                    hidden_layer_sizes=(100, 50), activation='relu',
                    solver='adam', alpha=0.001, random_state=42,
                    max_iter=500
                )
            
            if 'linear' in self.config.model_types:
                self.models['linear'] = Ridge(alpha=1.0, random_state=42)
                
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
    
    def train(self, training_data: TrainingData) -> Dict[str, ModelMetrics]:
        """
        Train all models in the ensemble
        
        Args:
            training_data: Training data container
            
        Returns:
            Dictionary of model metrics for each model
        """
        try:
            logger.info(f"Training ensemble on {len(training_data.features)} samples")
            
            metrics_dict = {}
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                training_data.features, training_data.targets,
                test_size=0.2, random_state=42
            )
            
            # Train each model
            for model_name, model in self.models.items():
                try:
                    start_time = datetime.now()
                    
                    logger.info(f"Training {model_name} model")
                    model.fit(X_train, y_train)
                    
                    # Make predictions
                    y_pred_test = model.predict(X_test)
                    
                    # Calculate metrics
                    metrics = ModelMetrics()
                    metrics.mse = mean_squared_error(y_test, y_pred_test)
                    metrics.rmse = np.sqrt(metrics.mse)
                    metrics.mae = mean_absolute_error(y_test, y_pred_test)
                    metrics.r2 = r2_score(y_test, y_pred_test)
                    
                    # Cross-validation
                    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
                    metrics.cross_val_scores = cv_scores.tolist()
                    
                    metrics.training_time = (datetime.now() - start_time).total_seconds()
                    metrics_dict[model_name] = metrics
                    
                    logger.info(f"{model_name} training completed. R² score: {metrics.r2:.4f}")
                    
                except Exception as e:
                    logger.error(f"Error training {model_name}: {e}")
                    metrics_dict[model_name] = ModelMetrics()
            
            self.is_trained = True
            return metrics_dict
            
        except Exception as e:
            logger.error(f"Error training ensemble: {e}")
            return {}
    
    def predict(self, edge_data: Dict, user_profile: UserProfile = None,
                real_time_context: RealTimeContext = None,
                historical_data: List[Dict] = None) -> CostPrediction:
        """
        Predict cost using ensemble of models
        
        Args:
            edge_data: Edge/segment information
            user_profile: User profile for personalization
            real_time_context: Current real-time conditions
            historical_data: Historical performance data
            
        Returns:
            Cost prediction with confidence and model contributions
        """
        try:
            if not self.is_trained:
                logger.warning("Models not trained, using default cost")
                default_cost = edge_data.get('length_meters', 100) / 1000
                return CostPrediction(predicted_cost=default_cost, confidence=0.1)
            
            # Check cache first
            cache_key = self._create_cache_key(edge_data, user_profile, real_time_context)
            if cache_key in self.prediction_cache:
                cached_prediction = self.prediction_cache[cache_key]
                # Use cached prediction if it's recent (within 1 minute)
                if (datetime.now() - cached_prediction.prediction_timestamp).seconds < 60:
                    return cached_prediction
            
            # Extract features
            features = self.feature_extractor.extract_dynamic_features(
                edge_data, user_profile, real_time_context, historical_data
            )
            
            if len(features) == 0:
                default_cost = edge_data.get('length_meters', 100) / 1000
                return CostPrediction(predicted_cost=default_cost, confidence=0.1)
            
            # Get predictions from all models
            model_predictions = {}
            for model_name, model in self.models.items():
                try:
                    if hasattr(model, 'predict'):
                        pred = model.predict([features])[0]
                        model_predictions[model_name] = max(0.01, pred)
                except Exception as e:
                    logger.error(f"Error predicting with {model_name}: {e}")
                    model_predictions[model_name] = edge_data.get('length_meters', 100) / 1000
            
            # Ensemble prediction with weights
            ensemble_prediction = 0.0
            total_weight = 0.0
            
            for model_name, prediction in model_predictions.items():
                weight = self.config.ensemble_weights.get(model_name, 1.0)
                ensemble_prediction += prediction * weight
                total_weight += weight
            
            if total_weight > 0:
                ensemble_prediction /= total_weight
            
            # Calculate confidence based on model agreement
            confidence = self._calculate_prediction_confidence(model_predictions)
            
            # Apply real-time adjustments
            real_time_adjustments = self._calculate_real_time_adjustments(
                real_time_context, ensemble_prediction
            )
            
            final_prediction = ensemble_prediction
            for adjustment_type, adjustment_value in real_time_adjustments.items():
                final_prediction *= (1 + adjustment_value)
            
            # Create prediction result
            prediction_result = CostPrediction(
                predicted_cost=max(0.01, final_prediction),
                confidence=confidence,
                model_contributions=model_predictions,
                real_time_adjustments=real_time_adjustments,
                prediction_timestamp=datetime.now()
            )
            
            # Cache the prediction
            self._cache_prediction(cache_key, prediction_result)
            
            return prediction_result
            
        except Exception as e:
            logger.error(f"Error predicting cost: {e}")
            default_cost = edge_data.get('length_meters', 100) / 1000
            return CostPrediction(predicted_cost=default_cost, confidence=0.1)
    
    def _calculate_prediction_confidence(self, model_predictions: Dict[str, float]) -> float:
        """Calculate confidence based on model agreement"""
        try:
            if len(model_predictions) < 2:
                return 0.5
            
            predictions = list(model_predictions.values())
            mean_pred = np.mean(predictions)
            std_pred = np.std(predictions)
            
            # Higher agreement (lower std) = higher confidence
            if mean_pred > 0:
                coefficient_of_variation = std_pred / mean_pred
                confidence = max(0.1, 1.0 - min(coefficient_of_variation, 1.0))
            else:
                confidence = 0.5
            
            return confidence
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5
    
    def _calculate_real_time_adjustments(self, context: RealTimeContext, base_cost: float) -> Dict[str, float]:
        """Calculate real-time adjustments to base cost"""
        adjustments = {}
        
        try:
            if not context:
                return adjustments
            
            # Traffic adjustment
            if context.traffic_density > 0.7:
                adjustments['high_traffic'] = 0.2  # 20% increase
            elif context.traffic_density > 0.5:
                adjustments['moderate_traffic'] = 0.1  # 10% increase
            
            # Weather adjustment
            weather = context.weather_conditions
            if weather:
                precipitation = weather.get('precipitation_mm', 0)
                if precipitation > 5:
                    adjustments['heavy_rain'] = 0.3  # 30% increase
                elif precipitation > 1:
                    adjustments['light_rain'] = 0.15  # 15% increase
                
                wind_speed = weather.get('wind_speed_ms', 0)
                if wind_speed > 10:
                    adjustments['strong_wind'] = 0.1  # 10% increase
            
            # Construction impact
            if len(context.construction_alerts) > 0:
                adjustments['construction'] = len(context.construction_alerts) * 0.15
            
            # Accessibility incidents
            if len(context.accessibility_incidents) > 0:
                adjustments['accessibility_issues'] = len(context.accessibility_incidents) * 0.25
            
            # User density adjustment
            if context.user_density > 50:
                adjustments['crowding'] = 0.1  # 10% increase for crowded areas
            
            return adjustments
            
        except Exception as e:
            logger.error(f"Error calculating real-time adjustments: {e}")
            return {}
    
    def _create_cache_key(self, edge_data: Dict, user_profile: UserProfile = None,
                         real_time_context: RealTimeContext = None) -> str:
        """Create cache key for prediction"""
        try:
            key_components = [
                str(edge_data.get('id', 'unknown')),
                str(user_profile.user_id if user_profile else 'anonymous'),
                str(real_time_context.timestamp.strftime('%Y%m%d%H%M') if real_time_context else 'no_context')
            ]
            return '_'.join(key_components)
            
        except Exception:
            return f"default_{datetime.now().strftime('%Y%m%d%H%M')}"
    
    def _cache_prediction(self, cache_key: str, prediction: CostPrediction):
        """Cache prediction result"""
        try:
            with self.cache_lock:
                # Limit cache size
                if len(self.prediction_cache) >= self.config.max_prediction_cache_size:
                    # Remove oldest entries
                    oldest_keys = sorted(
                        self.prediction_cache.keys(),
                        key=lambda k: self.prediction_cache[k].prediction_timestamp
                    )[:100]
                    for key in oldest_keys:
                        del self.prediction_cache[key]
                
                self.prediction_cache[cache_key] = prediction
                
        except Exception as e:
            logger.error(f"Error caching prediction: {e}")
    
    def update_models(self, new_training_data: TrainingData, retrain: bool = False) -> bool:
        """
        Update models with new training data
        
        Args:
            new_training_data: New training data
            retrain: Whether to retrain from scratch or update incrementally
            
        Returns:
            Success status
        """
        try:
            logger.info(f"Updating models with {len(new_training_data.features)} new samples")
            
            if retrain:
                # Retrain all models from scratch
                self.train(new_training_data)
            else:
                # For models that support incremental learning
                for model_name, model in self.models.items():
                    if hasattr(model, 'partial_fit'):
                        model.partial_fit(new_training_data.features, new_training_data.targets)
                    else:
                        # For models without incremental learning, retrain
                        model.fit(new_training_data.features, new_training_data.targets)
            
            self.last_update = datetime.now()
            
            # Clear cache after model update
            with self.cache_lock:
                self.prediction_cache.clear()
            
            logger.info("Model update completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error updating models: {e}")
            return False


def create_sample_dynamic_training_data(num_samples: int = 1000) -> TrainingData:
    """
    Create sample training data for dynamic cost prediction
    
    Args:
        num_samples: Number of samples to generate
        
    Returns:
        Training data container
    """
    try:
        logger.info(f"Creating {num_samples} dynamic cost training samples")
        
        np.random.seed(42)
        features = []
        targets = []
        
        for i in range(num_samples):
            # Basic edge features (14 features)
            edge_features = [
                np.random.uniform(10, 500),      # length
                np.random.uniform(0.8, 3.0),     # width
                np.random.uniform(0, 8.0),       # slope
                np.random.uniform(0.1, 1.0),     # surface_quality
                np.random.uniform(0.1, 1.0),     # accessibility_score
                np.random.uniform(0.1, 1.0),     # comfort_score
                np.random.uniform(0.1, 1.0),     # lighting_score
                np.random.randint(0, 2),         # has_steps
                np.random.randint(0, 2),         # has_barriers
                np.random.randint(0, 2),         # has_curb_cuts
                0,                               # is_blocked
                np.random.uniform(0.1, 1.0),     # safety_score
                np.random.uniform(0.1, 1.0),     # security_score
                np.random.uniform(0.1, 0.8)      # foot_traffic
            ]
            
            # User features (6 features)
            user_features = [
                np.random.uniform(1.0, 2.0),     # walking_speed
                np.random.uniform(0.8, 1.2),     # energy_efficiency
                np.random.uniform(0.8, 1.2),     # fatigue_factor
                np.random.uniform(3.0, 8.0),     # max_slope
                np.random.uniform(0.6, 1.2),     # min_width
                np.random.uniform(100, 800)      # max_segment_length
            ]
            
            # Real-time context features (6 features)
            context_features = [
                np.random.uniform(0.0, 1.0),     # traffic_density
                np.random.uniform(0.8, 2.0),     # average_speed
                np.random.uniform(0.0, 1.0),     # congestion_level
                np.random.uniform(0.0, 1.0),     # user_density
                np.random.uniform(0.0, 0.5),     # construction_impact
                np.random.uniform(0.0, 0.4)      # accessibility_impact
            ]
            
            # Historical features (5 features)
            historical_features = [
                np.random.uniform(0.0, 2.0),     # avg_actual_cost
                np.random.uniform(0.0, 0.5),     # cost_variance
                np.random.uniform(50, 300),      # avg_completion_time
                np.random.uniform(2.0, 5.0),     # avg_satisfaction
                np.random.uniform(0.5, 1.0)      # accuracy
            ]
            
            # Time features (7 features)
            time_features = [
                np.random.uniform(-1, 1),        # hour_sin
                np.random.uniform(-1, 1),        # hour_cos
                np.random.uniform(-1, 1),        # dow_sin
                np.random.uniform(-1, 1),        # dow_cos
                np.random.randint(0, 2),         # is_rush_hour
                np.random.randint(0, 2),         # is_weekend
                np.random.randint(0, 2)          # is_night
            ]
            
            # Weather features (4 features)
            weather_features = [
                np.random.uniform(10, 35),       # temperature
                np.random.uniform(30, 90),       # humidity
                np.random.uniform(0, 20),        # precipitation
                np.random.uniform(0, 15)         # wind_speed
            ]
            
            # Combine all features
            sample_features = (edge_features + user_features + context_features + 
                             historical_features + time_features + weather_features)
            
            # Calculate target cost with realistic factors
            base_cost = edge_features[0] / 1000  # length in km
            
            # Apply various factors
            surface_penalty = (1.0 - edge_features[3]) * 0.5
            slope_penalty = edge_features[2] * 0.1
            accessibility_penalty = (1.0 - edge_features[4]) * 0.3
            traffic_penalty = context_features[0] * 0.2
            weather_penalty = weather_features[2] * 0.02  # precipitation impact
            
            # Calculate final cost
            total_cost = base_cost * (1 + surface_penalty + slope_penalty + 
                                    accessibility_penalty + traffic_penalty + weather_penalty)
            
            features.append(sample_features)
            targets.append(max(0.01, total_cost))
        
        # Create DataFrame
        features_df = pd.DataFrame(features)
        targets_series = pd.Series(targets)
        
        # Create training data container
        training_data = TrainingData(
            features=features_df,
            targets=targets_series,
            feature_names=[f'feature_{i}' for i in range(features_df.shape[1])],
            target_name='dynamic_cost',
            data_source='synthetic_dynamic',
            collection_date=datetime.now().isoformat(),
            user_count=150,
            route_count=num_samples
        )
        
        logger.info("Dynamic cost training data created successfully")
        return training_data
        
    except Exception as e:
        logger.error(f"Error creating dynamic training data: {e}")
        return None


# Example usage and testing
if __name__ == "__main__":
    print("Testing Dynamic Cost Prediction")
    
    # Create sample training data
    training_data = create_sample_dynamic_training_data(800)
    
    if training_data:
        # Test ensemble predictor
        config = CostPredictionConfig(
            model_types=['random_forest', 'extra_trees', 'neural_network'],
            ensemble_weights={'random_forest': 0.4, 'extra_trees': 0.4, 'neural_network': 0.2}
        )
        
        predictor = EnsembleCostPredictor(config)
        metrics = predictor.train(training_data)
        
        for model_name, model_metrics in metrics.items():
            print(f"{model_name} R² score: {model_metrics.r2:.4f}")
        
        # Test prediction
        sample_edge = {
            'id': 'edge_123',
            'length_meters': 200,
            'width_meters': 2.0,
            'max_slope_degrees': 2.5,
            'surface_type': 'asphalt',
            'accessibility_score': 0.9,
            'comfort_score': 0.8,
            'safety_score': 0.7
        }
        
        real_time_context = RealTimeContext(
            timestamp=datetime.now(),
            traffic_density=0.6,
            average_speed=1.2,
            congestion_level='moderate',
            weather_conditions={'temperature_celsius': 25, 'precipitation_mm': 2},
            user_density=30
        )
        
        prediction = predictor.predict(sample_edge, real_time_context=real_time_context)
        print(f"Predicted cost: {prediction.predicted_cost:.4f}")
        print(f"Confidence: {prediction.confidence:.3f}")
        print(f"Model contributions: {prediction.model_contributions}")
        print(f"Real-time adjustments: {prediction.real_time_adjustments}")
    
    print("Dynamic cost prediction test completed")
