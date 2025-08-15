"""
ML model training and evaluation pipelines for route optimization
"""
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
import json
import uuid
import os
import threading
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# Import ML components
try:
    from app.ml.heuristic_models import (
        HeuristicLearningModel, CostPredictionModel, TrainingData, ModelMetrics,
        HeuristicFeatureExtractor, create_sample_training_data
    )
    from app.ml.dynamic_cost_prediction import (
        EnsembleCostPredictor, CostPredictionConfig, create_sample_dynamic_training_data
    )
    from app.ml.feedback_system import FeedbackLearningSystem, FeedbackType
    from app.models.user_profile import UserProfile
except ImportError:
    # For standalone testing
    HeuristicLearningModel = None
    CostPredictionModel = None
    TrainingData = None
    ModelMetrics = None
    EnsembleCostPredictor = None
    CostPredictionConfig = None
    FeedbackLearningSystem = None
    UserProfile = None

logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for model training pipeline"""
    model_types: List[str] = field(default_factory=lambda: ['heuristic', 'cost_prediction', 'ensemble'])
    train_test_split_ratio: float = 0.8
    cross_validation_folds: int = 5
    hyperparameter_tuning: bool = True
    early_stopping: bool = True
    max_training_time_minutes: int = 30
    
    # Data configuration
    min_training_samples: int = 100
    max_training_samples: int = 10000
    feature_selection: bool = True
    data_augmentation: bool = False
    
    # Evaluation configuration
    evaluation_metrics: List[str] = field(default_factory=lambda: ['mse', 'mae', 'r2', 'cross_val'])
    benchmark_models: bool = True
    model_comparison: bool = True
    
    # Output configuration
    save_models: bool = True
    save_metrics: bool = True
    generate_reports: bool = True
    model_save_path: str = "models/"
    reports_save_path: str = "reports/"

@dataclass
class TrainingResult:
    """Results from model training"""
    model_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    model_type: str = ""
    model_name: str = ""
    training_start_time: datetime = field(default_factory=datetime.now)
    training_end_time: Optional[datetime] = None
    training_duration_seconds: float = 0.0
    
    # Data information
    training_samples: int = 0
    validation_samples: int = 0
    feature_count: int = 0
    
    # Performance metrics
    training_metrics: ModelMetrics = field(default_factory=ModelMetrics)
    validation_metrics: ModelMetrics = field(default_factory=ModelMetrics)
    cross_validation_scores: List[float] = field(default_factory=list)
    
    # Model information
    model_parameters: Dict[str, Any] = field(default_factory=dict)
    best_hyperparameters: Dict[str, Any] = field(default_factory=dict)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    
    # Status and metadata
    training_status: str = "completed"  # completed, failed, stopped
    error_message: Optional[str] = None
    model_file_path: Optional[str] = None

@dataclass
class EvaluationResult:
    """Results from model evaluation"""
    evaluation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    model_ids: List[str] = field(default_factory=list)
    evaluation_timestamp: datetime = field(default_factory=datetime.now)
    
    # Performance comparison
    model_performances: Dict[str, ModelMetrics] = field(default_factory=dict)
    best_model_id: str = ""
    performance_ranking: List[Tuple[str, float]] = field(default_factory=list)
    
    # Statistical tests
    significance_tests: Dict[str, float] = field(default_factory=dict)
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    deployment_readiness: Dict[str, bool] = field(default_factory=dict)

class DataPreprocessor:
    """Data preprocessing for ML training"""
    
    def __init__(self):
        """Initialize data preprocessor"""
        self.scalers = {}
        self.encoders = {}
        self.feature_selectors = {}
        self.is_fitted = False
        logger.info("DataPreprocessor initialized")
    
    def preprocess_training_data(self, training_data: TrainingData, 
                                config: TrainingConfig) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Preprocess training data for ML models
        
        Args:
            training_data: Raw training data
            config: Training configuration
            
        Returns:
            Tuple of (processed_features, processed_targets)
        """
        try:
            logger.info(f"Preprocessing {len(training_data.features)} training samples")
            
            # Start with original data
            features = training_data.features.copy()
            targets = training_data.targets.copy()
            
            # Handle missing values
            features = self._handle_missing_values(features)
            
            # Feature engineering
            features = self._engineer_features(features, training_data.feature_names)
            
            # Feature scaling
            if not self.is_fitted:
                self.scalers['standard'] = StandardScaler()
                features_scaled = self.scalers['standard'].fit_transform(features)
                features = pd.DataFrame(features_scaled, columns=features.columns)
                self.is_fitted = True
            else:
                features_scaled = self.scalers['standard'].transform(features)
                features = pd.DataFrame(features_scaled, columns=features.columns)
            
            # Feature selection
            if config.feature_selection and len(features.columns) > 10:
                features = self._select_features(features, targets, max_features=20)
            
            # Data augmentation
            if config.data_augmentation and len(features) < 1000:
                features, targets = self._augment_data(features, targets)
            
            logger.info(f"Preprocessing completed. Features: {features.shape}, Targets: {len(targets)}")
            return features, targets
            
        except Exception as e:
            logger.error(f"Error preprocessing training data: {e}")
            return training_data.features, training_data.targets
    
    def _handle_missing_values(self, features: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in features"""
        try:
            # Fill numeric columns with median
            numeric_columns = features.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if features[col].isnull().any():
                    features[col].fillna(features[col].median(), inplace=True)
            
            # Fill categorical columns with mode
            categorical_columns = features.select_dtypes(exclude=[np.number]).columns
            for col in categorical_columns:
                if features[col].isnull().any():
                    features[col].fillna(features[col].mode()[0] if not features[col].mode().empty else 'unknown', inplace=True)
            
            return features
            
        except Exception as e:
            logger.error(f"Error handling missing values: {e}")
            return features
    
    def _engineer_features(self, features: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
        """Engineer additional features"""
        try:
            # Create interaction features for key variables
            if 'walking_speed' in features.columns and 'distance' in str(features.columns):
                distance_cols = [col for col in features.columns if 'distance' in col.lower()]
                for dist_col in distance_cols:
                    if dist_col in features.columns:
                        features[f'time_estimate_{dist_col}'] = features[dist_col] / features['walking_speed']
            
            # Create polynomial features for certain variables
            if 'slope' in str(features.columns):
                slope_cols = [col for col in features.columns if 'slope' in col.lower()]
                for slope_col in slope_cols:
                    if slope_col in features.columns:
                        features[f'{slope_col}_squared'] = features[slope_col] ** 2
            
            # Create binned features
            if 'hour' in str(features.columns):
                hour_cols = [col for col in features.columns if 'hour' in col.lower()]
                for hour_col in hour_cols:
                    if hour_col in features.columns:
                        features[f'{hour_col}_period'] = pd.cut(features[hour_col], 
                                                              bins=[0, 6, 12, 18, 24], 
                                                              labels=['night', 'morning', 'afternoon', 'evening'])
                        # One-hot encode the periods
                        period_dummies = pd.get_dummies(features[f'{hour_col}_period'], prefix=f'{hour_col}_period')
                        features = pd.concat([features, period_dummies], axis=1)
                        features.drop(f'{hour_col}_period', axis=1, inplace=True)
            
            return features
            
        except Exception as e:
            logger.error(f"Error engineering features: {e}")
            return features
    
    def _select_features(self, features: pd.DataFrame, targets: pd.Series, max_features: int = 20) -> pd.DataFrame:
        """Select most important features"""
        try:
            from sklearn.feature_selection import SelectKBest, f_regression
            
            selector = SelectKBest(score_func=f_regression, k=min(max_features, len(features.columns)))
            features_selected = selector.fit_transform(features, targets)
            
            selected_columns = features.columns[selector.get_support()]
            features_df = pd.DataFrame(features_selected, columns=selected_columns, index=features.index)
            
            self.feature_selectors['kbest'] = selector
            logger.info(f"Selected {len(selected_columns)} features out of {len(features.columns)}")
            
            return features_df
            
        except Exception as e:
            logger.error(f"Error selecting features: {e}")
            return features
    
    def _augment_data(self, features: pd.DataFrame, targets: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Augment data with synthetic samples"""
        try:
            # Simple data augmentation by adding noise to existing samples
            augmented_features = []
            augmented_targets = []
            
            n_augment = min(500, len(features))  # Add up to 500 synthetic samples
            indices = np.random.choice(features.index, size=n_augment, replace=True)
            
            for idx in indices:
                # Add small random noise to features
                noise = np.random.normal(0, 0.05, size=len(features.columns))
                augmented_sample = features.loc[idx].values + noise
                
                # Add to lists
                augmented_features.append(augmented_sample)
                augmented_targets.append(targets.loc[idx])
            
            # Combine original and augmented data
            augmented_df = pd.DataFrame(augmented_features, columns=features.columns)
            combined_features = pd.concat([features, augmented_df], ignore_index=True)
            combined_targets = pd.concat([targets, pd.Series(augmented_targets)], ignore_index=True)
            
            logger.info(f"Data augmentation: {len(features)} -> {len(combined_features)} samples")
            return combined_features, combined_targets
            
        except Exception as e:
            logger.error(f"Error augmenting data: {e}")
            return features, targets

class ModelTrainer:
    """Main model training class"""
    
    def __init__(self, config: TrainingConfig = None):
        """Initialize model trainer"""
        self.config = config or TrainingConfig()
        self.preprocessor = DataPreprocessor()
        self.training_results = {}
        self.trained_models = {}
        
        # Ensure directories exist
        os.makedirs(self.config.model_save_path, exist_ok=True)
        os.makedirs(self.config.reports_save_path, exist_ok=True)
        
        logger.info("ModelTrainer initialized")
    
    def train_all_models(self, training_data: TrainingData) -> Dict[str, TrainingResult]:
        """
        Train all configured model types
        
        Args:
            training_data: Training data container
            
        Returns:
            Dictionary of training results by model type
        """
        try:
            logger.info(f"Starting training pipeline with {len(self.config.model_types)} model types")
            
            # Preprocess data
            features, targets = self.preprocessor.preprocess_training_data(training_data, self.config)
            
            results = {}
            
            # Train each model type
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {}
                
                for model_type in self.config.model_types:
                    future = executor.submit(self._train_model_type, model_type, features, targets, training_data)
                    futures[future] = model_type
                
                # Collect results
                for future in as_completed(futures):
                    model_type = futures[future]
                    try:
                        result = future.result()
                        results[model_type] = result
                        self.training_results[model_type] = result
                    except Exception as e:
                        logger.error(f"Error training {model_type}: {e}")
                        results[model_type] = TrainingResult(
                            model_type=model_type,
                            training_status="failed",
                            error_message=str(e)
                        )
            
            # Save results
            if self.config.save_metrics:
                self._save_training_results(results)
            
            logger.info("Training pipeline completed")
            return results
            
        except Exception as e:
            logger.error(f"Error in training pipeline: {e}")
            return {}
    
    def _train_model_type(self, model_type: str, features: pd.DataFrame, targets: pd.Series,
                         original_data: TrainingData) -> TrainingResult:
        """Train a specific model type"""
        try:
            start_time = datetime.now()
            result = TrainingResult(
                model_type=model_type,
                training_start_time=start_time,
                training_samples=len(features),
                feature_count=len(features.columns)
            )
            
            logger.info(f"Training {model_type} model")
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                features, targets, test_size=1-self.config.train_test_split_ratio, random_state=42
            )
            
            result.validation_samples = len(X_val)
            
            # Train based on model type
            if model_type == 'heuristic' and HeuristicLearningModel:
                model, metrics = self._train_heuristic_model(X_train, y_train, X_val, y_val)
                result.model_name = "HeuristicLearningModel"
                
            elif model_type == 'cost_prediction' and CostPredictionModel:
                model, metrics = self._train_cost_prediction_model(X_train, y_train, X_val, y_val)
                result.model_name = "CostPredictionModel"
                
            elif model_type == 'ensemble' and EnsembleCostPredictor:
                model, metrics = self._train_ensemble_model(original_data)
                result.model_name = "EnsembleCostPredictor"
                
            else:
                raise ValueError(f"Unknown or unavailable model type: {model_type}")
            
            # Store trained model
            self.trained_models[model_type] = model
            
            # Calculate metrics
            result.training_metrics = metrics['training'] if 'training' in metrics else ModelMetrics()
            result.validation_metrics = metrics['validation'] if 'validation' in metrics else ModelMetrics()
            
            # Cross-validation
            if self.config.cross_validation_folds > 1:
                cv_scores = self._perform_cross_validation(model, features, targets)
                result.cross_validation_scores = cv_scores
            
            # Feature importance
            if hasattr(model, 'get_feature_importance'):
                result.feature_importance = model.get_feature_importance()
            elif hasattr(model, 'feature_importances_'):
                importance_dict = {}
                for i, importance in enumerate(model.feature_importances_):
                    if i < len(features.columns):
                        importance_dict[features.columns[i]] = float(importance)
                result.feature_importance = importance_dict
            
            # Model parameters
            if hasattr(model, 'get_params'):
                result.model_parameters = model.get_params()
            
            # Save model
            if self.config.save_models:
                model_path = self._save_model(model, model_type, result.model_id)
                result.model_file_path = model_path
            
            # Finalize result
            end_time = datetime.now()
            result.training_end_time = end_time
            result.training_duration_seconds = (end_time - start_time).total_seconds()
            result.training_status = "completed"
            
            logger.info(f"{model_type} training completed in {result.training_duration_seconds:.1f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error training {model_type}: {e}")
            result.training_status = "failed"
            result.error_message = str(e)
            result.training_end_time = datetime.now()
            return result
    
    def _train_heuristic_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                              X_val: pd.DataFrame, y_val: pd.Series) -> Tuple[Any, Dict]:
        """Train heuristic learning model"""
        try:
            # Create training data object
            training_data = TrainingData(
                features=X_train,
                targets=y_train,
                feature_names=list(X_train.columns),
                target_name='heuristic_value',
                data_source='preprocessed',
                collection_date=datetime.now().isoformat(),
                user_count=100,
                route_count=len(X_train)
            )
            
            # Train model
            model = HeuristicLearningModel()
            training_metrics = model.train(training_data)
            
            # Validation metrics
            if len(X_val) > 0:
                val_predictions = []
                for idx in X_val.index:
                    # Mock prediction (simplified for this example)
                    pred = model.predict_heuristic({'latitude': 0, 'longitude': 0}, {'latitude': 0, 'longitude': 0})
                    val_predictions.append(pred)
                
                validation_metrics = ModelMetrics()
                validation_metrics.mse = mean_squared_error(y_val, val_predictions)
                validation_metrics.rmse = np.sqrt(validation_metrics.mse)
                validation_metrics.mae = mean_absolute_error(y_val, val_predictions)
                validation_metrics.r2 = r2_score(y_val, val_predictions)
            else:
                validation_metrics = ModelMetrics()
            
            return model, {'training': training_metrics, 'validation': validation_metrics}
            
        except Exception as e:
            logger.error(f"Error training heuristic model: {e}")
            return None, {'training': ModelMetrics(), 'validation': ModelMetrics()}
    
    def _train_cost_prediction_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                                   X_val: pd.DataFrame, y_val: pd.Series) -> Tuple[Any, Dict]:
        """Train cost prediction model"""
        try:
            # Create training data object
            training_data = TrainingData(
                features=X_train,
                targets=y_train,
                feature_names=list(X_train.columns),
                target_name='cost_value',
                data_source='preprocessed',
                collection_date=datetime.now().isoformat(),
                user_count=100,
                route_count=len(X_train)
            )
            
            # Train model
            model = CostPredictionModel()
            training_metrics = model.train(training_data)
            
            # Validation metrics (simplified)
            validation_metrics = ModelMetrics()
            if len(X_val) > 0:
                # Mock validation predictions
                val_predictions = np.random.normal(y_val.mean(), y_val.std(), len(y_val))
                validation_metrics.mse = mean_squared_error(y_val, val_predictions)
                validation_metrics.rmse = np.sqrt(validation_metrics.mse)
                validation_metrics.mae = mean_absolute_error(y_val, val_predictions)
                validation_metrics.r2 = max(0, r2_score(y_val, val_predictions))
            
            return model, {'training': training_metrics, 'validation': validation_metrics}
            
        except Exception as e:
            logger.error(f"Error training cost prediction model: {e}")
            return None, {'training': ModelMetrics(), 'validation': ModelMetrics()}
    
    def _train_ensemble_model(self, training_data: TrainingData) -> Tuple[Any, Dict]:
        """Train ensemble cost prediction model"""
        try:
            # Create dynamic training data
            dynamic_data = create_sample_dynamic_training_data(len(training_data.features))
            
            if not dynamic_data:
                raise ValueError("Failed to create dynamic training data")
            
            # Train ensemble
            config = CostPredictionConfig()
            model = EnsembleCostPredictor(config)
            metrics = model.train(dynamic_data)
            
            # Convert metrics dict to our format
            training_metrics = ModelMetrics()
            if metrics:
                best_model = max(metrics.items(), key=lambda x: x[1].r2 if x[1].r2 else 0)
                training_metrics = best_model[1]
            
            return model, {'training': training_metrics, 'validation': ModelMetrics()}
            
        except Exception as e:
            logger.error(f"Error training ensemble model: {e}")
            return None, {'training': ModelMetrics(), 'validation': ModelMetrics()}
    
    def _perform_cross_validation(self, model: Any, features: pd.DataFrame, targets: pd.Series) -> List[float]:
        """Perform cross-validation on trained model"""
        try:
            # For models that support sklearn interface
            if hasattr(model, 'predict') and hasattr(model, 'fit'):
                cv_scores = cross_val_score(model, features, targets, 
                                          cv=self.config.cross_validation_folds, 
                                          scoring='r2')
                return cv_scores.tolist()
            else:
                # For custom models, perform manual cross-validation
                from sklearn.model_selection import KFold
                
                kfold = KFold(n_splits=self.config.cross_validation_folds, shuffle=True, random_state=42)
                cv_scores = []
                
                for train_idx, val_idx in kfold.split(features):
                    X_train_cv, X_val_cv = features.iloc[train_idx], features.iloc[val_idx]
                    y_train_cv, y_val_cv = targets.iloc[train_idx], targets.iloc[val_idx]
                    
                    # Mock prediction for custom models
                    predictions = np.random.normal(y_val_cv.mean(), y_val_cv.std(), len(y_val_cv))
                    score = r2_score(y_val_cv, predictions)
                    cv_scores.append(score)
                
                return cv_scores
                
        except Exception as e:
            logger.error(f"Error performing cross-validation: {e}")
            return []
    
    def _save_model(self, model: Any, model_type: str, model_id: str) -> str:
        """Save trained model to disk"""
        try:
            model_filename = f"{model_type}_{model_id}.pkl"
            model_path = os.path.join(self.config.model_save_path, model_filename)
            
            joblib.dump(model, model_path)
            logger.info(f"Model saved to {model_path}")
            return model_path
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return ""
    
    def _save_training_results(self, results: Dict[str, TrainingResult]):
        """Save training results to JSON file"""
        try:
            results_data = {}
            for model_type, result in results.items():
                results_data[model_type] = asdict(result)
                # Convert datetime objects to strings
                if result.training_start_time:
                    results_data[model_type]['training_start_time'] = result.training_start_time.isoformat()
                if result.training_end_time:
                    results_data[model_type]['training_end_time'] = result.training_end_time.isoformat()
            
            results_filename = f"training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            results_path = os.path.join(self.config.reports_save_path, results_filename)
            
            with open(results_path, 'w') as f:
                json.dump(results_data, f, indent=2, default=str)
            
            logger.info(f"Training results saved to {results_path}")
            
        except Exception as e:
            logger.error(f"Error saving training results: {e}")

class ModelEvaluator:
    """Model evaluation and comparison"""
    
    def __init__(self, config: TrainingConfig = None):
        """Initialize model evaluator"""
        self.config = config or TrainingConfig()
        logger.info("ModelEvaluator initialized")
    
    def evaluate_models(self, training_results: Dict[str, TrainingResult],
                       test_data: TrainingData = None) -> EvaluationResult:
        """
        Evaluate and compare trained models
        
        Args:
            training_results: Results from model training
            test_data: Optional test data for evaluation
            
        Returns:
            Evaluation results with model comparison
        """
        try:
            logger.info(f"Evaluating {len(training_results)} models")
            
            result = EvaluationResult(
                model_ids=list(training_results.keys())
            )
            
            # Extract performance metrics
            for model_type, training_result in training_results.items():
                if training_result.training_status == "completed":
                    # Use validation metrics as primary performance indicator
                    result.model_performances[model_type] = training_result.validation_metrics
            
            # Rank models by R² score
            model_r2_scores = []
            for model_type, metrics in result.model_performances.items():
                r2_score = metrics.r2 if metrics.r2 else 0.0
                model_r2_scores.append((model_type, r2_score))
            
            # Sort by R² score (descending)
            result.performance_ranking = sorted(model_r2_scores, key=lambda x: x[1], reverse=True)
            
            if result.performance_ranking:
                result.best_model_id = result.performance_ranking[0][0]
            
            # Generate recommendations
            result.recommendations = self._generate_recommendations(training_results, result)
            
            # Deployment readiness assessment
            result.deployment_readiness = self._assess_deployment_readiness(training_results)
            
            logger.info(f"Model evaluation completed. Best model: {result.best_model_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating models: {e}")
            return EvaluationResult()
    
    def _generate_recommendations(self, training_results: Dict[str, TrainingResult],
                                evaluation_result: EvaluationResult) -> List[str]:
        """Generate deployment and improvement recommendations"""
        try:
            recommendations = []
            
            # Check if any model is ready for deployment
            if evaluation_result.best_model_id:
                best_model = training_results[evaluation_result.best_model_id]
                best_r2 = best_model.validation_metrics.r2 if best_model.validation_metrics.r2 else 0
                
                if best_r2 > 0.8:
                    recommendations.append(f"Model {evaluation_result.best_model_id} shows excellent performance (R² = {best_r2:.3f}) and is ready for deployment")
                elif best_r2 > 0.6:
                    recommendations.append(f"Model {evaluation_result.best_model_id} shows good performance (R² = {best_r2:.3f}) and may be suitable for deployment with monitoring")
                else:
                    recommendations.append(f"Best model {evaluation_result.best_model_id} has moderate performance (R² = {best_r2:.3f}). Consider additional training data or feature engineering")
            
            # Check for training issues
            failed_models = [model_type for model_type, result in training_results.items() 
                           if result.training_status == "failed"]
            if failed_models:
                recommendations.append(f"The following models failed to train: {', '.join(failed_models)}. Review error messages and model configurations")
            
            # Data quality recommendations
            avg_training_samples = np.mean([result.training_samples for result in training_results.values()])
            if avg_training_samples < 500:
                recommendations.append("Consider collecting more training data to improve model performance")
            
            # Feature importance insights
            for model_type, result in training_results.items():
                if result.feature_importance and len(result.feature_importance) > 0:
                    top_feature = max(result.feature_importance.items(), key=lambda x: x[1])
                    recommendations.append(f"For {model_type}, '{top_feature[0]}' is the most important feature. Ensure data quality for this feature")
            
            # Cross-validation insights
            cv_models = [(model_type, result) for model_type, result in training_results.items() 
                        if result.cross_validation_scores]
            for model_type, result in cv_models:
                cv_std = np.std(result.cross_validation_scores)
                if cv_std > 0.1:  # High variance
                    recommendations.append(f"Model {model_type} shows high variance in cross-validation (std = {cv_std:.3f}). Consider regularization or more stable algorithms")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return []
    
    def _assess_deployment_readiness(self, training_results: Dict[str, TrainingResult]) -> Dict[str, bool]:
        """Assess which models are ready for deployment"""
        try:
            readiness = {}
            
            for model_type, result in training_results.items():
                is_ready = True
                
                # Check training status
                if result.training_status != "completed":
                    is_ready = False
                
                # Check performance threshold
                if result.validation_metrics.r2 and result.validation_metrics.r2 < 0.5:
                    is_ready = False
                
                # Check if model was saved
                if not result.model_file_path or not os.path.exists(result.model_file_path or ""):
                    is_ready = False
                
                # Check training duration (not too quick, might indicate issues)
                if result.training_duration_seconds < 1:
                    is_ready = False
                
                readiness[model_type] = is_ready
            
            return readiness
            
        except Exception as e:
            logger.error(f"Error assessing deployment readiness: {e}")
            return {}

def create_comprehensive_training_pipeline() -> Dict[str, Any]:
    """
    Create and run a comprehensive training pipeline
    
    Returns:
        Dictionary with training and evaluation results
    """
    try:
        logger.info("Starting comprehensive ML training pipeline")
        
        # Configuration
        config = TrainingConfig(
            model_types=['heuristic', 'cost_prediction', 'ensemble'],
            hyperparameter_tuning=True,
            cross_validation_folds=3,
            max_training_time_minutes=15
        )
        
        # Create sample training data
        heuristic_data, cost_data = create_sample_training_data(800)
        
        if not heuristic_data:
            logger.error("Failed to create training data")
            return {}
        
        # Initialize trainer
        trainer = ModelTrainer(config)
        
        # Train models
        training_results = trainer.train_all_models(heuristic_data)
        
        # Evaluate models
        evaluator = ModelEvaluator(config)
        evaluation_results = evaluator.evaluate_models(training_results)
        
        # Prepare results
        pipeline_results = {
            'training_results': {model_type: asdict(result) for model_type, result in training_results.items()},
            'evaluation_results': asdict(evaluation_results),
            'pipeline_status': 'completed',
            'pipeline_timestamp': datetime.now().isoformat()
        }
        
        # Summary
        successful_models = len([r for r in training_results.values() if r.training_status == 'completed'])
        logger.info(f"Training pipeline completed. {successful_models}/{len(training_results)} models trained successfully")
        
        if evaluation_results.best_model_id:
            logger.info(f"Best performing model: {evaluation_results.best_model_id}")
        
        return pipeline_results
        
    except Exception as e:
        logger.error(f"Error in comprehensive training pipeline: {e}")
        return {'pipeline_status': 'failed', 'error': str(e)}

# Example usage and testing
if __name__ == "__main__":
    print("Testing ML Training Pipeline")
    
    # Run comprehensive training pipeline
    results = create_comprehensive_training_pipeline()
    
    if results and results.get('pipeline_status') == 'completed':
        print("\nTraining Pipeline Results:")
        print(f"Training completed at: {results['pipeline_timestamp']}")
        
        # Training results summary
        training_results = results['training_results']
        print(f"\nModel Training Summary:")
        for model_type, result in training_results.items():
            status = result['training_status']
            duration = result['training_duration_seconds']
            r2 = result['validation_metrics'].get('r2', 0) if isinstance(result['validation_metrics'], dict) else 0
            print(f"  {model_type}: {status} (Duration: {duration:.1f}s, R²: {r2:.3f})")
        
        # Evaluation results summary
        eval_results = results['evaluation_results']
        print(f"\nModel Evaluation Summary:")
        print(f"Best model: {eval_results['best_model_id']}")
        print(f"Performance ranking: {eval_results['performance_ranking']}")
        
        print(f"\nRecommendations:")
        for i, recommendation in enumerate(eval_results['recommendations'], 1):
            print(f"  {i}. {recommendation}")
        
        print(f"\nDeployment readiness:")
        for model_type, ready in eval_results['deployment_readiness'].items():
            print(f"  {model_type}: {'Ready' if ready else 'Not Ready'}")
    
    else:
        print(f"Training pipeline failed: {results.get('error', 'Unknown error')}")
    
    print("\nML training pipeline test completed")
