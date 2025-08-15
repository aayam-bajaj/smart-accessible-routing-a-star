"""
ML model persistence, loading, and version management system
"""
import os
import json
import pickle
import joblib
import gzip
import hashlib
import shutil
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
import threading
import uuid
from contextlib import contextmanager
import warnings
warnings.filterwarnings('ignore')

# Import ML components
try:
    from app.ml.heuristic_models import HeuristicLearningModel, CostPredictionModel, ModelMetrics
    from app.ml.dynamic_cost_prediction import EnsembleCostPredictor, CostPredictionConfig
    from app.ml.training_pipeline import TrainingResult, TrainingConfig
    from app.models.user_profile import UserProfile
except ImportError:
    # For standalone testing
    HeuristicLearningModel = None
    CostPredictionModel = None
    ModelMetrics = None
    EnsembleCostPredictor = None
    CostPredictionConfig = None
    TrainingResult = None
    TrainingConfig = None
    UserProfile = None

logger = logging.getLogger(__name__)

@dataclass
class ModelVersion:
    """Model version information"""
    version_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    version_number: str = "1.0.0"
    model_type: str = ""
    model_name: str = ""
    created_date: datetime = field(default_factory=datetime.now)
    
    # Model metadata
    model_size_bytes: int = 0
    model_hash: str = ""
    training_data_hash: str = ""
    performance_metrics: ModelMetrics = field(default_factory=ModelMetrics)
    
    # File paths
    model_file_path: str = ""
    metadata_file_path: str = ""
    backup_file_path: str = ""
    
    # Deployment info
    deployment_status: str = "development"  # development, staging, production, deprecated
    deployment_date: Optional[datetime] = None
    last_used_date: Optional[datetime] = None
    usage_count: int = 0
    
    # Compatibility info
    framework_version: str = ""
    python_version: str = ""
    dependencies: Dict[str, str] = field(default_factory=dict)
    
    # Description and tags
    description: str = ""
    tags: List[str] = field(default_factory=list)
    author: str = ""

@dataclass
class ModelRegistry:
    """Registry of all model versions"""
    registry_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_date: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    # Models by type
    models: Dict[str, List[ModelVersion]] = field(default_factory=dict)
    active_versions: Dict[str, str] = field(default_factory=dict)  # model_type -> version_id
    
    # Registry settings
    max_versions_per_model: int = 10
    auto_cleanup_enabled: bool = True
    backup_enabled: bool = True

@dataclass
class ModelLoadingConfig:
    """Configuration for model loading"""
    cache_models: bool = True
    lazy_loading: bool = True
    verify_integrity: bool = True
    timeout_seconds: int = 30
    
    # Performance settings
    use_compression: bool = True
    parallel_loading: bool = False
    memory_limit_mb: int = 1024
    
    # Error handling
    retry_attempts: int = 3
    fallback_to_backup: bool = True
    strict_version_matching: bool = False

class ModelStorage:
    """Low-level model storage operations"""
    
    def __init__(self, base_path: str = "models/", config: ModelLoadingConfig = None):
        """Initialize model storage"""
        self.base_path = Path(base_path)
        self.config = config or ModelLoadingConfig()
        self.lock = threading.Lock()
        
        # Create directories
        self.base_path.mkdir(parents=True, exist_ok=True)
        (self.base_path / "active").mkdir(exist_ok=True)
        (self.base_path / "archive").mkdir(exist_ok=True)
        (self.base_path / "backup").mkdir(exist_ok=True)
        
        logger.info(f"ModelStorage initialized at {self.base_path}")
    
    def save_model(self, model: Any, model_version: ModelVersion, 
                   compress: bool = None) -> bool:
        """
        Save model to persistent storage
        
        Args:
            model: The model object to save
            model_version: Version metadata
            compress: Whether to compress the model file
            
        Returns:
            True if successful
        """
        try:
            with self.lock:
                logger.info(f"Saving model {model_version.model_name} v{model_version.version_number}")
                
                # Determine compression
                use_compression = compress if compress is not None else self.config.use_compression
                
                # Create model file path
                filename = f"{model_version.model_type}_{model_version.version_id}"
                if use_compression:
                    filename += ".pkl.gz"
                    model_file_path = self.base_path / "active" / filename
                else:
                    filename += ".pkl"
                    model_file_path = self.base_path / "active" / filename
                
                # Save model
                if use_compression:
                    with gzip.open(model_file_path, 'wb') as f:
                        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
                else:
                    joblib.dump(model, model_file_path, compress=3)
                
                # Update model version info
                model_version.model_file_path = str(model_file_path)
                model_version.model_size_bytes = model_file_path.stat().st_size
                model_version.model_hash = self._calculate_file_hash(model_file_path)
                
                # Save metadata
                metadata_path = self._save_metadata(model_version)
                model_version.metadata_file_path = str(metadata_path)
                
                # Create backup if enabled
                if self.config.fallback_to_backup:
                    backup_path = self._create_backup(model_file_path, model_version)
                    model_version.backup_file_path = str(backup_path)
                
                logger.info(f"Model saved successfully: {model_file_path}")
                return True
                
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def load_model(self, model_version: ModelVersion, 
                   verify_integrity: bool = None) -> Optional[Any]:
        """
        Load model from persistent storage
        
        Args:
            model_version: Version metadata
            verify_integrity: Whether to verify model integrity
            
        Returns:
            Loaded model object or None if failed
        """
        try:
            logger.info(f"Loading model {model_version.model_name} v{model_version.version_number}")
            
            # Determine verification
            verify = verify_integrity if verify_integrity is not None else self.config.verify_integrity
            
            model_path = Path(model_version.model_file_path)
            
            # Check if file exists
            if not model_path.exists():
                logger.warning(f"Model file not found: {model_path}")
                
                # Try backup if available
                if self.config.fallback_to_backup and model_version.backup_file_path:
                    backup_path = Path(model_version.backup_file_path)
                    if backup_path.exists():
                        logger.info("Falling back to backup model")
                        model_path = backup_path
                    else:
                        logger.error("Backup model also not found")
                        return None
                else:
                    return None
            
            # Verify integrity if requested
            if verify:
                current_hash = self._calculate_file_hash(model_path)
                if current_hash != model_version.model_hash:
                    logger.error(f"Model integrity check failed. Expected: {model_version.model_hash}, Got: {current_hash}")
                    return None
            
            # Load model
            if str(model_path).endswith('.gz'):
                with gzip.open(model_path, 'rb') as f:
                    model = pickle.load(f)
            else:
                model = joblib.load(model_path)
            
            # Update usage statistics
            model_version.last_used_date = datetime.now()
            model_version.usage_count += 1
            
            logger.info(f"Model loaded successfully: {model_path}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None
    
    def delete_model(self, model_version: ModelVersion, 
                     archive: bool = True) -> bool:
        """
        Delete model from storage
        
        Args:
            model_version: Version metadata
            archive: Whether to archive instead of permanent delete
            
        Returns:
            True if successful
        """
        try:
            with self.lock:
                logger.info(f"Deleting model {model_version.model_name} v{model_version.version_number}")
                
                model_path = Path(model_version.model_file_path)
                metadata_path = Path(model_version.metadata_file_path) if model_version.metadata_file_path else None
                backup_path = Path(model_version.backup_file_path) if model_version.backup_file_path else None
                
                if archive:
                    # Move to archive
                    archive_dir = self.base_path / "archive" / datetime.now().strftime("%Y%m")
                    archive_dir.mkdir(parents=True, exist_ok=True)
                    
                    if model_path.exists():
                        archive_model_path = archive_dir / model_path.name
                        shutil.move(str(model_path), str(archive_model_path))
                    
                    if metadata_path and metadata_path.exists():
                        archive_metadata_path = archive_dir / metadata_path.name
                        shutil.move(str(metadata_path), str(archive_metadata_path))
                        
                else:
                    # Permanent delete
                    if model_path.exists():
                        model_path.unlink()
                    if metadata_path and metadata_path.exists():
                        metadata_path.unlink()
                    if backup_path and backup_path.exists():
                        backup_path.unlink()
                
                logger.info(f"Model deleted successfully")
                return True
                
        except Exception as e:
            logger.error(f"Error deleting model: {e}")
            return False
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file"""
        try:
            hash_sha256 = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
            
        except Exception as e:
            logger.error(f"Error calculating file hash: {e}")
            return ""
    
    def _save_metadata(self, model_version: ModelVersion) -> Path:
        """Save model metadata to JSON file"""
        try:
            metadata_filename = f"{model_version.model_type}_{model_version.version_id}_metadata.json"
            metadata_path = self.base_path / "active" / metadata_filename
            
            # Convert to dictionary for JSON serialization
            metadata = asdict(model_version)
            
            # Convert datetime objects to ISO strings
            for key, value in metadata.items():
                if isinstance(value, datetime):
                    metadata[key] = value.isoformat()
                elif key == 'created_date' and value:
                    metadata[key] = value if isinstance(value, str) else value.isoformat()
                elif key == 'deployment_date' and value:
                    metadata[key] = value if isinstance(value, str) else value.isoformat()
                elif key == 'last_used_date' and value:
                    metadata[key] = value if isinstance(value, str) else value.isoformat()
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            return metadata_path
            
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
            return Path()
    
    def _create_backup(self, model_path: Path, model_version: ModelVersion) -> Path:
        """Create backup of model file"""
        try:
            backup_dir = self.base_path / "backup"
            backup_filename = f"{model_version.model_type}_{model_version.version_id}_backup{model_path.suffix}"
            backup_path = backup_dir / backup_filename
            
            shutil.copy2(model_path, backup_path)
            logger.info(f"Backup created: {backup_path}")
            
            return backup_path
            
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            return Path()

class ModelManager:
    """High-level model management operations"""
    
    def __init__(self, storage: ModelStorage = None, config: ModelLoadingConfig = None):
        """Initialize model manager"""
        self.storage = storage or ModelStorage()
        self.config = config or ModelLoadingConfig()
        self.registry = ModelRegistry()
        self.model_cache = {} if self.config.cache_models else None
        self.lock = threading.Lock()
        
        # Load existing registry
        self._load_registry()
        
        logger.info("ModelManager initialized")
    
    def register_model(self, model: Any, model_type: str, model_name: str,
                      version_number: str = None, description: str = "",
                      tags: List[str] = None, training_result: TrainingResult = None) -> ModelVersion:
        """
        Register a new model version
        
        Args:
            model: The model object
            model_type: Type of model (e.g., 'heuristic', 'cost_prediction')
            model_name: Human-readable model name
            version_number: Version string (auto-generated if None)
            description: Model description
            tags: Model tags for categorization
            training_result: Optional training results
            
        Returns:
            ModelVersion object
        """
        try:
            with self.lock:
                logger.info(f"Registering model {model_name} of type {model_type}")
                
                # Generate version number if not provided
                if version_number is None:
                    existing_versions = self.registry.models.get(model_type, [])
                    version_number = self._generate_version_number(existing_versions)
                
                # Create model version
                model_version = ModelVersion(
                    version_number=version_number,
                    model_type=model_type,
                    model_name=model_name,
                    description=description,
                    tags=tags or [],
                    framework_version=self._get_framework_version(),
                    python_version=self._get_python_version(),
                    dependencies=self._get_dependencies()
                )
                
                # Add training metrics if provided
                if training_result:
                    model_version.performance_metrics = training_result.validation_metrics
                    model_version.training_data_hash = self._calculate_training_data_hash(training_result)
                
                # Save model to storage
                if self.storage.save_model(model, model_version):
                    # Add to registry
                    if model_type not in self.registry.models:
                        self.registry.models[model_type] = []
                    
                    self.registry.models[model_type].append(model_version)
                    
                    # Set as active version if it's the first or performs better
                    if (model_type not in self.registry.active_versions or 
                        self._is_better_model(model_version, model_type)):
                        self.registry.active_versions[model_type] = model_version.version_id
                        model_version.deployment_status = "staging"
                    
                    # Clean up old versions if needed
                    self._cleanup_old_versions(model_type)
                    
                    # Save registry
                    self._save_registry()
                    
                    logger.info(f"Model registered successfully: {model_version.version_id}")
                    return model_version
                else:
                    raise RuntimeError("Failed to save model to storage")
                
        except Exception as e:
            logger.error(f"Error registering model: {e}")
            raise
    
    def load_model(self, model_type: str, version_id: str = None, 
                   use_cache: bool = None) -> Optional[Any]:
        """
        Load a model by type and version
        
        Args:
            model_type: Type of model to load
            version_id: Specific version ID (latest active if None)
            use_cache: Whether to use cached model
            
        Returns:
            Loaded model object or None
        """
        try:
            # Use cache setting if not specified
            use_cache = use_cache if use_cache is not None else self.config.cache_models
            
            # Get version ID if not specified
            if version_id is None:
                version_id = self.registry.active_versions.get(model_type)
                if not version_id:
                    logger.error(f"No active version found for model type: {model_type}")
                    return None
            
            # Check cache first
            cache_key = f"{model_type}:{version_id}"
            if use_cache and self.model_cache and cache_key in self.model_cache:
                logger.info(f"Loading model from cache: {cache_key}")
                return self.model_cache[cache_key]
            
            # Find model version
            model_version = self._find_model_version(model_type, version_id)
            if not model_version:
                logger.error(f"Model version not found: {model_type}:{version_id}")
                return None
            
            # Load from storage
            model = self.storage.load_model(model_version)
            
            if model:
                # Cache if enabled
                if use_cache and self.model_cache is not None:
                    self.model_cache[cache_key] = model
                
                # Update usage statistics
                model_version.last_used_date = datetime.now()
                model_version.usage_count += 1
                self._save_registry()
                
                logger.info(f"Model loaded successfully: {model_type}:{version_id}")
                
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None
    
    def get_model_info(self, model_type: str = None) -> Dict[str, List[Dict]]:
        """
        Get information about registered models
        
        Args:
            model_type: Specific model type (all types if None)
            
        Returns:
            Dictionary of model information
        """
        try:
            info = {}
            
            model_types = [model_type] if model_type else list(self.registry.models.keys())
            
            for mtype in model_types:
                if mtype in self.registry.models:
                    models_info = []
                    for version in self.registry.models[mtype]:
                        version_info = {
                            'version_id': version.version_id,
                            'version_number': version.version_number,
                            'model_name': version.model_name,
                            'created_date': version.created_date.isoformat() if isinstance(version.created_date, datetime) else version.created_date,
                            'deployment_status': version.deployment_status,
                            'size_mb': round(version.model_size_bytes / 1024 / 1024, 2),
                            'performance': asdict(version.performance_metrics),
                            'usage_count': version.usage_count,
                            'tags': version.tags,
                            'description': version.description
                        }
                        models_info.append(version_info)
                    
                    # Sort by creation date (newest first)
                    models_info.sort(key=lambda x: x['created_date'], reverse=True)
                    info[mtype] = models_info
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {}
    
    def set_active_version(self, model_type: str, version_id: str) -> bool:
        """
        Set active version for a model type
        
        Args:
            model_type: Model type
            version_id: Version to set as active
            
        Returns:
            True if successful
        """
        try:
            with self.lock:
                # Verify version exists
                model_version = self._find_model_version(model_type, version_id)
                if not model_version:
                    logger.error(f"Model version not found: {model_type}:{version_id}")
                    return False
                
                # Update active version
                old_version_id = self.registry.active_versions.get(model_type)
                self.registry.active_versions[model_type] = version_id
                
                # Update deployment status
                model_version.deployment_status = "staging"
                model_version.deployment_date = datetime.now()
                
                # Update old version status
                if old_version_id:
                    old_version = self._find_model_version(model_type, old_version_id)
                    if old_version:
                        old_version.deployment_status = "development"
                
                # Clear cache for this model type
                if self.model_cache:
                    cache_keys_to_remove = [key for key in self.model_cache.keys() 
                                          if key.startswith(f"{model_type}:")]
                    for key in cache_keys_to_remove:
                        del self.model_cache[key]
                
                # Save registry
                self._save_registry()
                
                logger.info(f"Active version set: {model_type} -> {version_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error setting active version: {e}")
            return False
    
    def delete_model_version(self, model_type: str, version_id: str,
                           archive: bool = True) -> bool:
        """
        Delete a specific model version
        
        Args:
            model_type: Model type
            version_id: Version to delete
            archive: Whether to archive instead of permanent delete
            
        Returns:
            True if successful
        """
        try:
            with self.lock:
                logger.info(f"Deleting model version: {model_type}:{version_id}")
                
                # Find model version
                model_version = self._find_model_version(model_type, version_id)
                if not model_version:
                    logger.error(f"Model version not found: {model_type}:{version_id}")
                    return False
                
                # Check if it's the active version
                if self.registry.active_versions.get(model_type) == version_id:
                    logger.warning(f"Attempting to delete active version: {model_type}:{version_id}")
                    
                    # Find another version to set as active
                    other_versions = [v for v in self.registry.models.get(model_type, []) 
                                    if v.version_id != version_id]
                    if other_versions:
                        # Set most recent as active
                        other_versions.sort(key=lambda x: x.created_date, reverse=True)
                        new_active = other_versions[0]
                        self.registry.active_versions[model_type] = new_active.version_id
                        new_active.deployment_status = "staging"
                        logger.info(f"Set new active version: {new_active.version_id}")
                    else:
                        # Remove from active versions
                        del self.registry.active_versions[model_type]
                        logger.info(f"No other versions available for {model_type}")
                
                # Delete from storage
                if self.storage.delete_model(model_version, archive):
                    # Remove from registry
                    self.registry.models[model_type] = [
                        v for v in self.registry.models[model_type] 
                        if v.version_id != version_id
                    ]
                    
                    # Remove from cache
                    if self.model_cache:
                        cache_key = f"{model_type}:{version_id}"
                        if cache_key in self.model_cache:
                            del self.model_cache[cache_key]
                    
                    # Save registry
                    self._save_registry()
                    
                    logger.info(f"Model version deleted successfully")
                    return True
                else:
                    logger.error("Failed to delete model from storage")
                    return False
                
        except Exception as e:
            logger.error(f"Error deleting model version: {e}")
            return False
    
    def cleanup_old_models(self, days_old: int = 30, 
                          keep_active: bool = True) -> int:
        """
        Clean up old model versions
        
        Args:
            days_old: Delete models older than this many days
            keep_active: Whether to preserve active versions
            
        Returns:
            Number of models cleaned up
        """
        try:
            cleanup_count = 0
            cutoff_date = datetime.now() - timedelta(days=days_old)
            
            with self.lock:
                for model_type, versions in self.registry.models.items():
                    active_version_id = self.registry.active_versions.get(model_type)
                    
                    versions_to_delete = []
                    for version in versions:
                        # Skip if it's the active version and keep_active is True
                        if keep_active and version.version_id == active_version_id:
                            continue
                        
                        # Check if old enough
                        version_date = version.created_date
                        if isinstance(version_date, str):
                            version_date = datetime.fromisoformat(version_date)
                        
                        if version_date < cutoff_date:
                            versions_to_delete.append(version.version_id)
                    
                    # Delete old versions
                    for version_id in versions_to_delete:
                        if self.delete_model_version(model_type, version_id, archive=True):
                            cleanup_count += 1
            
            logger.info(f"Cleaned up {cleanup_count} old model versions")
            return cleanup_count
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return 0
    
    def _find_model_version(self, model_type: str, version_id: str) -> Optional[ModelVersion]:
        """Find a specific model version"""
        versions = self.registry.models.get(model_type, [])
        for version in versions:
            if version.version_id == version_id:
                return version
        return None
    
    def _is_better_model(self, new_version: ModelVersion, model_type: str) -> bool:
        """Check if new model is better than current active version"""
        try:
            current_version_id = self.registry.active_versions.get(model_type)
            if not current_version_id:
                return True
            
            current_version = self._find_model_version(model_type, current_version_id)
            if not current_version:
                return True
            
            # Compare R² scores (higher is better)
            new_r2 = new_version.performance_metrics.r2 or 0.0
            current_r2 = current_version.performance_metrics.r2 or 0.0
            
            return new_r2 > current_r2
            
        except Exception as e:
            logger.error(f"Error comparing models: {e}")
            return False
    
    def _generate_version_number(self, existing_versions: List[ModelVersion]) -> str:
        """Generate next version number"""
        try:
            if not existing_versions:
                return "1.0.0"
            
            # Find highest version
            max_version = [0, 0, 0]
            for version in existing_versions:
                try:
                    parts = version.version_number.split('.')
                    version_parts = [int(p) for p in parts[:3]]
                    if version_parts > max_version:
                        max_version = version_parts
                except (ValueError, IndexError):
                    continue
            
            # Increment minor version
            max_version[1] += 1
            return '.'.join(map(str, max_version))
            
        except Exception as e:
            logger.error(f"Error generating version number: {e}")
            return f"{len(existing_versions) + 1}.0.0"
    
    def _cleanup_old_versions(self, model_type: str):
        """Clean up old versions to maintain max count"""
        try:
            versions = self.registry.models.get(model_type, [])
            if len(versions) <= self.registry.max_versions_per_model:
                return
            
            # Sort by creation date (oldest first)
            versions.sort(key=lambda x: x.created_date if isinstance(x.created_date, datetime) 
                         else datetime.fromisoformat(x.created_date))
            
            # Keep active version and most recent versions
            active_version_id = self.registry.active_versions.get(model_type)
            versions_to_keep = []
            versions_to_delete = []
            
            # Always keep active version
            for version in reversed(versions):  # Start with newest
                if len(versions_to_keep) < self.registry.max_versions_per_model:
                    versions_to_keep.append(version)
                elif version.version_id == active_version_id:
                    versions_to_keep.append(version)
                else:
                    versions_to_delete.append(version)
            
            # Delete excess versions
            for version in versions_to_delete:
                self.storage.delete_model(version, archive=True)
            
            # Update registry
            self.registry.models[model_type] = versions_to_keep
            
            if versions_to_delete:
                logger.info(f"Cleaned up {len(versions_to_delete)} old versions for {model_type}")
                
        except Exception as e:
            logger.error(f"Error cleaning up old versions: {e}")
    
    def _load_registry(self):
        """Load model registry from file"""
        try:
            registry_path = self.storage.base_path / "model_registry.json"
            if registry_path.exists():
                with open(registry_path, 'r') as f:
                    registry_data = json.load(f)
                
                # Reconstruct registry
                self.registry = ModelRegistry(**registry_data)
                
                # Convert model data back to ModelVersion objects
                models_dict = {}
                for model_type, versions_data in registry_data.get('models', {}).items():
                    versions = []
                    for version_data in versions_data:
                        # Convert datetime strings back to datetime objects
                        for date_field in ['created_date', 'deployment_date', 'last_used_date']:
                            if date_field in version_data and version_data[date_field]:
                                if isinstance(version_data[date_field], str):
                                    version_data[date_field] = datetime.fromisoformat(version_data[date_field])
                        
                        # Convert performance metrics
                        if 'performance_metrics' in version_data:
                            metrics_data = version_data['performance_metrics']
                            if isinstance(metrics_data, dict) and ModelMetrics:
                                version_data['performance_metrics'] = ModelMetrics(**metrics_data)
                        
                        version = ModelVersion(**version_data)
                        versions.append(version)
                    
                    models_dict[model_type] = versions
                
                self.registry.models = models_dict
                logger.info(f"Loaded registry with {len(models_dict)} model types")
            else:
                logger.info("No existing registry found, starting fresh")
                
        except Exception as e:
            logger.error(f"Error loading registry: {e}")
            self.registry = ModelRegistry()
    
    def _save_registry(self):
        """Save model registry to file"""
        try:
            registry_path = self.storage.base_path / "model_registry.json"
            
            # Convert to dictionary for JSON serialization
            registry_data = asdict(self.registry)
            
            # Convert datetime objects and ModelMetrics to serializable format
            for model_type, versions in registry_data['models'].items():
                for version in versions:
                    # Convert datetime objects
                    for date_field in ['created_date', 'deployment_date', 'last_used_date']:
                        if date_field in version and version[date_field]:
                            if isinstance(version[date_field], datetime):
                                version[date_field] = version[date_field].isoformat()
                    
                    # Convert ModelMetrics to dict
                    if 'performance_metrics' in version:
                        metrics = version['performance_metrics']
                        if hasattr(metrics, '__dict__'):
                            version['performance_metrics'] = asdict(metrics)
            
            # Convert registry datetime fields
            for date_field in ['created_date', 'last_updated']:
                if date_field in registry_data and registry_data[date_field]:
                    if isinstance(registry_data[date_field], datetime):
                        registry_data[date_field] = registry_data[date_field].isoformat()
            
            # Save to file
            with open(registry_path, 'w') as f:
                json.dump(registry_data, f, indent=2, default=str)
            
            logger.debug("Registry saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving registry: {e}")
    
    def _get_framework_version(self) -> str:
        """Get current ML framework version"""
        try:
            import sklearn
            return f"scikit-learn=={sklearn.__version__}"
        except ImportError:
            return "unknown"
    
    def _get_python_version(self) -> str:
        """Get current Python version"""
        import sys
        return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    
    def _get_dependencies(self) -> Dict[str, str]:
        """Get current dependency versions"""
        dependencies = {}
        try:
            import numpy as np
            dependencies['numpy'] = np.__version__
        except ImportError:
            pass
        
        try:
            import pandas as pd
            dependencies['pandas'] = pd.__version__
        except ImportError:
            pass
        
        try:
            import joblib
            dependencies['joblib'] = joblib.__version__
        except ImportError:
            pass
        
        return dependencies
    
    def _calculate_training_data_hash(self, training_result: TrainingResult) -> str:
        """Calculate hash of training data for reproducibility"""
        try:
            # Create a simple hash based on training metadata
            hash_input = f"{training_result.training_samples}:{training_result.feature_count}:{training_result.model_type}"
            return hashlib.sha256(hash_input.encode()).hexdigest()[:16]
        except Exception as e:
            logger.error(f"Error calculating training data hash: {e}")
            return ""

@contextmanager
def model_loading_context(model_manager: ModelManager, model_type: str, 
                         version_id: str = None):
    """Context manager for safe model loading and unloading"""
    model = None
    try:
        model = model_manager.load_model(model_type, version_id)
        if model is None:
            raise RuntimeError(f"Failed to load model: {model_type}:{version_id}")
        yield model
    finally:
        # Cleanup if needed (for now, just rely on garbage collection)
        if model is not None:
            del model

# Example usage and testing
def test_model_persistence():
    """Test model persistence system"""
    print("Testing Model Persistence System")
    
    # Create mock model
    from sklearn.ensemble import RandomForestRegressor
    mock_model = RandomForestRegressor(n_estimators=10, random_state=42)
    
    # Create manager
    manager = ModelManager()
    
    try:
        # Register model
        print("Registering model...")
        version = manager.register_model(
            model=mock_model,
            model_type="test_model",
            model_name="Test Random Forest",
            description="Test model for persistence system",
            tags=["test", "random_forest"]
        )
        print(f"Model registered: {version.version_id}")
        
        # Load model
        print("Loading model...")
        loaded_model = manager.load_model("test_model")
        if loaded_model:
            print("Model loaded successfully")
        else:
            print("Failed to load model")
        
        # Get model info
        print("Model information:")
        info = manager.get_model_info()
        for model_type, versions in info.items():
            print(f"  {model_type}: {len(versions)} versions")
            for version in versions:
                print(f"    v{version['version_number']} ({version['deployment_status']}) - {version['size_mb']} MB")
        
        # Test context manager
        print("Testing context manager...")
        with model_loading_context(manager, "test_model") as model:
            print(f"Model loaded in context: {type(model).__name__}")
        
        print("Model persistence test completed successfully")
        return True
        
    except Exception as e:
        print(f"Model persistence test failed: {e}")
        return False

if __name__ == "__main__":
    test_model_persistence()
