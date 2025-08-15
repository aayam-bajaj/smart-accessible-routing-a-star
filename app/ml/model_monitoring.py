"""
ML model monitoring, performance tracking, and automated updating system
"""
import time
import logging
import threading
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
import json
import uuid
import statistics
from enum import Enum
from collections import deque, defaultdict
import schedule
import warnings
warnings.filterwarnings('ignore')

# Import ML components
try:
    from app.ml.model_persistence import ModelManager, ModelVersion
    from app.ml.training_pipeline import ModelTrainer, TrainingConfig, TrainingResult
    from app.ml.heuristic_models import ModelMetrics, TrainingData, create_sample_training_data
    from app.ml.dynamic_cost_prediction import EnsembleCostPredictor
    from app.ml.feedback_system import FeedbackLearningSystem, FeedbackType, FeedbackData
    from app.models.user_profile import UserProfile
except ImportError:
    # For standalone testing
    ModelManager = None
    ModelVersion = None
    ModelTrainer = None
    TrainingConfig = None
    TrainingResult = None
    ModelMetrics = None
    TrainingData = None
    EnsembleCostPredictor = None
    FeedbackLearningSystem = None
    FeedbackType = None
    FeedbackData = None
    UserProfile = None
    create_sample_training_data = lambda x: (None, None)

logger = logging.getLogger(__name__)

class ModelStatus(Enum):
    """Model status enumeration"""
    HEALTHY = "healthy"
    DEGRADED = "degraded" 
    FAILING = "failing"
    OFFLINE = "offline"

class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class PerformanceMetric:
    """Individual performance metric"""
    name: str
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    baseline_value: Optional[float] = None
    threshold_low: Optional[float] = None
    threshold_high: Optional[float] = None
    
    def is_degraded(self) -> bool:
        """Check if metric indicates degraded performance"""
        if self.threshold_low is not None and self.value < self.threshold_low:
            return True
        if self.threshold_high is not None and self.value > self.threshold_high:
            return True
        return False
    
    def deviation_from_baseline(self) -> Optional[float]:
        """Calculate deviation from baseline"""
        if self.baseline_value is not None:
            return abs(self.value - self.baseline_value) / self.baseline_value
        return None

@dataclass
class ModelPerformanceReport:
    """Performance report for a model"""
    model_type: str
    model_version_id: str
    report_timestamp: datetime = field(default_factory=datetime.now)
    
    # Performance metrics
    accuracy_metrics: Dict[str, PerformanceMetric] = field(default_factory=dict)
    latency_metrics: Dict[str, PerformanceMetric] = field(default_factory=dict)
    throughput_metrics: Dict[str, PerformanceMetric] = field(default_factory=dict)
    
    # System metrics
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    error_rate: float = 0.0
    
    # Status and alerts
    overall_status: ModelStatus = ModelStatus.HEALTHY
    alerts: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

@dataclass
class MonitoringAlert:
    """Monitoring alert"""
    alert_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    model_type: str = ""
    model_version_id: str = ""
    alert_type: str = ""
    severity: AlertSeverity = AlertSeverity.LOW
    message: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolution_timestamp: Optional[datetime] = None
    
    # Alert context
    metric_name: str = ""
    metric_value: float = 0.0
    threshold_value: float = 0.0
    
    # Actions taken
    actions_taken: List[str] = field(default_factory=list)

@dataclass
class RetrainingTrigger:
    """Trigger conditions for model retraining"""
    trigger_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    model_type: str = ""
    trigger_name: str = ""
    description: str = ""
    
    # Trigger conditions
    performance_threshold: float = 0.8  # Minimum R² score
    error_rate_threshold: float = 0.1   # Maximum error rate
    data_drift_threshold: float = 0.2   # Maximum data drift
    time_since_training_days: int = 30  # Days since last training
    
    # Feedback-based triggers
    min_negative_feedback_count: int = 50
    negative_feedback_rate_threshold: float = 0.3
    
    # Status
    enabled: bool = True
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0

class ModelMonitor:
    """Model performance monitoring system"""
    
    def __init__(self, model_manager: ModelManager = None):
        """Initialize model monitor"""
        self.model_manager = model_manager or ModelManager()
        self.performance_history = defaultdict(lambda: deque(maxlen=1000))
        self.active_alerts = {}
        self.monitoring_config = {}
        self.is_running = False
        self.monitor_thread = None
        self.lock = threading.Lock()
        
        # Default monitoring configuration
        self.default_thresholds = {
            'accuracy': {'low': 0.7, 'high': None},
            'latency': {'low': None, 'high': 5.0},  # seconds
            'error_rate': {'low': None, 'high': 0.1},  # 10%
            'memory_usage': {'low': None, 'high': 1000.0}  # MB
        }
        
        logger.info("ModelMonitor initialized")
    
    def start_monitoring(self, interval_minutes: int = 5):
        """Start continuous model monitoring"""
        try:
            if self.is_running:
                logger.warning("Model monitoring is already running")
                return
            
            self.is_running = True
            
            # Schedule monitoring tasks
            schedule.every(interval_minutes).minutes.do(self._monitor_all_models)
            schedule.every().hour.do(self._check_retraining_triggers)
            schedule.every().day.do(self._cleanup_old_metrics)
            
            # Start monitoring thread
            self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitor_thread.start()
            
            logger.info(f"Model monitoring started with {interval_minutes} minute interval")
            
        except Exception as e:
            logger.error(f"Error starting model monitoring: {e}")
            self.is_running = False
    
    def stop_monitoring(self):
        """Stop model monitoring"""
        try:
            self.is_running = False
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=5)
            
            schedule.clear()
            logger.info("Model monitoring stopped")
            
        except Exception as e:
            logger.error(f"Error stopping model monitoring: {e}")
    
    def record_prediction_metrics(self, model_type: str, model_version_id: str,
                                 prediction_time: float, prediction_accuracy: float = None,
                                 error_occurred: bool = False):
        """Record metrics from a model prediction"""
        try:
            with self.lock:
                timestamp = datetime.now()
                
                # Record latency
                latency_metric = PerformanceMetric(
                    name="prediction_latency",
                    value=prediction_time,
                    timestamp=timestamp,
                    **self.default_thresholds.get('latency', {})
                )
                self.performance_history[f"{model_type}:latency"].append(latency_metric)
                
                # Record accuracy if provided
                if prediction_accuracy is not None:
                    accuracy_metric = PerformanceMetric(
                        name="prediction_accuracy",
                        value=prediction_accuracy,
                        timestamp=timestamp,
                        **self.default_thresholds.get('accuracy', {})
                    )
                    self.performance_history[f"{model_type}:accuracy"].append(accuracy_metric)
                
                # Record errors
                error_metric = PerformanceMetric(
                    name="prediction_error",
                    value=1.0 if error_occurred else 0.0,
                    timestamp=timestamp,
                    **self.default_thresholds.get('error_rate', {})
                )
                self.performance_history[f"{model_type}:errors"].append(error_metric)
                
        except Exception as e:
            logger.error(f"Error recording prediction metrics: {e}")
    
    def get_model_performance_report(self, model_type: str, 
                                   hours_back: int = 24) -> Optional[ModelPerformanceReport]:
        """Generate performance report for a model"""
        try:
            # Get active version
            model_info = self.model_manager.get_model_info(model_type)
            if not model_info or model_type not in model_info:
                logger.error(f"No model info found for type: {model_type}")
                return None
            
            active_version = None
            for version in model_info[model_type]:
                if version['deployment_status'] in ['staging', 'production']:
                    active_version = version['version_id']
                    break
            
            if not active_version:
                logger.error(f"No active version found for model type: {model_type}")
                return None
            
            # Create report
            report = ModelPerformanceReport(
                model_type=model_type,
                model_version_id=active_version
            )
            
            # Calculate metrics from recent history
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            
            # Accuracy metrics
            accuracy_history = self.performance_history.get(f"{model_type}:accuracy", deque())
            recent_accuracy = [m for m in accuracy_history if m.timestamp >= cutoff_time]
            if recent_accuracy:
                avg_accuracy = statistics.mean([m.value for m in recent_accuracy])
                report.accuracy_metrics['average_accuracy'] = PerformanceMetric(
                    name="average_accuracy",
                    value=avg_accuracy,
                    **self.default_thresholds.get('accuracy', {})
                )
            
            # Latency metrics
            latency_history = self.performance_history.get(f"{model_type}:latency", deque())
            recent_latency = [m for m in latency_history if m.timestamp >= cutoff_time]
            if recent_latency:
                avg_latency = statistics.mean([m.value for m in recent_latency])
                p95_latency = sorted([m.value for m in recent_latency])[int(0.95 * len(recent_latency))]
                
                report.latency_metrics['average_latency'] = PerformanceMetric(
                    name="average_latency",
                    value=avg_latency,
                    **self.default_thresholds.get('latency', {})
                )
                
                report.latency_metrics['p95_latency'] = PerformanceMetric(
                    name="p95_latency", 
                    value=p95_latency,
                    **self.default_thresholds.get('latency', {})
                )
            
            # Error rate
            error_history = self.performance_history.get(f"{model_type}:errors", deque())
            recent_errors = [m for m in error_history if m.timestamp >= cutoff_time]
            if recent_errors:
                error_rate = statistics.mean([m.value for m in recent_errors])
                report.error_rate = error_rate
            
            # Determine overall status
            report.overall_status = self._determine_model_status(report)
            
            # Generate recommendations
            report.recommendations = self._generate_recommendations(report)
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return None
    
    def _monitor_all_models(self):
        """Monitor all registered models"""
        try:
            model_info = self.model_manager.get_model_info()
            
            for model_type, versions in model_info.items():
                # Find active version
                active_versions = [v for v in versions if v['deployment_status'] in ['staging', 'production']]
                
                for version in active_versions:
                    report = self.get_model_performance_report(model_type)
                    if report:
                        self._check_for_alerts(report)
                        logger.debug(f"Monitored {model_type}: {report.overall_status.value}")
            
        except Exception as e:
            logger.error(f"Error in model monitoring: {e}")
    
    def _check_for_alerts(self, report: ModelPerformanceReport):
        """Check for performance alerts"""
        try:
            alerts_generated = []
            
            # Check accuracy metrics
            for metric_name, metric in report.accuracy_metrics.items():
                if metric.is_degraded():
                    alert = MonitoringAlert(
                        model_type=report.model_type,
                        model_version_id=report.model_version_id,
                        alert_type="performance_degradation",
                        severity=AlertSeverity.HIGH if metric.value < 0.6 else AlertSeverity.MEDIUM,
                        message=f"Model accuracy degraded: {metric.value:.3f}",
                        metric_name=metric_name,
                        metric_value=metric.value,
                        threshold_value=metric.threshold_low or 0.0
                    )
                    alerts_generated.append(alert)
            
            # Check latency metrics
            for metric_name, metric in report.latency_metrics.items():
                if metric.is_degraded():
                    alert = MonitoringAlert(
                        model_type=report.model_type,
                        model_version_id=report.model_version_id,
                        alert_type="latency_increase",
                        severity=AlertSeverity.MEDIUM,
                        message=f"Model latency increased: {metric.value:.3f}s",
                        metric_name=metric_name,
                        metric_value=metric.value,
                        threshold_value=metric.threshold_high or 0.0
                    )
                    alerts_generated.append(alert)
            
            # Check error rate
            if report.error_rate > 0.1:  # 10% error rate threshold
                alert = MonitoringAlert(
                    model_type=report.model_type,
                    model_version_id=report.model_version_id,
                    alert_type="high_error_rate",
                    severity=AlertSeverity.HIGH,
                    message=f"High error rate: {report.error_rate:.1%}",
                    metric_name="error_rate",
                    metric_value=report.error_rate,
                    threshold_value=0.1
                )
                alerts_generated.append(alert)
            
            # Store new alerts
            for alert in alerts_generated:
                alert_key = f"{alert.model_type}:{alert.alert_type}:{alert.metric_name}"
                if alert_key not in self.active_alerts:
                    self.active_alerts[alert_key] = alert
                    logger.warning(f"Alert generated: {alert.message}")
                    
                    # Take automated actions
                    self._handle_alert(alert)
            
        except Exception as e:
            logger.error(f"Error checking for alerts: {e}")
    
    def _handle_alert(self, alert: MonitoringAlert):
        """Handle generated alerts"""
        try:
            actions_taken = []
            
            if alert.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]:
                # High severity alerts - consider automatic actions
                
                if alert.alert_type == "performance_degradation":
                    # Consider triggering retraining
                    actions_taken.append("Triggered retraining evaluation")
                    logger.info(f"High severity performance alert for {alert.model_type} - evaluating retraining")
                    
                elif alert.alert_type == "high_error_rate":
                    # Consider rolling back to previous version
                    actions_taken.append("Evaluated model rollback")
                    logger.info(f"High error rate alert for {alert.model_type} - evaluating rollback")
            
            # Update alert with actions taken
            alert.actions_taken = actions_taken
            
        except Exception as e:
            logger.error(f"Error handling alert: {e}")
    
    def _determine_model_status(self, report: ModelPerformanceReport) -> ModelStatus:
        """Determine overall model status from report"""
        try:
            degraded_metrics = 0
            total_metrics = 0
            
            # Check accuracy metrics
            for metric in report.accuracy_metrics.values():
                total_metrics += 1
                if metric.is_degraded():
                    degraded_metrics += 1
            
            # Check latency metrics
            for metric in report.latency_metrics.values():
                total_metrics += 1
                if metric.is_degraded():
                    degraded_metrics += 1
            
            # Check error rate
            if report.error_rate > 0.1:
                degraded_metrics += 1
            total_metrics += 1
            
            if total_metrics == 0:
                return ModelStatus.HEALTHY
            
            # Determine status based on degraded ratio
            degradation_ratio = degraded_metrics / total_metrics
            
            if degradation_ratio >= 0.5:
                return ModelStatus.FAILING
            elif degradation_ratio >= 0.3:
                return ModelStatus.DEGRADED
            else:
                return ModelStatus.HEALTHY
                
        except Exception as e:
            logger.error(f"Error determining model status: {e}")
            return ModelStatus.OFFLINE
    
    def _generate_recommendations(self, report: ModelPerformanceReport) -> List[str]:
        """Generate recommendations based on performance report"""
        try:
            recommendations = []
            
            # Accuracy-based recommendations
            for metric_name, metric in report.accuracy_metrics.items():
                if metric.is_degraded():
                    if metric.value < 0.6:
                        recommendations.append(f"Critical: {metric_name} is very low ({metric.value:.3f}). Consider immediate retraining or rollback.")
                    else:
                        recommendations.append(f"Consider retraining to improve {metric_name} ({metric.value:.3f}).")
            
            # Latency-based recommendations  
            for metric_name, metric in report.latency_metrics.items():
                if metric.is_degraded():
                    recommendations.append(f"Optimize model for {metric_name} ({metric.value:.3f}s). Consider model compression or hardware upgrade.")
            
            # Error rate recommendations
            if report.error_rate > 0.2:
                recommendations.append(f"High error rate ({report.error_rate:.1%}). Investigate data quality and model stability.")
            elif report.error_rate > 0.1:
                recommendations.append(f"Elevated error rate ({report.error_rate:.1%}). Monitor closely and consider improvements.")
            
            # Overall status recommendations
            if report.overall_status == ModelStatus.FAILING:
                recommendations.append("Model is failing. Consider immediate rollback to previous version.")
            elif report.overall_status == ModelStatus.DEGRADED:
                recommendations.append("Model performance is degraded. Plan retraining or optimization.")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return []
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        logger.info("Model monitoring loop started")
        
        while self.is_running:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)
        
        logger.info("Model monitoring loop stopped")
    
    def _check_retraining_triggers(self):
        """Check if any models need retraining"""
        try:
            logger.debug("Checking retraining triggers")
            
            # This would integrate with feedback system and performance metrics
            # For now, just log the check
            
        except Exception as e:
            logger.error(f"Error checking retraining triggers: {e}")
    
    def _cleanup_old_metrics(self):
        """Clean up old performance metrics"""
        try:
            cutoff_time = datetime.now() - timedelta(days=7)  # Keep 7 days
            
            for key, metrics in self.performance_history.items():
                # Remove old metrics
                while metrics and metrics[0].timestamp < cutoff_time:
                    metrics.popleft()
            
            logger.debug("Cleaned up old performance metrics")
            
        except Exception as e:
            logger.error(f"Error cleaning up metrics: {e}")

class AutoRetrainer:
    """Automated model retraining system"""
    
    def __init__(self, model_manager: ModelManager = None, 
                 model_trainer: ModelTrainer = None):
        """Initialize auto retrainer"""
        self.model_manager = model_manager or ModelManager()
        self.model_trainer = model_trainer or ModelTrainer()
        self.retraining_triggers = {}
        self.retraining_history = deque(maxlen=100)
        self.lock = threading.Lock()
        
        # Default triggers
        self._setup_default_triggers()
        
        logger.info("AutoRetrainer initialized")
    
    def add_retraining_trigger(self, trigger: RetrainingTrigger):
        """Add a retraining trigger"""
        try:
            self.retraining_triggers[trigger.trigger_id] = trigger
            logger.info(f"Added retraining trigger: {trigger.trigger_name}")
            
        except Exception as e:
            logger.error(f"Error adding retraining trigger: {e}")
    
    def evaluate_retraining_needs(self, model_type: str) -> bool:
        """Evaluate if a model needs retraining"""
        try:
            # Get model performance
            monitor = ModelMonitor(self.model_manager)
            report = monitor.get_model_performance_report(model_type)
            
            if not report:
                return False
            
            # Check triggers
            for trigger in self.retraining_triggers.values():
                if trigger.model_type != model_type or not trigger.enabled:
                    continue
                
                if self._should_trigger_retraining(trigger, report):
                    logger.info(f"Retraining triggered for {model_type} by {trigger.trigger_name}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error evaluating retraining needs: {e}")
            return False
    
    def retrain_model(self, model_type: str) -> bool:
        """Retrain a specific model"""
        try:
            logger.info(f"Starting retraining for model type: {model_type}")
            
            # Create new training data (simplified)
            if create_sample_training_data:
                training_data, _ = create_sample_training_data(1000)
                
                if not training_data:
                    logger.error("Failed to create training data")
                    return False
                
                # Train new model
                results = self.model_trainer.train_all_models(training_data)
                
                if model_type in results and results[model_type].training_status == "completed":
                    # Register new model version
                    new_model = self.model_trainer.trained_models.get(model_type)
                    if new_model:
                        self.model_manager.register_model(
                            model=new_model,
                            model_type=model_type,
                            model_name=f"Retrained {model_type.title()} Model",
                            description=f"Automatically retrained on {datetime.now().isoformat()}",
                            tags=["retrained", "automated"],
                            training_result=results[model_type]
                        )
                        
                        logger.info(f"Successfully retrained {model_type}")
                        return True
                    
            logger.error(f"Failed to retrain {model_type}")
            return False
            
        except Exception as e:
            logger.error(f"Error retraining model {model_type}: {e}")
            return False
    
    def _setup_default_triggers(self):
        """Setup default retraining triggers"""
        try:
            # Performance degradation trigger
            performance_trigger = RetrainingTrigger(
                model_type="all",
                trigger_name="Performance Degradation",
                description="Trigger when model performance drops below threshold",
                performance_threshold=0.75,
                error_rate_threshold=0.15
            )
            self.add_retraining_trigger(performance_trigger)
            
            # Time-based trigger
            time_trigger = RetrainingTrigger(
                model_type="all",
                trigger_name="Scheduled Retraining", 
                description="Trigger retraining after specified time period",
                time_since_training_days=14
            )
            self.add_retraining_trigger(time_trigger)
            
        except Exception as e:
            logger.error(f"Error setting up default triggers: {e}")
    
    def _should_trigger_retraining(self, trigger: RetrainingTrigger,
                                  report: ModelPerformanceReport) -> bool:
        """Check if trigger conditions are met"""
        try:
            # Check performance threshold
            for metric in report.accuracy_metrics.values():
                if metric.value < trigger.performance_threshold:
                    return True
            
            # Check error rate threshold
            if report.error_rate > trigger.error_rate_threshold:
                return True
            
            # Check time since last training (simplified check)
            if trigger.last_triggered is None:
                # Never triggered, check if model is old enough
                cutoff_date = datetime.now() - timedelta(days=trigger.time_since_training_days)
                # This would normally check actual training date
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking trigger conditions: {e}")
            return False

class ABTester:
    """A/B testing system for model improvements"""
    
    def __init__(self, model_manager: ModelManager = None):
        """Initialize A/B tester"""
        self.model_manager = model_manager or ModelManager()
        self.active_experiments = {}
        self.experiment_results = {}
        self.lock = threading.Lock()
        
        logger.info("ABTester initialized")
    
    def create_ab_experiment(self, experiment_name: str, control_model_type: str,
                            treatment_model_type: str, traffic_split: float = 0.5,
                            duration_days: int = 7) -> str:
        """Create a new A/B experiment"""
        try:
            experiment_id = str(uuid.uuid4())
            
            experiment = {
                'experiment_id': experiment_id,
                'experiment_name': experiment_name,
                'control_model_type': control_model_type,
                'treatment_model_type': treatment_model_type,
                'traffic_split': traffic_split,
                'start_time': datetime.now(),
                'end_time': datetime.now() + timedelta(days=duration_days),
                'control_metrics': deque(maxlen=10000),
                'treatment_metrics': deque(maxlen=10000),
                'status': 'active'
            }
            
            self.active_experiments[experiment_id] = experiment
            logger.info(f"Created A/B experiment: {experiment_name} ({experiment_id})")
            
            return experiment_id
            
        except Exception as e:
            logger.error(f"Error creating A/B experiment: {e}")
            return ""
    
    def record_experiment_result(self, experiment_id: str, model_type: str,
                               prediction_accuracy: float, user_satisfaction: float):
        """Record result for A/B experiment"""
        try:
            if experiment_id not in self.active_experiments:
                return
            
            experiment = self.active_experiments[experiment_id]
            result = {
                'timestamp': datetime.now(),
                'accuracy': prediction_accuracy,
                'satisfaction': user_satisfaction
            }
            
            if model_type == experiment['control_model_type']:
                experiment['control_metrics'].append(result)
            elif model_type == experiment['treatment_model_type']:
                experiment['treatment_metrics'].append(result)
            
        except Exception as e:
            logger.error(f"Error recording experiment result: {e}")
    
    def analyze_experiment_results(self, experiment_id: str) -> Dict[str, Any]:
        """Analyze A/B experiment results"""
        try:
            if experiment_id not in self.active_experiments:
                return {}
            
            experiment = self.active_experiments[experiment_id]
            control_metrics = list(experiment['control_metrics'])
            treatment_metrics = list(experiment['treatment_metrics'])
            
            if len(control_metrics) < 10 or len(treatment_metrics) < 10:
                return {'status': 'insufficient_data'}
            
            # Calculate means
            control_accuracy = statistics.mean([m['accuracy'] for m in control_metrics])
            treatment_accuracy = statistics.mean([m['accuracy'] for m in treatment_metrics])
            
            control_satisfaction = statistics.mean([m['satisfaction'] for m in control_metrics])
            treatment_satisfaction = statistics.mean([m['satisfaction'] for m in treatment_metrics])
            
            # Simple significance test (would use proper statistical test in production)
            accuracy_improvement = (treatment_accuracy - control_accuracy) / control_accuracy
            satisfaction_improvement = (treatment_satisfaction - control_satisfaction) / control_satisfaction
            
            # Determine winner
            if accuracy_improvement > 0.05 and satisfaction_improvement > 0.05:
                winner = 'treatment'
            elif accuracy_improvement < -0.05 or satisfaction_improvement < -0.05:
                winner = 'control'
            else:
                winner = 'inconclusive'
            
            results = {
                'experiment_id': experiment_id,
                'status': 'completed',
                'winner': winner,
                'control_accuracy': control_accuracy,
                'treatment_accuracy': treatment_accuracy,
                'accuracy_improvement': accuracy_improvement,
                'control_satisfaction': control_satisfaction,
                'treatment_satisfaction': treatment_satisfaction,
                'satisfaction_improvement': satisfaction_improvement,
                'sample_sizes': {
                    'control': len(control_metrics),
                    'treatment': len(treatment_metrics)
                }
            }
            
            self.experiment_results[experiment_id] = results
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing experiment results: {e}")
            return {}

def create_comprehensive_monitoring_system() -> Dict[str, Any]:
    """Create and configure comprehensive model monitoring system"""
    try:
        logger.info("Setting up comprehensive model monitoring system")
        
        # Initialize components
        model_manager = ModelManager()
        monitor = ModelMonitor(model_manager)
        retrainer = AutoRetrainer(model_manager)
        ab_tester = ABTester(model_manager)
        
        # Start monitoring
        monitor.start_monitoring(interval_minutes=1)
        
        # Simulate some model usage and monitoring
        import time
        
        # Record some sample metrics
        monitor.record_prediction_metrics("heuristic", "test-version", 0.15, 0.85)
        monitor.record_prediction_metrics("cost_prediction", "test-version", 0.25, 0.92)
        monitor.record_prediction_metrics("ensemble", "test-version", 0.35, 0.78)
        
        # Generate performance reports
        time.sleep(1)  # Brief wait for metrics to be processed
        
        reports = {}
        for model_type in ["heuristic", "cost_prediction", "ensemble"]:
            report = monitor.get_model_performance_report(model_type, hours_back=1)
            if report:
                reports[model_type] = {
                    'status': report.overall_status.value,
                    'error_rate': report.error_rate,
                    'accuracy_metrics': len(report.accuracy_metrics),
                    'latency_metrics': len(report.latency_metrics),
                    'recommendations_count': len(report.recommendations)
                }
        
        # Test A/B experiment
        experiment_id = ab_tester.create_ab_experiment(
            "Heuristic Model Improvement",
            "heuristic_v1", 
            "heuristic_v2",
            traffic_split=0.5,
            duration_days=3
        )
        
        # Record some experiment data
        for i in range(20):
            # Control model results
            ab_tester.record_experiment_result(experiment_id, "heuristic_v1", 
                                             0.80 + (i % 3) * 0.05, 0.75 + (i % 4) * 0.05)
            # Treatment model results  
            ab_tester.record_experiment_result(experiment_id, "heuristic_v2",
                                             0.85 + (i % 3) * 0.05, 0.82 + (i % 4) * 0.05)
        
        # Analyze experiment
        experiment_results = ab_tester.analyze_experiment_results(experiment_id)
        
        # Stop monitoring
        time.sleep(2)
        monitor.stop_monitoring()
        
        # Compile results
        system_results = {
            'monitoring_system': 'operational',
            'performance_reports': reports,
            'active_alerts': len(monitor.active_alerts),
            'retraining_triggers': len(retrainer.retraining_triggers),
            'ab_experiment': {
                'experiment_id': experiment_id,
                'results': experiment_results
            },
            'system_status': 'healthy'
        }
        
        logger.info("Comprehensive monitoring system setup completed")
        return system_results
        
    except Exception as e:
        logger.error(f"Error setting up monitoring system: {e}")
        return {'system_status': 'error', 'error': str(e)}

# Example usage and testing
if __name__ == "__main__":
    print("Testing ML Model Monitoring System")
    
    # Run comprehensive monitoring system test
    results = create_comprehensive_monitoring_system()
    
    if results and results.get('system_status') == 'healthy':
        print("\nModel Monitoring System Results:")
        print(f"System status: {results['system_status']}")
        print(f"Active alerts: {results['active_alerts']}")
        print(f"Retraining triggers: {results['retraining_triggers']}")
        
        print("\nPerformance Reports:")
        for model_type, report in results['performance_reports'].items():
            print(f"  {model_type}: {report['status']} (Error rate: {report['error_rate']:.1%})")
        
        experiment = results['ab_experiment']
        experiment_results = experiment['results']
        if experiment_results:
            print(f"\nA/B Experiment Results:")
            print(f"  Experiment ID: {experiment['experiment_id'][:8]}...")
            print(f"  Winner: {experiment_results.get('winner', 'unknown')}")
            print(f"  Accuracy improvement: {experiment_results.get('accuracy_improvement', 0):.1%}")
            print(f"  Satisfaction improvement: {experiment_results.get('satisfaction_improvement', 0):.1%}")
        
        print("\nModel monitoring system test completed successfully")
    else:
        print(f"Model monitoring system test failed: {results.get('error', 'Unknown error')}")
    
    print("\nML monitoring system test completed")
