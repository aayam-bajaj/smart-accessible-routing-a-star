"""
User feedback collection and learning system for route optimization
"""
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any, Union, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import json
import uuid
import threading
from collections import defaultdict, deque
import sqlite3
import os
from pathlib import Path

# Import routing and ML components
try:
    from app.services.routing_algorithm import RouteResult, RouteSegment
    from app.models.user_profile import UserProfile, MobilityAidType
    from app.ml.heuristic_models import TrainingData, ModelMetrics
except ImportError:
    # For standalone testing
    RouteResult = None
    RouteSegment = None
    UserProfile = None
    MobilityAidType = None
    TrainingData = None
    ModelMetrics = None

logger = logging.getLogger(__name__)

class FeedbackType(Enum):
    """Types of user feedback"""
    ROUTE_RATING = "route_rating"
    SEGMENT_RATING = "segment_rating"
    ACCESSIBILITY_ISSUE = "accessibility_issue"
    ROUTE_COMPLETION = "route_completion"
    ALTERNATIVE_PREFERENCE = "alternative_preference"
    SAFETY_CONCERN = "safety_concern"
    COMFORT_RATING = "comfort_rating"
    TIME_ACCURACY = "time_accuracy"
    DIFFICULTY_RATING = "difficulty_rating"

class FeedbackSeverity(Enum):
    """Severity levels for feedback"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class UserFeedback:
    """Individual feedback entry from user"""
    feedback_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    route_id: str = ""
    segment_id: Optional[str] = None
    feedback_type: FeedbackType = FeedbackType.ROUTE_RATING
    severity: FeedbackSeverity = FeedbackSeverity.MEDIUM
    rating: Optional[float] = None  # 1-5 scale
    text_comment: Optional[str] = None
    location_lat: Optional[float] = None
    location_lng: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Specific feedback data
    accessibility_issues: List[str] = field(default_factory=list)
    safety_concerns: List[str] = field(default_factory=list)
    suggested_improvements: List[str] = field(default_factory=list)
    
    # Context information
    weather_conditions: Dict = field(default_factory=dict)
    time_of_day: str = ""
    mobility_context: Dict = field(default_factory=dict)
    
    # Validation and processing
    is_validated: bool = False
    processing_status: str = "pending"  # pending, processed, ignored
    validation_score: float = 0.0

@dataclass
class FeedbackAggregation:
    """Aggregated feedback for a route or segment"""
    item_id: str  # route_id or segment_id
    item_type: str  # "route" or "segment"
    total_feedback_count: int = 0
    average_rating: float = 0.0
    rating_distribution: Dict[int, int] = field(default_factory=dict)
    
    # Issue frequency
    accessibility_issues: Dict[str, int] = field(default_factory=dict)
    safety_concerns: Dict[str, int] = field(default_factory=dict)
    comfort_complaints: Dict[str, int] = field(default_factory=dict)
    
    # Time patterns
    feedback_by_time: Dict[str, List[float]] = field(default_factory=dict)  # hour -> ratings
    feedback_by_weather: Dict[str, List[float]] = field(default_factory=dict)
    
    # User patterns
    feedback_by_mobility_aid: Dict[str, List[float]] = field(default_factory=dict)
    
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class FeedbackInsights:
    """Insights derived from feedback analysis"""
    insight_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    insight_type: str = ""  # "accessibility", "safety", "comfort", "timing", "route_preference"
    description: str = ""
    confidence: float = 0.0
    affected_items: List[str] = field(default_factory=list)  # route/segment IDs
    user_groups: List[str] = field(default_factory=list)  # mobility aid types affected
    recommended_actions: List[str] = field(default_factory=list)
    priority: FeedbackSeverity = FeedbackSeverity.MEDIUM
    created_at: datetime = field(default_factory=datetime.now)

class FeedbackValidator:
    """Validates and filters user feedback for quality"""
    
    def __init__(self):
        """Initialize feedback validator"""
        self.spam_keywords = {
            'test', 'spam', 'fake', 'random', 'nonsense', 'blah', 'asdf'
        }
        self.min_comment_length = 5
        self.max_comment_length = 1000
        logger.info("FeedbackValidator initialized")
    
    def validate_feedback(self, feedback: UserFeedback) -> Tuple[bool, float, List[str]]:
        """
        Validate feedback entry
        
        Args:
            feedback: UserFeedback object to validate
            
        Returns:
            Tuple of (is_valid, validation_score, issues)
        """
        try:
            is_valid = True
            validation_score = 1.0
            issues = []
            
            # Check required fields
            if not feedback.user_id:
                issues.append("Missing user ID")
                validation_score -= 0.3
            
            if not feedback.route_id and not feedback.segment_id:
                issues.append("Missing route or segment ID")
                validation_score -= 0.4
                is_valid = False
            
            # Validate rating
            if feedback.rating is not None:
                if feedback.rating < 1 or feedback.rating > 5:
                    issues.append("Rating out of valid range (1-5)")
                    validation_score -= 0.2
                    is_valid = False
            
            # Validate text comment
            if feedback.text_comment:
                comment = feedback.text_comment.lower().strip()
                
                # Check length
                if len(comment) < self.min_comment_length:
                    issues.append("Comment too short")
                    validation_score -= 0.1
                elif len(comment) > self.max_comment_length:
                    issues.append("Comment too long")
                    validation_score -= 0.1
                
                # Check for spam keywords
                spam_count = sum(1 for keyword in self.spam_keywords if keyword in comment)
                if spam_count > 0:
                    issues.append("Potential spam content detected")
                    validation_score -= spam_count * 0.2
                    if spam_count >= 2:
                        is_valid = False
            
            # Validate location data
            if feedback.location_lat is not None and feedback.location_lng is not None:
                if not (-90 <= feedback.location_lat <= 90):
                    issues.append("Invalid latitude")
                    validation_score -= 0.1
                if not (-180 <= feedback.location_lng <= 180):
                    issues.append("Invalid longitude")
                    validation_score -= 0.1
            
            # Check timestamp reasonableness
            now = datetime.now()
            if feedback.timestamp > now:
                issues.append("Future timestamp")
                validation_score -= 0.2
            elif (now - feedback.timestamp).days > 30:
                issues.append("Very old feedback")
                validation_score -= 0.1
            
            # Ensure minimum validation score
            validation_score = max(0.0, min(1.0, validation_score))
            
            return is_valid, validation_score, issues
            
        except Exception as e:
            logger.error(f"Error validating feedback: {e}")
            return False, 0.0, ["Validation error occurred"]

class FeedbackDatabase:
    """SQLite database for storing feedback data"""
    
    def __init__(self, db_path: str = "data/feedback.db"):
        """Initialize feedback database"""
        self.db_path = db_path
        self.db_lock = threading.Lock()
        self._create_tables()
        logger.info(f"FeedbackDatabase initialized at {db_path}")
    
    def _create_tables(self):
        """Create database tables if they don't exist"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # User feedback table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS user_feedback (
                        feedback_id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        route_id TEXT,
                        segment_id TEXT,
                        feedback_type TEXT NOT NULL,
                        severity TEXT NOT NULL,
                        rating REAL,
                        text_comment TEXT,
                        location_lat REAL,
                        location_lng REAL,
                        timestamp TEXT NOT NULL,
                        accessibility_issues TEXT,
                        safety_concerns TEXT,
                        suggested_improvements TEXT,
                        weather_conditions TEXT,
                        time_of_day TEXT,
                        mobility_context TEXT,
                        is_validated INTEGER,
                        processing_status TEXT,
                        validation_score REAL
                    )
                ''')
                
                # Feedback aggregations table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS feedback_aggregations (
                        item_id TEXT PRIMARY KEY,
                        item_type TEXT NOT NULL,
                        total_feedback_count INTEGER,
                        average_rating REAL,
                        rating_distribution TEXT,
                        accessibility_issues TEXT,
                        safety_concerns TEXT,
                        comfort_complaints TEXT,
                        feedback_by_time TEXT,
                        feedback_by_weather TEXT,
                        feedback_by_mobility_aid TEXT,
                        last_updated TEXT
                    )
                ''')
                
                # Feedback insights table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS feedback_insights (
                        insight_id TEXT PRIMARY KEY,
                        insight_type TEXT NOT NULL,
                        description TEXT NOT NULL,
                        confidence REAL,
                        affected_items TEXT,
                        user_groups TEXT,
                        recommended_actions TEXT,
                        priority TEXT,
                        created_at TEXT
                    )
                ''')
                
                # Create indexes
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_feedback_user_id ON user_feedback(user_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_feedback_route_id ON user_feedback(route_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_feedback_timestamp ON user_feedback(timestamp)')
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error creating database tables: {e}")
    
    def store_feedback(self, feedback: UserFeedback) -> bool:
        """Store feedback in database"""
        try:
            with self.db_lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    cursor.execute('''
                        INSERT OR REPLACE INTO user_feedback VALUES (
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                        )
                    ''', (
                        feedback.feedback_id,
                        feedback.user_id,
                        feedback.route_id,
                        feedback.segment_id,
                        feedback.feedback_type.value,
                        feedback.severity.value,
                        feedback.rating,
                        feedback.text_comment,
                        feedback.location_lat,
                        feedback.location_lng,
                        feedback.timestamp.isoformat(),
                        json.dumps(feedback.accessibility_issues),
                        json.dumps(feedback.safety_concerns),
                        json.dumps(feedback.suggested_improvements),
                        json.dumps(feedback.weather_conditions),
                        feedback.time_of_day,
                        json.dumps(feedback.mobility_context),
                        int(feedback.is_validated),
                        feedback.processing_status,
                        feedback.validation_score
                    ))
                    
                    conn.commit()
                    return True
                    
        except Exception as e:
            logger.error(f"Error storing feedback: {e}")
            return False
    
    def get_feedback(self, user_id: str = None, route_id: str = None, 
                    segment_id: str = None, limit: int = 100) -> List[UserFeedback]:
        """Retrieve feedback from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                query = "SELECT * FROM user_feedback WHERE 1=1"
                params = []
                
                if user_id:
                    query += " AND user_id = ?"
                    params.append(user_id)
                
                if route_id:
                    query += " AND route_id = ?"
                    params.append(route_id)
                
                if segment_id:
                    query += " AND segment_id = ?"
                    params.append(segment_id)
                
                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                feedback_list = []
                for row in rows:
                    feedback = UserFeedback(
                        feedback_id=row[0],
                        user_id=row[1],
                        route_id=row[2],
                        segment_id=row[3],
                        feedback_type=FeedbackType(row[4]),
                        severity=FeedbackSeverity(row[5]),
                        rating=row[6],
                        text_comment=row[7],
                        location_lat=row[8],
                        location_lng=row[9],
                        timestamp=datetime.fromisoformat(row[10]),
                        accessibility_issues=json.loads(row[11]) if row[11] else [],
                        safety_concerns=json.loads(row[12]) if row[12] else [],
                        suggested_improvements=json.loads(row[13]) if row[13] else [],
                        weather_conditions=json.loads(row[14]) if row[14] else {},
                        time_of_day=row[15] or "",
                        mobility_context=json.loads(row[16]) if row[16] else {},
                        is_validated=bool(row[17]),
                        processing_status=row[18] or "pending",
                        validation_score=row[19] or 0.0
                    )
                    feedback_list.append(feedback)
                
                return feedback_list
                
        except Exception as e:
            logger.error(f"Error retrieving feedback: {e}")
            return []

class FeedbackAnalyzer:
    """Analyzes feedback to extract insights and patterns"""
    
    def __init__(self):
        """Initialize feedback analyzer"""
        self.min_feedback_count = 5  # Minimum feedback needed for analysis
        self.confidence_threshold = 0.7
        logger.info("FeedbackAnalyzer initialized")
    
    def analyze_route_feedback(self, route_id: str, feedback_list: List[UserFeedback]) -> FeedbackAggregation:
        """
        Analyze feedback for a specific route
        
        Args:
            route_id: Route identifier
            feedback_list: List of feedback for the route
            
        Returns:
            FeedbackAggregation object with analysis results
        """
        try:
            aggregation = FeedbackAggregation(item_id=route_id, item_type="route")
            
            if not feedback_list:
                return aggregation
            
            # Filter valid feedback
            valid_feedback = [f for f in feedback_list if f.is_validated and f.validation_score > 0.5]
            aggregation.total_feedback_count = len(valid_feedback)
            
            if not valid_feedback:
                return aggregation
            
            # Calculate average rating
            ratings = [f.rating for f in valid_feedback if f.rating is not None]
            if ratings:
                aggregation.average_rating = np.mean(ratings)
                
                # Rating distribution
                for rating in ratings:
                    rounded_rating = int(round(rating))
                    aggregation.rating_distribution[rounded_rating] = \
                        aggregation.rating_distribution.get(rounded_rating, 0) + 1
            
            # Analyze accessibility issues
            for feedback in valid_feedback:
                for issue in feedback.accessibility_issues:
                    aggregation.accessibility_issues[issue] = \
                        aggregation.accessibility_issues.get(issue, 0) + 1
            
            # Analyze safety concerns
            for feedback in valid_feedback:
                for concern in feedback.safety_concerns:
                    aggregation.safety_concerns[concern] = \
                        aggregation.safety_concerns.get(concern, 0) + 1
            
            # Time-based patterns
            for feedback in valid_feedback:
                if feedback.time_of_day and feedback.rating is not None:
                    hour = feedback.time_of_day
                    if hour not in aggregation.feedback_by_time:
                        aggregation.feedback_by_time[hour] = []
                    aggregation.feedback_by_time[hour].append(feedback.rating)
            
            # Weather patterns
            for feedback in valid_feedback:
                weather = feedback.weather_conditions.get('condition', 'unknown')
                if feedback.rating is not None:
                    if weather not in aggregation.feedback_by_weather:
                        aggregation.feedback_by_weather[weather] = []
                    aggregation.feedback_by_weather[weather].append(feedback.rating)
            
            # Mobility aid patterns
            for feedback in valid_feedback:
                mobility_aid = feedback.mobility_context.get('mobility_aid', 'unknown')
                if feedback.rating is not None:
                    if mobility_aid not in aggregation.feedback_by_mobility_aid:
                        aggregation.feedback_by_mobility_aid[mobility_aid] = []
                    aggregation.feedback_by_mobility_aid[mobility_aid].append(feedback.rating)
            
            aggregation.last_updated = datetime.now()
            return aggregation
            
        except Exception as e:
            logger.error(f"Error analyzing route feedback: {e}")
            return FeedbackAggregation(item_id=route_id, item_type="route")
    
    def generate_insights(self, aggregations: List[FeedbackAggregation]) -> List[FeedbackInsights]:
        """
        Generate insights from feedback aggregations
        
        Args:
            aggregations: List of feedback aggregations to analyze
            
        Returns:
            List of generated insights
        """
        try:
            insights = []
            
            for aggregation in aggregations:
                if aggregation.total_feedback_count < self.min_feedback_count:
                    continue
                
                # Low rating insight
                if aggregation.average_rating < 2.5:
                    insight = FeedbackInsights(
                        insight_type="low_satisfaction",
                        description=f"Route/segment {aggregation.item_id} has consistently low ratings "
                                  f"(average: {aggregation.average_rating:.1f}/5.0)",
                        confidence=min(1.0, aggregation.total_feedback_count / 10.0),
                        affected_items=[aggregation.item_id],
                        recommended_actions=[
                            "Investigate route quality",
                            "Check for accessibility barriers",
                            "Consider alternative routing"
                        ],
                        priority=FeedbackSeverity.HIGH if aggregation.average_rating < 2.0 else FeedbackSeverity.MEDIUM
                    )
                    insights.append(insight)
                
                # Accessibility issues insight
                if aggregation.accessibility_issues:
                    most_common_issue = max(aggregation.accessibility_issues.items(), key=lambda x: x[1])
                    issue_frequency = most_common_issue[1] / aggregation.total_feedback_count
                    
                    if issue_frequency > 0.3:  # 30% of users report this issue
                        insight = FeedbackInsights(
                            insight_type="accessibility",
                            description=f"Frequent accessibility issue in {aggregation.item_id}: "
                                      f"{most_common_issue[0]} (reported by {issue_frequency*100:.1f}% of users)",
                            confidence=min(1.0, issue_frequency * 2),
                            affected_items=[aggregation.item_id],
                            recommended_actions=[
                                f"Address {most_common_issue[0]}",
                                "Conduct accessibility audit",
                                "Update route metadata"
                            ],
                            priority=FeedbackSeverity.HIGH if issue_frequency > 0.5 else FeedbackSeverity.MEDIUM
                        )
                        insights.append(insight)
                
                # Safety concerns insight
                if aggregation.safety_concerns:
                    most_common_concern = max(aggregation.safety_concerns.items(), key=lambda x: x[1])
                    concern_frequency = most_common_concern[1] / aggregation.total_feedback_count
                    
                    if concern_frequency > 0.2:  # 20% of users report safety concerns
                        insight = FeedbackInsights(
                            insight_type="safety",
                            description=f"Safety concern in {aggregation.item_id}: "
                                      f"{most_common_concern[0]} (reported by {concern_frequency*100:.1f}% of users)",
                            confidence=min(1.0, concern_frequency * 2.5),
                            affected_items=[aggregation.item_id],
                            recommended_actions=[
                                f"Investigate {most_common_concern[0]}",
                                "Contact local authorities if needed",
                                "Add safety warnings to route"
                            ],
                            priority=FeedbackSeverity.CRITICAL if concern_frequency > 0.4 else FeedbackSeverity.HIGH
                        )
                        insights.append(insight)
                
                # Time-based patterns
                if len(aggregation.feedback_by_time) > 2:
                    time_ratings = {hour: np.mean(ratings) for hour, ratings in aggregation.feedback_by_time.items()}
                    worst_time = min(time_ratings.items(), key=lambda x: x[1])
                    best_time = max(time_ratings.items(), key=lambda x: x[1])
                    
                    if best_time[1] - worst_time[1] > 1.0:  # Significant time-based difference
                        insight = FeedbackInsights(
                            insight_type="timing",
                            description=f"Route {aggregation.item_id} shows time-dependent quality variation. "
                                      f"Worst at {worst_time[0]} (avg: {worst_time[1]:.1f}), "
                                      f"best at {best_time[0]} (avg: {best_time[1]:.1f})",
                            confidence=0.7,
                            affected_items=[aggregation.item_id],
                            recommended_actions=[
                                "Investigate time-specific issues",
                                "Consider time-based route preferences",
                                "Update routing algorithm with time factors"
                            ],
                            priority=FeedbackSeverity.MEDIUM
                        )
                        insights.append(insight)
            
            # Filter insights by confidence
            high_confidence_insights = [i for i in insights if i.confidence >= self.confidence_threshold]
            
            return high_confidence_insights
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return []

class FeedbackLearningSystem:
    """Main system for collecting, analyzing, and learning from user feedback"""
    
    def __init__(self, db_path: str = "data/feedback.db"):
        """Initialize feedback learning system"""
        self.database = FeedbackDatabase(db_path)
        self.validator = FeedbackValidator()
        self.analyzer = FeedbackAnalyzer()
        
        # Cache for aggregations
        self.aggregation_cache = {}
        self.cache_lock = threading.Lock()
        self.cache_expiry = timedelta(hours=1)
        
        logger.info("FeedbackLearningSystem initialized")
    
    def collect_feedback(self, user_id: str, route_id: str = None, segment_id: str = None,
                        feedback_type: FeedbackType = FeedbackType.ROUTE_RATING,
                        rating: float = None, comment: str = None,
                        accessibility_issues: List[str] = None,
                        safety_concerns: List[str] = None,
                        location: Tuple[float, float] = None,
                        context: Dict = None) -> bool:
        """
        Collect user feedback
        
        Args:
            user_id: User identifier
            route_id: Route identifier (optional)
            segment_id: Segment identifier (optional)
            feedback_type: Type of feedback
            rating: Rating score (1-5)
            comment: Text comment
            accessibility_issues: List of accessibility issues
            safety_concerns: List of safety concerns
            location: (latitude, longitude) tuple
            context: Additional context information
            
        Returns:
            Success status
        """
        try:
            # Create feedback object
            feedback = UserFeedback(
                user_id=user_id,
                route_id=route_id or "",
                segment_id=segment_id,
                feedback_type=feedback_type,
                rating=rating,
                text_comment=comment,
                accessibility_issues=accessibility_issues or [],
                safety_concerns=safety_concerns or [],
                time_of_day=str(datetime.now().hour)
            )
            
            if location:
                feedback.location_lat, feedback.location_lng = location
            
            if context:
                feedback.weather_conditions = context.get('weather', {})
                feedback.mobility_context = context.get('mobility', {})
            
            # Validate feedback
            is_valid, validation_score, issues = self.validator.validate_feedback(feedback)
            feedback.is_validated = is_valid
            feedback.validation_score = validation_score
            feedback.processing_status = "validated" if is_valid else "rejected"
            
            if issues:
                logger.info(f"Feedback validation issues: {issues}")
            
            # Store in database
            success = self.database.store_feedback(feedback)
            
            if success:
                logger.info(f"Feedback collected from user {user_id} for {route_id or segment_id}")
                
                # Clear relevant cache entries
                self._clear_cache_for_item(route_id or segment_id)
                
            return success
            
        except Exception as e:
            logger.error(f"Error collecting feedback: {e}")
            return False
    
    def get_route_insights(self, route_id: str) -> Dict[str, Any]:
        """
        Get insights for a specific route
        
        Args:
            route_id: Route identifier
            
        Returns:
            Dictionary containing route insights and recommendations
        """
        try:
            # Get feedback for route
            feedback_list = self.database.get_feedback(route_id=route_id, limit=500)
            
            if not feedback_list:
                return {"message": "No feedback available for this route"}
            
            # Analyze feedback
            aggregation = self.analyzer.analyze_route_feedback(route_id, feedback_list)
            insights = self.analyzer.generate_insights([aggregation])
            
            # Prepare response
            result = {
                "route_id": route_id,
                "total_feedback": aggregation.total_feedback_count,
                "average_rating": aggregation.average_rating,
                "rating_distribution": aggregation.rating_distribution,
                "common_issues": {
                    "accessibility": dict(list(aggregation.accessibility_issues.items())[:5]),
                    "safety": dict(list(aggregation.safety_concerns.items())[:5])
                },
                "insights": [asdict(insight) for insight in insights],
                "recommendations": []
            }
            
            # Add recommendations based on insights
            for insight in insights:
                result["recommendations"].extend(insight.recommended_actions)
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting route insights: {e}")
            return {"error": "Failed to retrieve route insights"}
    
    def generate_training_data_from_feedback(self, min_feedback_count: int = 10) -> Optional[TrainingData]:
        """
        Generate training data from collected feedback for ML model improvement
        
        Args:
            min_feedback_count: Minimum feedback entries needed
            
        Returns:
            TrainingData object or None
        """
        try:
            if not TrainingData:
                logger.warning("TrainingData class not available")
                return None
            
            # Get all validated feedback
            all_feedback = self.database.get_feedback(limit=10000)
            valid_feedback = [f for f in all_feedback if f.is_validated and f.rating is not None]
            
            if len(valid_feedback) < min_feedback_count:
                logger.info(f"Insufficient feedback for training data generation: {len(valid_feedback)} < {min_feedback_count}")
                return None
            
            # Prepare features and targets
            features = []
            targets = []
            feature_names = [
                'hour_of_day', 'day_of_week', 'rating', 'has_accessibility_issues',
                'has_safety_concerns', 'temperature', 'precipitation',
                'mobility_aid_wheelchair', 'mobility_aid_walker', 'mobility_aid_none'
            ]
            
            for feedback in valid_feedback:
                try:
                    # Extract features
                    hour = int(feedback.time_of_day) if feedback.time_of_day.isdigit() else 12
                    day_of_week = feedback.timestamp.weekday()
                    rating = feedback.rating
                    has_accessibility_issues = 1.0 if feedback.accessibility_issues else 0.0
                    has_safety_concerns = 1.0 if feedback.safety_concerns else 0.0
                    
                    # Weather features
                    weather = feedback.weather_conditions
                    temperature = weather.get('temperature', 20.0)
                    precipitation = weather.get('precipitation', 0.0)
                    
                    # Mobility aid features (one-hot encoded)
                    mobility_aid = feedback.mobility_context.get('mobility_aid', 'none')
                    wheelchair = 1.0 if 'wheelchair' in mobility_aid.lower() else 0.0
                    walker = 1.0 if 'walker' in mobility_aid.lower() else 0.0
                    none_aid = 1.0 if mobility_aid == 'none' else 0.0
                    
                    feature_vector = [
                        hour, day_of_week, rating, has_accessibility_issues, has_safety_concerns,
                        temperature, precipitation, wheelchair, walker, none_aid
                    ]
                    
                    features.append(feature_vector)
                    
                    # Target: satisfaction score (normalized rating)
                    normalized_rating = (rating - 1) / 4  # Scale 1-5 to 0-1
                    targets.append(normalized_rating)
                    
                except Exception as e:
                    logger.warning(f"Error processing feedback for training data: {e}")
                    continue
            
            if len(features) < min_feedback_count:
                return None
            
            # Create training data
            features_df = pd.DataFrame(features, columns=feature_names)
            targets_series = pd.Series(targets)
            
            training_data = TrainingData(
                features=features_df,
                targets=targets_series,
                feature_names=feature_names,
                target_name='satisfaction_score',
                data_source='user_feedback',
                collection_date=datetime.now().isoformat(),
                user_count=len(set(f.user_id for f in valid_feedback)),
                route_count=len(set(f.route_id for f in valid_feedback if f.route_id))
            )
            
            logger.info(f"Generated training data with {len(features)} samples from user feedback")
            return training_data
            
        except Exception as e:
            logger.error(f"Error generating training data from feedback: {e}")
            return None
    
    def get_user_feedback_summary(self, user_id: str) -> Dict[str, Any]:
        """Get summary of feedback provided by a specific user"""
        try:
            user_feedback = self.database.get_feedback(user_id=user_id, limit=1000)
            
            if not user_feedback:
                return {"message": "No feedback found for this user"}
            
            # Calculate summary statistics
            ratings = [f.rating for f in user_feedback if f.rating is not None]
            
            summary = {
                "user_id": user_id,
                "total_feedback_entries": len(user_feedback),
                "average_rating_given": np.mean(ratings) if ratings else None,
                "feedback_types": {},
                "most_recent_feedback": user_feedback[0].timestamp.isoformat() if user_feedback else None,
                "accessibility_issues_reported": 0,
                "safety_concerns_reported": 0
            }
            
            # Count feedback types
            for feedback in user_feedback:
                feedback_type = feedback.feedback_type.value
                summary["feedback_types"][feedback_type] = summary["feedback_types"].get(feedback_type, 0) + 1
                
                summary["accessibility_issues_reported"] += len(feedback.accessibility_issues)
                summary["safety_concerns_reported"] += len(feedback.safety_concerns)
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting user feedback summary: {e}")
            return {"error": "Failed to retrieve user feedback summary"}
    
    def _clear_cache_for_item(self, item_id: str):
        """Clear cache entries for a specific item"""
        try:
            with self.cache_lock:
                keys_to_remove = [key for key in self.aggregation_cache.keys() if item_id in key]
                for key in keys_to_remove:
                    del self.aggregation_cache[key]
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")

# Example usage and testing
if __name__ == "__main__":
    print("Testing Feedback Learning System")
    
    # Initialize system
    feedback_system = FeedbackLearningSystem()
    
    # Test feedback collection
    print("\nTesting feedback collection...")
    
    # Simulate user feedback
    test_feedback_data = [
        {
            "user_id": "user_001",
            "route_id": "route_123",
            "feedback_type": FeedbackType.ROUTE_RATING,
            "rating": 4.5,
            "comment": "Great route, very accessible and safe",
            "accessibility_issues": [],
            "safety_concerns": []
        },
        {
            "user_id": "user_002",
            "route_id": "route_123", 
            "feedback_type": FeedbackType.ACCESSIBILITY_ISSUE,
            "rating": 2.0,
            "comment": "Steps without ramps at the intersection",
            "accessibility_issues": ["missing_ramp", "steep_curb"],
            "safety_concerns": []
        },
        {
            "user_id": "user_003",
            "route_id": "route_123",
            "feedback_type": FeedbackType.SAFETY_CONCERN,
            "rating": 2.5,
            "comment": "Poor lighting in the evening",
            "accessibility_issues": [],
            "safety_concerns": ["poor_lighting", "isolated_area"]
        }
    ]
    
    # Collect feedback
    for feedback_data in test_feedback_data:
        success = feedback_system.collect_feedback(**feedback_data)
        print(f"Feedback collected: {success}")
    
    # Test route insights
    print("\nTesting route insights...")
    insights = feedback_system.get_route_insights("route_123")
    print(f"Route insights: {json.dumps(insights, indent=2)}")
    
    # Test user summary
    print("\nTesting user feedback summary...")
    user_summary = feedback_system.get_user_feedback_summary("user_001")
    print(f"User summary: {json.dumps(user_summary, indent=2)}")
    
    # Test training data generation
    print("\nTesting training data generation...")
    training_data = feedback_system.generate_training_data_from_feedback(min_feedback_count=1)
    if training_data:
        print(f"Training data generated: {len(training_data.features)} samples")
        print(f"Feature names: {training_data.feature_names}")
    else:
        print("Training data generation skipped")
    
    print("\nFeedback learning system test completed")
