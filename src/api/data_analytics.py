"""
Data Management and Analytics API Endpoints
==========================================

Provides endpoints for map data management, route analytics, system metrics,
performance monitoring, and administrative functions with data export and
reporting capabilities.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import json
import csv
import io
from flask import request, jsonify, Response, stream_template
from dataclasses import asdict
import sqlite3
import os
import hashlib
import time
from collections import defaultdict, Counter
import statistics
import pandas as pd

from ..models.user_profile import UserProfile, MobilityAid
from ..models.graph import Node, Edge, AccessibilityGraph
from ..models.route import RouteResult, RouteSegment
from .validation import (
    validate_request_data, handle_api_errors, require_auth,
    validate_query_params, ValidationAPIError, format_success_response,
    PAGINATION_PARAMS, LOCATION_PARAMS, SORTING_PARAMS
)
from .schemas import API_SCHEMAS


class DataAnalyticsManager:
    """Manages system data, analytics, and reporting."""
    
    def __init__(self, db_path: str = "analytics.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize analytics and management database tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Route usage analytics
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS route_analytics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    route_hash TEXT NOT NULL,
                    user_id TEXT,
                    start_lat REAL NOT NULL,
                    start_lon REAL NOT NULL,
                    end_lat REAL NOT NULL,
                    end_lon REAL NOT NULL,
                    total_distance REAL,
                    total_duration REAL,
                    accessibility_score REAL,
                    algorithm_used TEXT,
                    execution_time_ms INTEGER,
                    success BOOLEAN DEFAULT TRUE,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    mobility_aid TEXT,
                    route_type TEXT -- e.g., 'accessibility', 'fastest', 'shortest'
                )
            """)
            
            # System performance metrics
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    metric_unit TEXT,
                    category TEXT, -- e.g., 'performance', 'usage', 'errors'
                    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT -- JSON for additional context
                )
            """)
            
            # Map data status
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS map_data_status (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    region_name TEXT NOT NULL,
                    data_source TEXT,
                    last_updated TIMESTAMP,
                    nodes_count INTEGER DEFAULT 0,
                    edges_count INTEGER DEFAULT 0,
                    accessibility_data_coverage REAL DEFAULT 0.0,
                    data_quality_score REAL DEFAULT 0.0,
                    update_status TEXT DEFAULT 'current' -- 'current', 'updating', 'outdated'
                )
            """)
            
            # API usage tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS api_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    endpoint TEXT NOT NULL,
                    method TEXT NOT NULL,
                    user_id TEXT,
                    response_status INTEGER,
                    response_time_ms INTEGER,
                    request_size INTEGER,
                    response_size INTEGER,
                    ip_address TEXT,
                    user_agent TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    rate_limit_hit BOOLEAN DEFAULT FALSE
                )
            """)
            
            # Error tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS error_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    error_type TEXT NOT NULL,
                    error_message TEXT,
                    stack_trace TEXT,
                    endpoint TEXT,
                    user_id TEXT,
                    request_data TEXT, -- JSON
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    resolved BOOLEAN DEFAULT FALSE,
                    severity TEXT DEFAULT 'medium' -- 'low', 'medium', 'high', 'critical'
                )
            """)
            
            conn.commit()
    
    def log_route_usage(self, route_data: Dict[str, Any]) -> int:
        """Log route calculation usage for analytics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO route_analytics 
                (route_hash, user_id, start_lat, start_lon, end_lat, end_lon,
                 total_distance, total_duration, accessibility_score, 
                 algorithm_used, execution_time_ms, success, mobility_aid, route_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                route_data.get('route_hash', ''),
                route_data.get('user_id', ''),
                route_data.get('start_lat', 0),
                route_data.get('start_lon', 0),
                route_data.get('end_lat', 0),
                route_data.get('end_lon', 0),
                route_data.get('total_distance', 0),
                route_data.get('total_duration', 0),
                route_data.get('accessibility_score', 0),
                route_data.get('algorithm_used', 'astar'),
                route_data.get('execution_time_ms', 0),
                route_data.get('success', True),
                route_data.get('mobility_aid', 'none'),
                route_data.get('route_type', 'accessibility')
            ))
            
            route_id = cursor.lastrowid
            conn.commit()
            return route_id
    
    def log_system_metric(self, name: str, value: float, unit: str = None, 
                         category: str = "performance", metadata: Dict = None) -> None:
        """Log system metric for monitoring."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO system_metrics 
                (metric_name, metric_value, metric_unit, category, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, (name, value, unit, category, json.dumps(metadata or {})))
            
            conn.commit()
    
    def log_api_usage(self, usage_data: Dict[str, Any]) -> None:
        """Log API usage for analytics and rate limiting."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO api_usage 
                (endpoint, method, user_id, response_status, response_time_ms,
                 request_size, response_size, ip_address, user_agent, rate_limit_hit)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                usage_data.get('endpoint', ''),
                usage_data.get('method', ''),
                usage_data.get('user_id', ''),
                usage_data.get('response_status', 200),
                usage_data.get('response_time_ms', 0),
                usage_data.get('request_size', 0),
                usage_data.get('response_size', 0),
                usage_data.get('ip_address', ''),
                usage_data.get('user_agent', ''),
                usage_data.get('rate_limit_hit', False)
            ))
            
            conn.commit()
    
    def get_route_analytics(self, filters: Dict = None, limit: int = 100) -> Dict[str, Any]:
        """Get route usage analytics with optional filters."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM route_analytics WHERE 1=1"
            params = []
            
            if filters:
                if 'start_date' in filters:
                    query += " AND created_at >= ?"
                    params.append(filters['start_date'])
                
                if 'end_date' in filters:
                    query += " AND created_at <= ?"
                    params.append(filters['end_date'])
                
                if 'user_id' in filters:
                    query += " AND user_id = ?"
                    params.append(filters['user_id'])
                
                if 'mobility_aid' in filters:
                    query += " AND mobility_aid = ?"
                    params.append(filters['mobility_aid'])
                
                if 'success_only' in filters and filters['success_only']:
                    query += " AND success = TRUE"
            
            query += f" ORDER BY created_at DESC LIMIT {limit}"
            
            cursor.execute(query, params)
            results = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            
            routes = [dict(zip(columns, result)) for result in results]
            
            # Calculate summary statistics
            if routes:
                distances = [r['total_distance'] for r in routes if r['total_distance']]
                durations = [r['total_duration'] for r in routes if r['total_duration']]
                accessibility_scores = [r['accessibility_score'] for r in routes if r['accessibility_score']]
                exec_times = [r['execution_time_ms'] for r in routes if r['execution_time_ms']]
                
                summary = {
                    'total_routes': len(routes),
                    'successful_routes': sum(1 for r in routes if r['success']),
                    'average_distance': statistics.mean(distances) if distances else 0,
                    'average_duration': statistics.mean(durations) if durations else 0,
                    'average_accessibility_score': statistics.mean(accessibility_scores) if accessibility_scores else 0,
                    'average_execution_time_ms': statistics.mean(exec_times) if exec_times else 0,
                    'mobility_aids_distribution': Counter(r['mobility_aid'] for r in routes),
                    'algorithms_used': Counter(r['algorithm_used'] for r in routes)
                }
            else:
                summary = {'total_routes': 0}
            
            return {
                'routes': routes,
                'summary': summary,
                'filters_applied': filters or {}
            }
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get current system health metrics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get recent metrics (last hour)
            one_hour_ago = datetime.now() - timedelta(hours=1)
            
            cursor.execute("""
                SELECT metric_name, AVG(metric_value) as avg_value, 
                       COUNT(*) as sample_count, metric_unit
                FROM system_metrics 
                WHERE recorded_at >= ? 
                GROUP BY metric_name, metric_unit
            """, (one_hour_ago.isoformat(),))
            
            metrics = cursor.fetchall()
            
            # Get error count in last hour
            cursor.execute("""
                SELECT COUNT(*) FROM error_logs 
                WHERE timestamp >= ? AND resolved = FALSE
            """, (one_hour_ago.isoformat(),))
            
            unresolved_errors = cursor.fetchone()[0]
            
            # Get API success rate in last hour
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_requests,
                    SUM(CASE WHEN response_status < 400 THEN 1 ELSE 0 END) as successful_requests,
                    AVG(response_time_ms) as avg_response_time
                FROM api_usage 
                WHERE timestamp >= ?
            """, (one_hour_ago.isoformat(),))
            
            api_stats = cursor.fetchone()
            
            health_status = {
                'overall_status': 'healthy',
                'metrics': {metric[0]: {
                    'value': metric[1],
                    'unit': metric[3],
                    'samples': metric[2]
                } for metric in metrics},
                'errors': {
                    'unresolved_count': unresolved_errors,
                    'status': 'ok' if unresolved_errors < 10 else 'warning' if unresolved_errors < 50 else 'critical'
                },
                'api_performance': {
                    'total_requests': api_stats[0] if api_stats[0] else 0,
                    'success_rate': (api_stats[1] / api_stats[0] * 100) if api_stats[0] else 100,
                    'avg_response_time_ms': api_stats[2] if api_stats[2] else 0
                },
                'timestamp': datetime.now().isoformat()
            }
            
            # Determine overall health status
            if unresolved_errors > 50 or (api_stats[0] and (api_stats[1] / api_stats[0]) < 0.95):
                health_status['overall_status'] = 'critical'
            elif unresolved_errors > 10 or (api_stats[0] and (api_stats[1] / api_stats[0]) < 0.99):
                health_status['overall_status'] = 'warning'
            
            return health_status


# Initialize analytics manager
analytics_manager = DataAnalyticsManager()


# API Endpoints

@handle_api_errors
@require_auth
@validate_query_params({
    **PAGINATION_PARAMS,
    "start_date": {"type": "string", "format": "date"},
    "end_date": {"type": "string", "format": "date"},
    "mobility_aid": {"type": "string", "enum": ["none", "wheelchair", "mobility_scooter", "walker", "cane", "crutches", "prosthetic"]},
    "success_only": {"type": "boolean", "default": False}
})
def get_route_analytics():
    """Get route usage analytics and statistics."""
    # Check if user has admin permissions (simplified check)
    user_id = request.current_user['user_id']
    is_admin = request.current_user.get('is_admin', False)
    
    filters = {}
    
    # Non-admin users can only see their own data
    if not is_admin:
        filters['user_id'] = str(user_id)
    
    # Add query parameter filters
    validated_params = getattr(request, 'validated_params', {})
    if 'start_date' in validated_params:
        filters['start_date'] = validated_params['start_date']
    if 'end_date' in validated_params:
        filters['end_date'] = validated_params['end_date']
    if 'mobility_aid' in validated_params:
        filters['mobility_aid'] = validated_params['mobility_aid']
    if validated_params.get('success_only'):
        filters['success_only'] = True
    
    limit = validated_params.get('limit', 100)
    
    analytics = analytics_manager.get_route_analytics(filters, limit)
    
    return format_success_response(analytics)


@handle_api_errors
@require_auth
def get_system_health():
    """Get system health metrics and status."""
    # Only admin users can access system health
    is_admin = request.current_user.get('is_admin', False)
    if not is_admin:
        return jsonify({"error": "Admin access required"}), 403
    
    health = analytics_manager.get_system_health()
    
    return format_success_response(health)


@handle_api_errors
@require_auth
@validate_query_params({
    **PAGINATION_PARAMS,
    "category": {"type": "string", "enum": ["performance", "usage", "errors", "ml"]},
    "metric_name": {"type": "string"},
    "start_date": {"type": "string"},
    "end_date": {"type": "string"}
})
def get_system_metrics():
    """Get system metrics with filtering and aggregation."""
    is_admin = request.current_user.get('is_admin', False)
    if not is_admin:
        return jsonify({"error": "Admin access required"}), 403
    
    validated_params = getattr(request, 'validated_params', {})
    
    with sqlite3.connect(analytics_manager.db_path) as conn:
        cursor = conn.cursor()
        
        query = "SELECT * FROM system_metrics WHERE 1=1"
        params = []
        
        if 'category' in validated_params:
            query += " AND category = ?"
            params.append(validated_params['category'])
        
        if 'metric_name' in validated_params:
            query += " AND metric_name LIKE ?"
            params.append(f"%{validated_params['metric_name']}%")
        
        if 'start_date' in validated_params:
            query += " AND recorded_at >= ?"
            params.append(validated_params['start_date'])
        
        if 'end_date' in validated_params:
            query += " AND recorded_at <= ?"
            params.append(validated_params['end_date'])
        
        query += " ORDER BY recorded_at DESC"
        limit = validated_params.get('limit', 100)
        query += f" LIMIT {limit}"
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        
        metrics = []
        for result in results:
            metric = dict(zip(columns, result))
            if metric['metadata']:
                try:
                    metric['metadata'] = json.loads(metric['metadata'])
                except:
                    metric['metadata'] = {}
            metrics.append(metric)
    
    return format_success_response({
        'metrics': metrics,
        'pagination': {
            'limit': limit,
            'total': len(metrics)
        }
    })


@handle_api_errors
@require_auth
def get_api_usage_stats():
    """Get API usage statistics."""
    is_admin = request.current_user.get('is_admin', False)
    if not is_admin:
        return jsonify({"error": "Admin access required"}), 403
    
    # Get stats for different time periods
    now = datetime.now()
    periods = {
        'last_hour': now - timedelta(hours=1),
        'last_day': now - timedelta(days=1),
        'last_week': now - timedelta(weeks=1),
        'last_month': now - timedelta(days=30)
    }
    
    stats = {}
    
    with sqlite3.connect(analytics_manager.db_path) as conn:
        cursor = conn.cursor()
        
        for period_name, start_time in periods.items():
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_requests,
                    COUNT(DISTINCT user_id) as unique_users,
                    AVG(response_time_ms) as avg_response_time,
                    SUM(CASE WHEN response_status < 400 THEN 1 ELSE 0 END) as successful_requests,
                    SUM(CASE WHEN rate_limit_hit = 1 THEN 1 ELSE 0 END) as rate_limited_requests
                FROM api_usage 
                WHERE timestamp >= ?
            """, (start_time.isoformat(),))
            
            result = cursor.fetchone()
            
            # Get top endpoints
            cursor.execute("""
                SELECT endpoint, COUNT(*) as request_count
                FROM api_usage 
                WHERE timestamp >= ?
                GROUP BY endpoint
                ORDER BY request_count DESC
                LIMIT 10
            """, (start_time.isoformat(),))
            
            top_endpoints = [{'endpoint': row[0], 'count': row[1]} for row in cursor.fetchall()]
            
            stats[period_name] = {
                'total_requests': result[0],
                'unique_users': result[1],
                'avg_response_time_ms': result[2] or 0,
                'success_rate': (result[3] / result[0] * 100) if result[0] else 100,
                'rate_limited_requests': result[4],
                'top_endpoints': top_endpoints
            }
    
    return format_success_response(stats)


@handle_api_errors
@require_auth
def get_map_data_status():
    """Get map data coverage and quality status."""
    with sqlite3.connect(analytics_manager.db_path) as conn:
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM map_data_status ORDER BY region_name")
        results = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        
        regions = [dict(zip(columns, result)) for result in results]
        
        # Calculate overall statistics
        if regions:
            total_nodes = sum(r['nodes_count'] for r in regions)
            total_edges = sum(r['edges_count'] for r in regions)
            avg_coverage = statistics.mean(r['accessibility_data_coverage'] for r in regions)
            avg_quality = statistics.mean(r['data_quality_score'] for r in regions)
            outdated_regions = len([r for r in regions if r['update_status'] == 'outdated'])
            
            summary = {
                'total_regions': len(regions),
                'total_nodes': total_nodes,
                'total_edges': total_edges,
                'average_accessibility_coverage': avg_coverage,
                'average_data_quality': avg_quality,
                'outdated_regions': outdated_regions,
                'regions_needing_update': outdated_regions
            }
        else:
            summary = {'total_regions': 0}
    
    return format_success_response({
        'regions': regions,
        'summary': summary
    })


@handle_api_errors
@require_auth
@validate_request_data({
    "type": "object",
    "required": ["region_name"],
    "properties": {
        "region_name": {"type": "string", "minLength": 1},
        "data_source": {"type": "string"},
        "nodes_count": {"type": "integer", "minimum": 0},
        "edges_count": {"type": "integer", "minimum": 0},
        "accessibility_data_coverage": {"type": "number", "minimum": 0, "maximum": 100},
        "data_quality_score": {"type": "number", "minimum": 0, "maximum": 10},
        "update_status": {"type": "string", "enum": ["current", "updating", "outdated"]}
    }
})
def update_map_data_status():
    """Update map data status for a region."""
    is_admin = request.current_user.get('is_admin', False)
    if not is_admin:
        return jsonify({"error": "Admin access required"}), 403
    
    data = request.json
    
    with sqlite3.connect(analytics_manager.db_path) as conn:
        cursor = conn.cursor()
        
        # Check if region exists
        cursor.execute("SELECT id FROM map_data_status WHERE region_name = ?", (data['region_name'],))
        existing = cursor.fetchone()
        
        if existing:
            # Update existing region
            update_fields = []
            values = []
            
            for field in ['data_source', 'nodes_count', 'edges_count', 
                         'accessibility_data_coverage', 'data_quality_score', 'update_status']:
                if field in data:
                    update_fields.append(f"{field} = ?")
                    values.append(data[field])
            
            if update_fields:
                update_fields.append("last_updated = ?")
                values.append(datetime.now().isoformat())
                values.append(data['region_name'])
                
                query = f"UPDATE map_data_status SET {', '.join(update_fields)} WHERE region_name = ?"
                cursor.execute(query, values)
        else:
            # Insert new region
            cursor.execute("""
                INSERT INTO map_data_status 
                (region_name, data_source, nodes_count, edges_count, 
                 accessibility_data_coverage, data_quality_score, update_status, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data['region_name'],
                data.get('data_source', ''),
                data.get('nodes_count', 0),
                data.get('edges_count', 0),
                data.get('accessibility_data_coverage', 0.0),
                data.get('data_quality_score', 0.0),
                data.get('update_status', 'current'),
                datetime.now().isoformat()
            ))
        
        conn.commit()
    
    return format_success_response({"updated": True}, "Map data status updated successfully")


@handle_api_errors
@require_auth
def export_analytics_data():
    """Export analytics data in various formats."""
    is_admin = request.current_user.get('is_admin', False)
    if not is_admin:
        return jsonify({"error": "Admin access required"}), 403
    
    export_format = request.args.get('format', 'json').lower()
    data_type = request.args.get('type', 'routes')  # routes, metrics, api_usage
    
    if export_format not in ['json', 'csv']:
        return jsonify({"error": "Unsupported format. Use 'json' or 'csv'"}), 400
    
    if data_type not in ['routes', 'metrics', 'api_usage']:
        return jsonify({"error": "Unsupported data type"}), 400
    
    # Get data based on type
    with sqlite3.connect(analytics_manager.db_path) as conn:
        if data_type == 'routes':
            df = pd.read_sql_query("SELECT * FROM route_analytics ORDER BY created_at DESC LIMIT 10000", conn)
        elif data_type == 'metrics':
            df = pd.read_sql_query("SELECT * FROM system_metrics ORDER BY recorded_at DESC LIMIT 10000", conn)
        elif data_type == 'api_usage':
            df = pd.read_sql_query("SELECT * FROM api_usage ORDER BY timestamp DESC LIMIT 10000", conn)
    
    # Generate response based on format
    if export_format == 'csv':
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        
        return Response(
            output.getvalue(),
            mimetype='text/csv',
            headers={'Content-Disposition': f'attachment; filename={data_type}_export.csv'}
        )
    else:  # JSON
        data = df.to_dict(orient='records')
        return format_success_response({
            'data': data,
            'export_info': {
                'type': data_type,
                'format': export_format,
                'record_count': len(data),
                'exported_at': datetime.now().isoformat()
            }
        })


@handle_api_errors
@require_auth
def get_error_logs():
    """Get system error logs."""
    is_admin = request.current_user.get('is_admin', False)
    if not is_admin:
        return jsonify({"error": "Admin access required"}), 403
    
    severity = request.args.get('severity')
    resolved = request.args.get('resolved')
    limit = min(int(request.args.get('limit', 100)), 1000)
    
    with sqlite3.connect(analytics_manager.db_path) as conn:
        cursor = conn.cursor()
        
        query = "SELECT * FROM error_logs WHERE 1=1"
        params = []
        
        if severity:
            query += " AND severity = ?"
            params.append(severity)
        
        if resolved is not None:
            query += " AND resolved = ?"
            params.append(resolved.lower() == 'true')
        
        query += f" ORDER BY timestamp DESC LIMIT {limit}"
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        
        errors = []
        for result in results:
            error = dict(zip(columns, result))
            if error['request_data']:
                try:
                    error['request_data'] = json.loads(error['request_data'])
                except:
                    pass
            errors.append(error)
    
    return format_success_response({
        'errors': errors,
        'pagination': {
            'limit': limit,
            'total': len(errors)
        }
    })


# Export endpoint functions for Flask app registration
DATA_ANALYTICS_ENDPOINTS = {
    'get_route_analytics': get_route_analytics,
    'get_system_health': get_system_health,
    'get_system_metrics': get_system_metrics,
    'get_api_usage_stats': get_api_usage_stats,
    'get_map_data_status': get_map_data_status,
    'update_map_data_status': update_map_data_status,
    'export_analytics_data': export_analytics_data,
    'get_error_logs': get_error_logs
}
