"""
Main routes for the Smart Accessible Routing System
"""
from flask import Blueprint, render_template, request, jsonify, current_app
import logging

# Create blueprint
main_bp = Blueprint('main', __name__)

# Get logger
logger = logging.getLogger(__name__)

@main_bp.route('/')
def index():
    """
    Main index page
    """
    logger.info("Index page accessed")
    return render_template('main/index.html')

@main_bp.route('/dashboard')
def dashboard():
    """
    User dashboard page
    """
    logger.info("Dashboard page accessed")
    return render_template('main/dashboard.html')

@main_bp.route('/route-planner')
def route_planner():
    """
    Route planner page
    """
    logger.info("Route planner page accessed")
    return render_template('main/route_planner.html')

@main_bp.route('/about')
def about():
    """
    About page
    """
    logger.info("About page accessed")
    return render_template('main/about.html')

@main_bp.route('/contact')
def contact():
    """
    Contact page
    """
    logger.info("Contact page accessed")
    return render_template('main/contact.html')

@main_bp.route('/api/health')
def api_health():
    """
    API health check endpoint
    """
    logger.info("API health check accessed")
    return jsonify({
        'status': 'healthy',
        'service': 'Smart Accessible Routing System',
        'version': '1.0.0'
    })

@main_bp.route('/api/stats')
def api_stats():
    """
    API statistics endpoint
    """
    logger.info("API stats accessed")
    # Placeholder for actual statistics
    stats = {
        'total_routes': 0,
        'total_users': 0,
        'total_feedback': 0,
        'uptime': '100%'
    }
    return jsonify(stats)

@main_bp.errorhandler(404)
def not_found(error):
    """
    Handle 404 errors
    """
    logger.warning(f"404 error: {error}")
    return render_template('main/404.html'), 404

@main_bp.errorhandler(500)
def internal_error(error):
    """
    Handle 500 errors
    """
    logger.error(f"500 error: {error}")
    return render_template('main/500.html'), 500