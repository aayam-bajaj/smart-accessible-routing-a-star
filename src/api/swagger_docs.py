"""
Swagger/OpenAPI Documentation Generator
=====================================

Generates comprehensive OpenAPI 3.0 documentation for all API endpoints
including request/response schemas, authentication, error handling, and
interactive API exploration.
"""

from typing import Dict, Any, List
from flask import Flask, jsonify, request, redirect, url_for
from flask_swagger_ui import get_swaggerui_blueprint
import yaml
import json
from datetime import datetime

from .schemas import API_SCHEMAS, SCHEMA_DEFINITIONS


class SwaggerDocGenerator:
    """Generates and manages OpenAPI documentation."""
    
    def __init__(self, app: Flask = None):
        self.app = app
        self.openapi_spec = None
        self._generate_openapi_spec()
        
        if app:
            self.init_app(app)
    
    def init_app(self, app: Flask):
        """Initialize Swagger UI with Flask app."""
        self.app = app
        
        # Swagger UI configuration
        SWAGGER_URL = '/api/docs'
        API_URL = '/api/openapi.json'
        
        swaggerui_blueprint = get_swaggerui_blueprint(
            SWAGGER_URL,
            API_URL,
            config={
                'app_name': "Smart Accessible Routing API",
                'dom_id': '#swagger-ui',
                'url': API_URL,
                'layout': 'StandaloneLayout',
                'deepLinking': True,
                'displayRequestDuration': True,
                'docExpansion': 'list',
                'filter': True,
                'showRequestHeaders': True,
                'supportedSubmitMethods': ['get', 'post', 'put', 'delete', 'patch']
            }
        )
        
        app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)
        
        # Register OpenAPI spec endpoint
        @app.route('/api/openapi.json')
        def get_openapi_spec():
            return jsonify(self.openapi_spec)
        
        @app.route('/api/openapi.yaml')
        def get_openapi_yaml():
            return yaml.dump(self.openapi_spec, default_flow_style=False), 200, {'Content-Type': 'application/yaml'}
        
        # Redirect root docs to Swagger UI
        @app.route('/docs')
        @app.route('/api-docs')
        def redirect_to_docs():
            return redirect(SWAGGER_URL)
    
    def _generate_openapi_spec(self):
        """Generate complete OpenAPI 3.0 specification."""
        self.openapi_spec = {
            "openapi": "3.0.3",
            "info": {
                "title": "Smart Accessible Routing API",
                "description": self._get_api_description(),
                "version": "1.0.0",
                "contact": {
                    "name": "Smart Accessible Routing Team",
                    "email": "support@smart-routing.com"
                },
                "license": {
                    "name": "MIT License",
                    "url": "https://opensource.org/licenses/MIT"
                }
            },
            "servers": [
                {
                    "url": "http://localhost:5000/api/v1",
                    "description": "Development server"
                },
                {
                    "url": "https://api.smart-routing.com/v1",
                    "description": "Production server"
                }
            ],
            "paths": self._generate_paths(),
            "components": {
                "schemas": self._generate_schemas(),
                "securitySchemes": self._generate_security_schemes(),
                "responses": self._generate_common_responses(),
                "parameters": self._generate_common_parameters()
            },
            "security": [{"BearerAuth": []}],
            "tags": self._generate_tags()
        }
    
    def _get_api_description(self) -> str:
        """Get comprehensive API description."""
        return """
        # Smart Accessible Routing API
        
        A comprehensive API for calculating accessible routes with advanced personalization,
        machine learning integration, and real-time accessibility feedback.
        
        ## Features
        
        - **Accessible Route Calculation**: Advanced A* algorithms optimized for accessibility
        - **Personalization**: Machine learning-driven route personalization based on user feedback
        - **Multi-criteria Optimization**: Balance distance, time, accessibility, safety, and comfort
        - **Real-time Learning**: Continuous improvement through user feedback integration
        - **Accessibility Management**: Comprehensive accessibility constraint and preference management
        - **Analytics & Monitoring**: Detailed route analytics and system performance monitoring
        
        ## Authentication
        
        Most endpoints require authentication using JWT tokens. Include the token in the Authorization header:
        
        ```
        Authorization: Bearer YOUR_JWT_TOKEN
        ```
        
        ## Rate Limiting
        
        API requests are rate limited to ensure fair usage:
        - Authenticated users: 1000 requests/hour
        - Unauthenticated users: 100 requests/hour
        
        ## Error Handling
        
        The API uses standard HTTP status codes and returns detailed error information:
        
        - `400 Bad Request`: Invalid request data or parameters
        - `401 Unauthorized`: Authentication required or invalid token
        - `403 Forbidden`: Insufficient permissions
        - `404 Not Found`: Resource not found
        - `429 Too Many Requests`: Rate limit exceeded
        - `500 Internal Server Error`: Unexpected server error
        
        ## Versioning
        
        The API uses URL-based versioning. Current version is `v1`.
        """
    
    def _generate_paths(self) -> Dict[str, Any]:
        """Generate OpenAPI paths for all endpoints."""
        paths = {}
        
        # Routing endpoints
        paths.update(self._get_routing_paths())
        
        # Authentication endpoints
        paths.update(self._get_auth_paths())
        
        # Accessibility and ML endpoints
        paths.update(self._get_accessibility_ml_paths())
        
        # Analytics endpoints
        paths.update(self._get_analytics_paths())
        
        return paths
    
    def _get_routing_paths(self) -> Dict[str, Any]:
        """Generate routing endpoint paths."""
        return {
            "/routes/calculate": {
                "post": {
                    "tags": ["Routing"],
                    "summary": "Calculate accessible route",
                    "description": "Calculate an accessible route between two points with user-specific constraints",
                    "operationId": "calculateRoute",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/RouteRequest"},
                                "examples": {
                                    "basic_route": {
                                        "summary": "Basic route calculation",
                                        "value": {
                                            "start": {"lat": 40.7128, "lon": -74.0060, "name": "New York City"},
                                            "end": {"lat": 40.7589, "lon": -73.9851, "name": "Times Square"},
                                            "user_profile": {
                                                "mobility_aid": "wheelchair",
                                                "constraints": {
                                                    "max_slope_percent": 5.0,
                                                    "requires_elevator": True
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Route calculated successfully",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/RouteResponse"}
                                }
                            }
                        },
                        "400": {"$ref": "#/components/responses/BadRequest"},
                        "401": {"$ref": "#/components/responses/Unauthorized"},
                        "500": {"$ref": "#/components/responses/InternalError"}
                    }
                }
            },
            "/routes/multi-criteria": {
                "post": {
                    "tags": ["Routing"],
                    "summary": "Multi-criteria route optimization",
                    "description": "Calculate routes optimized for multiple criteria with custom weights",
                    "operationId": "calculateMultiCriteriaRoute",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/MultiCriteriaRouteRequest"}
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Multi-criteria route calculated successfully",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/RouteResponse"}
                                }
                            }
                        },
                        "400": {"$ref": "#/components/responses/BadRequest"},
                        "401": {"$ref": "#/components/responses/Unauthorized"}
                    }
                }
            },
            "/routes/personalized": {
                "post": {
                    "tags": ["Routing"],
                    "summary": "Get personalized route recommendations",
                    "description": "Calculate routes using machine learning personalization based on user history",
                    "operationId": "getPersonalizedRoute",
                    "security": [{"BearerAuth": []}],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/RouteRequest"}
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Personalized route calculated successfully",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/PersonalizedRouteResponse"}
                                }
                            }
                        },
                        "401": {"$ref": "#/components/responses/Unauthorized"}
                    }
                }
            },
            "/routes/compare": {
                "post": {
                    "tags": ["Routing"],
                    "summary": "Compare multiple routes",
                    "description": "Compare and rank multiple route options based on user preferences",
                    "operationId": "compareRoutes",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/RouteComparisonRequest"}
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Route comparison completed successfully",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/RouteComparisonResponse"}
                                }
                            }
                        }
                    }
                }
            }
        }
    
    def _get_auth_paths(self) -> Dict[str, Any]:
        """Generate authentication endpoint paths."""
        return {
            "/auth/register": {
                "post": {
                    "tags": ["Authentication"],
                    "summary": "Register new user",
                    "description": "Create a new user account with accessibility profile",
                    "operationId": "registerUser",
                    "security": [],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/UserRegistration"},
                                "examples": {
                                    "wheelchair_user": {
                                        "summary": "Wheelchair user registration",
                                        "value": {
                                            "email": "user@example.com",
                                            "username": "wheelchairuser",
                                            "password": "SecurePass123",
                                            "profile": {
                                                "mobility_aid": "wheelchair",
                                                "constraints": {
                                                    "requires_elevator": True,
                                                    "max_slope_percent": 5.0
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    },
                    "responses": {
                        "201": {
                            "description": "User registered successfully",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/AuthResponse"}
                                }
                            }
                        },
                        "400": {"$ref": "#/components/responses/BadRequest"}
                    }
                }
            },
            "/auth/login": {
                "post": {
                    "tags": ["Authentication"],
                    "summary": "User login",
                    "description": "Authenticate user and receive JWT token",
                    "operationId": "loginUser",
                    "security": [],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/UserLogin"}
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Login successful",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/AuthResponse"}
                                }
                            }
                        },
                        "401": {"$ref": "#/components/responses/Unauthorized"}
                    }
                }
            },
            "/auth/profile": {
                "get": {
                    "tags": ["Authentication"],
                    "summary": "Get current user profile",
                    "description": "Retrieve current authenticated user's profile and accessibility settings",
                    "operationId": "getCurrentUser",
                    "security": [{"BearerAuth": []}],
                    "responses": {
                        "200": {
                            "description": "User profile retrieved successfully",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/UserProfile"}
                                }
                            }
                        },
                        "401": {"$ref": "#/components/responses/Unauthorized"}
                    }
                },
                "put": {
                    "tags": ["Authentication"],
                    "summary": "Update user profile",
                    "description": "Update user's accessibility profile and preferences",
                    "operationId": "updateUserProfile",
                    "security": [{"BearerAuth": []}],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/UserProfileUpdate"}
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Profile updated successfully",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/UserProfile"}
                                }
                            }
                        },
                        "401": {"$ref": "#/components/responses/Unauthorized"}
                    }
                }
            }
        }
    
    def _get_accessibility_ml_paths(self) -> Dict[str, Any]:
        """Generate accessibility and ML endpoint paths."""
        return {
            "/accessibility/templates": {
                "get": {
                    "tags": ["Accessibility"],
                    "summary": "Get accessibility templates",
                    "description": "Retrieve available accessibility constraint templates",
                    "operationId": "getAccessibilityTemplates",
                    "security": [],
                    "responses": {
                        "200": {
                            "description": "Templates retrieved successfully",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "success": {"type": "boolean"},
                                            "data": {
                                                "type": "array",
                                                "items": {"$ref": "#/components/schemas/AccessibilityTemplate"}
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
                "post": {
                    "tags": ["Accessibility"],
                    "summary": "Create accessibility template",
                    "description": "Create a new accessibility constraint template",
                    "operationId": "createAccessibilityTemplate",
                    "security": [{"BearerAuth": []}],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/AccessibilityTemplate"}
                            }
                        }
                    },
                    "responses": {
                        "201": {
                            "description": "Template created successfully"
                        },
                        "401": {"$ref": "#/components/responses/Unauthorized"}
                    }
                }
            },
            "/ml/feedback": {
                "post": {
                    "tags": ["Machine Learning"],
                    "summary": "Submit route feedback",
                    "description": "Submit feedback for route quality and accessibility",
                    "operationId": "submitFeedback",
                    "security": [{"BearerAuth": []}],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/FeedbackRequest"}
                            }
                        }
                    },
                    "responses": {
                        "201": {
                            "description": "Feedback submitted successfully"
                        },
                        "401": {"$ref": "#/components/responses/Unauthorized"}
                    }
                }
            },
            "/ml/recommendations": {
                "post": {
                    "tags": ["Machine Learning"],
                    "summary": "Get personalized recommendations",
                    "description": "Get ML-powered personalized routing recommendations",
                    "operationId": "getPersonalizedRecommendations",
                    "security": [{"BearerAuth": []}],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/PersonalizationRequest"}
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Recommendations generated successfully",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/PersonalizationResponse"}
                                }
                            }
                        },
                        "401": {"$ref": "#/components/responses/Unauthorized"}
                    }
                }
            }
        }
    
    def _get_analytics_paths(self) -> Dict[str, Any]:
        """Generate analytics endpoint paths."""
        return {
            "/analytics/routes": {
                "get": {
                    "tags": ["Analytics"],
                    "summary": "Get route analytics",
                    "description": "Retrieve route usage analytics and statistics",
                    "operationId": "getRouteAnalytics",
                    "security": [{"BearerAuth": []}],
                    "parameters": [
                        {"$ref": "#/components/parameters/StartDate"},
                        {"$ref": "#/components/parameters/EndDate"},
                        {"$ref": "#/components/parameters/Limit"},
                        {"$ref": "#/components/parameters/MobilityAid"}
                    ],
                    "responses": {
                        "200": {
                            "description": "Analytics retrieved successfully",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/RouteAnalytics"}
                                }
                            }
                        },
                        "401": {"$ref": "#/components/responses/Unauthorized"}
                    }
                }
            },
            "/analytics/system/health": {
                "get": {
                    "tags": ["Analytics"],
                    "summary": "Get system health",
                    "description": "Retrieve current system health metrics (Admin only)",
                    "operationId": "getSystemHealth",
                    "security": [{"BearerAuth": ["admin"]}],
                    "responses": {
                        "200": {
                            "description": "System health retrieved successfully",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/SystemHealth"}
                                }
                            }
                        },
                        "401": {"$ref": "#/components/responses/Unauthorized"},
                        "403": {"$ref": "#/components/responses/Forbidden"}
                    }
                }
            }
        }
    
    def _generate_schemas(self) -> Dict[str, Any]:
        """Generate OpenAPI schemas from our schema definitions."""
        schemas = {}
        
        # Convert our existing schemas to OpenAPI format
        for schema_name, schema_def in API_SCHEMAS.items():
            schemas[self._to_camel_case(schema_name)] = schema_def
        
        # Add common definitions
        for def_name, def_schema in SCHEMA_DEFINITIONS.items():
            schemas[self._to_camel_case(def_name)] = def_schema
        
        # Add response schemas
        schemas.update({
            "RouteResponse": {
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "data": {"$ref": "#/components/schemas/RouteResult"},
                    "message": {"type": "string"}
                }
            },
            "PersonalizedRouteResponse": {
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "data": {"$ref": "#/components/schemas/RouteResult"},
                    "recommendations": {"$ref": "#/components/schemas/PersonalizationResponse"},
                    "message": {"type": "string"}
                }
            },
            "RouteComparisonResponse": {
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "data": {
                        "type": "object",
                        "properties": {
                            "comparison": {
                                "type": "array",
                                "items": {"$ref": "#/components/schemas/RouteComparison"}
                            },
                            "recommendation": {"type": "string"}
                        }
                    }
                }
            },
            "AuthResponse": {
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "data": {
                        "type": "object",
                        "properties": {
                            "user": {"$ref": "#/components/schemas/User"},
                            "token": {"type": "string"},
                            "profile": {"$ref": "#/components/schemas/UserProfile"}
                        }
                    },
                    "message": {"type": "string"}
                }
            },
            "ErrorResponse": {
                "type": "object",
                "properties": {
                    "success": {"type": "boolean", "example": False},
                    "error": {"type": "string"},
                    "error_type": {"type": "string"},
                    "details": {"type": "object"}
                }
            }
        })
        
        return schemas
    
    def _generate_security_schemes(self) -> Dict[str, Any]:
        """Generate security scheme definitions."""
        return {
            "BearerAuth": {
                "type": "http",
                "scheme": "bearer",
                "bearerFormat": "JWT",
                "description": "JWT token for authentication. Include as: `Authorization: Bearer <token>`"
            }
        }
    
    def _generate_common_responses(self) -> Dict[str, Any]:
        """Generate common response definitions."""
        return {
            "BadRequest": {
                "description": "Bad request - invalid input data",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/ErrorResponse"},
                        "example": {
                            "success": False,
                            "error": "Validation error: Invalid coordinates",
                            "error_type": "validation_error",
                            "details": {"field": "lat", "value": 200}
                        }
                    }
                }
            },
            "Unauthorized": {
                "description": "Unauthorized - authentication required",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/ErrorResponse"},
                        "example": {
                            "success": False,
                            "error": "Invalid or expired token",
                            "error_type": "authentication_error"
                        }
                    }
                }
            },
            "Forbidden": {
                "description": "Forbidden - insufficient permissions",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/ErrorResponse"},
                        "example": {
                            "success": False,
                            "error": "Admin access required",
                            "error_type": "authorization_error"
                        }
                    }
                }
            },
            "NotFound": {
                "description": "Resource not found",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/ErrorResponse"},
                        "example": {
                            "success": False,
                            "error": "Template not found",
                            "error_type": "not_found_error"
                        }
                    }
                }
            },
            "InternalError": {
                "description": "Internal server error",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/ErrorResponse"},
                        "example": {
                            "success": False,
                            "error": "Internal server error",
                            "error_type": "server_error"
                        }
                    }
                }
            }
        }
    
    def _generate_common_parameters(self) -> Dict[str, Any]:
        """Generate common parameter definitions."""
        return {
            "Limit": {
                "name": "limit",
                "in": "query",
                "description": "Maximum number of results to return",
                "schema": {"type": "integer", "minimum": 1, "maximum": 100, "default": 20}
            },
            "Offset": {
                "name": "offset",
                "in": "query",
                "description": "Number of results to skip",
                "schema": {"type": "integer", "minimum": 0, "default": 0}
            },
            "StartDate": {
                "name": "start_date",
                "in": "query",
                "description": "Filter results from this date (ISO 8601)",
                "schema": {"type": "string", "format": "date"}
            },
            "EndDate": {
                "name": "end_date",
                "in": "query",
                "description": "Filter results to this date (ISO 8601)",
                "schema": {"type": "string", "format": "date"}
            },
            "MobilityAid": {
                "name": "mobility_aid",
                "in": "query",
                "description": "Filter by mobility aid type",
                "schema": {
                    "type": "string",
                    "enum": ["none", "wheelchair", "mobility_scooter", "walker", "cane", "crutches", "prosthetic"]
                }
            }
        }
    
    def _generate_tags(self) -> List[Dict[str, str]]:
        """Generate API tags for grouping endpoints."""
        return [
            {
                "name": "Routing",
                "description": "Route calculation and optimization endpoints"
            },
            {
                "name": "Authentication",
                "description": "User authentication and profile management"
            },
            {
                "name": "Accessibility",
                "description": "Accessibility constraint and template management"
            },
            {
                "name": "Machine Learning",
                "description": "ML-powered personalization and feedback"
            },
            {
                "name": "Analytics",
                "description": "System analytics and performance monitoring"
            }
        ]
    
    def _to_camel_case(self, snake_str: str) -> str:
        """Convert snake_case to CamelCase for OpenAPI schema names."""
        components = snake_str.split('_')
        return ''.join(word.capitalize() for word in components)


def setup_swagger_documentation(app: Flask) -> SwaggerDocGenerator:
    """Setup Swagger documentation for Flask app."""
    doc_generator = SwaggerDocGenerator(app)
    
    # Add additional documentation endpoints
    @app.route('/api/docs/redoc')
    def redoc_ui():
        """Alternative documentation UI using ReDoc."""
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Smart Accessible Routing API - ReDoc</title>
            <meta charset="utf-8"/>
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <link href="https://fonts.googleapis.com/css?family=Montserrat:300,400,700|Roboto:300,400,700" rel="stylesheet">
            <style>
                body {{ margin: 0; padding: 0; }}
            </style>
        </head>
        <body>
            <redoc spec-url='/api/openapi.json'></redoc>
            <script src="https://cdn.jsdelivr.net/npm/redoc@latest/bundles/redoc.standalone.js"></script>
        </body>
        </html>
        """
    
    @app.route('/api/docs/postman')
    def get_postman_collection():
        """Generate Postman collection for API testing."""
        collection = {
            "info": {
                "name": "Smart Accessible Routing API",
                "description": "Complete API collection for testing all endpoints",
                "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
            },
            "auth": {
                "type": "bearer",
                "bearer": [{"key": "token", "value": "{{jwt_token}}", "type": "string"}]
            },
            "variable": [
                {"key": "base_url", "value": "http://localhost:5000/api/v1"},
                {"key": "jwt_token", "value": ""}
            ],
            "item": _generate_postman_requests()
        }
        
        return jsonify(collection)
    
    return doc_generator


def _generate_postman_requests() -> List[Dict[str, Any]]:
    """Generate Postman request collection."""
    return [
        {
            "name": "Authentication",
            "item": [
                {
                    "name": "Register User",
                    "request": {
                        "method": "POST",
                        "header": [{"key": "Content-Type", "value": "application/json"}],
                        "body": {
                            "mode": "raw",
                            "raw": json.dumps({
                                "email": "test@example.com",
                                "username": "testuser",
                                "password": "TestPass123",
                                "profile": {
                                    "mobility_aid": "wheelchair",
                                    "constraints": {"requires_elevator": True}
                                }
                            }, indent=2)
                        },
                        "url": "{{base_url}}/auth/register"
                    }
                },
                {
                    "name": "Login User",
                    "request": {
                        "method": "POST",
                        "header": [{"key": "Content-Type", "value": "application/json"}],
                        "body": {
                            "mode": "raw",
                            "raw": json.dumps({
                                "login": "test@example.com",
                                "password": "TestPass123"
                            }, indent=2)
                        },
                        "url": "{{base_url}}/auth/login"
                    }
                }
            ]
        },
        {
            "name": "Routing",
            "item": [
                {
                    "name": "Calculate Route",
                    "request": {
                        "method": "POST",
                        "header": [{"key": "Content-Type", "value": "application/json"}],
                        "body": {
                            "mode": "raw",
                            "raw": json.dumps({
                                "start": {"lat": 40.7128, "lon": -74.0060},
                                "end": {"lat": 40.7589, "lon": -73.9851},
                                "user_profile": {
                                    "mobility_aid": "wheelchair",
                                    "constraints": {"max_slope_percent": 5.0}
                                }
                            }, indent=2)
                        },
                        "url": "{{base_url}}/routes/calculate"
                    }
                }
            ]
        }
    ]


# Export the documentation generator
__all__ = ['SwaggerDocGenerator', 'setup_swagger_documentation']
