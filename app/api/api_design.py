"""
Smart Accessible Routing API Design and Architecture
"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

class APIVersion(Enum):
    """API version enumeration"""
    V1 = "v1"
    V2 = "v2"

class HTTPMethod(Enum):
    """HTTP methods supported by API"""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"
    OPTIONS = "OPTIONS"

class ResponseFormat(Enum):
    """API response formats"""
    JSON = "application/json"
    XML = "application/xml"
    GEOJSON = "application/geo+json"

@dataclass
class APIEndpoint:
    """API endpoint definition"""
    path: str
    method: HTTPMethod
    version: APIVersion
    description: str
    
    # Request/Response specifications
    request_schema: Optional[Dict[str, Any]] = None
    response_schema: Optional[Dict[str, Any]] = None
    error_codes: List[int] = field(default_factory=lambda: [400, 401, 403, 404, 500])
    
    # Authentication and permissions
    requires_auth: bool = False
    required_permissions: List[str] = field(default_factory=list)
    rate_limit: Optional[int] = None  # requests per minute
    
    # Documentation
    tags: List[str] = field(default_factory=list)
    examples: Dict[str, Any] = field(default_factory=dict)

# ============================================================================
# API ARCHITECTURE SPECIFICATION
# ============================================================================

class APIArchitecture:
    """Comprehensive API architecture definition"""
    
    def __init__(self):
        """Initialize API architecture"""
        self.base_url = "/api"
        self.current_version = APIVersion.V1
        self.endpoints = self._define_endpoints()
        self.middleware_stack = self._define_middleware()
        self.security_config = self._define_security()
        self.documentation_config = self._define_documentation()
    
    def _define_endpoints(self) -> Dict[str, APIEndpoint]:
        """Define all API endpoints"""
        endpoints = {}
        
        # ====================================================================
        # ROUTING ENDPOINTS
        # ====================================================================
        
        # Basic route calculation
        endpoints["calculate_route"] = APIEndpoint(
            path="/v1/routes/calculate",
            method=HTTPMethod.POST,
            version=APIVersion.V1,
            description="Calculate optimal route between two points",
            request_schema={
                "type": "object",
                "required": ["start", "end"],
                "properties": {
                    "start": {
                        "type": "object",
                        "properties": {
                            "latitude": {"type": "number", "minimum": -90, "maximum": 90},
                            "longitude": {"type": "number", "minimum": -180, "maximum": 180}
                        },
                        "required": ["latitude", "longitude"]
                    },
                    "end": {
                        "type": "object",
                        "properties": {
                            "latitude": {"type": "number", "minimum": -90, "maximum": 90},
                            "longitude": {"type": "number", "minimum": -180, "maximum": 180}
                        },
                        "required": ["latitude", "longitude"]
                    },
                    "user_profile": {
                        "type": "object",
                        "properties": {
                            "mobility_aid": {"type": "string", "enum": ["none", "wheelchair", "walker", "cane", "scooter"]},
                            "walking_speed": {"type": "number", "minimum": 0.5, "maximum": 10.0},
                            "accessibility_needs": {"type": "array", "items": {"type": "string"}},
                            "preferences": {"type": "object"}
                        }
                    },
                    "routing_options": {
                        "type": "object",
                        "properties": {
                            "algorithm": {"type": "string", "enum": ["astar", "dijkstra", "bidirectional", "anytime"]},
                            "optimization_criteria": {"type": "array", "items": {"type": "string"}},
                            "avoid_obstacles": {"type": "array", "items": {"type": "string"}},
                            "include_alternatives": {"type": "boolean", "default": False},
                            "max_alternatives": {"type": "integer", "minimum": 1, "maximum": 5}
                        }
                    }
                }
            },
            response_schema={
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": ["success", "error"]},
                    "route": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "array", "items": {"type": "object"}},
                            "total_distance": {"type": "number"},
                            "total_time": {"type": "number"},
                            "accessibility_score": {"type": "number", "minimum": 0, "maximum": 1},
                            "warnings": {"type": "array", "items": {"type": "string"}},
                            "alternatives": {"type": "array", "items": {"type": "object"}}
                        }
                    },
                    "metadata": {
                        "type": "object",
                        "properties": {
                            "calculation_time": {"type": "number"},
                            "algorithm_used": {"type": "string"},
                            "nodes_explored": {"type": "integer"}
                        }
                    }
                }
            },
            tags=["routing", "core"],
            rate_limit=100,
            examples={
                "basic_request": {
                    "start": {"latitude": 40.7128, "longitude": -74.0060},
                    "end": {"latitude": 40.7589, "longitude": -73.9851}
                },
                "wheelchair_accessible": {
                    "start": {"latitude": 40.7128, "longitude": -74.0060},
                    "end": {"latitude": 40.7589, "longitude": -73.9851},
                    "user_profile": {
                        "mobility_aid": "wheelchair",
                        "accessibility_needs": ["wheelchair_accessible", "avoid_stairs"]
                    }
                }
            }
        )
        
        # Multi-criteria routing
        endpoints["multicriteria_route"] = APIEndpoint(
            path="/v1/routes/multicriteria",
            method=HTTPMethod.POST,
            version=APIVersion.V1,
            description="Calculate route optimized for multiple criteria",
            request_schema={
                "type": "object",
                "required": ["start", "end", "criteria"],
                "properties": {
                    "start": {"$ref": "#/components/schemas/Coordinate"},
                    "end": {"$ref": "#/components/schemas/Coordinate"},
                    "criteria": {
                        "type": "object",
                        "properties": {
                            "distance_weight": {"type": "number", "minimum": 0, "maximum": 1},
                            "time_weight": {"type": "number", "minimum": 0, "maximum": 1},
                            "accessibility_weight": {"type": "number", "minimum": 0, "maximum": 1},
                            "safety_weight": {"type": "number", "minimum": 0, "maximum": 1},
                            "comfort_weight": {"type": "number", "minimum": 0, "maximum": 1}
                        }
                    }
                }
            },
            tags=["routing", "advanced"],
            requires_auth=True,
            rate_limit=50
        )
        
        # Personalized routing
        endpoints["personalized_route"] = APIEndpoint(
            path="/v1/routes/personalized",
            method=HTTPMethod.POST,
            version=APIVersion.V1,
            description="Calculate personalized route based on user history and ML models",
            requires_auth=True,
            tags=["routing", "ml", "personalization"],
            rate_limit=50
        )
        
        # Route comparison
        endpoints["compare_routes"] = APIEndpoint(
            path="/v1/routes/compare",
            method=HTTPMethod.POST,
            version=APIVersion.V1,
            description="Compare multiple routing options",
            tags=["routing", "analysis"],
            rate_limit=25
        )
        
        # Real-time route updates
        endpoints["update_route"] = APIEndpoint(
            path="/v1/routes/{route_id}/update",
            method=HTTPMethod.GET,
            version=APIVersion.V1,
            description="Get real-time updates for an active route",
            requires_auth=True,
            tags=["routing", "realtime"],
            rate_limit=200
        )
        
        # ====================================================================
        # USER MANAGEMENT ENDPOINTS
        # ====================================================================
        
        # User registration
        endpoints["register_user"] = APIEndpoint(
            path="/v1/users/register",
            method=HTTPMethod.POST,
            version=APIVersion.V1,
            description="Register a new user account",
            request_schema={
                "type": "object",
                "required": ["email", "password"],
                "properties": {
                    "email": {"type": "string", "format": "email"},
                    "password": {"type": "string", "minLength": 8},
                    "name": {"type": "string", "minLength": 1, "maxLength": 100},
                    "accessibility_profile": {"$ref": "#/components/schemas/AccessibilityProfile"}
                }
            },
            tags=["users", "auth"],
            rate_limit=10
        )
        
        # User authentication
        endpoints["login_user"] = APIEndpoint(
            path="/v1/users/login",
            method=HTTPMethod.POST,
            version=APIVersion.V1,
            description="Authenticate user and obtain access token",
            request_schema={
                "type": "object",
                "required": ["email", "password"],
                "properties": {
                    "email": {"type": "string", "format": "email"},
                    "password": {"type": "string"},
                    "remember_me": {"type": "boolean", "default": False}
                }
            },
            tags=["users", "auth"],
            rate_limit=20
        )
        
        # User profile management
        endpoints["get_user_profile"] = APIEndpoint(
            path="/v1/users/profile",
            method=HTTPMethod.GET,
            version=APIVersion.V1,
            description="Get user profile information",
            requires_auth=True,
            tags=["users", "profile"],
            rate_limit=100
        )
        
        endpoints["update_user_profile"] = APIEndpoint(
            path="/v1/users/profile",
            method=HTTPMethod.PUT,
            version=APIVersion.V1,
            description="Update user profile information",
            requires_auth=True,
            tags=["users", "profile"],
            rate_limit=50
        )
        
        # User preferences
        endpoints["get_user_preferences"] = APIEndpoint(
            path="/v1/users/preferences",
            method=HTTPMethod.GET,
            version=APIVersion.V1,
            description="Get user routing preferences",
            requires_auth=True,
            tags=["users", "preferences"],
            rate_limit=100
        )
        
        endpoints["update_user_preferences"] = APIEndpoint(
            path="/v1/users/preferences",
            method=HTTPMethod.PUT,
            version=APIVersion.V1,
            description="Update user routing preferences",
            requires_auth=True,
            tags=["users", "preferences"],
            rate_limit=50
        )
        
        # ====================================================================
        # ACCESSIBILITY ENDPOINTS
        # ====================================================================
        
        # Accessibility constraints
        endpoints["get_accessibility_constraints"] = APIEndpoint(
            path="/v1/accessibility/constraints",
            method=HTTPMethod.GET,
            version=APIVersion.V1,
            description="Get available accessibility constraints",
            tags=["accessibility"],
            rate_limit=200
        )
        
        # Accessibility assessment
        endpoints["assess_route_accessibility"] = APIEndpoint(
            path="/v1/accessibility/assess",
            method=HTTPMethod.POST,
            version=APIVersion.V1,
            description="Assess accessibility of a specific route",
            tags=["accessibility", "analysis"],
            rate_limit=50
        )
        
        # Report accessibility issues
        endpoints["report_accessibility_issue"] = APIEndpoint(
            path="/v1/accessibility/report",
            method=HTTPMethod.POST,
            version=APIVersion.V1,
            description="Report accessibility issues on routes",
            requires_auth=True,
            tags=["accessibility", "feedback"],
            rate_limit=20
        )
        
        # ====================================================================
        # ML INTEGRATION ENDPOINTS
        # ====================================================================
        
        # Submit feedback for ML training
        endpoints["submit_feedback"] = APIEndpoint(
            path="/v1/ml/feedback",
            method=HTTPMethod.POST,
            version=APIVersion.V1,
            description="Submit route feedback for ML model improvement",
            requires_auth=True,
            tags=["ml", "feedback"],
            rate_limit=100
        )
        
        # Get personalized recommendations
        endpoints["get_recommendations"] = APIEndpoint(
            path="/v1/ml/recommendations",
            method=HTTPMethod.GET,
            version=APIVersion.V1,
            description="Get personalized route recommendations",
            requires_auth=True,
            tags=["ml", "recommendations"],
            rate_limit=50
        )
        
        # Model performance metrics
        endpoints["get_model_metrics"] = APIEndpoint(
            path="/v1/ml/metrics",
            method=HTTPMethod.GET,
            version=APIVersion.V1,
            description="Get ML model performance metrics",
            requires_auth=True,
            required_permissions=["admin"],
            tags=["ml", "admin"],
            rate_limit=10
        )
        
        # ====================================================================
        # DATA MANAGEMENT ENDPOINTS
        # ====================================================================
        
        # Map data endpoints
        endpoints["get_map_data"] = APIEndpoint(
            path="/v1/data/map",
            method=HTTPMethod.GET,
            version=APIVersion.V1,
            description="Get map data for specified area",
            tags=["data", "maps"],
            rate_limit=100
        )
        
        endpoints["upload_map_data"] = APIEndpoint(
            path="/v1/data/map",
            method=HTTPMethod.POST,
            version=APIVersion.V1,
            description="Upload new map data",
            requires_auth=True,
            required_permissions=["data_manager"],
            tags=["data", "maps", "admin"],
            rate_limit=5
        )
        
        # Analytics endpoints
        endpoints["get_route_analytics"] = APIEndpoint(
            path="/v1/analytics/routes",
            method=HTTPMethod.GET,
            version=APIVersion.V1,
            description="Get route usage analytics",
            requires_auth=True,
            required_permissions=["analyst"],
            tags=["analytics", "admin"],
            rate_limit=20
        )
        
        endpoints["get_system_metrics"] = APIEndpoint(
            path="/v1/analytics/system",
            method=HTTPMethod.GET,
            version=APIVersion.V1,
            description="Get system performance metrics",
            requires_auth=True,
            required_permissions=["admin"],
            tags=["analytics", "admin"],
            rate_limit=10
        )
        
        # ====================================================================
        # ADMIN ENDPOINTS
        # ====================================================================
        
        # System health
        endpoints["health_check"] = APIEndpoint(
            path="/v1/health",
            method=HTTPMethod.GET,
            version=APIVersion.V1,
            description="System health check",
            tags=["system"],
            rate_limit=1000
        )
        
        # System status
        endpoints["system_status"] = APIEndpoint(
            path="/v1/status",
            method=HTTPMethod.GET,
            version=APIVersion.V1,
            description="Detailed system status information",
            requires_auth=True,
            required_permissions=["admin"],
            tags=["system", "admin"],
            rate_limit=50
        )
        
        return endpoints
    
    def _define_middleware(self) -> List[str]:
        """Define middleware stack order"""
        return [
            "security_headers",
            "cors",
            "rate_limiting",
            "request_logging",
            "authentication",
            "authorization",
            "request_validation",
            "response_formatting",
            "error_handling"
        ]
    
    def _define_security(self) -> Dict[str, Any]:
        """Define security configuration"""
        return {
            "jwt_secret_key": "your-secret-key",  # Should be from environment
            "jwt_access_token_expires": 3600,     # 1 hour
            "jwt_refresh_token_expires": 2592000, # 30 days
            "password_hash_rounds": 12,
            "rate_limit_storage": "redis://localhost:6379/0",
            "cors_origins": ["*"],  # Configure for production
            "cors_methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "cors_headers": ["Content-Type", "Authorization", "X-Requested-With"],
            "security_headers": {
                "X-Content-Type-Options": "nosniff",
                "X-Frame-Options": "DENY",
                "X-XSS-Protection": "1; mode=block",
                "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
                "Content-Security-Policy": "default-src 'self'"
            }
        }
    
    def _define_documentation(self) -> Dict[str, Any]:
        """Define API documentation configuration"""
        return {
            "title": "Smart Accessible Routing API",
            "description": "Comprehensive API for intelligent accessible route planning and navigation",
            "version": "1.0.0",
            "contact": {
                "name": "API Support",
                "email": "api-support@smartrouting.com",
                "url": "https://smartrouting.com/support"
            },
            "license": {
                "name": "MIT",
                "url": "https://opensource.org/licenses/MIT"
            },
            "servers": [
                {"url": "https://api.smartrouting.com", "description": "Production server"},
                {"url": "https://staging-api.smartrouting.com", "description": "Staging server"},
                {"url": "http://localhost:5000", "description": "Development server"}
            ],
            "tags": [
                {"name": "routing", "description": "Route calculation and optimization"},
                {"name": "users", "description": "User management and authentication"},
                {"name": "accessibility", "description": "Accessibility features and constraints"},
                {"name": "ml", "description": "Machine learning integration"},
                {"name": "data", "description": "Data management operations"},
                {"name": "analytics", "description": "Analytics and reporting"},
                {"name": "admin", "description": "Administrative functions"},
                {"name": "system", "description": "System health and status"}
            ]
        }
    
    def get_endpoint(self, endpoint_name: str) -> Optional[APIEndpoint]:
        """Get endpoint definition by name"""
        return self.endpoints.get(endpoint_name)
    
    def get_endpoints_by_tag(self, tag: str) -> List[APIEndpoint]:
        """Get all endpoints with specified tag"""
        return [endpoint for endpoint in self.endpoints.values() if tag in endpoint.tags]
    
    def get_public_endpoints(self) -> List[APIEndpoint]:
        """Get all public endpoints (no authentication required)"""
        return [endpoint for endpoint in self.endpoints.values() if not endpoint.requires_auth]
    
    def get_authenticated_endpoints(self) -> List[APIEndpoint]:
        """Get all endpoints requiring authentication"""
        return [endpoint for endpoint in self.endpoints.values() if endpoint.requires_auth]
    
    def generate_openapi_spec(self) -> Dict[str, Any]:
        """Generate OpenAPI 3.0 specification"""
        spec = {
            "openapi": "3.0.0",
            "info": self.documentation_config,
            "servers": self.documentation_config["servers"],
            "tags": self.documentation_config["tags"],
            "paths": {},
            "components": {
                "schemas": self._generate_schemas(),
                "securitySchemes": self._generate_security_schemes()
            }
        }
        
        # Add paths
        for endpoint_name, endpoint in self.endpoints.items():
            path = endpoint.path
            method = endpoint.method.value.lower()
            
            if path not in spec["paths"]:
                spec["paths"][path] = {}
            
            spec["paths"][path][method] = {
                "summary": endpoint.description,
                "tags": endpoint.tags,
                "operationId": endpoint_name,
                "responses": self._generate_responses(endpoint),
            }
            
            if endpoint.request_schema:
                spec["paths"][path][method]["requestBody"] = {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": endpoint.request_schema
                        }
                    }
                }
            
            if endpoint.requires_auth:
                spec["paths"][path][method]["security"] = [{"BearerAuth": []}]
        
        return spec
    
    def _generate_schemas(self) -> Dict[str, Any]:
        """Generate common schemas for OpenAPI spec"""
        return {
            "Coordinate": {
                "type": "object",
                "required": ["latitude", "longitude"],
                "properties": {
                    "latitude": {"type": "number", "minimum": -90, "maximum": 90},
                    "longitude": {"type": "number", "minimum": -180, "maximum": 180}
                }
            },
            "AccessibilityProfile": {
                "type": "object",
                "properties": {
                    "mobility_aid": {"type": "string", "enum": ["none", "wheelchair", "walker", "cane", "scooter"]},
                    "walking_speed": {"type": "number", "minimum": 0.5, "maximum": 10.0},
                    "accessibility_needs": {"type": "array", "items": {"type": "string"}},
                    "preferences": {"type": "object"}
                }
            },
            "Route": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "path": {"type": "array", "items": {"$ref": "#/components/schemas/Coordinate"}},
                    "total_distance": {"type": "number"},
                    "total_time": {"type": "number"},
                    "accessibility_score": {"type": "number", "minimum": 0, "maximum": 1},
                    "warnings": {"type": "array", "items": {"type": "string"}}
                }
            },
            "Error": {
                "type": "object",
                "required": ["error", "message"],
                "properties": {
                    "error": {"type": "string"},
                    "message": {"type": "string"},
                    "details": {"type": "object"},
                    "timestamp": {"type": "string", "format": "date-time"},
                    "request_id": {"type": "string"}
                }
            }
        }
    
    def _generate_security_schemes(self) -> Dict[str, Any]:
        """Generate security schemes for OpenAPI spec"""
        return {
            "BearerAuth": {
                "type": "http",
                "scheme": "bearer",
                "bearerFormat": "JWT"
            }
        }
    
    def _generate_responses(self, endpoint: APIEndpoint) -> Dict[str, Any]:
        """Generate response definitions for endpoint"""
        responses = {
            "200": {
                "description": "Success",
                "content": {
                    "application/json": {
                        "schema": endpoint.response_schema or {"type": "object"}
                    }
                }
            }
        }
        
        # Add error responses
        error_responses = {
            400: {"description": "Bad Request", "schema": {"$ref": "#/components/schemas/Error"}},
            401: {"description": "Unauthorized", "schema": {"$ref": "#/components/schemas/Error"}},
            403: {"description": "Forbidden", "schema": {"$ref": "#/components/schemas/Error"}},
            404: {"description": "Not Found", "schema": {"$ref": "#/components/schemas/Error"}},
            429: {"description": "Rate Limit Exceeded", "schema": {"$ref": "#/components/schemas/Error"}},
            500: {"description": "Internal Server Error", "schema": {"$ref": "#/components/schemas/Error"}}
        }
        
        for error_code in endpoint.error_codes:
            if error_code in error_responses:
                responses[str(error_code)] = {
                    "description": error_responses[error_code]["description"],
                    "content": {
                        "application/json": {
                            "schema": error_responses[error_code]["schema"]
                        }
                    }
                }
        
        return responses

# Global API architecture instance
api_architecture = APIArchitecture()

# Example usage and testing
if __name__ == "__main__":
    print("Smart Accessible Routing API Architecture")
    print("=" * 50)
    
    # Print endpoint summary
    print(f"\nTotal endpoints defined: {len(api_architecture.endpoints)}")
    
    # Group by tags
    tag_groups = {}
    for endpoint in api_architecture.endpoints.values():
        for tag in endpoint.tags:
            if tag not in tag_groups:
                tag_groups[tag] = []
            tag_groups[tag].append(endpoint)
    
    print("\nEndpoints by category:")
    for tag, endpoints in tag_groups.items():
        print(f"  {tag}: {len(endpoints)} endpoints")
    
    # Show authentication requirements
    public_endpoints = api_architecture.get_public_endpoints()
    auth_endpoints = api_architecture.get_authenticated_endpoints()
    
    print(f"\nAuthentication requirements:")
    print(f"  Public endpoints: {len(public_endpoints)}")
    print(f"  Authenticated endpoints: {len(auth_endpoints)}")
    
    # Show example OpenAPI spec structure
    print(f"\nGenerating OpenAPI specification...")
    openapi_spec = api_architecture.generate_openapi_spec()
    print(f"  OpenAPI version: {openapi_spec['openapi']}")
    print(f"  API title: {openapi_spec['info']['title']}")
    print(f"  Paths defined: {len(openapi_spec['paths'])}")
    print(f"  Schemas defined: {len(openapi_spec['components']['schemas'])}")
    
    print("\nAPI architecture design completed successfully!")
