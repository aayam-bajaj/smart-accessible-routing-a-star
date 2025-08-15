"""
API Request and Response Schemas
===============================

Comprehensive JSON schemas for validating API requests and responses
across all endpoints including routing, user management, ML integration,
and data management APIs.
"""

from typing import Dict, Any


# Routing API Schemas
ROUTING_SCHEMAS = {
    "route_request": {
        "type": "object",
        "required": ["start", "end"],
        "properties": {
            "start": {
                "type": "object",
                "required": ["lat", "lon"],
                "properties": {
                    "lat": {"type": "number", "minimum": -90, "maximum": 90},
                    "lon": {"type": "number", "minimum": -180, "maximum": 180},
                    "name": {"type": "string", "description": "Optional location name"}
                }
            },
            "end": {
                "type": "object",
                "required": ["lat", "lon"],
                "properties": {
                    "lat": {"type": "number", "minimum": -90, "maximum": 90},
                    "lon": {"type": "number", "minimum": -180, "maximum": 180},
                    "name": {"type": "string", "description": "Optional location name"}
                }
            },
            "waypoints": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["lat", "lon"],
                    "properties": {
                        "lat": {"type": "number", "minimum": -90, "maximum": 90},
                        "lon": {"type": "number", "minimum": -180, "maximum": 180},
                        "name": {"type": "string"}
                    }
                },
                "description": "Optional intermediate waypoints"
            },
            "user_profile": {
                "type": "object",
                "description": "User accessibility profile and preferences",
                "properties": {
                    "mobility_aid": {
                        "type": "string",
                        "enum": ["none", "wheelchair", "mobility_scooter", "walker", "cane", "crutches", "prosthetic"]
                    },
                    "constraints": {"$ref": "#/definitions/accessibility_constraints"},
                    "preferences": {"$ref": "#/definitions/user_preferences"}
                }
            },
            "routing_options": {
                "type": "object",
                "properties": {
                    "algorithm": {
                        "type": "string",
                        "enum": ["astar", "bidirectional_astar", "anytime_astar", "multi_destination"],
                        "default": "astar"
                    },
                    "optimization_criteria": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["distance", "time", "accessibility", "safety", "comfort"]
                        },
                        "default": ["accessibility", "distance"]
                    },
                    "max_alternatives": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 5,
                        "default": 3
                    },
                    "avoid_areas": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "lat": {"type": "number"},
                                "lon": {"type": "number"},
                                "radius": {"type": "number", "minimum": 0}
                            }
                        }
                    },
                    "time_constraints": {
                        "type": "object",
                        "properties": {
                            "departure_time": {"type": "string", "format": "date-time"},
                            "arrival_time": {"type": "string", "format": "date-time"},
                            "max_duration": {"type": "integer", "minimum": 0}
                        }
                    }
                }
            }
        }
    },
    
    "multi_criteria_route_request": {
        "allOf": [
            {"$ref": "#/schemas/route_request"},
            {
                "type": "object",
                "required": ["criteria_weights"],
                "properties": {
                    "criteria_weights": {
                        "type": "object",
                        "properties": {
                            "distance": {"type": "number", "minimum": 0, "maximum": 1},
                            "time": {"type": "number", "minimum": 0, "maximum": 1},
                            "accessibility": {"type": "number", "minimum": 0, "maximum": 1},
                            "safety": {"type": "number", "minimum": 0, "maximum": 1},
                            "comfort": {"type": "number", "minimum": 0, "maximum": 1},
                            "environmental": {"type": "number", "minimum": 0, "maximum": 1}
                        },
                        "description": "Weights for different optimization criteria (sum should be 1.0)"
                    },
                    "normalization_method": {
                        "type": "string",
                        "enum": ["linear", "logarithmic", "sigmoid"],
                        "default": "logarithmic"
                    }
                }
            }
        ]
    },
    
    "route_comparison_request": {
        "type": "object",
        "required": ["routes"],
        "properties": {
            "routes": {
                "type": "array",
                "minItems": 2,
                "items": {
                    "type": "object",
                    "properties": {
                        "route_id": {"type": "string"},
                        "route_data": {"$ref": "#/definitions/route_result"}
                    }
                }
            },
            "comparison_criteria": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["distance", "time", "accessibility_score", "safety_score", "comfort_score"]
                },
                "default": ["distance", "time", "accessibility_score"]
            },
            "user_preferences": {"$ref": "#/definitions/user_preferences"}
        }
    }
}

# Machine Learning API Schemas
ML_SCHEMAS = {
    "feedback_request": {
        "type": "object",
        "required": ["route_id", "feedback_type", "rating"],
        "properties": {
            "route_id": {"type": "string", "description": "Unique route identifier"},
            "feedback_type": {
                "type": "string",
                "enum": ["route_quality", "accessibility", "safety", "comfort", "navigation"],
                "description": "Type of feedback being provided"
            },
            "rating": {
                "type": "integer",
                "minimum": 1,
                "maximum": 5,
                "description": "Rating from 1 (poor) to 5 (excellent)"
            },
            "issues": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["blocked_path", "steep_slope", "poor_surface", "missing_ramp", 
                            "no_curb_cuts", "unsafe_area", "construction", "other"]
                },
                "description": "Specific issues encountered"
            },
            "comments": {
                "type": "string",
                "maxLength": 1000,
                "description": "Additional feedback comments"
            },
            "location": {
                "type": "object",
                "properties": {
                    "lat": {"type": "number", "minimum": -90, "maximum": 90},
                    "lon": {"type": "number", "minimum": -180, "maximum": 180}
                },
                "description": "Location where issue occurred"
            },
            "context": {
                "type": "object",
                "properties": {
                    "weather": {"type": "string"},
                    "time_of_day": {"type": "string"},
                    "mobility_aid_used": {"type": "string"}
                },
                "description": "Context information for the feedback"
            }
        }
    },
    
    "personalization_request": {
        "type": "object",
        "required": ["user_id"],
        "properties": {
            "user_id": {"type": "string", "description": "Unique user identifier"},
            "route_history": {
                "type": "array",
                "items": {"$ref": "#/definitions/route_result"},
                "description": "Recent route history for personalization"
            },
            "preferences_update": {
                "type": "object",
                "properties": {
                    "learned_preferences": {
                        "type": "object",
                        "additionalProperties": {"type": "number"},
                        "description": "ML-learned user preferences"
                    },
                    "adaptation_rate": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.1,
                        "description": "Rate of preference adaptation"
                    }
                }
            },
            "context": {
                "type": "object",
                "properties": {
                    "current_conditions": {
                        "type": "object",
                        "properties": {
                            "weather": {"type": "string"},
                            "time_of_day": {"type": "string"},
                            "day_of_week": {"type": "string"}
                        }
                    },
                    "recent_feedback": {
                        "type": "array",
                        "items": {"$ref": "#/schemas/feedback_request"}
                    }
                }
            }
        }
    }
}

# User Management API Schemas
USER_MANAGEMENT_SCHEMAS = {
    "register_user": {
        "type": "object",
        "required": ["email", "username", "password"],
        "properties": {
            "email": {
                "type": "string",
                "format": "email",
                "description": "User email address"
            },
            "username": {
                "type": "string",
                "minLength": 3,
                "maxLength": 20,
                "pattern": "^[a-zA-Z0-9_]+$",
                "description": "Unique username"
            },
            "password": {
                "type": "string",
                "minLength": 8,
                "description": "Password with uppercase, lowercase, and numeric characters"
            },
            "profile": {
                "type": "object",
                "description": "Optional initial profile data",
                "properties": {
                    "mobility_aid": {"type": "string"},
                    "constraints": {"$ref": "#/definitions/accessibility_constraints"},
                    "preferences": {"$ref": "#/definitions/user_preferences"}
                }
            }
        }
    },
    
    "login_user": {
        "type": "object",
        "required": ["login", "password"],
        "properties": {
            "login": {
                "type": "string",
                "description": "Email address or username"
            },
            "password": {
                "type": "string",
                "description": "User password"
            },
            "remember_me": {
                "type": "boolean",
                "default": False,
                "description": "Extend session duration"
            }
        }
    },
    
    "update_profile": {
        "type": "object",
        "properties": {
            "mobility_aid": {
                "type": "string",
                "enum": ["none", "wheelchair", "mobility_scooter", "walker", "cane", "crutches", "prosthetic"],
                "description": "Primary mobility aid used"
            },
            "constraints": {
                "type": "object",
                "properties": {
                    "max_walking_distance": {
                        "type": "number",
                        "minimum": 0,
                        "description": "Maximum walking distance in meters"
                    },
                    "max_slope_percent": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 100,
                        "description": "Maximum acceptable slope percentage"
                    },
                    "requires_elevator": {
                        "type": "boolean",
                        "description": "Must use elevators instead of stairs"
                    },
                    "requires_ramp": {
                        "type": "boolean",
                        "description": "Requires ramp access"
                    },
                    "avoids_stairs": {
                        "type": "boolean",
                        "description": "Avoids stairs when possible"
                    },
                    "needs_rest_areas": {
                        "type": "boolean",
                        "description": "Needs rest areas along route"
                    }
                }
            },
            "preferences": {
                "type": "object",
                "properties": {
                    "walking_speed_factor": {
                        "type": "number",
                        "minimum": 0.1,
                        "maximum": 3.0,
                        "description": "Walking speed relative to average"
                    },
                    "surface_preferences": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["paved", "concrete", "gravel", "grass", "dirt"]
                        },
                        "description": "Preferred surface types"
                    },
                    "weather_considerations": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["covered", "shaded", "indoor_preferred", "weather_protected"]
                        },
                        "description": "Weather-related preferences"
                    },
                    "time_preferences": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["avoid_rush_hour", "daylight_only", "well_lit", "quiet_routes"]
                        },
                        "description": "Time and environmental preferences"
                    },
                    "route_complexity_preference": {
                        "type": "string",
                        "enum": ["simple", "balanced", "complex"],
                        "description": "Preferred route complexity"
                    }
                }
            },
            "learning_enabled": {
                "type": "boolean",
                "description": "Enable machine learning personalization"
            },
            "data_sharing_consent": {
                "type": "boolean",
                "description": "Consent to share data for research"
            }
        }
    },
    
    "change_password": {
        "type": "object",
        "required": ["current_password", "new_password"],
        "properties": {
            "current_password": {
                "type": "string",
                "description": "Current user password"
            },
            "new_password": {
                "type": "string",
                "minLength": 8,
                "description": "New password with uppercase, lowercase, and numeric characters"
            },
            "confirm_password": {
                "type": "string",
                "description": "Confirmation of new password"
            }
        }
    }
}

# Common definitions used across schemas
COMMON_DEFINITIONS = {
    "accessibility_constraints": {
        "type": "object",
        "properties": {
            "max_walking_distance": {
                "type": "number",
                "minimum": 0,
                "description": "Maximum walking distance in meters"
            },
            "max_slope_percent": {
                "type": "number",
                "minimum": 0,
                "maximum": 100,
                "description": "Maximum acceptable slope percentage"
            },
            "requires_elevator": {
                "type": "boolean",
                "description": "Must use elevators instead of stairs"
            },
            "requires_ramp": {
                "type": "boolean",
                "description": "Requires ramp access"
            },
            "avoids_stairs": {
                "type": "boolean",
                "description": "Avoids stairs when possible"
            },
            "needs_rest_areas": {
                "type": "boolean",
                "description": "Needs rest areas along route"
            }
        }
    },
    
    "user_preferences": {
        "type": "object",
        "properties": {
            "walking_speed_factor": {
                "type": "number",
                "minimum": 0.1,
                "maximum": 3.0,
                "default": 1.0,
                "description": "Walking speed relative to average"
            },
            "surface_preferences": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["paved", "concrete", "gravel", "grass", "dirt"]
                },
                "description": "Preferred surface types"
            },
            "weather_considerations": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["covered", "shaded", "indoor_preferred", "weather_protected"]
                },
                "description": "Weather-related preferences"
            },
            "time_preferences": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["avoid_rush_hour", "daylight_only", "well_lit", "quiet_routes"]
                },
                "description": "Time and environmental preferences"
            },
            "route_complexity_preference": {
                "type": "string",
                "enum": ["simple", "balanced", "complex"],
                "default": "balanced",
                "description": "Preferred route complexity"
            }
        }
    },
    
    "route_result": {
        "type": "object",
        "properties": {
            "route_id": {"type": "string"},
            "segments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "start": {"$ref": "#/definitions/location"},
                        "end": {"$ref": "#/definitions/location"},
                        "distance": {"type": "number"},
                        "duration": {"type": "number"},
                        "accessibility_score": {"type": "number"},
                        "instructions": {"type": "string"}
                    }
                }
            },
            "total_distance": {"type": "number"},
            "total_duration": {"type": "number"},
            "accessibility_rating": {"type": "number"},
            "warnings": {
                "type": "array",
                "items": {"type": "string"}
            }
        }
    },
    
    "location": {
        "type": "object",
        "required": ["lat", "lon"],
        "properties": {
            "lat": {"type": "number", "minimum": -90, "maximum": 90},
            "lon": {"type": "number", "minimum": -180, "maximum": 180},
            "name": {"type": "string"},
            "address": {"type": "string"}
        }
    }
}

# Export all schemas
API_SCHEMAS = {
    **ROUTING_SCHEMAS,
    **ML_SCHEMAS,
    **USER_MANAGEMENT_SCHEMAS
}

# Export definitions
SCHEMA_DEFINITIONS = COMMON_DEFINITIONS
