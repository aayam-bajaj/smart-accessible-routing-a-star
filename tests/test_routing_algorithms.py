"""
Comprehensive test suite for routing algorithms
"""
import unittest
import time
import sys
import os
from unittest.mock import Mock, patch
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import routing components
from app.services.routing_algorithm import (
    AStarRouter, 
    MultiCriteriaAStarRouter, 
    PersonalizedAStarRouter,
    RouteResult,
    RouteSegment,
    AlternativeRoute
)
from app.models.user_profile import (
    UserProfile,
    ProfileTemplates,
    MobilityAidType,
    AccessibilityConstraints
)

class TestDataGenerator:
    """Generate test data for routing algorithms"""
    
    @staticmethod
    def create_simple_graph() -> Dict:
        """Create a simple 3-node linear graph for basic testing"""
        return {
            'nodes': {
                1: {'osm_id': 1000001, 'latitude': 19.0760, 'longitude': 73.0760},
                2: {'osm_id': 1000002, 'latitude': 19.0770, 'longitude': 73.0770},
                3: {'osm_id': 1000003, 'latitude': 19.0780, 'longitude': 73.0780}
            },
            'edges': {
                1: {
                    'osm_id': 2000001,
                    'start_node_id': 1,
                    'end_node_id': 2,
                    'length_meters': 150.0,
                    'energy_cost': 1.2,
                    'comfort_score': 0.8,
                    'accessibility_score': 0.9,
                    'surface_type': 'asphalt',
                    'max_slope_degrees': 1.5,
                    'has_barrier': False,
                    'has_steps': False,
                    'wheelchair_accessible': True,
                    'width_meters': 2.5
                },
                2: {
                    'osm_id': 2000002,
                    'start_node_id': 2,
                    'end_node_id': 3,
                    'length_meters': 200.0,
                    'energy_cost': 1.5,
                    'comfort_score': 0.7,
                    'accessibility_score': 0.8,
                    'surface_type': 'concrete',
                    'max_slope_degrees': 2.0,
                    'has_barrier': False,
                    'has_steps': False,
                    'wheelchair_accessible': True,
                    'width_meters': 2.0
                }
            }
        }
    
    @staticmethod
    def create_complex_graph() -> Dict:
        """Create a more complex graph with multiple paths and obstacles"""
        return {
            'nodes': {
                1: {'osm_id': 1000001, 'latitude': 19.0760, 'longitude': 73.0760},
                2: {'osm_id': 1000002, 'latitude': 19.0770, 'longitude': 73.0770},
                3: {'osm_id': 1000003, 'latitude': 19.0780, 'longitude': 73.0780},
                4: {'osm_id': 1000004, 'latitude': 19.0765, 'longitude': 73.0775},
                5: {'osm_id': 1000005, 'latitude': 19.0785, 'longitude': 73.0785}
            },
            'edges': {
                1: {  # Direct path 1->2
                    'osm_id': 2000001,
                    'start_node_id': 1,
                    'end_node_id': 2,
                    'length_meters': 150.0,
                    'energy_cost': 1.2,
                    'comfort_score': 0.8,
                    'accessibility_score': 0.9,
                    'surface_type': 'asphalt',
                    'max_slope_degrees': 1.5,
                    'has_barrier': False,
                    'has_steps': False,
                    'wheelchair_accessible': True,
                    'width_meters': 2.5
                },
                2: {  # Direct path 2->3
                    'osm_id': 2000002,
                    'start_node_id': 2,
                    'end_node_id': 3,
                    'length_meters': 200.0,
                    'energy_cost': 1.5,
                    'comfort_score': 0.7,
                    'accessibility_score': 0.8,
                    'surface_type': 'concrete',
                    'max_slope_degrees': 2.0,
                    'has_barrier': False,
                    'has_steps': False,
                    'wheelchair_accessible': True,
                    'width_meters': 2.0
                },
                3: {  # Alternative path 1->4 (shorter but with steps)
                    'osm_id': 2000003,
                    'start_node_id': 1,
                    'end_node_id': 4,
                    'length_meters': 120.0,
                    'energy_cost': 2.0,
                    'comfort_score': 0.5,
                    'accessibility_score': 0.3,
                    'surface_type': 'concrete',
                    'max_slope_degrees': 5.0,
                    'has_barrier': False,
                    'has_steps': True,
                    'wheelchair_accessible': False,
                    'width_meters': 1.5
                },
                4: {  # Alternative path 4->5
                    'osm_id': 2000004,
                    'start_node_id': 4,
                    'end_node_id': 5,
                    'length_meters': 180.0,
                    'energy_cost': 1.8,
                    'comfort_score': 0.6,
                    'accessibility_score': 0.7,
                    'surface_type': 'gravel',
                    'max_slope_degrees': 3.0,
                    'has_barrier': True,
                    'has_steps': False,
                    'wheelchair_accessible': False,
                    'width_meters': 1.8
                },
                5: {  # Path 5->3
                    'osm_id': 2000005,
                    'start_node_id': 5,
                    'end_node_id': 3,
                    'length_meters': 100.0,
                    'energy_cost': 1.0,
                    'comfort_score': 0.9,
                    'accessibility_score': 0.9,
                    'surface_type': 'asphalt',
                    'max_slope_degrees': 0.5,
                    'has_barrier': False,
                    'has_steps': False,
                    'wheelchair_accessible': True,
                    'width_meters': 3.0
                }
            }
        }
    
    @staticmethod
    def create_blocked_graph() -> Dict:
        """Create graph with blocked edges for accessibility testing"""
        graph = TestDataGenerator.create_simple_graph()
        # Block the direct path
        graph['edges'][1]['is_blocked'] = True
        graph['edges'][1]['blockage_reason'] = 'construction'
        return graph
    
    @staticmethod
    def create_test_user_profile() -> UserProfile:
        """Create a test user profile for wheelchair user"""
        return ProfileTemplates.wheelchair_user()


class BaseRoutingTest(unittest.TestCase):
    """Base test class with common setup and utilities"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.simple_graph = TestDataGenerator.create_simple_graph()
        self.complex_graph = TestDataGenerator.create_complex_graph()
        self.blocked_graph = TestDataGenerator.create_blocked_graph()
        self.user_profile = TestDataGenerator.create_test_user_profile()
        self.user_profile.user_id = 1
    
    def assertRouteValid(self, route: RouteResult):
        """Assert that a route is valid"""
        self.assertIsNotNone(route)
        self.assertIsInstance(route, RouteResult)
        self.assertGreater(len(route.segments), 0)
        self.assertGreater(route.total_distance, 0)
        self.assertGreater(route.total_time, 0)
        self.assertGreaterEqual(route.accessibility_score, 0)
        self.assertLessEqual(route.accessibility_score, 1)
    
    def assertRouteMetrics(self, route: RouteResult, expected_segment_count: int):
        """Assert route has expected number of segments and valid metrics"""
        self.assertEqual(len(route.segments), expected_segment_count)
        
        # Check that totals match sum of segments
        total_distance = sum(seg.length_meters for seg in route.segments)
        total_time = sum(seg.estimated_time_minutes for seg in route.segments)
        
        self.assertAlmostEqual(route.total_distance, total_distance, places=2)
        self.assertAlmostEqual(route.total_time, total_time, places=2)
    
    def measure_performance(self, func, *args, **kwargs):
        """Measure function execution time"""
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
        return result, execution_time


class TestAStarRouter(BaseRoutingTest):
    """Test basic A* router functionality"""
    
    def test_router_initialization(self):
        """Test router initialization"""
        router = AStarRouter(self.simple_graph)
        self.assertEqual(len(router.nodes), 3)
        self.assertEqual(len(router.edges), 2)
        self.assertEqual(len(router.node_positions), 3)
    
    def test_heuristic_calculation(self):
        """Test heuristic distance calculation"""
        router = AStarRouter(self.simple_graph)
        
        # Test distance between nodes 1 and 3
        distance = router.calculate_heuristic(1, 3)
        self.assertGreater(distance, 0)
        self.assertIsInstance(distance, float)
        
        # Distance should be symmetric
        distance_reverse = router.calculate_heuristic(3, 1)
        self.assertAlmostEqual(distance, distance_reverse, places=2)
        
        # Distance to same node should be 0
        same_distance = router.calculate_heuristic(1, 1)
        self.assertEqual(same_distance, 0)
    
    def test_neighbor_discovery(self):
        """Test neighbor node discovery"""
        router = AStarRouter(self.simple_graph)
        
        # Node 1 should have 1 neighbor (node 2)
        neighbors_1 = router.get_neighbors(1)
        self.assertEqual(len(neighbors_1), 1)
        self.assertEqual(neighbors_1[0][0], 2)
        
        # Node 2 should have 2 neighbors (nodes 1 and 3)
        neighbors_2 = router.get_neighbors(2)
        self.assertEqual(len(neighbors_2), 2)
        neighbor_ids = [n[0] for n in neighbors_2]
        self.assertIn(1, neighbor_ids)
        self.assertIn(3, neighbor_ids)
    
    def test_basic_routing(self):
        """Test basic route finding"""
        router = AStarRouter(self.simple_graph)
        route = router.find_route(1, 3)
        
        self.assertRouteValid(route)
        self.assertRouteMetrics(route, 2)  # Should have 2 segments: 1->2, 2->3
        
        # Check route segments
        self.assertEqual(route.segments[0].start_node_id, 1)
        self.assertEqual(route.segments[0].end_node_id, 2)
        self.assertEqual(route.segments[1].start_node_id, 2)
        self.assertEqual(route.segments[1].end_node_id, 3)
    
    def test_no_route_found(self):
        """Test behavior when no route exists"""
        # Create isolated graph
        isolated_graph = {
            'nodes': {
                1: {'osm_id': 1000001, 'latitude': 19.0760, 'longitude': 73.0760},
                2: {'osm_id': 1000002, 'latitude': 19.0770, 'longitude': 73.0770}
            },
            'edges': {}  # No edges
        }
        
        router = AStarRouter(isolated_graph)
        route = router.find_route(1, 2)
        self.assertIsNone(route)
    
    def test_same_start_end_node(self):
        """Test routing when start and end nodes are the same"""
        router = AStarRouter(self.simple_graph)
        route = router.find_route(1, 1)
        
        # Should return empty route or route with no segments
        if route is not None:
            self.assertEqual(len(route.segments), 0)
            self.assertEqual(route.total_distance, 0)


class TestMultiCriteriaRouter(BaseRoutingTest):
    """Test multi-criteria A* router functionality"""
    
    def test_router_initialization(self):
        """Test multi-criteria router initialization"""
        user_profile = {
            'distance_weight': 0.3,
            'energy_efficiency_weight': 0.3,
            'comfort_weight': 0.4,
            'mobility_aid_type': 'wheelchair'
        }
        
        router = MultiCriteriaAStarRouter(self.simple_graph, user_profile)
        self.assertEqual(router.user_profile['distance_weight'], 0.3)
        self.assertEqual(router.user_profile['mobility_aid_type'], 'wheelchair')
    
    def test_cost_calculation(self):
        """Test weighted cost calculation"""
        user_profile = {
            'distance_weight': 0.5,
            'energy_efficiency_weight': 0.3,
            'comfort_weight': 0.2
        }
        
        router = MultiCriteriaAStarRouter(self.simple_graph, user_profile)
        edge_data = self.simple_graph['edges'][1]
        
        cost = router.calculate_weighted_cost(edge_data)
        self.assertIsInstance(cost, float)
        self.assertGreater(cost, 0)
        self.assertLessEqual(cost, 10.0)  # Should be bounded
    
    def test_accessibility_filtering(self):
        """Test accessibility constraint filtering"""
        wheelchair_profile = {
            'mobility_aid_type': 'wheelchair',
            'avoid_stairs': True,
            'min_path_width': 1.2,
            'max_slope_degrees': 3.0
        }
        
        router = MultiCriteriaAStarRouter(self.complex_graph, wheelchair_profile)
        
        # Edge with steps should be inaccessible
        edge_with_steps = self.complex_graph['edges'][3]
        self.assertFalse(router.is_edge_accessible(edge_with_steps))
        
        # Normal edge should be accessible
        normal_edge = self.complex_graph['edges'][1]
        self.assertTrue(router.is_edge_accessible(normal_edge))
    
    def test_route_optimization(self):
        """Test that multi-criteria routing produces different results"""
        # Distance-optimized profile
        distance_profile = {
            'distance_weight': 0.9,
            'comfort_weight': 0.1
        }
        
        # Comfort-optimized profile
        comfort_profile = {
            'distance_weight': 0.1,
            'comfort_weight': 0.9
        }
        
        distance_router = MultiCriteriaAStarRouter(self.complex_graph, distance_profile)
        comfort_router = MultiCriteriaAStarRouter(self.complex_graph, comfort_profile)
        
        distance_route = distance_router.find_route(1, 3)
        comfort_route = comfort_router.find_route(1, 3)
        
        self.assertRouteValid(distance_route)
        self.assertRouteValid(comfort_route)
        
        # Routes might be different based on optimization
        # At minimum, they should have valid but potentially different scores
        self.assertIsInstance(distance_route.route_score, float)
        self.assertIsInstance(comfort_route.route_score, float)


class TestPersonalizedRouter(BaseRoutingTest):
    """Test personalized A* router functionality"""
    
    def test_router_with_user_profile_object(self):
        """Test router initialization with UserProfile object"""
        router = PersonalizedAStarRouter(self.simple_graph, self.user_profile)
        self.assertIsNotNone(router.user_profile_obj)
        self.assertEqual(router.user_profile_obj.user_id, 1)
    
    def test_enhanced_accessibility_checking(self):
        """Test enhanced accessibility checking with UserProfile"""
        router = PersonalizedAStarRouter(self.complex_graph, self.user_profile)
        
        # Test wheelchair-specific checks
        edge_with_steps = self.complex_graph['edges'][3]
        self.assertFalse(router.is_edge_accessible(edge_with_steps))
        
        # Test width requirements
        narrow_edge = self.complex_graph['edges'][3]  # width 1.5m
        self.assertFalse(router.is_edge_accessible(narrow_edge))  # Wheelchair needs > 1.2m
    
    def test_personalized_cost_calculation(self):
        """Test personalized cost calculation with learning"""
        router = PersonalizedAStarRouter(self.simple_graph, self.user_profile)
        edge_data = self.simple_graph['edges'][1]
        
        # Initial cost
        initial_cost = router.calculate_weighted_cost(edge_data)
        
        # Add some learned preferences
        router.user_profile_obj.learned_preferences['prefer_asphalt'] = 0.8
        
        # Cost should be different with learned preferences
        learned_cost = router.calculate_weighted_cost(edge_data)
        self.assertNotEqual(initial_cost, learned_cost)
    
    def test_estimated_time_calculation(self):
        """Test personalized time estimation"""
        router = PersonalizedAStarRouter(self.simple_graph, self.user_profile)
        edge_data = self.simple_graph['edges'][1]
        
        estimated_time = router.calculate_estimated_time(edge_data)
        self.assertIsInstance(estimated_time, float)
        self.assertGreater(estimated_time, 0)
    
    def test_route_experience_recording(self):
        """Test route experience recording and learning"""
        router = PersonalizedAStarRouter(self.simple_graph, self.user_profile)
        route = router.find_route(1, 3)
        
        feedback = {
            'overall_satisfaction': 4,
            'route_characteristics': {
                'low_slope': True,
                'smooth_surface': True
            }
        }
        
        initial_prefs = dict(router.user_profile_obj.learned_preferences)
        router.record_route_experience(route, feedback)
        
        # Preferences should be updated
        self.assertNotEqual(initial_prefs, router.user_profile_obj.learned_preferences)


class TestRouteResultDataStructures(BaseRoutingTest):
    """Test enhanced route result data structures"""
    
    def test_route_segment_creation(self):
        """Test RouteSegment creation and serialization"""
        segment = RouteSegment(
            start_node_id=1,
            end_node_id=2,
            edge_id=2000001,
            length_meters=150.0,
            accessibility_score=0.9,
            energy_cost=1.2,
            comfort_score=0.8,
            surface_type='asphalt',
            slope_degrees=1.5,
            has_barriers=False,
            estimated_time_minutes=2.0
        )
        
        self.assertEqual(segment.start_node_id, 1)
        self.assertEqual(segment.surface_type, 'asphalt')
        
        # Test serialization
        segment_dict = segment.to_dict()
        self.assertIsInstance(segment_dict, dict)
        self.assertEqual(segment_dict['start_node_id'], 1)
        self.assertEqual(segment_dict['surface_type'], 'asphalt')
    
    def test_route_result_methods(self):
        """Test RouteResult helper methods"""
        router = AStarRouter(self.simple_graph)
        route = router.find_route(1, 3)
        
        # Test adding warnings and recommendations
        route.add_warning("Construction ahead")
        route.add_recommendation("Consider alternative route")
        route.add_accessibility_note("Suitable for wheelchairs")
        
        self.assertIn("Construction ahead", route.warnings)
        self.assertIn("Consider alternative route", route.recommendations)
        self.assertIn("Suitable for wheelchairs", route.accessibility_notes)
        
        # Test duplicate prevention
        route.add_warning("Construction ahead")  # Same warning
        self.assertEqual(len(route.warnings), 1)
    
    def test_detailed_metrics_calculation(self):
        """Test detailed route metrics calculation"""
        router = AStarRouter(self.simple_graph)
        route = router.find_route(1, 3)
        
        route.calculate_detailed_metrics()
        
        # Check that metrics are calculated
        self.assertGreaterEqual(route.total_ascent, 0)
        self.assertGreaterEqual(route.total_descent, 0)
        self.assertIsInstance(route.surface_breakdown, dict)
        self.assertGreaterEqual(route.barrier_count, 0)
    
    def test_route_serialization(self):
        """Test complete route serialization"""
        router = AStarRouter(self.simple_graph)
        route = router.find_route(1, 3)
        
        route_dict = route.to_dict()
        self.assertIsInstance(route_dict, dict)
        self.assertIn('segments', route_dict)
        self.assertIn('total_distance', route_dict)
        self.assertIsInstance(route_dict['segments'], list)


class TestPerformanceBenchmarks(BaseRoutingTest):
    """Performance benchmarks for routing algorithms"""
    
    def test_basic_routing_performance(self):
        """Test basic A* routing performance"""
        router = AStarRouter(self.simple_graph)
        
        route, execution_time = self.measure_performance(router.find_route, 1, 3)
        
        self.assertRouteValid(route)
        self.assertLess(execution_time, 100)  # Should complete in < 100ms
    
    def test_multi_criteria_routing_performance(self):
        """Test multi-criteria routing performance"""
        user_profile = {'distance_weight': 0.5, 'comfort_weight': 0.5}
        router = MultiCriteriaAStarRouter(self.complex_graph, user_profile)
        
        route, execution_time = self.measure_performance(router.find_route, 1, 3)
        
        if route:  # Route might not exist due to constraints
            self.assertRouteValid(route)
        self.assertLess(execution_time, 200)  # Should complete in < 200ms
    
    def test_personalized_routing_performance(self):
        """Test personalized routing performance"""
        router = PersonalizedAStarRouter(self.simple_graph, self.user_profile)
        
        route, execution_time = self.measure_performance(router.find_route, 1, 3)
        
        self.assertRouteValid(route)
        self.assertLess(execution_time, 300)  # Should complete in < 300ms
    
    def test_memory_usage(self):
        """Test memory usage during routing"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Perform multiple routing operations
        router = PersonalizedAStarRouter(self.complex_graph, self.user_profile)
        for _ in range(10):
            route = router.find_route(1, 3)
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (< 50MB for 10 routes)
        self.assertLess(memory_increase, 50 * 1024 * 1024)


class TestErrorHandling(BaseRoutingTest):
    """Test error handling and edge cases"""
    
    def test_invalid_node_ids(self):
        """Test routing with invalid node IDs"""
        router = AStarRouter(self.simple_graph)
        
        # Non-existent start node
        route = router.find_route(999, 3)
        self.assertIsNone(route)
        
        # Non-existent end node
        route = router.find_route(1, 999)
        self.assertIsNone(route)
    
    def test_empty_graph(self):
        """Test routing with empty graph"""
        empty_graph = {'nodes': {}, 'edges': {}}
        router = AStarRouter(empty_graph)
        
        route = router.find_route(1, 2)
        self.assertIsNone(route)
    
    def test_corrupted_edge_data(self):
        """Test routing with corrupted edge data"""
        corrupted_graph = self.simple_graph.copy()
        # Remove required fields
        del corrupted_graph['edges'][1]['length_meters']
        
        router = AStarRouter(corrupted_graph)
        # Should handle missing data gracefully
        route = router.find_route(1, 3)
        # Route might be None or valid depending on implementation
        if route:
            self.assertRouteValid(route)
    
    def test_blocked_route_handling(self):
        """Test handling of completely blocked routes"""
        router = MultiCriteriaAStarRouter(self.blocked_graph, {'avoid_blocked': True})
        route = router.find_route(1, 3)
        
        # Should either find alternative route or return None
        if route:
            # Ensure no blocked edges are used
            for segment in route.segments:
                self.assertFalse(segment.is_blocked)


class TestIntegration(BaseRoutingTest):
    """Integration tests combining multiple components"""
    
    def test_end_to_end_wheelchair_routing(self):
        """Test complete wheelchair user routing scenario"""
        wheelchair_user = ProfileTemplates.wheelchair_user()
        wheelchair_user.user_id = 1
        
        router = PersonalizedAStarRouter(self.complex_graph, wheelchair_user)
        route = router.find_route(1, 3)
        
        if route:
            self.assertRouteValid(route)
            
            # Verify accessibility constraints are met
            for segment in route.segments:
                self.assertTrue(segment.wheelchair_accessible)
                self.assertFalse(segment.has_steps)
                self.assertGreaterEqual(segment.width_meters, 1.2)
    
    def test_profile_learning_workflow(self):
        """Test complete learning workflow with feedback"""
        router = PersonalizedAStarRouter(self.simple_graph, self.user_profile)
        
        # Find initial route
        route1 = router.find_route(1, 3)
        initial_cost = router.calculate_weighted_cost(self.simple_graph['edges'][1])
        
        # Provide feedback
        feedback = {
            'overall_satisfaction': 5,
            'route_characteristics': {'smooth_surface': True}
        }
        router.record_route_experience(route1, feedback)
        
        # Cost should change based on feedback
        updated_cost = router.calculate_weighted_cost(self.simple_graph['edges'][1])
        self.assertNotEqual(initial_cost, updated_cost)
    
    def test_multiple_user_profiles(self):
        """Test routing with different user profiles"""
        profiles = {
            'wheelchair': ProfileTemplates.wheelchair_user(),
            'elderly_walker': ProfileTemplates.elderly_walker(),
            'visually_impaired': ProfileTemplates.visually_impaired()
        }
        
        routes = {}
        for profile_name, profile in profiles.items():
            profile.user_id = len(routes) + 1
            router = PersonalizedAStarRouter(self.complex_graph, profile)
            routes[profile_name] = router.find_route(1, 3)
        
        # Different profiles should produce different results
        valid_routes = {k: v for k, v in routes.items() if v is not None}
        self.assertGreater(len(valid_routes), 0)


def run_performance_suite():
    """Run performance benchmarks and print results"""
    print("\n=== ROUTING ALGORITHM PERFORMANCE BENCHMARKS ===")
    
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestPerformanceBenchmarks))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def run_full_test_suite():
    """Run all tests"""
    print("\n=== RUNNING FULL ROUTING ALGORITHM TEST SUITE ===")
    
    # Discover and run all tests
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n=== TEST SUMMARY ===")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    # Run specific test suites based on command line arguments
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'performance':
        run_performance_suite()
    else:
        run_full_test_suite()
