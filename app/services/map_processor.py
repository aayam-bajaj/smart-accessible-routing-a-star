"""
Map data processing service for Smart Accessible Routing System
"""
import logging
import osmnx as ox
import networkx as nx
import geopandas as gpd
from shapely.geometry import Point, LineString
import pandas as pd
from typing import Dict, List, Tuple, Optional
import json

# Get logger
logger = logging.getLogger(__name__)

class MapProcessor:
    """Service for processing map data from OpenStreetMap"""
    
    def __init__(self, region: str = "Kharghar, India"):
        """
        Initialize the map processor
        
        Args:
            region: The region to download map data for
        """
        self.region = region
        self.graph = None
        self.nodes_gdf = None
        self.edges_gdf = None
        
    def download_map_data(self) -> bool:
        """
        Download map data from OpenStreetMap for the specified region
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(f"Downloading map data for region: {self.region}")
            
            # Download the street network as a graph
            self.graph = ox.graph_from_place(
                self.region, 
                network_type='walk',
                simplify=True
            )
            
            # Convert to GeoDataFrames for easier processing
            self.nodes_gdf, self.edges_gdf = ox.graph_to_gdfs(self.graph)
            
            logger.info(f"Downloaded map data with {len(self.nodes_gdf)} nodes and {len(self.edges_gdf)} edges")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading map data: {e}")
            return False
    
    def extract_accessibility_attributes(self) -> bool:
        """
        Extract accessibility attributes from OSM data
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info("Extracting accessibility attributes from OSM data")
            
            # Add accessibility attributes to edges
            for idx, edge in self.edges_gdf.iterrows():
                # Extract surface type
                surface_type = edge.get('surface', 'unknown')
                
                # Extract width
                width = edge.get('width', None)
                if width is not None:
                    try:
                        width = float(width)
                    except ValueError:
                        width = None
                
                # Extract slope information
                slope = edge.get('incline', None)
                if slope is not None:
                    try:
                        slope = float(slope.replace('%', ''))
                    except ValueError:
                        slope = None
                
                # Check for stairs
                has_stairs = edge.get('highway') == 'steps'
                
                # Check for barriers
                has_barrier = edge.get('barrier', False)
                
                # Check wheelchair accessibility
                wheelchair_accessible = edge.get('wheelchair', 'yes') == 'yes'
                
                # Update the edge with accessibility attributes
                self.edges_gdf.at[idx, 'surface_type'] = surface_type
                self.edges_gdf.at[idx, 'width_meters'] = width
                self.edges_gdf.at[idx, 'slope_percent'] = slope
                self.edges_gdf.at[idx, 'has_stairs'] = has_stairs
                self.edges_gdf.at[idx, 'has_barrier'] = has_barrier
                self.edges_gdf.at[idx, 'wheelchair_accessible'] = wheelchair_accessible
                
                # Calculate accessibility score (simplified)
                accessibility_score = self._calculate_accessibility_score(
                    surface_type, width, slope, has_stairs, has_barrier, wheelchair_accessible
                )
                self.edges_gdf.at[idx, 'accessibility_score'] = accessibility_score
            
            logger.info("Accessibility attributes extracted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error extracting accessibility attributes: {e}")
            return False
    
    def _calculate_accessibility_score(self, surface_type: str, width: float, slope: float, 
                                     has_stairs: bool, has_barrier: bool, wheelchair_accessible: bool) -> float:
        """
        Calculate accessibility score for an edge
        
        Args:
            surface_type: Type of surface
            width: Width in meters
            slope: Slope in percent
            has_stairs: Whether the edge has stairs
            has_barrier: Whether the edge has barriers
            wheelchair_accessible: Whether the edge is wheelchair accessible
            
        Returns:
            float: Accessibility score between 0 and 1
        """
        score = 1.0
        
        # Surface type scoring
        surface_scores = {
            'asphalt': 1.0,
            'concrete': 1.0,
            'paved': 0.9,
            'gravel': 0.7,
            'grass': 0.5,
            'unknown': 0.8
        }
        score *= surface_scores.get(surface_type, 0.8)
        
        # Width scoring (minimum 0.8m for wheelchair)
        if width is not None:
            if width < 0.8:
                score *= 0.3
            elif width < 1.0:
                score *= 0.7
            # Width >= 1.0m is good
        
        # Slope scoring (maximum 5% for accessibility)
        if slope is not None:
            if slope > 5:
                score *= max(0.1, 1 - (slope - 5) / 10)
            elif slope > 2:
                score *= 0.8
            # Slope <= 2% is good
        
        # Stairs penalty
        if has_stairs:
            score *= 0.1
        
        # Barrier penalty
        if has_barrier:
            score *= 0.2
        
        # Wheelchair accessibility bonus
        if wheelchair_accessible:
            score = min(1.0, score * 1.2)
        
        return max(0.0, min(1.0, score))
    
    def save_map_data(self, filepath: str) -> bool:
        """
        Save processed map data to file
        
        Args:
            filepath: Path to save the data
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(f"Saving map data to {filepath}")
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save graph in GraphML format
            ox.save_graphml(self.graph, filepath)
            
            logger.info("Map data saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error saving map data: {e}")
            return False
    
    def load_map_data(self, filepath: str) -> bool:
        """
        Load processed map data from file
        
        Args:
            filepath: Path to load the data from
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(f"Loading map data from {filepath}")
            
            # Load graph from GraphML format
            self.graph = ox.load_graphml(filepath)
            
            # Convert to GeoDataFrames
            self.nodes_gdf, self.edges_gdf = ox.graph_to_gdfs(self.graph)
            
            logger.info("Map data loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading map data: {e}")
            return False
    
    def get_map_statistics(self) -> Dict:
        """
        Get statistics about the map data
        
        Returns:
            Dict: Dictionary with map statistics
        """
        if self.graph is None:
            return {}
        
        stats = {
            'total_nodes': len(self.nodes_gdf) if self.nodes_gdf is not None else 0,
            'total_edges': len(self.edges_gdf) if self.edges_gdf is not None else 0,
            'avg_accessibility_score': self.edges_gdf['accessibility_score'].mean() if self.edges_gdf is not None else 0,
            'accessible_edges': len(self.edges_gdf[self.edges_gdf['accessibility_score'] > 0.7]) if self.edges_gdf is not None else 0,
            'region': self.region
        }
        
        return stats

# Example usage
if __name__ == "__main__":
    # Create map processor
    processor = MapProcessor("Kharghar, India")
    
    # Download map data
    if processor.download_map_data():
        print("Map data downloaded successfully")
        
        # Extract accessibility attributes
        if processor.extract_accessibility_attributes():
            print("Accessibility attributes extracted successfully")
            
            # Save map data
            if processor.save_map_data("data/maps/kharghar_walk_network.graphml"):
                print("Map data saved successfully")
            
            # Print statistics
            stats = processor.get_map_statistics()
            print("Map Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")