"""
Graph Utilities

Utility functions for graph operations and analysis.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class GraphUtils:
    """Utility functions for graph operations"""
    
    @staticmethod
    def calculate_graph_metrics(graph_data):
        """Calculate basic graph metrics"""
        num_nodes = graph_data.x.size(0)
        num_edges = graph_data.edge_index.size(1)
        
        return {
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'density': num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0
        }
    
    @staticmethod
    def find_connected_components(edge_index, num_nodes):
        """Find connected components in the graph"""
        # Placeholder implementation
        return [list(range(num_nodes))]
    
    @staticmethod
    def export_graph_for_visualization(graph_data):
        """Export graph data for visualization"""
        return {
            'nodes': [{'id': i} for i in range(graph_data.x.size(0))],
            'edges': [
                {'source': int(graph_data.edge_index[0, i]), 
                 'target': int(graph_data.edge_index[1, i])}
                for i in range(graph_data.edge_index.size(1))
            ]
        }
