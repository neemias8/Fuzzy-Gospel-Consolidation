"""
Fuzzy Event Graph Implementation

This module implements the graph structure for Gospel events
with fuzzy relationship edges.
"""

import torch
from torch_geometric.data import Data
import numpy as np
from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class FuzzyEventGraph:
    """Graph representation of Gospel events with fuzzy relationships"""
    
    def __init__(self, events, fuzzy_relations, config):
        """
        Initialize the fuzzy event graph.
        
        Args:
            events: List of Event objects
            fuzzy_relations: Dictionary of fuzzy relations
            config: Configuration dictionary
        """
        self.events = events
        self.fuzzy_relations = fuzzy_relations
        self.config = config
        self.graph_data = None
        
        logger.info("Building fuzzy event graph...")
        self._build_graph()
    
    def _build_graph(self):
        """Build the PyTorch Geometric graph using real fuzzy relations"""
        num_events = len(self.events)
        
        # Create node features from event data
        node_features = self._create_node_features()
        
        # Create edges from fuzzy relations
        edge_indices = []
        edge_features = []
        
        # Thresholds for creating edges
        same_threshold = self.config.get('graph', {}).get('same_event_threshold', 0.3)
        conflict_threshold = self.config.get('graph', {}).get('conflict_threshold', 0.2)
        temporal_threshold = self.config.get('graph', {}).get('temporal_threshold', 0.1)
        
        # Build edges from fuzzy relations
        for (event1_id, event2_id), relation in self.fuzzy_relations.items():
            # Find event indices
            idx1 = self._get_event_index(event1_id)
            idx2 = self._get_event_index(event2_id)
            
            if idx1 is None or idx2 is None:
                continue
            
            # Create edge if any relation is above threshold
            create_edge = (
                relation.mu_same >= same_threshold or
                relation.mu_conflict >= conflict_threshold or
                relation.mu_before >= temporal_threshold
            )
            
            if create_edge:
                # Add bidirectional edges
                edge_indices.extend([[idx1, idx2], [idx2, idx1]])
                
                # Edge features: [mu_same, mu_conflict, mu_before, mu_after]
                mu_after = self._get_reverse_temporal(event2_id, event1_id)
                edge_feature = [relation.mu_same, relation.mu_conflict, relation.mu_before, mu_after]
                edge_features.extend([edge_feature, [relation.mu_same, relation.mu_conflict, mu_after, relation.mu_before]])
        
        # Convert to tensors
        if edge_indices:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_features, dtype=torch.float)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 4), dtype=torch.float)
        
        self.graph_data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr
        )
        
        logger.info(f"Graph built: {num_events} nodes, {edge_index.size(1)} edges from {len(self.fuzzy_relations)} fuzzy relations")
    
    def _create_node_features(self):
        """Create node features from event data"""
        features = []
        
        for event in self.events:
            # Basic features for each event
            feature_vector = []
            
            # Day encoding (one-hot for days of Holy Week)
            day_encoding = self._encode_day(event.day)
            feature_vector.extend(day_encoding)
            
            # Gospel encoding (one-hot for Matthew, Mark, Luke, John)
            gospel_encoding = self._encode_gospels(event.participating_gospels)
            feature_vector.extend(gospel_encoding)
            
            # Event ID (normalized)
            event_id_norm = float(event.id) / 144.0  # Normalize by max events
            feature_vector.append(event_id_norm)
            
            # Text length (normalized)
            total_text_length = sum(len(text) for text in event.texts.values())
            text_length_norm = min(total_text_length / 1000.0, 1.0)  # Cap at 1000 chars
            feature_vector.append(text_length_norm)
            
            # Number of gospels reporting this event
            num_gospels = len(event.participating_gospels)
            num_gospels_norm = num_gospels / 4.0  # Normalize by max gospels
            feature_vector.append(num_gospels_norm)
            
            # Pad or truncate to ensure consistent size (128 features)
            if len(feature_vector) < 128:
                feature_vector.extend([0.0] * (128 - len(feature_vector)))
            else:
                feature_vector = feature_vector[:128]
            
            features.append(feature_vector)
        
        return torch.tensor(features, dtype=torch.float)
    
    def _encode_day(self, day):
        """Encode day as one-hot vector"""
        days = ['Saturday', 'Palm Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday ', 'Sunday', 'Resurrection', 'Easter']
        encoding = [0.0] * len(days)
        try:
            if day in days:
                encoding[days.index(day)] = 1.0
        except:
            pass  # Use zero encoding for unknown days
        return encoding
    
    def _encode_gospels(self, gospels):
        """Encode gospels as multi-hot vector"""
        gospel_names = ['Matthew', 'Mark', 'Luke', 'John']
        encoding = [0.0] * len(gospel_names)
        for gospel in gospels:
            if gospel in gospel_names:
                encoding[gospel_names.index(gospel)] = 1.0
        return encoding
    
    def _get_event_index(self, event_id):
        """Get the index of an event by its ID"""
        for idx, event in enumerate(self.events):
            if event.id == event_id:
                return idx
        return None
    
    def _get_reverse_temporal(self, event1_id, event2_id):
        """Get reverse temporal relation (event2 before event1)"""
        reverse_key = (event2_id, event1_id)
        if reverse_key in self.fuzzy_relations:
            return self.fuzzy_relations[reverse_key].mu_before
        return 0.0
    
    def get_pytorch_data(self):
        """Get PyTorch Geometric Data object"""
        return self.graph_data
    
    def get_statistics(self):
        """Get graph statistics"""
        return {
            'num_nodes': self.graph_data.x.size(0),
            'num_edges': self.graph_data.edge_index.size(1),
            'node_features': self.graph_data.x.size(1),
            'edge_features': self.graph_data.edge_attr.size(1) if self.graph_data.edge_attr.numel() > 0 else 0
        }
    
    def export_for_analysis(self):
        """Export graph data for analysis"""
        return {
            'nodes': self.graph_data.x.tolist(),
            'edges': self.graph_data.edge_index.tolist(),
            'edge_features': self.graph_data.edge_attr.tolist() if self.graph_data.edge_attr.numel() > 0 else []
        }
