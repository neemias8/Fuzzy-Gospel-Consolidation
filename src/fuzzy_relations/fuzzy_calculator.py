"""
Fuzzy Relation Calculator

This module implements the core fuzzy logic calculations for determining
relationships between Gospel events.
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from data_processing.data_structures import Event, FuzzyRelation
from data_processing.text_extractor import TextExtractor
from fuzzy_relations.membership_functions import MembershipFunctions
from fuzzy_relations.temporal_reasoning import TemporalReasoner

logger = logging.getLogger(__name__)


class FuzzyRelationCalculator:
    """Calculates fuzzy relationships between Gospel events"""
    
    def __init__(self, config: Dict):
        """
        Initialize the fuzzy relation calculator.
        
        Args:
            config: Configuration dictionary with fuzzy parameters
        """
        self.config = config
        self.fuzzy_config = config.get('fuzzy', {})
        
        # Initialize components
        self.sentence_transformer = SentenceTransformer(
            config['models']['sentence_transformer']
        )
        self.membership_functions = MembershipFunctions(self.fuzzy_config)
        self.temporal_reasoner = TemporalReasoner()
        
        # Cache for embeddings to avoid recomputation
        self._embedding_cache = {}
        
        logger.info("FuzzyRelationCalculator initialized")
    
    def get_text_embedding(self, text: str) -> np.ndarray:
        """
        Get sentence embedding for text, with caching.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        if not text or not text.strip():
            return np.zeros(self.sentence_transformer.get_sentence_embedding_dimension())
        
        # Use text hash as cache key
        text_key = hash(text.strip())
        
        if text_key not in self._embedding_cache:
            self._embedding_cache[text_key] = self.sentence_transformer.encode(text.strip())
        
        return self._embedding_cache[text_key]
    
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        if not text1 or not text2:
            return 0.0
        
        emb1 = self.get_text_embedding(text1)
        emb2 = self.get_text_embedding(text2)
        
        # Calculate cosine similarity
        similarity = cosine_similarity([emb1], [emb2])[0][0]
        
        # Ensure non-negative
        return max(0.0, float(similarity))
    
    def calculate_description_similarity(self, event1: Event, event2: Event) -> float:
        """
        Calculate similarity between event descriptions.
        
        Args:
            event1: First event
            event2: Second event
            
        Returns:
            Description similarity score
        """
        if not event1.description or not event2.description:
            return 0.0
        
        return self.calculate_semantic_similarity(event1.description, event2.description)
    
    def calculate_temporal_proximity(self, event1: Event, event2: Event) -> float:
        """
        Calculate temporal proximity between events.
        
        Args:
            event1: First event
            event2: Second event
            
        Returns:
            Temporal proximity score
        """
        return self.temporal_reasoner.calculate_day_proximity(event1.day, event2.day)
    
    def calculate_gospel_overlap(self, event1: Event, event2: Event) -> float:
        """
        Calculate overlap in gospel coverage between events.
        
        Args:
            event1: First event
            event2: Second event
            
        Returns:
            Gospel overlap score
        """
        gospels1 = event1.participating_gospels
        gospels2 = event2.participating_gospels
        
        if not gospels1 and not gospels2:
            return 0.0
        
        intersection = len(gospels1.intersection(gospels2))
        union = len(gospels1.union(gospels2))
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_text_similarity(self, event1: Event, event2: Event) -> float:
        """
        Calculate average text similarity across common gospels.
        
        Args:
            event1: First event
            event2: Second event
            
        Returns:
            Average text similarity score
        """
        similarities = []
        
        for gospel in ['matthew', 'mark', 'luke', 'john']:
            text1 = event1.texts.get(gospel, '')
            text2 = event2.texts.get(gospel, '')
            
            if text1 and text2:
                sim = self.calculate_semantic_similarity(text1, text2)
                similarities.append(sim)
        
        return np.mean(similarities) if similarities else 0.0
    
    def calculate_participant_overlap(self, event1: Event, event2: Event, 
                                    text_extractor: TextExtractor) -> float:
        """
        Calculate overlap in participants between events.
        
        Args:
            event1: First event
            event2: Second event
            text_extractor: TextExtractor instance
            
        Returns:
            Participant overlap score
        """
        participants1 = text_extractor.get_event_participants(event1)
        participants2 = text_extractor.get_event_participants(event2)
        
        if not participants1 and not participants2:
            return 0.0
        
        intersection = len(participants1.intersection(participants2))
        union = len(participants1.union(participants2))
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_same_event_membership(self, event1: Event, event2: Event,
                                      text_extractor: Optional[TextExtractor] = None) -> float:
        """
        Calculate fuzzy membership for 'same event' relation.
        
        Args:
            event1: First event
            event2: Second event
            text_extractor: Optional TextExtractor for participant analysis
            
        Returns:
            Membership degree for 'same event' relation
        """
        # Calculate individual similarity components
        desc_sim = self.calculate_description_similarity(event1, event2)
        temporal_sim = self.calculate_temporal_proximity(event1, event2)
        gospel_overlap = self.calculate_gospel_overlap(event1, event2)
        text_sim = self.calculate_text_similarity(event1, event2)
        
        # Optional participant overlap
        participant_sim = 0.0
        if text_extractor:
            participant_sim = self.calculate_participant_overlap(event1, event2, text_extractor)
        
        # Get weights from configuration
        weights = self.fuzzy_config.get('weights', {})
        w_desc = weights.get('description_similarity', 0.3)
        w_temporal = weights.get('temporal_proximity', 0.2)
        w_gospel = weights.get('gospel_overlap', 0.2)
        w_text = weights.get('text_similarity', 0.3)
        w_participant = weights.get('participant_overlap', 0.0)
        
        # Normalize weights
        total_weight = w_desc + w_temporal + w_gospel + w_text + w_participant
        if total_weight > 0:
            w_desc /= total_weight
            w_temporal /= total_weight
            w_gospel /= total_weight
            w_text /= total_weight
            w_participant /= total_weight
        
        # Calculate weighted average
        mu_same = (w_desc * desc_sim + 
                   w_temporal * temporal_sim + 
                   w_gospel * gospel_overlap + 
                   w_text * text_sim + 
                   w_participant * participant_sim)
        
        # Apply membership function
        return self.membership_functions.same_event_membership(mu_same)
    
    def calculate_conflict_membership(self, event1: Event, event2: Event) -> float:
        """
        Calculate fuzzy membership for 'conflict' relation.
        
        Args:
            event1: First event
            event2: Second event
            
        Returns:
            Membership degree for 'conflict' relation
        """
        # Events can only conflict if they're related
        same_membership = self.calculate_same_event_membership(event1, event2)
        
        if same_membership < 0.3:  # Not related enough to conflict
            return 0.0
        
        # Look for textual differences in same gospel accounts
        conflicts = []
        
        for gospel in ['matthew', 'mark', 'luke', 'john']:
            text1 = event1.texts.get(gospel, '')
            text2 = event2.texts.get(gospel, '')
            
            if text1 and text2:
                # High semantic similarity but different events suggests conflict
                text_sim = self.calculate_semantic_similarity(text1, text2)
                
                # Conflict occurs when texts are similar but not identical
                if 0.3 < text_sim < 0.8:
                    conflict_degree = 1.0 - text_sim
                    conflicts.append(conflict_degree)
        
        if not conflicts:
            return 0.0
        
        # Average conflict across gospels
        avg_conflict = np.mean(conflicts)
        
        # Scale by how related the events are
        scaled_conflict = avg_conflict * same_membership
        
        return self.membership_functions.conflict_membership(scaled_conflict)
    
    def calculate_temporal_before_membership(self, event1: Event, event2: Event) -> float:
        """
        Calculate fuzzy membership for 'event1 before event2' relation.
        
        Args:
            event1: First event
            event2: Second event
            
        Returns:
            Membership degree for temporal 'before' relation
        """
        return self.temporal_reasoner.calculate_before_membership(event1, event2)
    
    def calculate_temporal_after_membership(self, event1: Event, event2: Event) -> float:
        """
        Calculate fuzzy membership for 'event1 after event2' relation.
        
        Args:
            event1: First event
            event2: Second event
            
        Returns:
            Membership degree for temporal 'after' relation
        """
        return self.temporal_reasoner.calculate_before_membership(event2, event1)
    
    def calculate_all_relations(self, event1: Event, event2: Event,
                              text_extractor: Optional[TextExtractor] = None) -> FuzzyRelation:
        """
        Calculate all fuzzy relations between two events.
        
        Args:
            event1: First event
            event2: Second event
            text_extractor: Optional TextExtractor for enhanced analysis
            
        Returns:
            FuzzyRelation object with all membership degrees
        """
        mu_same = self.calculate_same_event_membership(event1, event2, text_extractor)
        mu_conflict = self.calculate_conflict_membership(event1, event2)
        mu_before = self.calculate_temporal_before_membership(event1, event2)
        mu_after = self.calculate_temporal_after_membership(event1, event2)
        
        return FuzzyRelation(
            mu_same=mu_same,
            mu_conflict=mu_conflict,
            mu_before=mu_before,
            mu_after=mu_after
        )
    
    def calculate_relation_matrix(self, events: List[Event],
                                text_extractor: Optional[TextExtractor] = None) -> Dict[Tuple[int, int], FuzzyRelation]:
        """
        Calculate fuzzy relations for all event pairs.
        
        Args:
            events: List of events to analyze
            text_extractor: Optional TextExtractor for enhanced analysis
            
        Returns:
            Dictionary mapping event ID pairs to FuzzyRelation objects
        """
        logger.info(f"Calculating fuzzy relations for {len(events)} events...")
        
        relations = {}
        total_pairs = len(events) * (len(events) - 1) // 2
        processed = 0
        
        for i, event1 in enumerate(events):
            for j, event2 in enumerate(events[i+1:], i+1):
                relation = self.calculate_all_relations(event1, event2, text_extractor)
                relations[(event1.id, event2.id)] = relation
                
                processed += 1
                if processed % 100 == 0:
                    logger.info(f"Processed {processed}/{total_pairs} event pairs")
        
        logger.info(f"Fuzzy relation calculation complete: {len(relations)} relations")
        return relations
    
    def get_high_similarity_pairs(self, relations: Dict[Tuple[int, int], FuzzyRelation],
                                threshold: float = None) -> List[Tuple[int, int, float]]:
        """
        Get event pairs with high similarity scores.
        
        Args:
            relations: Dictionary of fuzzy relations
            threshold: Similarity threshold (uses config default if None)
            
        Returns:
            List of (event1_id, event2_id, similarity_score) tuples
        """
        if threshold is None:
            threshold = self.fuzzy_config.get('same_event_threshold', 0.8)
        
        high_similarity = []
        
        for (event1_id, event2_id), relation in relations.items():
            if relation.mu_same >= threshold:
                high_similarity.append((event1_id, event2_id, relation.mu_same))
        
        # Sort by similarity score (descending)
        high_similarity.sort(key=lambda x: x[2], reverse=True)
        
        return high_similarity
    
    def get_conflict_pairs(self, relations: Dict[Tuple[int, int], FuzzyRelation],
                          threshold: float = None) -> List[Tuple[int, int, float]]:
        """
        Get event pairs with high conflict scores.
        
        Args:
            relations: Dictionary of fuzzy relations
            threshold: Conflict threshold (uses config default if None)
            
        Returns:
            List of (event1_id, event2_id, conflict_score) tuples
        """
        if threshold is None:
            threshold = self.fuzzy_config.get('conflict_threshold', 0.6)
        
        conflicts = []
        
        for (event1_id, event2_id), relation in relations.items():
            if relation.mu_conflict >= threshold:
                conflicts.append((event1_id, event2_id, relation.mu_conflict))
        
        # Sort by conflict score (descending)
        conflicts.sort(key=lambda x: x[2], reverse=True)
        
        return conflicts
    
    def export_relation_statistics(self, relations: Dict[Tuple[int, int], FuzzyRelation]) -> Dict[str, any]:
        """
        Export statistics about the calculated relations.
        
        Args:
            relations: Dictionary of fuzzy relations
            
        Returns:
            Dictionary with relation statistics
        """
        if not relations:
            return {}
        
        # Extract all membership values
        same_values = [r.mu_same for r in relations.values()]
        conflict_values = [r.mu_conflict for r in relations.values()]
        before_values = [r.mu_before for r in relations.values()]
        after_values = [r.mu_after for r in relations.values()]
        
        stats = {
            'total_relations': len(relations),
            'same_event': {
                'mean': np.mean(same_values),
                'std': np.std(same_values),
                'min': np.min(same_values),
                'max': np.max(same_values),
                'above_threshold': sum(1 for v in same_values if v >= 0.8)
            },
            'conflict': {
                'mean': np.mean(conflict_values),
                'std': np.std(conflict_values),
                'min': np.min(conflict_values),
                'max': np.max(conflict_values),
                'above_threshold': sum(1 for v in conflict_values if v >= 0.6)
            },
            'temporal_before': {
                'mean': np.mean(before_values),
                'std': np.std(before_values),
                'min': np.min(before_values),
                'max': np.max(before_values)
            },
            'temporal_after': {
                'mean': np.mean(after_values),
                'std': np.std(after_values),
                'min': np.min(after_values),
                'max': np.max(after_values)
            }
        }
        
        return stats
