"""
Membership Functions for Fuzzy Logic

This module defines the fuzzy membership functions used to convert
crisp values into fuzzy membership degrees.
"""

import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class MembershipFunctions:
    """Fuzzy membership functions for Gospel event relationships"""
    
    def __init__(self, config: Dict):
        """
        Initialize membership functions with configuration.
        
        Args:
            config: Configuration dictionary with fuzzy parameters
        """
        self.config = config
        
    def triangular_membership(self, x: float, a: float, b: float, c: float) -> float:
        """
        Triangular membership function.
        
        Args:
            x: Input value
            a: Left base point
            b: Peak point
            c: Right base point
            
        Returns:
            Membership degree [0, 1]
        """
        if x <= a or x >= c:
            return 0.0
        elif a < x <= b:
            return (x - a) / (b - a)
        else:  # b < x < c
            return (c - x) / (c - b)
    
    def trapezoidal_membership(self, x: float, a: float, b: float, c: float, d: float) -> float:
        """
        Trapezoidal membership function.
        
        Args:
            x: Input value
            a: Left base point
            b: Left top point
            c: Right top point
            d: Right base point
            
        Returns:
            Membership degree [0, 1]
        """
        if x <= a or x >= d:
            return 0.0
        elif a < x <= b:
            return (x - a) / (b - a)
        elif b < x <= c:
            return 1.0
        else:  # c < x < d
            return (d - x) / (d - c)
    
    def gaussian_membership(self, x: float, center: float, sigma: float) -> float:
        """
        Gaussian membership function.
        
        Args:
            x: Input value
            center: Center of the Gaussian
            sigma: Standard deviation
            
        Returns:
            Membership degree [0, 1]
        """
        return np.exp(-0.5 * ((x - center) / sigma) ** 2)
    
    def sigmoid_membership(self, x: float, a: float, c: float) -> float:
        """
        Sigmoid membership function.
        
        Args:
            x: Input value
            a: Slope parameter
            c: Center point
            
        Returns:
            Membership degree [0, 1]
        """
        return 1.0 / (1.0 + np.exp(-a * (x - c)))
    
    def same_event_membership(self, similarity: float) -> float:
        """
        Membership function for 'same event' relation.
        
        Uses a sigmoid function to model the transition from
        "definitely different" to "definitely same" events.
        
        Args:
            similarity: Similarity score [0, 1]
            
        Returns:
            Membership degree for 'same event'
        """
        # Sigmoid with steep transition around 0.7
        return self.sigmoid_membership(similarity, a=10.0, c=0.7)
    
    def conflict_membership(self, conflict_score: float) -> float:
        """
        Membership function for 'conflict' relation.
        
        Uses a triangular function with peak around 0.5-0.7
        (moderate similarity but clear differences).
        
        Args:
            conflict_score: Conflict score [0, 1]
            
        Returns:
            Membership degree for 'conflict'
        """
        # Triangular function peaking at moderate conflict
        return self.triangular_membership(conflict_score, a=0.2, b=0.6, c=0.9)
    
    def temporal_certainty_membership(self, certainty: float) -> float:
        """
        Membership function for temporal certainty.
        
        Args:
            certainty: Temporal certainty score [0, 1]
            
        Returns:
            Membership degree for temporal certainty
        """
        # Trapezoidal function with high plateau
        return self.trapezoidal_membership(certainty, a=0.3, b=0.7, c=1.0, d=1.0)
    
    def proximity_membership(self, distance: float, max_distance: float = 7.0) -> float:
        """
        Membership function for temporal proximity.
        
        Args:
            distance: Distance in days
            max_distance: Maximum meaningful distance
            
        Returns:
            Membership degree for proximity
        """
        if distance >= max_distance:
            return 0.0
        
        # Exponential decay
        return np.exp(-distance / 2.0)
    
    def gospel_coverage_membership(self, coverage: float) -> float:
        """
        Membership function for gospel coverage overlap.
        
        Args:
            coverage: Coverage overlap ratio [0, 1]
            
        Returns:
            Membership degree for coverage
        """
        # Quadratic function emphasizing high coverage
        return coverage ** 0.5
    
    def text_similarity_membership(self, similarity: float) -> float:
        """
        Membership function for text similarity.
        
        Args:
            similarity: Text similarity score [0, 1]
            
        Returns:
            Membership degree for text similarity
        """
        # S-shaped curve with threshold around 0.3
        if similarity < 0.3:
            return 0.0
        elif similarity > 0.8:
            return 1.0
        else:
            # Smooth transition between 0.3 and 0.8
            normalized = (similarity - 0.3) / 0.5
            return 3 * normalized**2 - 2 * normalized**3  # Smooth S-curve
    
    def participant_overlap_membership(self, overlap: float) -> float:
        """
        Membership function for participant overlap.
        
        Args:
            overlap: Participant overlap ratio [0, 1]
            
        Returns:
            Membership degree for participant overlap
        """
        # Linear function with slight emphasis on higher overlap
        return overlap ** 0.8
    
    def combine_memberships_and(self, *memberships: float) -> float:
        """
        Combine multiple membership degrees using AND (minimum).
        
        Args:
            memberships: Variable number of membership degrees
            
        Returns:
            Combined membership degree
        """
        return min(memberships) if memberships else 0.0
    
    def combine_memberships_or(self, *memberships: float) -> float:
        """
        Combine multiple membership degrees using OR (maximum).
        
        Args:
            memberships: Variable number of membership degrees
            
        Returns:
            Combined membership degree
        """
        return max(memberships) if memberships else 0.0
    
    def combine_memberships_weighted(self, memberships: List[float], 
                                   weights: List[float]) -> float:
        """
        Combine membership degrees using weighted average.
        
        Args:
            memberships: List of membership degrees
            weights: List of weights (should sum to 1.0)
            
        Returns:
            Weighted combined membership degree
        """
        if not memberships or not weights or len(memberships) != len(weights):
            return 0.0
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight == 0:
            return 0.0
        
        normalized_weights = [w / total_weight for w in weights]
        
        # Calculate weighted sum
        return sum(m * w for m, w in zip(memberships, normalized_weights))
    
    def defuzzify_centroid(self, membership_values: List[float], 
                          x_values: List[float]) -> float:
        """
        Defuzzify using centroid method.
        
        Args:
            membership_values: List of membership degrees
            x_values: Corresponding x values
            
        Returns:
            Defuzzified crisp value
        """
        if not membership_values or not x_values or len(membership_values) != len(x_values):
            return 0.0
        
        numerator = sum(m * x for m, x in zip(membership_values, x_values))
        denominator = sum(membership_values)
        
        return numerator / denominator if denominator > 0 else 0.0
    
    def defuzzify_maximum(self, membership_values: List[float], 
                         x_values: List[float]) -> float:
        """
        Defuzzify using maximum method (return x with highest membership).
        
        Args:
            membership_values: List of membership degrees
            x_values: Corresponding x values
            
        Returns:
            X value with maximum membership
        """
        if not membership_values or not x_values or len(membership_values) != len(x_values):
            return 0.0
        
        max_index = np.argmax(membership_values)
        return x_values[max_index]
    
    def linguistic_hedge_very(self, membership: float) -> float:
        """
        Apply 'very' linguistic hedge (concentration).
        
        Args:
            membership: Original membership degree
            
        Returns:
            Modified membership degree
        """
        return membership ** 2
    
    def linguistic_hedge_somewhat(self, membership: float) -> float:
        """
        Apply 'somewhat' linguistic hedge (dilation).
        
        Args:
            membership: Original membership degree
            
        Returns:
            Modified membership degree
        """
        return membership ** 0.5
    
    def linguistic_hedge_not(self, membership: float) -> float:
        """
        Apply 'not' linguistic hedge (complement).
        
        Args:
            membership: Original membership degree
            
        Returns:
            Complement membership degree
        """
        return 1.0 - membership
    
    def evaluate_fuzzy_rule(self, antecedent_memberships: List[float],
                           consequent_membership: float,
                           rule_weight: float = 1.0) -> float:
        """
        Evaluate a fuzzy rule using minimum for AND operations.
        
        Args:
            antecedent_memberships: List of antecedent membership degrees
            consequent_membership: Consequent membership degree
            rule_weight: Weight of the rule
            
        Returns:
            Rule activation strength
        """
        if not antecedent_memberships:
            return 0.0
        
        # AND operation on antecedents
        antecedent_strength = min(antecedent_memberships)
        
        # Apply rule weight
        weighted_strength = antecedent_strength * rule_weight
        
        # Clip consequent by rule strength
        return min(weighted_strength, consequent_membership)
    
    def create_membership_plot_data(self, function_name: str, 
                                  x_range: Tuple[float, float] = (0.0, 1.0),
                                  num_points: int = 100) -> Tuple[List[float], List[float]]:
        """
        Generate data points for plotting membership functions.
        
        Args:
            function_name: Name of the membership function
            x_range: Range of x values
            num_points: Number of points to generate
            
        Returns:
            Tuple of (x_values, y_values) for plotting
        """
        x_values = np.linspace(x_range[0], x_range[1], num_points)
        y_values = []
        
        function_map = {
            'same_event': self.same_event_membership,
            'conflict': self.conflict_membership,
            'temporal_certainty': self.temporal_certainty_membership,
            'gospel_coverage': self.gospel_coverage_membership,
            'text_similarity': self.text_similarity_membership,
            'participant_overlap': self.participant_overlap_membership
        }
        
        if function_name not in function_map:
            logger.warning(f"Unknown membership function: {function_name}")
            return [], []
        
        func = function_map[function_name]
        
        for x in x_values:
            try:
                y = func(x)
                y_values.append(y)
            except Exception as e:
                logger.warning(f"Error evaluating {function_name} at x={x}: {e}")
                y_values.append(0.0)
        
        return x_values.tolist(), y_values
