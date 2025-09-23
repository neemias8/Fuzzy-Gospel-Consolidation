"""
Temporal Reasoning Module

This module handles temporal relationships and reasoning for Gospel events,
including day-based ordering and temporal proximity calculations.
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

from data_processing.data_structures import Event

logger = logging.getLogger(__name__)


class TemporalReasoner:
    """Handles temporal reasoning for Gospel events"""
    
    def __init__(self):
        """Initialize the temporal reasoner with Holy Week day ordering"""
        
        # Define the canonical order of days during Holy Week
        self.day_order = {
            'Saturday': 0,      # Saturday before Palm Sunday
            'Palm Sunday': 1,   # Triumphal Entry
            'Monday': 2,        # Monday of Holy Week
            'Tuesday': 3,       # Tuesday of Holy Week
            'Wednesday': 4,     # Wednesday of Holy Week
            'Thursday': 5,      # Maundy Thursday
            'Friday': 6,        # Good Friday
            'Saturday ': 7,     # Saturday in the tomb (note space to distinguish)
            'Sunday': 8,        # Easter Sunday
            'Resurrection': 8,  # Alternative name for Easter
            'Easter': 8         # Alternative name for Easter
        }
        
        # Define day names for reverse lookup
        self.order_to_day = {v: k for k, v in self.day_order.items()}
        
        # Maximum meaningful temporal distance (in days)
        self.max_temporal_distance = 8
        
        logger.info("TemporalReasoner initialized with Holy Week chronology")
    
    def get_day_order(self, day: str) -> Optional[int]:
        """
        Get the numerical order for a day.
        
        Args:
            day: Day name (e.g., "Palm Sunday", "Friday")
            
        Returns:
            Numerical order or None if day not recognized
        """
        if not day:
            return None
        
        # Try exact match first
        if day in self.day_order:
            return self.day_order[day]
        
        # Try case-insensitive match
        day_lower = day.lower()
        for known_day, order in self.day_order.items():
            if known_day.lower() == day_lower:
                return order
        
        # Try partial matches for common variations
        day_lower = day_lower.strip()
        if 'palm' in day_lower and 'sunday' in day_lower:
            return self.day_order['Palm Sunday']
        elif 'easter' in day_lower or 'resurrection' in day_lower:
            return self.day_order['Sunday']
        elif day_lower in ['mon', 'monday']:
            return self.day_order['Monday']
        elif day_lower in ['tue', 'tuesday']:
            return self.day_order['Tuesday']
        elif day_lower in ['wed', 'wednesday']:
            return self.day_order['Wednesday']
        elif day_lower in ['thu', 'thursday', 'maundy']:
            return self.day_order['Thursday']
        elif day_lower in ['fri', 'friday', 'good friday']:
            return self.day_order['Friday']
        elif day_lower in ['sat', 'saturday']:
            # Distinguish between Saturday before and Saturday after
            # Default to Saturday before if ambiguous
            return self.day_order['Saturday']
        elif day_lower in ['sun', 'sunday']:
            return self.day_order['Sunday']
        
        logger.warning(f"Unknown day: {day}")
        return None
    
    def calculate_day_distance(self, day1: str, day2: str) -> Optional[int]:
        """
        Calculate the distance in days between two days.
        
        Args:
            day1: First day
            day2: Second day
            
        Returns:
            Distance in days (absolute value) or None if either day is unknown
        """
        order1 = self.get_day_order(day1)
        order2 = self.get_day_order(day2)
        
        if order1 is None or order2 is None:
            return None
        
        return abs(order1 - order2)
    
    def calculate_day_proximity(self, day1: str, day2: str) -> float:
        """
        Calculate temporal proximity between two days (0.0 to 1.0).
        
        Args:
            day1: First day
            day2: Second day
            
        Returns:
            Proximity score (1.0 = same day, 0.0 = maximum distance)
        """
        distance = self.calculate_day_distance(day1, day2)
        
        if distance is None:
            return 0.5  # Unknown proximity
        
        if distance == 0:
            return 1.0  # Same day
        
        # Exponential decay with distance
        proximity = np.exp(-distance / 2.0)
        return max(0.0, min(1.0, proximity))
    
    def is_before(self, day1: str, day2: str) -> Optional[bool]:
        """
        Check if day1 comes before day2 in the Holy Week sequence.
        
        Args:
            day1: First day
            day2: Second day
            
        Returns:
            True if day1 is before day2, False if after, None if unknown
        """
        order1 = self.get_day_order(day1)
        order2 = self.get_day_order(day2)
        
        if order1 is None or order2 is None:
            return None
        
        return order1 < order2
    
    def calculate_before_membership(self, event1: Event, event2: Event) -> float:
        """
        Calculate fuzzy membership for 'event1 before event2' relation.
        
        Args:
            event1: First event
            event2: Second event
            
        Returns:
            Membership degree for 'before' relation
        """
        # First check day-based ordering
        day_before = self.is_before(event1.day, event2.day)
        
        if day_before is True:
            return 1.0  # Definitely before
        elif day_before is False:
            return 0.0  # Definitely after
        elif day_before is None:
            # Unknown days - use event ID as fallback
            return 1.0 if event1.id < event2.id else 0.0
        
        # Same day - use event ID and additional heuristics
        if event1.id < event2.id:
            base_membership = 0.8  # Likely before based on ID
        else:
            base_membership = 0.2  # Likely after based on ID
        
        # Adjust based on event descriptions if available
        membership_adjustment = self._analyze_temporal_indicators(event1, event2)
        
        final_membership = base_membership + membership_adjustment
        return max(0.0, min(1.0, final_membership))
    
    def _analyze_temporal_indicators(self, event1: Event, event2: Event) -> float:
        """
        Analyze event descriptions for temporal indicators.
        
        Args:
            event1: First event
            event2: Second event
            
        Returns:
            Adjustment to membership degree (-0.3 to +0.3)
        """
        # Keywords that suggest temporal ordering
        early_indicators = [
            'early', 'morning', 'dawn', 'first', 'begin', 'start',
            'arrive', 'approach', 'enter', 'come'
        ]
        
        late_indicators = [
            'late', 'evening', 'night', 'last', 'end', 'finish',
            'leave', 'depart', 'go', 'after'
        ]
        
        desc1 = (event1.description or '').lower()
        desc2 = (event2.description or '').lower()
        
        # Count indicators in each description
        early1 = sum(1 for word in early_indicators if word in desc1)
        late1 = sum(1 for word in late_indicators if word in desc1)
        early2 = sum(1 for word in early_indicators if word in desc2)
        late2 = sum(1 for word in late_indicators if word in desc2)
        
        # Calculate temporal bias
        bias1 = early1 - late1  # Positive = early bias, negative = late bias
        bias2 = early2 - late2
        
        # If event1 has early bias and event2 has late bias, increase "before" membership
        if bias1 > 0 and bias2 < 0:
            return 0.2
        elif bias1 < 0 and bias2 > 0:
            return -0.2
        elif bias1 > bias2:
            return 0.1
        elif bias1 < bias2:
            return -0.1
        
        return 0.0  # No clear temporal indicators
    
    def get_temporal_sequence(self, events: List[Event]) -> List[Event]:
        """
        Sort events by temporal order.
        
        Args:
            events: List of events to sort
            
        Returns:
            Events sorted by temporal order
        """
        def temporal_sort_key(event: Event) -> Tuple[int, int]:
            """Sort key function for temporal ordering"""
            day_order = self.get_day_order(event.day)
            if day_order is None:
                day_order = 999  # Put unknown days at the end
            
            return (day_order, event.id)
        
        return sorted(events, key=temporal_sort_key)
    
    def group_events_by_day(self, events: List[Event]) -> Dict[str, List[Event]]:
        """
        Group events by day.
        
        Args:
            events: List of events to group
            
        Returns:
            Dictionary mapping day names to event lists
        """
        day_groups = {}
        
        for event in events:
            day = event.day or 'Unknown'
            if day not in day_groups:
                day_groups[day] = []
            day_groups[day].append(event)
        
        # Sort events within each day by ID
        for day in day_groups:
            day_groups[day].sort(key=lambda e: e.id)
        
        return day_groups
    
    def calculate_temporal_coherence(self, event_sequence: List[Event]) -> float:
        """
        Calculate how well a sequence of events follows temporal order.
        
        Args:
            event_sequence: Sequence of events to analyze
            
        Returns:
            Coherence score (0.0 to 1.0)
        """
        if len(event_sequence) < 2:
            return 1.0  # Single event is perfectly coherent
        
        correct_pairs = 0
        total_pairs = 0
        
        for i in range(len(event_sequence)):
            for j in range(i + 1, len(event_sequence)):
                event1 = event_sequence[i]
                event2 = event_sequence[j]
                
                # Check if the order is correct
                should_be_before = self.is_before(event1.day, event2.day)
                
                if should_be_before is not None:
                    total_pairs += 1
                    if should_be_before or event1.day == event2.day:
                        correct_pairs += 1
        
        return correct_pairs / total_pairs if total_pairs > 0 else 1.0
    
    def find_temporal_conflicts(self, events: List[Event]) -> List[Tuple[Event, Event, str]]:
        """
        Find events that have temporal conflicts.
        
        Args:
            events: List of events to analyze
            
        Returns:
            List of (event1, event2, conflict_description) tuples
        """
        conflicts = []
        
        for i, event1 in enumerate(events):
            for event2 in events[i+1:]:
                # Check for same-day events with conflicting implications
                if event1.day == event2.day and event1.day:
                    # Look for conflicting temporal indicators in descriptions
                    desc1 = (event1.description or '').lower()
                    desc2 = (event2.description or '').lower()
                    
                    # Check for explicit time conflicts
                    if ('morning' in desc1 and 'evening' in desc2) or \
                       ('evening' in desc1 and 'morning' in desc2):
                        conflicts.append((event1, event2, "Time of day conflict"))
                    
                    # Check for sequence conflicts
                    if ('first' in desc1 and 'last' in desc2) or \
                       ('last' in desc1 and 'first' in desc2):
                        conflicts.append((event1, event2, "Sequence conflict"))
        
        return conflicts
    
    def create_temporal_graph(self, events: List[Event]) -> Dict[int, List[int]]:
        """
        Create a directed graph representing temporal relationships.
        
        Args:
            events: List of events
            
        Returns:
            Adjacency list representation of temporal graph
        """
        graph = {event.id: [] for event in events}
        
        for i, event1 in enumerate(events):
            for event2 in events[i+1:]:
                before_membership = self.calculate_before_membership(event1, event2)
                
                # Add edge if there's strong temporal relationship
                if before_membership > 0.7:
                    graph[event1.id].append(event2.id)
                elif before_membership < 0.3:
                    graph[event2.id].append(event1.id)
        
        return graph
    
    def validate_temporal_consistency(self, events: List[Event]) -> Dict[str, any]:
        """
        Validate temporal consistency of events.
        
        Args:
            events: List of events to validate
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'total_events': len(events),
            'events_with_known_days': 0,
            'temporal_coherence': 0.0,
            'conflicts': [],
            'day_distribution': {},
            'warnings': []
        }
        
        # Count events with known days
        known_day_events = [e for e in events if self.get_day_order(e.day) is not None]
        results['events_with_known_days'] = len(known_day_events)
        
        # Calculate temporal coherence
        if known_day_events:
            results['temporal_coherence'] = self.calculate_temporal_coherence(known_day_events)
        
        # Find conflicts
        results['conflicts'] = self.find_temporal_conflicts(events)
        
        # Day distribution
        day_groups = self.group_events_by_day(events)
        results['day_distribution'] = {day: len(events) for day, events in day_groups.items()}
        
        # Generate warnings
        if results['events_with_known_days'] < len(events) * 0.8:
            results['warnings'].append("Many events have unknown or unrecognized days")
        
        if results['temporal_coherence'] < 0.8:
            results['warnings'].append("Low temporal coherence detected")
        
        if len(results['conflicts']) > 0:
            results['warnings'].append(f"{len(results['conflicts'])} temporal conflicts found")
        
        return results
