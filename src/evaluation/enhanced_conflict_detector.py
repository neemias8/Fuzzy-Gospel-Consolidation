"""
Enhanced Gospel Difference Detection Implementation

This module implements methods to identify and document differences between 
Gospel accounts. Rather than trying to "resolve" conflicts, it focuses on:

1. IDENTIFYING differences in participant details, timing, locations, and numerical values
2. DOCUMENTING these variations as comparative analysis results
3. ANNOTATING the differences for scholarly study and transparency

The goal is comparative analysis, not conflict resolution - preserving the 
distinct perspectives each Gospel offers while noting where they differ.
"""

import re
import logging
from typing import Dict, List, Set, Tuple, Any
from collections import Counter

logger = logging.getLogger(__name__)


class ConflictDetector:
    """Enhanced conflict detection for Gospel events"""
    
    def __init__(self):
        """Initialize conflict detector with knowledge bases"""
        
        # Dictionary of key participants with variants
        self.participants = {
            'jesus': {'jesus', 'christ', 'lord', 'master', 'teacher', 'son of man'},
            'peter': {'peter', 'simon', 'simon peter', 'cephas'},
            'john': {'john', 'beloved disciple'},
            'judas': {'judas', 'judas iscariot'},
            'mary_magdalene': {'mary', 'mary magdalene'},
            'pilate': {'pilate', 'pontius pilate', 'governor'},
            'disciples': {'disciples', 'apostles', 'twelve'},
            'crowd': {'crowd', 'people', 'multitude'}
        }
        
        # Dictionary of locations with variants
        self.locations = {
            'jerusalem': {'jerusalem', 'holy city'},
            'temple': {'temple', 'temple courts', 'house of god'},
            'mount_olives': {'mount of olives', 'olivet'},
            'gethsemane': {'gethsemane', 'garden'},
            'golgotha': {'golgotha', 'calvary', 'skull', 'place of the skull'},
            'bethany': {'bethany'},
            'bethphage': {'bethphage'}
        }
        
        # Dictionary of objects/animals with variants
        self.objects = {
            'donkey': {'donkey', 'ass', 'colt', 'young donkey'},
            'cross': {'cross', 'tree'},
            'cup': {'cup', 'chalice'},
            'bread': {'bread', 'loaf'},
            'coins': {'coins', 'silver', 'thirty pieces', 'money'}
        }
        
        # Numerical conflict patterns (what to look for)
        self.numerical_patterns = {
            'rooster_crows': r'(?:rooster|cock).{0,50}(?:crow|crows?)',
            'times_denied': r'(?:deny|disown).{0,30}(?:three|3|twice|two|2)',
            'days': r'(?:after|in).{0,20}(?:three|3|two|2).{0,20}days?',
            'hours': r'(?:sixth|third|ninth).{0,20}hour',
            'pieces_silver': r'(?:thirty|30).{0,20}(?:silver|pieces|coins)'
        }
        
        # Temporal conflict keywords
        self.temporal_keywords = {
            'early': {'early', 'dawn', 'morning', 'daybreak'},
            'late': {'late', 'evening', 'night', 'dusk'},
            'before': {'before', 'prior to'},
            'after': {'after', 'following', 'then'},
            'during': {'during', 'while', 'as'}
        }
    
    def detect_participant_conflicts(self, event1, event2) -> List[Dict[str, Any]]:
        """Detect conflicts in participant descriptions"""
        conflicts = []
        
        # Extract participants from both events
        participants1 = self._extract_participants_enhanced(event1)
        participants2 = self._extract_participants_enhanced(event2)
        
        # Find conflicts (participants in one but not the other)
        only_in_1 = participants1 - participants2
        only_in_2 = participants2 - participants1
        
        if only_in_1 or only_in_2:
            conflicts.append({
                'type': 'participant_difference',
                'event1_only': list(only_in_1),
                'event2_only': list(only_in_2),
                'severity': 'medium'
            })
        
        return conflicts
    
    def detect_numerical_conflicts(self, event1, event2) -> List[Dict[str, Any]]:
        """Detect numerical conflicts (times, quantities, etc.)"""
        conflicts = []
        
        for pattern_name, pattern in self.numerical_patterns.items():
            matches1 = self._find_numerical_matches(event1, pattern)
            matches2 = self._find_numerical_matches(event2, pattern)
            
            if matches1 and matches2 and matches1 != matches2:
                conflicts.append({
                    'type': 'numerical_conflict',
                    'pattern': pattern_name,
                    'event1_values': matches1,
                    'event2_values': matches2,
                    'severity': 'high'  # Numbers are usually important
                })
        
        return conflicts
    
    def detect_temporal_conflicts(self, event1, event2) -> List[Dict[str, Any]]:
        """Detect temporal/timing conflicts"""
        conflicts = []
        
        # Extract temporal indicators from both events
        temporal1 = self._extract_temporal_indicators(event1)
        temporal2 = self._extract_temporal_indicators(event2)
        
        # Look for contradictory temporal information
        for category in ['early', 'late']:
            if (category in temporal1 and 
                any(opposite in temporal2 for opposite in ['late' if category == 'early' else 'early'])):
                conflicts.append({
                    'type': 'temporal_conflict',
                    'category': category,
                    'event1_indicators': temporal1.get(category, []),
                    'event2_indicators': temporal2.get('late' if category == 'early' else 'early', []),
                    'severity': 'medium'
                })
        
        return conflicts
    
    def detect_location_conflicts(self, event1, event2) -> List[Dict[str, Any]]:
        """Detect location/setting conflicts"""
        conflicts = []
        
        locations1 = self._extract_locations(event1)
        locations2 = self._extract_locations(event2)
        
        # Different locations for same event type
        if locations1 and locations2 and not locations1.intersection(locations2):
            conflicts.append({
                'type': 'location_conflict',
                'event1_locations': list(locations1),
                'event2_locations': list(locations2),
                'severity': 'high'  # Location is usually important
            })
        
        return conflicts
    
    def detect_sequence_conflicts(self, events: List) -> List[Dict[str, Any]]:
        """Detect conflicts in event sequence/order"""
        conflicts = []
        
        # Group events by day
        day_groups = {}
        for event in events:
            day = getattr(event, 'day', 'unknown')
            if day not in day_groups:
                day_groups[day] = []
            day_groups[day].append(event)
        
        # Check for sequence conflicts within each day
        for day, day_events in day_groups.items():
            if len(day_events) > 1:
                sequence_conflicts = self._analyze_event_sequence(day_events)
                conflicts.extend(sequence_conflicts)
        
        return conflicts
    
    def detect_all_conflicts(self, event1, event2) -> Dict[str, Any]:
        """Comprehensive conflict detection between two events"""
        all_conflicts = []
        
        # Only check for conflicts if events are related
        if self._events_are_related(event1, event2):
            all_conflicts.extend(self.detect_participant_conflicts(event1, event2))
            all_conflicts.extend(self.detect_numerical_conflicts(event1, event2))
            all_conflicts.extend(self.detect_temporal_conflicts(event1, event2))
            all_conflicts.extend(self.detect_location_conflicts(event1, event2))
        
        # Calculate overall conflict score
        conflict_score = self._calculate_conflict_score(all_conflicts)
        
        return {
            'conflicts': all_conflicts,
            'conflict_score': conflict_score,
            'has_conflicts': len(all_conflicts) > 0,
            'severity_breakdown': self._get_severity_breakdown(all_conflicts)
        }
    
    def analyze_known_conflicts(self, corpus, config_test_cases) -> Dict[str, Any]:
        """Analyze specific known conflicts from configuration"""
        results = {}
        
        for test_case in config_test_cases:
            case_name = test_case['name']
            event_ids = test_case['events']
            expected_conflict = test_case['conflict']
            
            # Get events by ID
            events = [event for event in corpus.events if event.id in event_ids]
            
            if len(events) >= 2:
                # Analyze pairwise conflicts
                case_conflicts = []
                for i in range(len(events)):
                    for j in range(i+1, len(events)):
                        conflict_result = self.detect_all_conflicts(events[i], events[j])
                        if conflict_result['has_conflicts']:
                            case_conflicts.append(conflict_result)
                
                results[case_name] = {
                    'expected_conflict': expected_conflict,
                    'events_analyzed': len(events),
                    'conflicts_found': len(case_conflicts),
                    'details': case_conflicts
                }
        
        return results
    
    # Helper methods
    def _extract_participants_enhanced(self, event) -> Set[str]:
        """Extract participants using enhanced name recognition"""
        participants = set()
        
        # Combine all text from event
        all_text = self._get_all_event_text(event).lower()
        
        # Check for each participant category
        for participant_type, variants in self.participants.items():
            if any(variant in all_text for variant in variants):
                participants.add(participant_type)
        
        return participants
    
    def _find_numerical_matches(self, event, pattern) -> List[str]:
        """Find numerical matches in event text"""
        all_text = self._get_all_event_text(event)
        matches = re.findall(pattern, all_text, re.IGNORECASE)
        return matches
    
    def _extract_temporal_indicators(self, event) -> Dict[str, List[str]]:
        """Extract temporal indicators from event"""
        indicators = {}
        all_text = self._get_all_event_text(event).lower()
        
        for category, keywords in self.temporal_keywords.items():
            found = [kw for kw in keywords if kw in all_text]
            if found:
                indicators[category] = found
        
        return indicators
    
    def _extract_locations(self, event) -> Set[str]:
        """Extract locations from event text"""
        locations = set()
        all_text = self._get_all_event_text(event).lower()
        
        for location_type, variants in self.locations.items():
            if any(variant in all_text for variant in variants):
                locations.add(location_type)
        
        return locations
    
    def _get_all_event_text(self, event) -> str:
        """Get combined text from all sources of an event"""
        texts = []
        
        if hasattr(event, 'description') and event.description:
            texts.append(event.description)
        
        if hasattr(event, 'texts'):
            for text in event.texts.values():
                if text:
                    texts.append(text)
        
        return ' '.join(texts)
    
    def _events_are_related(self, event1, event2) -> bool:
        """Check if two events are related enough to have conflicts"""
        # Simple heuristic: same day or overlapping participants
        same_day = getattr(event1, 'day', '') == getattr(event2, 'day', '')
        
        participants1 = self._extract_participants_enhanced(event1)
        participants2 = self._extract_participants_enhanced(event2)
        shared_participants = len(participants1.intersection(participants2)) > 0
        
        return same_day or shared_participants
    
    def _calculate_conflict_score(self, conflicts) -> float:
        """Calculate numerical conflict score from conflicts list"""
        if not conflicts:
            return 0.0
        
        severity_weights = {'low': 0.3, 'medium': 0.6, 'high': 1.0}
        total_score = sum(severity_weights.get(c.get('severity', 'medium'), 0.6) for c in conflicts)
        
        # Normalize by number of conflicts (max 1.0)
        return min(total_score / len(conflicts), 1.0)
    
    def _get_severity_breakdown(self, conflicts) -> Dict[str, int]:
        """Get count of conflicts by severity"""
        breakdown = {'low': 0, 'medium': 0, 'high': 0}
        for conflict in conflicts:
            severity = conflict.get('severity', 'medium')
            breakdown[severity] += 1
        return breakdown
    
    def _analyze_event_sequence(self, events) -> List[Dict[str, Any]]:
        """Analyze sequence conflicts within a group of events"""
        # Placeholder for sequence analysis
        return []