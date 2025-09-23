"""
Fuzzy Relations Module

This module implements fuzzy logic calculations for determining relationships
between Gospel events, including similarity, conflict, and temporal ordering.
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
current_path = Path(__file__).parent
src_path = current_path.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from fuzzy_relations.fuzzy_calculator import FuzzyRelationCalculator
from fuzzy_relations.membership_functions import MembershipFunctions
from fuzzy_relations.temporal_reasoning import TemporalReasoner

__all__ = [
    'FuzzyRelationCalculator',
    'MembershipFunctions', 
    'TemporalReasoner'
]
