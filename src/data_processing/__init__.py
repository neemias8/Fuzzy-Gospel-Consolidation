"""
Data Processing Module

This module handles XML parsing, text extraction, and data structure creation
for the Fuzzy Gospel Consolidation project.
"""

from .data_structures import Event, VerseReference, GospelCorpus
from .xml_parser import XMLParser
from .text_extractor import TextExtractor

__all__ = [
    'Event',
    'VerseReference', 
    'GospelCorpus',
    'XMLParser',
    'TextExtractor'
]
