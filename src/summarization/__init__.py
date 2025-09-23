"""
Summarization Module

This module handles the generation of consolidated summaries
from the fuzzy-enhanced event representations.
"""

from .consolidation_summarizer import ConsolidationSummarizer
from .text_generator import TextGenerator

__all__ = [
    'ConsolidationSummarizer',
    'TextGenerator'
]
