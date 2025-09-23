"""
Text Generator

Utility functions for text generation and processing.
"""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class TextGenerator:
    """Utility class for text generation operations"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the text generator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
    def generate_event_narrative(self, event, context=None):
        """
        Generate narrative text for a single event.
        
        Args:
            event: Event object
            context: Optional context information
            
        Returns:
            Generated narrative text
        """
        # Placeholder implementation
        narrative = event.description or "An event occurred"
        
        if event.when_where:
            narrative += f" ({event.when_where})"
        
        return narrative
    
    def combine_narratives(self, narratives: List[str]) -> str:
        """
        Combine multiple narrative texts into a coherent whole.
        
        Args:
            narratives: List of narrative texts
            
        Returns:
            Combined narrative
        """
        return " ".join(narratives)
    
    def post_process_text(self, text: str) -> str:
        """
        Post-process generated text for better readability.
        
        Args:
            text: Raw generated text
            
        Returns:
            Post-processed text
        """
        # Basic post-processing
        text = text.strip()
        
        # Ensure proper sentence endings
        if text and not text.endswith(('.', '!', '?')):
            text += '.'
        
        return text
