"""
Text extraction and preprocessing utilities.

This module provides functionality for extracting and preprocessing text
from the parsed Gospel data.
"""

import re
from typing import Dict, List, Set, Tuple
import logging
from collections import Counter

from .data_structures import Event, GospelCorpus

logger = logging.getLogger(__name__)


class TextExtractor:
    """Handles text extraction and preprocessing for Gospel events"""
    
    def __init__(self, corpus: GospelCorpus):
        self.corpus = corpus
        self.stop_words = self._load_stop_words()
    
    def _load_stop_words(self) -> Set[str]:
        """Load common English stop words"""
        # Basic stop words - in production, you might use NLTK
        return {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'but', 'had', 'have', 'his', 'her',
            'him', 'they', 'them', 'their', 'said', 'then', 'when', 'who',
            'you', 'your', 'all', 'any', 'can', 'did', 'do', 'does', 'not',
            'or', 'so', 'up', 'out', 'if', 'about', 'into', 'than', 'only',
            'other', 'new', 'some', 'could', 'time', 'very', 'what', 'know',
            'just', 'first', 'get', 'may', 'way', 'day', 'man', 'old', 'see',
            'now', 'come', 'made', 'over', 'also', 'back', 'after', 'use',
            'two', 'how', 'our', 'work', 'life', 'only', 'years', 'way',
            'even', 'good', 'much', 'should', 'well', 'people', 'down',
            'own', 'go', 'would', 'like', 'been', 'more', 'no', 'one'
        }
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove quotation marks that might interfere with processing
        text = re.sub(r'["""]', '"', text)
        text = re.sub(r'['']', "'", text)
        
        # Normalize punctuation spacing
        text = re.sub(r'\s*([,.!?;:])\s*', r'\1 ', text)
        
        return text.strip()
    
    def extract_keywords(self, text: str, min_length: int = 3) -> List[str]:
        """
        Extract keywords from text by removing stop words and short words.
        
        Args:
            text: Text to extract keywords from
            min_length: Minimum word length to consider
            
        Returns:
            List of keywords
        """
        if not text:
            return []
        
        # Convert to lowercase and split into words
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        
        # Filter out stop words and short words
        keywords = [
            word for word in words 
            if len(word) >= min_length and word not in self.stop_words
        ]
        
        return keywords
    
    def get_event_keywords(self, event: Event) -> Dict[str, List[str]]:
        """
        Extract keywords from an event's description and texts.
        
        Args:
            event: Event to extract keywords from
            
        Returns:
            Dictionary mapping sources to keyword lists
        """
        keywords = {}
        
        # Keywords from description
        if event.description:
            keywords['description'] = self.extract_keywords(event.description)
        
        # Keywords from each gospel text
        for gospel, text in event.texts.items():
            if text:
                keywords[gospel] = self.extract_keywords(text)
        
        return keywords
    
    def get_common_keywords(self, event1: Event, event2: Event) -> Set[str]:
        """
        Find common keywords between two events.
        
        Args:
            event1: First event
            event2: Second event
            
        Returns:
            Set of common keywords
        """
        keywords1 = set()
        keywords2 = set()
        
        # Collect all keywords from event1
        for keyword_list in self.get_event_keywords(event1).values():
            keywords1.update(keyword_list)
        
        # Collect all keywords from event2
        for keyword_list in self.get_event_keywords(event2).values():
            keywords2.update(keyword_list)
        
        return keywords1.intersection(keywords2)
    
    def extract_participants(self, text: str) -> Set[str]:
        """
        Extract participant names from text using simple heuristics.
        
        Args:
            text: Text to extract participants from
            
        Returns:
            Set of participant names
        """
        if not text:
            return set()
        
        # Common biblical names and titles
        biblical_names = {
            'jesus', 'christ', 'lord', 'peter', 'john', 'james', 'andrew',
            'philip', 'bartholomew', 'matthew', 'thomas', 'simon', 'judas',
            'mary', 'martha', 'lazarus', 'pilate', 'herod', 'caiaphas',
            'barabbas', 'joseph', 'nicodemus', 'disciples', 'apostles',
            'pharisees', 'sadducees', 'scribes', 'priests', 'crowd',
            'people', 'soldiers', 'centurion', 'governor'
        }
        
        # Extract words that might be names
        words = re.findall(r'\b[A-Z][a-z]+\b', text)
        
        # Filter for known biblical names
        participants = set()
        for word in words:
            if word.lower() in biblical_names:
                participants.add(word.lower())
        
        return participants
    
    def get_event_participants(self, event: Event) -> Set[str]:
        """
        Get all participants mentioned in an event.
        
        Args:
            event: Event to extract participants from
            
        Returns:
            Set of participant names
        """
        participants = set()
        
        # Extract from description
        if event.description:
            participants.update(self.extract_participants(event.description))
        
        # Extract from all gospel texts
        for text in event.texts.values():
            if text:
                participants.update(self.extract_participants(text))
        
        return participants
    
    def calculate_text_overlap(self, text1: str, text2: str) -> float:
        """
        Calculate word overlap between two texts using Jaccard similarity.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Jaccard similarity coefficient (0.0 to 1.0)
        """
        if not text1 or not text2:
            return 0.0
        
        words1 = set(self.extract_keywords(text1))
        words2 = set(self.extract_keywords(text2))
        
        if not words1 and not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def get_text_statistics(self, event: Event) -> Dict[str, int]:
        """
        Get text statistics for an event.
        
        Args:
            event: Event to analyze
            
        Returns:
            Dictionary with text statistics
        """
        stats = {
            'total_chars': 0,
            'total_words': 0,
            'total_sentences': 0,
            'gospels_with_text': 0
        }
        
        for gospel, text in event.texts.items():
            if text:
                stats['gospels_with_text'] += 1
                stats['total_chars'] += len(text)
                stats['total_words'] += len(text.split())
                stats['total_sentences'] += len(re.split(r'[.!?]+', text))
        
        return stats
    
    def create_event_summary(self, event: Event, max_length: int = 200) -> str:
        """
        Create a brief summary of an event combining information from all gospels.
        
        Args:
            event: Event to summarize
            max_length: Maximum length of summary
            
        Returns:
            Event summary text
        """
        # Start with the description
        summary_parts = [event.description] if event.description else []
        
        # Add location/timing if available
        if event.when_where:
            summary_parts.append(f"({event.when_where})")
        
        # Combine texts from all gospels, avoiding repetition
        all_keywords = set()
        unique_details = []
        
        for gospel, text in event.texts.items():
            if text:
                keywords = set(self.extract_keywords(text))
                new_keywords = keywords - all_keywords
                
                if new_keywords:
                    # Extract sentences containing new keywords
                    sentences = re.split(r'[.!?]+', text)
                    for sentence in sentences:
                        sentence = sentence.strip()
                        if sentence and any(kw in sentence.lower() for kw in new_keywords):
                            unique_details.append(sentence)
                            all_keywords.update(new_keywords)
                            break  # One sentence per gospel to avoid repetition
        
        # Combine all parts
        if unique_details:
            summary_parts.extend(unique_details[:2])  # Limit to 2 additional details
        
        summary = ". ".join(summary_parts)
        
        # Truncate if too long
        if len(summary) > max_length:
            summary = summary[:max_length-3] + "..."
        
        return summary
    
    def export_corpus_statistics(self) -> Dict[str, any]:
        """
        Export comprehensive statistics about the corpus.
        
        Returns:
            Dictionary with corpus statistics
        """
        stats = {
            'total_events': len(self.corpus.events),
            'events_by_day': {},
            'events_by_gospel_count': {},
            'text_statistics': {
                'total_characters': 0,
                'total_words': 0,
                'average_words_per_event': 0
            },
            'top_keywords': [],
            'top_participants': []
        }
        
        # Count events by day
        day_counts = Counter(event.day for event in self.corpus.events if event.day)
        stats['events_by_day'] = dict(day_counts)
        
        # Count events by number of gospels
        gospel_count_distribution = Counter(
            len(event.participating_gospels) for event in self.corpus.events
        )
        stats['events_by_gospel_count'] = dict(gospel_count_distribution)
        
        # Collect text statistics and keywords
        all_keywords = []
        all_participants = []
        total_words = 0
        
        for event in self.corpus.events:
            event_stats = self.get_text_statistics(event)
            total_words += event_stats['total_words']
            stats['text_statistics']['total_characters'] += event_stats['total_chars']
            
            # Collect keywords and participants
            keywords = self.get_event_keywords(event)
            for keyword_list in keywords.values():
                all_keywords.extend(keyword_list)
            
            participants = self.get_event_participants(event)
            all_participants.extend(participants)
        
        stats['text_statistics']['total_words'] = total_words
        stats['text_statistics']['average_words_per_event'] = (
            total_words / len(self.corpus.events) if self.corpus.events else 0
        )
        
        # Top keywords and participants
        keyword_counts = Counter(all_keywords)
        participant_counts = Counter(all_participants)
        
        stats['top_keywords'] = keyword_counts.most_common(20)
        stats['top_participants'] = participant_counts.most_common(15)
        
        return stats
