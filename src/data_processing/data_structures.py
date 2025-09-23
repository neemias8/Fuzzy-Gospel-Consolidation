"""
Core data structures for the Fuzzy Gospel Consolidation project.

This module defines the main data classes used throughout the system:
- VerseReference: Represents biblical verse references (e.g., "21:1-7")
- Event: Represents a single event from the chronology
- GospelCorpus: Container for all events and gospel texts
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
import logging

logger = logging.getLogger(__name__)


@dataclass
class VerseReference:
    """Represents a biblical verse reference like '21:1-7'"""
    
    chapter: int
    verse_start: int
    verse_end: int
    
    @classmethod
    def parse(cls, ref_str: str) -> Optional['VerseReference']:
        """
        Parse a verse reference string into a VerseReference object.
        
        Args:
            ref_str: String like "21:1-7" or "21:5"
            
        Returns:
            VerseReference object or None if parsing fails
            
        Examples:
            >>> VerseReference.parse("21:1-7")
            VerseReference(chapter=21, verse_start=1, verse_end=7)
            >>> VerseReference.parse("21:5")
            VerseReference(chapter=21, verse_start=5, verse_end=5)
        """
        if not ref_str or not ref_str.strip():
            return None
            
        # Pattern: chapter:verse or chapter:verse-verse
        pattern = r'(\d+):(\d+)(?:-(\d+))?'
        match = re.match(pattern, ref_str.strip())
        
        if not match:
            logger.warning(f"Could not parse verse reference: {ref_str}")
            return None
            
        chapter = int(match.group(1))
        verse_start = int(match.group(2))
        verse_end = int(match.group(3)) if match.group(3) else verse_start
        
        return cls(chapter, verse_start, verse_end)
    
    def __str__(self) -> str:
        """String representation of the verse reference"""
        if self.verse_start == self.verse_end:
            return f"{self.chapter}:{self.verse_start}"
        return f"{self.chapter}:{self.verse_start}-{self.verse_end}"
    
    def contains_verse(self, chapter: int, verse: int) -> bool:
        """Check if this reference contains a specific verse"""
        return (self.chapter == chapter and 
                self.verse_start <= verse <= self.verse_end)


@dataclass
class Event:
    """Represents a single event from the Gospel chronology"""
    
    id: int
    day: str
    description: str
    when_where: str
    gospel_refs: Dict[str, Optional[VerseReference]]
    texts: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize empty texts dict if not provided"""
        if not self.texts:
            self.texts = {}
    
    @property
    def participating_gospels(self) -> Set[str]:
        """Get set of gospels that mention this event"""
        return {gospel for gospel, ref in self.gospel_refs.items() if ref is not None}
    
    @property
    def has_text(self) -> bool:
        """Check if event has extracted text from any gospel"""
        return any(text.strip() for text in self.texts.values())
    
    def get_combined_text(self, separator: str = " ") -> str:
        """Get all texts combined into a single string"""
        valid_texts = [text for text in self.texts.values() if text.strip()]
        return separator.join(valid_texts)
    
    def get_text_for_gospel(self, gospel: str) -> str:
        """Get text for a specific gospel, empty string if not available"""
        return self.texts.get(gospel, "")
    
    def __str__(self) -> str:
        """String representation of the event"""
        gospels = ", ".join(self.participating_gospels)
        return f"Event {self.id}: {self.description} ({gospels})"


@dataclass
class FuzzyRelation:
    """Represents fuzzy relationships between two events"""
    
    mu_same: float  # Membership degree for "same event"
    mu_conflict: float  # Membership degree for "conflicting accounts"
    mu_before: float  # Membership degree for "event1 before event2"
    mu_after: float  # Membership degree for "event1 after event2"
    
    def __post_init__(self):
        """Validate membership degrees are in [0, 1]"""
        for attr_name in ['mu_same', 'mu_conflict', 'mu_before', 'mu_after']:
            value = getattr(self, attr_name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{attr_name} must be in [0, 1], got {value}")
    
    def to_edge_features(self) -> List[float]:
        """Convert to list of features for GNN edge attributes"""
        return [self.mu_same, self.mu_conflict, self.mu_before, self.mu_after]
    
    def dominant_relation(self) -> str:
        """Get the dominant relationship type"""
        relations = {
            'same': self.mu_same,
            'conflict': self.mu_conflict,
            'before': self.mu_before,
            'after': self.mu_after
        }
        return max(relations, key=relations.get)


class GospelCorpus:
    """Container for all events and gospel texts"""
    
    def __init__(self):
        self.events: List[Event] = []
        self.gospel_texts: Dict[str, Dict[int, Dict[int, str]]] = {}
        # Structure: gospel -> chapter -> verse -> text
        self._event_index: Dict[int, Event] = {}
    
    def add_event(self, event: Event) -> None:
        """Add an event to the corpus"""
        self.events.append(event)
        self._event_index[event.id] = event
        logger.debug(f"Added event {event.id}: {event.description}")
    
    def get_event_by_id(self, event_id: int) -> Optional[Event]:
        """Get event by ID"""
        return self._event_index.get(event_id)
    
    def add_gospel_text(self, gospel_name: str, chapter: int, verse: int, text: str) -> None:
        """Add verse text to the corpus"""
        if gospel_name not in self.gospel_texts:
            self.gospel_texts[gospel_name] = {}
        if chapter not in self.gospel_texts[gospel_name]:
            self.gospel_texts[gospel_name][chapter] = {}
        
        self.gospel_texts[gospel_name][chapter][verse] = text
    
    def extract_text_for_event(self, event: Event, gospel: str) -> str:
        """Extract text for an event from a specific gospel"""
        ref = event.gospel_refs.get(gospel)
        if not ref or gospel not in self.gospel_texts:
            return ""
        
        if ref.chapter not in self.gospel_texts[gospel]:
            logger.warning(f"Chapter {ref.chapter} not found in {gospel}")
            return ""
        
        chapter_texts = self.gospel_texts[gospel][ref.chapter]
        verses = []
        
        for verse_num in range(ref.verse_start, ref.verse_end + 1):
            if verse_num in chapter_texts:
                verses.append(chapter_texts[verse_num])
            else:
                logger.warning(f"Verse {gospel} {ref.chapter}:{verse_num} not found")
        
        return " ".join(verses)
    
    def extract_all_event_texts(self) -> None:
        """Extract texts for all events from all available gospels"""
        logger.info("Extracting texts for all events...")
        
        for event in self.events:
            for gospel in ['matthew', 'mark', 'luke', 'john']:
                text = self.extract_text_for_event(event, gospel)
                if text:
                    event.texts[gospel] = text
        
        logger.info(f"Text extraction complete. {len(self.events)} events processed.")
    
    def get_events_by_day(self, day: str) -> List[Event]:
        """Get all events for a specific day"""
        return [event for event in self.events if event.day == day]
    
    def get_events_with_gospel(self, gospel: str) -> List[Event]:
        """Get all events that appear in a specific gospel"""
        return [event for event in self.events 
                if gospel in event.participating_gospels]
    
    def get_statistics(self) -> Dict[str, int]:
        """Get corpus statistics"""
        stats = {
            'total_events': len(self.events),
            'events_with_text': sum(1 for event in self.events if event.has_text),
            'gospels_loaded': len(self.gospel_texts)
        }
        
        # Events per gospel
        for gospel in ['matthew', 'mark', 'luke', 'john']:
            stats[f'events_in_{gospel}'] = len(self.get_events_with_gospel(gospel))
        
        # Events per day
        days = set(event.day for event in self.events)
        for day in days:
            if day:  # Skip empty days
                stats[f'events_on_{day.lower().replace(" ", "_")}'] = len(
                    self.get_events_by_day(day)
                )
        
        return stats
    
    def __len__(self) -> int:
        """Number of events in the corpus"""
        return len(self.events)
    
    def __iter__(self):
        """Iterate over events"""
        return iter(self.events)
    
    def __getitem__(self, index: int) -> Event:
        """Get event by index"""
        return self.events[index]
