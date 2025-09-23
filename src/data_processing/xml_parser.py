"""
XML Parser for Gospel texts and chronology data.

This module handles parsing of the XML files containing:
- Gospel texts (Matthew, Mark, Luke, John)
- Chronology of events during Holy Week
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional
import logging

from .data_structures import Event, VerseReference, GospelCorpus

logger = logging.getLogger(__name__)


class XMLParser:
    """Parser for Gospel XML files and chronology data"""
    
    def __init__(self):
        self.corpus = GospelCorpus()
    
    def parse_chronology(self, chronology_file: Path) -> None:
        """
        Parse the chronology XML file to extract events.
        
        Args:
            chronology_file: Path to ChronologyOfTheFourGospels_PW.xml
        """
        logger.info(f"Parsing chronology file: {chronology_file}")
        
        try:
            tree = ET.parse(chronology_file)
            root = tree.getroot()
            
            events_element = root.find('events')
            if events_element is None:
                raise ValueError("No 'events' element found in chronology XML")
            
            event_count = 0
            for event_elem in events_element.findall('event'):
                event = self._parse_event_element(event_elem)
                if event:
                    self.corpus.add_event(event)
                    event_count += 1
            
            logger.info(f"Successfully parsed {event_count} events from chronology")
            
        except ET.ParseError as e:
            logger.error(f"XML parsing error in chronology file: {e}")
            raise
        except Exception as e:
            logger.error(f"Error parsing chronology file: {e}")
            raise
    
    def _parse_event_element(self, event_elem: ET.Element) -> Optional[Event]:
        """Parse a single event element from the chronology XML"""
        try:
            # Get event ID
            event_id = event_elem.get('id')
            if not event_id:
                logger.warning("Event element missing 'id' attribute")
                return None
            event_id = int(event_id)
            
            # Get basic event information
            day = self._get_element_text(event_elem, 'day', '')
            description = self._get_element_text(event_elem, 'description', '')
            when_where = self._get_element_text(event_elem, 'when_where', '')
            
            # Parse gospel references
            gospel_refs = {}
            for gospel in ['matthew', 'mark', 'luke', 'john']:
                ref_text = self._get_element_text(event_elem, gospel, '')
                gospel_refs[gospel] = VerseReference.parse(ref_text)
            
            return Event(
                id=event_id,
                day=day,
                description=description,
                when_where=when_where,
                gospel_refs=gospel_refs
            )
            
        except Exception as e:
            logger.warning(f"Error parsing event element: {e}")
            return None
    
    def _get_element_text(self, parent: ET.Element, tag: str, default: str = '') -> str:
        """Safely get text content from an XML element"""
        element = parent.find(tag)
        if element is not None and element.text:
            return element.text.strip()
        return default
    
    def parse_gospel(self, gospel_file: Path, gospel_name: str) -> None:
        """
        Parse a Gospel XML file to extract verse texts.
        
        Args:
            gospel_file: Path to the Gospel XML file
            gospel_name: Name of the gospel ('matthew', 'mark', 'luke', 'john')
        """
        logger.info(f"Parsing {gospel_name} from: {gospel_file}")
        
        try:
            tree = ET.parse(gospel_file)
            root = tree.getroot()
            
            # Navigate to the book element
            book = root.find('.//book')
            if book is None:
                raise ValueError(f"No 'book' element found in {gospel_name} XML")
            
            verse_count = 0
            for chapter in book.findall('chapter'):
                chapter_num = int(chapter.get('number', 0))
                
                for verse in chapter.findall('verse'):
                    verse_num = int(verse.get('number', 0))
                    verse_text = verse.text or ''
                    
                    if verse_text.strip():
                        self.corpus.add_gospel_text(
                            gospel_name, chapter_num, verse_num, verse_text.strip()
                        )
                        verse_count += 1
            
            logger.info(f"Successfully parsed {verse_count} verses from {gospel_name}")
            
        except ET.ParseError as e:
            logger.error(f"XML parsing error in {gospel_name} file: {e}")
            raise
        except Exception as e:
            logger.error(f"Error parsing {gospel_name} file: {e}")
            raise
    
    def parse_all_gospels(self, data_dir: Path, gospel_files: Dict[str, str]) -> None:
        """
        Parse all Gospel XML files.
        
        Args:
            data_dir: Directory containing the XML files
            gospel_files: Dictionary mapping gospel names to filenames
        """
        for gospel_name, filename in gospel_files.items():
            gospel_path = data_dir / filename
            if gospel_path.exists():
                self.parse_gospel(gospel_path, gospel_name)
            else:
                logger.warning(f"Gospel file not found: {gospel_path}")
    
    def load_complete_dataset(self, data_dir: Path, config: Dict) -> GospelCorpus:
        """
        Load the complete dataset (chronology + all gospels).
        
        Args:
            data_dir: Directory containing XML files
            config: Configuration dictionary with file names
            
        Returns:
            Populated GospelCorpus object
        """
        logger.info("Loading complete Gospel dataset...")
        
        # Parse chronology first
        chronology_file = data_dir / config['chronology_file']
        if chronology_file.exists():
            self.parse_chronology(chronology_file)
        else:
            raise FileNotFoundError(f"Chronology file not found: {chronology_file}")
        
        # Parse all gospels
        self.parse_all_gospels(data_dir, config['gospels'])
        
        # Extract texts for all events
        self.corpus.extract_all_event_texts()
        
        # Log statistics
        stats = self.corpus.get_statistics()
        logger.info("Dataset loading complete:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
        
        return self.corpus
    
    def validate_corpus(self) -> Dict[str, List[str]]:
        """
        Validate the loaded corpus and return any issues found.
        
        Returns:
            Dictionary with validation results
        """
        issues = {
            'warnings': [],
            'errors': []
        }
        
        # Check for events without any text
        events_without_text = [
            event for event in self.corpus.events if not event.has_text
        ]
        if events_without_text:
            issues['warnings'].append(
                f"{len(events_without_text)} events have no extracted text"
            )
        
        # Check for events with only one gospel
        single_gospel_events = [
            event for event in self.corpus.events 
            if len(event.participating_gospels) == 1
        ]
        if single_gospel_events:
            issues['warnings'].append(
                f"{len(single_gospel_events)} events appear in only one gospel"
            )
        
        # Check for missing verse references
        missing_refs = []
        for event in self.corpus.events:
            for gospel, ref in event.gospel_refs.items():
                if ref and gospel in self.corpus.gospel_texts:
                    # Check if the referenced verses exist
                    if ref.chapter not in self.corpus.gospel_texts[gospel]:
                        missing_refs.append(f"Event {event.id}: {gospel} {ref}")
        
        if missing_refs:
            issues['errors'].extend(missing_refs)
        
        return issues
