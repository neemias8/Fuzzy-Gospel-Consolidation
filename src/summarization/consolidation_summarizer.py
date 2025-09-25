"""
Consolidation Summarizer

This module implements the main summarization logic that combines
fuzzy relations and GNN outputs to generate consolidated narratives.
"""

import logging
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

logger = logging.getLogger(__name__)


class ConsolidationSummarizer:
    """Main summarizer for consolidating Gospel narratives"""
    
    def __init__(self, model_name: str):
        """
        Initialize the consolidation summarizer.
        
        Args:
            model_name: Name or path of the summarization model
        """
        self.model_name = model_name
        
        # Initialize the text generation model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        logger.info(f"ConsolidationSummarizer initialized with {model_name}")
    
    def generate_summary_from_clusters(self, clusters: Dict[int, Dict], use_gnn_embeddings: bool = True):
        """
        Generate summary from simple clusters (for ablation test)
        
        Args:
            clusters: Dictionary mapping cluster_id -> {'events': [...], 'embeddings': [...]}
            use_gnn_embeddings: Whether to use GNN embeddings (False for ablation)
            
        Returns:
            Consolidated summary text
        """
        logger.info(f"Generating summary from {len(clusters)} clusters (GNN: {use_gnn_embeddings})")
        
        summary_parts = []
        
        for cluster_id, cluster_data in clusters.items():
            events = cluster_data['events']
            if not events:
                continue
                
            # Generate summary for this cluster
            cluster_summary = self._generate_simple_cluster_summary(events)
            if cluster_summary:
                summary_parts.append(cluster_summary)
        
        # Combine all cluster summaries
        full_summary = "\n\n".join(summary_parts)
        
        logger.info(f"Summary generated: {len(full_summary)} characters from {len(clusters)} clusters")
        return full_summary
    
    def generate_consolidated_summary(self, events, graph, model):
        """
        Generate a consolidated summary of the Gospel events.
        
        Args:
            events: List of Event objects
            graph: FuzzyEventGraph object
            model: Trained FuzzyGNN model
            
        Returns:
            Consolidated summary text
        """
        logger.info("Generating consolidated summary using GNN embeddings...")
        
        # Step 1: Get event embeddings from trained GNN
        graph_data = graph.get_pytorch_data()
        event_embeddings = model.get_embeddings(graph_data)
        
        # Step 2: Cluster similar events based on embeddings and fuzzy relations
        event_clusters = self._cluster_events(events, event_embeddings, graph.fuzzy_relations)
        
        # Step 3: Generate consolidated narrative
        summary_parts = []
        
        # Step 4: Organize clusters by chronological order
        chronological_clusters = self._order_clusters_chronologically(event_clusters)
        
        # Step 5: Generate text for each cluster
        for cluster_info in chronological_clusters:
            cluster_summary = self._generate_cluster_summary(cluster_info)
            if cluster_summary:
                summary_parts.append(cluster_summary)
        
        # Combine all parts
        consolidated_summary = "\\n\\n".join(summary_parts)
        
        logger.info(f"Summary generated: {len(consolidated_summary)} characters from {len(chronological_clusters)} event clusters")
        return consolidated_summary
    
    def _cluster_events(self, events, embeddings, fuzzy_relations):
        """
        Cluster events based on GNN embeddings and fuzzy relations.
        
        Args:
            events: List of Event objects
            embeddings: Tensor of event embeddings from GNN
            fuzzy_relations: Dictionary of fuzzy relations
            
        Returns:
            List of event clusters with metadata
        """
        import torch
        from sklearn.cluster import AgglomerativeClustering
        import numpy as np
        
        # Convert embeddings to numpy for clustering
        embeddings_np = embeddings.detach().numpy()
        
        # Determine number of clusters (roughly one per day with some splits)
        n_clusters = min(15, len(events) // 3)  # Adaptive clustering
        
        # Perform hierarchical clustering
        clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        cluster_labels = clustering.fit_predict(embeddings_np)
        
        # Group events by cluster
        clusters = {}
        for idx, (event, label) in enumerate(zip(events, cluster_labels)):
            if label not in clusters:
                clusters[label] = {
                    'events': [],
                    'embeddings': [],
                    'same_relations': [],
                    'conflicts': []
                }
            clusters[label]['events'].append(event)
            clusters[label]['embeddings'].append(embeddings[idx])
        
        # Analyze relations within each cluster
        for cluster_id, cluster_info in clusters.items():
            cluster_events = cluster_info['events']
            
            # Find high-similarity pairs (same events)
            for i, event1 in enumerate(cluster_events):
                for j, event2 in enumerate(cluster_events[i+1:], i+1):
                    relation_key = (event1.id, event2.id)
                    reverse_key = (event2.id, event1.id)
                    
                    relation = fuzzy_relations.get(relation_key) or fuzzy_relations.get(reverse_key)
                    if relation:
                        if relation.mu_same > 0.7:
                            cluster_info['same_relations'].append((event1, event2, relation.mu_same))
                        if relation.mu_conflict > 0.3:
                            cluster_info['conflicts'].append((event1, event2, relation.mu_conflict))
        
        return list(clusters.values())
    
    def _order_clusters_chronologically(self, clusters):
        """Order clusters chronologically based on events' days and IDs"""
        day_order = {
            'Saturday': 0, 'Palm Sunday': 1, 'Monday': 2, 'Tuesday': 3,
            'Wednesday': 4, 'Thursday': 5, 'Friday': 6, 'Saturday ': 7, 
            'Sunday': 8, 'Resurrection': 8, 'Easter': 8
        }
        
        def get_cluster_order(cluster):
            events = cluster['events']
            # Use the earliest event's day and ID as the cluster's position
            min_day_order = float('inf')
            min_event_id = float('inf')
            
            for event in events:
                day_num = day_order.get(event.day, 999)
                min_day_order = min(min_day_order, day_num)
                min_event_id = min(min_event_id, event.id)
            
            return (min_day_order, min_event_id)
        
        return sorted(clusters, key=get_cluster_order)
    
    def _generate_cluster_summary(self, cluster_info):
        """
        Generate summary text for a cluster of related events.
        
        Args:
            cluster_info: Dictionary with cluster data
            
        Returns:
            Summary text for the cluster
        """
        events = cluster_info['events']
        same_relations = cluster_info['same_relations']
        conflicts = cluster_info['conflicts']
        
        if not events:
            return ""
        
        # Determine the primary day and context
        primary_event = min(events, key=lambda e: e.id)
        day = primary_event.day or "During the Passion Week"
        
        summary_parts = []
        
        # Handle same events (merge similar accounts)  
        if same_relations:
            # Group highly similar events
            merged_text = self._merge_similar_events(same_relations)
            summary_parts.append(merged_text)
        
        # Handle conflicts (present alternatives)
        if conflicts:
            conflict_text = self._present_conflicting_accounts(conflicts)
            summary_parts.append(conflict_text)
        
        # Handle unique events in cluster
        unique_events = self._get_unique_events(events, same_relations)
        for event in unique_events:
            event_text = self._format_single_event(event)
            summary_parts.append(event_text)
        
        if summary_parts:
            cluster_summary = f"## {day}\n\n" + "\n\n".join(summary_parts) + "\n"
            return cluster_summary
        
        return ""
    
    def _merge_similar_events(self, same_relations):
        """Merge similar events into a consolidated description"""
        if not same_relations:
            return ""
        
        # Take the pair with highest similarity
        best_pair = max(same_relations, key=lambda x: x[2])
        event1, event2, similarity = best_pair
        
        # Combine information from both events
        combined_gospels = list(set(list(event1.participating_gospels) + list(event2.participating_gospels)))
        gospel_str = ", ".join(combined_gospels)
        
        # Use the description from the event with more gospels
        if len(event1.participating_gospels) >= len(event2.participating_gospels):
            primary_desc = event1.description
            primary_texts = event1.texts
        else:
            primary_desc = event2.description
            primary_texts = event2.texts
        
        # Create merged narrative
        merged_text = f"**{primary_desc}** (reported in {gospel_str})\n\n"
        
        # Add the richest text available
        if primary_texts:
            best_gospel = max(primary_texts.keys(), key=lambda k: len(primary_texts[k]))
            merged_text += f"{primary_texts[best_gospel]}"
        
        return merged_text
    
    def _present_conflicting_accounts(self, conflicts):
        """Present conflicting accounts as alternatives"""
        if not conflicts:
            return ""
        
        conflict_texts = []
        for event1, event2, conflict_score in conflicts:
            conflict_text = f"**Alternative Accounts:**\n\n"
            conflict_text += f"*According to {', '.join(event1.participating_gospels)}:* {event1.description}\n\n"
            conflict_text += f"*According to {', '.join(event2.participating_gospels)}:* {event2.description}\n"
            conflict_texts.append(conflict_text)
        
        return "\n\n".join(conflict_texts)
    
    def _get_unique_events(self, events, same_relations):
        """Get events that are not part of same_relations pairs"""
        paired_events = set()
        for event1, event2, _ in same_relations:
            paired_events.add(event1.id)
            paired_events.add(event2.id)
        
        return [event for event in events if event.id not in paired_events]
    
    def _format_single_event(self, event):
        """Format a single event for the summary"""
        gospel_str = ", ".join(event.participating_gospels)
        text = f"**{event.description}** ({gospel_str})\n\n"
        
        # Add gospel text if available
        if event.texts:
            best_gospel = max(event.texts.keys(), key=lambda k: len(event.texts[k]))
            text += f"{event.texts[best_gospel]}"
        
        return text
    
    def _generate_day_summary(self, day: str, events: List) -> str:
        """
        Generate summary for events on a specific day.
        
        Args:
            day: Day name
            events: List of events for that day
            
        Returns:
            Summary text for the day
        """
        if not events:
            return ""
        
        # Create a simple narrative for the day
        day_text = f"**{day}**\\n\\n"
        
        # Combine event descriptions and texts
        event_texts = []
        for event in events[:5]:  # Limit to first 5 events per day
            if event.description:
                event_texts.append(event.description)
            
            # Add text from gospels if available
            for gospel, text in event.texts.items():
                if text and len(text) > 20:  # Only include substantial texts
                    # Truncate long texts
                    if len(text) > 200:
                        text = text[:200] + "..."
                    event_texts.append(f"({gospel.title()}: {text})")
        
        if event_texts:
            day_text += " ".join(event_texts[:3])  # Limit to avoid too long summaries
        
        return day_text
    
    def _generate_simple_cluster_summary(self, events):
        """
        Generate summary for a cluster WITHOUT using GNN embeddings or fuzzy relations
        Simple approach for ablation test
        """
        if not events:
            return ""
        
        # Group events by day
        day_groups = {}
        for event in events:
            day = getattr(event, 'day', 'Unknown Day')
            if day == 'Unknown Day' and hasattr(event, 'description'):
                day = event.description[:30] + "..." if len(event.description) > 30 else event.description
            
            if day not in day_groups:
                day_groups[day] = []
            day_groups[day].append(event)
        
        summary_parts = []
        
        for day, day_events in day_groups.items():
            # Sort events by ID to maintain some order
            day_events.sort(key=lambda e: getattr(e, 'id', 0))
            
            # Create simple summary for this day
            day_summary = f"## {day}\n\n"
            
            event_descriptions = []
            for event in day_events[:3]:  # Limit to 3 events per day to avoid too long summaries
                if hasattr(event, 'description') and event.description:
                    # Get participating gospels
                    gospels = getattr(event, 'participating_gospels', ['Unknown Gospel'])
                    gospel_str = ', '.join(gospels)
                    
                    event_text = f"**{event.description}** (reported in {gospel_str})"
                    
                    # Add actual text from one gospel if available
                    if hasattr(event, 'texts') and event.texts:
                        # Get text from first available gospel
                        first_gospel = list(event.texts.keys())[0]
                        text = event.texts[first_gospel]
                        if text and len(text.strip()) > 10:
                            # Truncate long texts
                            if len(text) > 150:
                                text = text[:150] + "..."
                            event_text += f"\n\n{text}"
                    
                    event_descriptions.append(event_text)
            
            if event_descriptions:
                day_summary += "\n\n".join(event_descriptions)
                summary_parts.append(day_summary)
        
        return "\n\n".join(summary_parts) if summary_parts else ""
