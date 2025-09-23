#!/usr/bin/env python3
"""
Debug Evaluation

Debugs why metrics are returning zero.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from main import FuzzyGospelConsolidator

def debug_evaluation():
    """Debug the evaluation process"""
    print("🔍 DEBUGGING EVALUATION METRICS")
    print("=" * 40)
    
    # Initialize consolidator
    consolidator = FuzzyGospelConsolidator()
    
    # Load data
    print("\n1️⃣ Loading data...")
    consolidator.load_data()
    print(f"   Loaded {len(consolidator.corpus.events)} events")
    
    # Check event structure
    print("\n2️⃣ Examining event structure...")
    sample_event = consolidator.corpus.events[0]
    print(f"   Sample event attributes: {dir(sample_event)}")
    
    if hasattr(sample_event, 'extracted_text'):
        print(f"   Has extracted_text: {sample_event.extracted_text}")
    else:
        print("   ❌ No extracted_text attribute")
    
    if hasattr(sample_event, 'text'):
        print(f"   Has text: {sample_event.text[:100] if sample_event.text else 'None'}...")
    else:
        print("   ❌ No text attribute")
    
    # Check what text fields are available
    text_fields = []
    for attr in dir(sample_event):
        if 'text' in attr.lower() and not attr.startswith('_'):
            text_fields.append(attr)
    print(f"   Available text fields: {text_fields}")
    
    # Create references manually
    print("\n3️⃣ Creating references...")
    references = []
    
    # Try different approaches
    for event in consolidator.corpus.events[:5]:  # Test first 5 events
        print(f"   Event: {getattr(event, 'title', 'No title')}")
        
        # Check extracted_text
        if hasattr(event, 'extracted_text') and event.extracted_text:
            print(f"     extracted_text: {type(event.extracted_text)} with {len(event.extracted_text)} items")
            references.extend(event.extracted_text.values())
        
        # Check raw text
        if hasattr(event, 'text') and event.text:
            print(f"     text: {len(event.text)} chars")
            references.append(event.text)
        
        # Check gospel-specific texts
        for gospel in ['matthew', 'mark', 'luke', 'john']:
            if hasattr(event, gospel) and getattr(event, gospel):
                gospel_text = getattr(event, gospel)
                if isinstance(gospel_text, str):
                    print(f"     {gospel}: {len(gospel_text)} chars")
                    references.append(gospel_text)
    
    print(f"\n   Total references collected: {len(references)}")
    if references:
        print(f"   Sample reference: {references[0][:200]}...")
    
    # Generate a small summary for testing
    print("\n4️⃣ Testing with sample texts...")
    test_summary = "Jesus went to Jerusalem and taught in the temple. The disciples followed him."
    
    if references:
        from evaluation.metrics import AutomaticMetrics
        metrics = AutomaticMetrics()
        
        # Test ROUGE
        rouge_scores = metrics.calculate_rouge(test_summary, references[:3])  # Use first 3 refs
        print(f"   ROUGE scores: {rouge_scores}")
        
        # Test BERTScore
        bert_scores = metrics.calculate_bertscore(test_summary, references[:3])
        print(f"   BERTScore: {bert_scores}")
        
        # Test METEOR
        meteor_score = metrics.calculate_meteor(test_summary, references[:3])
        print(f"   METEOR: {meteor_score}")
    else:
        print("   ❌ No references to test with!")
    
    # Test temporal ordering
    print("\n5️⃣ Testing temporal ordering...")
    
    # Check if events have titles
    event_titles = []
    for event in consolidator.corpus.events[:10]:
        title = getattr(event, 'title', None)
        if title:
            event_titles.append(title)
            print(f"   Event title: {title}")
    
    print(f"   Found {len(event_titles)} event titles")

if __name__ == "__main__":
    debug_evaluation()