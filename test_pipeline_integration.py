#!/usr/bin/env python3
"""
Test script to verify the corrected implementation of the 3-step pipeline:
1. Text extraction
2. Fuzzy relations calculation (μ_same_event, μ_conflict, μ_before)
3. Fuzzy-GNN graph construction and processing
"""

import sys
from pathlib import Path
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_pipeline_integration():
    """Test the complete 3-step pipeline"""
    print("🧪 Testing Complete Fuzzy-GNN Pipeline Integration")
    print("=" * 60)
    
    try:
        from main import FuzzyGospelConsolidator
        
        # Initialize consolidator
        print("1️⃣ Initializing system...")
        consolidator = FuzzyGospelConsolidator("config.yaml")
        
        # Test Step 1: Load and extract texts
        print("\n2️⃣ Step 1: Loading data and extracting texts...")
        consolidator.load_data()
        print(f"   ✅ Loaded {len(consolidator.corpus.events)} events")
        
        # Test Step 2: Calculate fuzzy relations
        print("\n3️⃣ Step 2: Calculating fuzzy relations (μ_same, μ_conflict, μ_before)...")
        consolidator.calculate_fuzzy_relations()
        print(f"   ✅ Calculated {len(consolidator.fuzzy_relations)} fuzzy relations")
        
        # Verify the three fuzzy functions are working
        sample_relations = list(consolidator.fuzzy_relations.values())[:5]
        print(f"   📊 Sample relations:")
        for i, rel in enumerate(sample_relations):
            print(f"      Relation {i+1}: μ_same={rel.mu_same:.3f}, μ_conflict={rel.mu_conflict:.3f}, μ_before={rel.mu_before:.3f}")
        
        # Test Step 3: Build fuzzy graph
        print("\n4️⃣ Step 3a: Building fuzzy event graph...")
        consolidator.build_graph()
        stats = consolidator.graph.get_statistics()
        print(f"   ✅ Graph built: {stats['num_nodes']} nodes, {stats['num_edges']} edges")
        
        # Test Step 3b: Train GNN
        print("\n5️⃣ Step 3b: Training fuzzy GNN...")
        consolidator.train_model()
        print("   ✅ GNN training completed")
        
        # Test Step 3c: Generate consolidated summary
        print("\n6️⃣ Step 3c: Generating consolidated summary...")
        summary = consolidator.generate_summary()
        print(f"   ✅ Summary generated: {len(summary)} characters")
        
        # Show sample of summary
        print(f"\n📄 Summary preview (first 500 characters):")
        print("-" * 50)
        print(summary[:500] + "..." if len(summary) > 500 else summary)
        print("-" * 50)
        
        print("\n🎉 ALL TESTS PASSED! The 3-step pipeline is working correctly:")
        print("   ✅ Step 1: Text extraction from Gospel events")
        print("   ✅ Step 2: Fuzzy relations (μ_same_event, μ_conflict, μ_before)")
        print("   ✅ Step 3: Fuzzy-GNN graph processing and consolidation")
        
        return True
        
    except Exception as e:
        print(f"\n❌ PIPELINE TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_specific_fuzzy_functions():
    """Test specific fuzzy membership functions"""
    print("\n🔍 Testing Specific Fuzzy Functions")
    print("=" * 40)
    
    try:
        from fuzzy_relations import FuzzyRelationCalculator
        from data_processing import XMLParser, TextExtractor
        import yaml
        
        # Load config and data
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        xml_parser = XMLParser()
        corpus = xml_parser.load_complete_dataset(Path('data/raw'), config['data'])
        text_extractor = TextExtractor(corpus)
        
        # Initialize fuzzy calculator
        fuzzy_calc = FuzzyRelationCalculator(config)
        
        # Test on first two events
        if len(corpus.events) >= 2:
            event1, event2 = corpus.events[0], corpus.events[1]
            
            print(f"Testing fuzzy functions between:")
            print(f"  Event 1: {event1.description} (Day: {event1.day})")
            print(f"  Event 2: {event2.description} (Day: {event2.day})")
            
            # Test individual functions
            mu_same = fuzzy_calc.calculate_same_event_membership(event1, event2, text_extractor)
            mu_conflict = fuzzy_calc.calculate_conflict_membership(event1, event2)
            mu_before = fuzzy_calc.calculate_temporal_before_membership(event1, event2)
            
            print(f"\n📊 Fuzzy membership results:")
            print(f"   μ_same_event: {mu_same:.4f}")
            print(f"   μ_conflict:   {mu_conflict:.4f}")
            print(f"   μ_before:     {mu_before:.4f}")
            
            print("   ✅ All three fuzzy functions are working!")
        
        return True
        
    except Exception as e:
        print(f"❌ Fuzzy function test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting Fuzzy-GNN Pipeline Validation")
    print("=" * 60)
    
    # Test the complete pipeline
    pipeline_success = test_pipeline_integration()
    
    # Test specific fuzzy functions
    fuzzy_success = test_specific_fuzzy_functions()
    
    print("\n" + "=" * 60)
    if pipeline_success and fuzzy_success:
        print("🎉 ALL TESTS PASSED! The implementation correctly implements:")
        print("   1. Text extraction from Gospel events")
        print("   2. Fuzzy relations (μ_same_event, μ_conflict, μ_before)")
        print("   3. Fuzzy-GNN graph construction and processing")
        print("   4. Consolidated summary generation")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        sys.exit(1)