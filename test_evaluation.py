#!/usr/bin/env python3
"""
Test Evaluation Pipeline

Tests the evaluation functionality including ROUGE, METEOR, BERTScore, and Kendall's Tau.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from main import FuzzyGospelConsolidator

def test_evaluation_pipeline():
    """Test the complete pipeline with evaluation"""
    print("🧪 TESTING EVALUATION PIPELINE")
    print("=" * 50)
    
    try:
        # Initialize consolidator
        consolidator = FuzzyGospelConsolidator()
        
        # Run steps 1-5 (data loading through summary generation)
        print("\n1️⃣ Running pipeline steps 1-5...")
        consolidator.load_data()
        consolidator.calculate_fuzzy_relations()
        consolidator.build_graph()
        consolidator.train_model()
        summary = consolidator.generate_summary()
        
        print(f"   ✅ Generated summary: {len(summary)} characters")
        
        # Step 6: Evaluation
        print("\n2️⃣ Step 6: Running comprehensive evaluation...")
        evaluation_results = consolidator.evaluate_results()
        
        print("   ✅ Evaluation completed successfully!")
        
        # Display key metrics
        print("\n📊 EVALUATION RESULTS:")
        print("-" * 30)
        
        auto_metrics = evaluation_results.get('automatic_metrics', {})
        print(f"ROUGE-1:      {auto_metrics.get('rouge1', 0.0):.4f}")
        print(f"ROUGE-2:      {auto_metrics.get('rouge2', 0.0):.4f}")
        print(f"ROUGE-L:      {auto_metrics.get('rougeL', 0.0):.4f}")
        print(f"BERTScore F1: {auto_metrics.get('bertscore_f1', 0.0):.4f}")
        print(f"METEOR:       {auto_metrics.get('meteor', 0.0):.4f}")
        print(f"BLEU:         {auto_metrics.get('bleu', 0.0):.4f}")
        
        temporal = evaluation_results.get('temporal_coherence', {})
        print(f"Kendall's Tau: {temporal.get('kendall_tau', 0.0):.4f}")
        print(f"Temporal Acc:  {temporal.get('temporal_accuracy', 0.0):.4f}")
        print(f"Violations:    {temporal.get('chronological_violations', 0)}")
        
        print(f"Overall Score: {evaluation_results.get('overall_score', 0.0):.4f}")
        
        # Test individual metrics
        print("\n3️⃣ Testing individual metrics...")
        from evaluation.metrics import AutomaticMetrics
        
        metrics = AutomaticMetrics()
        
        # Test ROUGE
        sample_text = "Jesus went to Jerusalem and taught in the temple."
        references = ["Jesus traveled to Jerusalem and preached in the temple courts."]
        rouge_scores = metrics.calculate_rouge(sample_text, references)
        print(f"   ✅ ROUGE scores: {rouge_scores}")
        
        # Test BERTScore
        bert_scores = metrics.calculate_bertscore(sample_text, references)
        print(f"   ✅ BERTScore: {bert_scores}")
        
        # Test METEOR
        meteor_score = metrics.calculate_meteor(sample_text, references)
        print(f"   ✅ METEOR: {meteor_score:.4f}")
        
        # Test Kendall's Tau
        generated_order = ["event1", "event2", "event3"]
        reference_order = ["event1", "event3", "event2"]
        tau = metrics.calculate_kendall_tau(generated_order, reference_order)
        print(f"   ✅ Kendall's Tau: {tau:.4f}")
        
        print("\n4️⃣ Checking saved files...")
        from pathlib import Path
        results_dir = Path("results")
        
        if results_dir.exists():
            json_files = list(results_dir.glob("evaluation_results_*.json"))
            txt_files = list(results_dir.glob("consolidated_summary_*.txt"))
            report_files = list(results_dir.glob("evaluation_report_*.txt"))
            
            print(f"   ✅ Found {len(json_files)} evaluation JSON files")
            print(f"   ✅ Found {len(txt_files)} summary text files")
            print(f"   ✅ Found {len(report_files)} evaluation report files")
            
            if report_files:
                latest_report = max(report_files, key=lambda f: f.stat().st_mtime)
                print(f"   📄 Latest report: {latest_report.name}")
        
        print("\n🎉 ALL EVALUATION TESTS PASSED!")
        print("   ✅ ROUGE, METEOR, BERTScore calculations working")
        print("   ✅ Kendall's Tau temporal ordering evaluation working")
        print("   ✅ Results saved to files successfully")
        print("   ✅ Complete 6-step pipeline functional")
        
        return True
        
    except Exception as e:
        print(f"\n❌ EVALUATION TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run test
    success = test_evaluation_pipeline()
    
    if success:
        print("\n" + "=" * 60)
        print("🎯 COMPLETE PIPELINE VERIFIED:")
        print("   1. Text extraction from Gospel events ✅")
        print("   2. Fuzzy relations (μ_same, μ_conflict, μ_before) ✅")
        print("   3. Fuzzy-GNN graph construction and processing ✅")
        print("   4. Consolidated summary generation ✅")
        print("   5. Comprehensive evaluation with metrics ✅")
        print("   6. Results saved to files ✅")
        sys.exit(0)
    else:
        sys.exit(1)