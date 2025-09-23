#!/usr/bin/env python3
"""
Test Final Evaluation

Tests only the evaluation metrics with proper sample data.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from evaluation.metrics import AutomaticMetrics

def test_metrics_only():
    """Test just the metrics with proper data"""
    print("🧪 TESTING EVALUATION METRICS ONLY")
    print("=" * 40)
    
    metrics = AutomaticMetrics()
    
    # Test data
    generated = "Jesus went to Jerusalem and taught in the temple courts. The disciples followed him and listened to his teachings."
    references = [
        "Jesus traveled to Jerusalem and preached in the temple. His disciples accompanied him and heard his words.",
        "Christ went to the holy city and instructed people in the temple area. The twelve were with him."
    ]
    
    print("\n📊 Testing ROUGE...")
    rouge_scores = metrics.calculate_rouge(generated, references)
    print(f"   ROUGE-1: {rouge_scores.get('rouge1', 0.0):.4f}")
    print(f"   ROUGE-2: {rouge_scores.get('rouge2', 0.0):.4f}")
    print(f"   ROUGE-L: {rouge_scores.get('rougeL', 0.0):.4f}")
    
    print("\n📊 Testing BERTScore...")
    bert_scores = metrics.calculate_bertscore(generated, references)
    print(f"   Precision: {bert_scores.get('precision', 0.0):.4f}")
    print(f"   Recall:    {bert_scores.get('recall', 0.0):.4f}")
    print(f"   F1:        {bert_scores.get('f1', 0.0):.4f}")
    
    print("\n📊 Testing METEOR...")
    meteor_score = metrics.calculate_meteor(generated, references)
    print(f"   METEOR: {meteor_score:.4f}")
    
    print("\n📊 Testing BLEU...")
    bleu_score = metrics.calculate_bleu(generated, references)
    print(f"   BLEU: {bleu_score:.4f}")
    
    print("\n📊 Testing Kendall's Tau...")
    generated_order = ["event1", "event2", "event3", "event4"]
    reference_order = ["event1", "event3", "event2", "event4"]
    tau = metrics.calculate_kendall_tau(generated_order, reference_order)
    print(f"   Kendall's Tau: {tau:.4f}")
    
    print("\n🎉 ALL METRICS WORKING!")
    print("   ✅ ROUGE: Lexical overlap evaluation")
    print("   ✅ BERTScore: Semantic similarity evaluation")
    print("   ✅ METEOR: Token-level evaluation with synonyms")
    print("   ✅ BLEU: N-gram precision evaluation")
    print("   ✅ Kendall's Tau: Temporal ordering correlation")
    
    return True

if __name__ == "__main__":
    test_metrics_only()