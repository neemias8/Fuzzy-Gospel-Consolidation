#!/usr/bin/env python3
"""
Script para corrigir o relatório de avaliação com os valores corretos do JSON
"""
import json
from datetime import datetime

# Carrega os resultados JSON
with open('results/evaluation_results_20250925_150440.json', 'r') as f:
    results = json.load(f)

# Gera o relatório corrigido
with open('results/evaluation_report_corrected.txt', 'w', encoding='utf-8') as f:
    f.write("FUZZY GOSPEL CONSOLIDATION - EVALUATION REPORT (CORRIGIDO)\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    # Summary statistics
    f.write("SUMMARY STATISTICS\n")
    f.write("-" * 20 + "\n")
    f.write(f"Summary length: {results.get('summary_length', 0)} characters\n")
    f.write(f"Summary word count: {results.get('summary_word_count', 0)} words\n")
    f.write(f"Overall score: {results.get('overall_score', 0.0):.4f}\n\n")
    
    # Automatic metrics - CORRIGIDO
    auto_metrics = results.get('automatic_metrics', {})
    f.write("AUTOMATIC METRICS\n")
    f.write("-" * 17 + "\n")
    f.write(f"ROUGE-1: {auto_metrics.get('rouge', {}).get('rouge1', 0.0):.4f}\n")
    f.write(f"ROUGE-2: {auto_metrics.get('rouge', {}).get('rouge2', 0.0):.4f}\n")
    f.write(f"ROUGE-L: {auto_metrics.get('rouge', {}).get('rougeL', 0.0):.4f}\n")
    f.write(f"BERTScore F1: {auto_metrics.get('bertscore', {}).get('f1', 0.0):.4f}\n")
    f.write(f"METEOR: {auto_metrics.get('meteor', 0.0):.4f}\n")
    f.write(f"BLEU: {auto_metrics.get('bleu', 0.0):.4f}\n\n")
    
    # Temporal coherence
    temp_coherence = results.get('temporal_coherence', {})
    f.write("TEMPORAL COHERENCE\n")
    f.write("-" * 18 + "\n")
    f.write(f"Kendall's Tau: {temp_coherence.get('kendall_tau', 0.0):.4f}\n")
    f.write(f"Temporal Accuracy: {temp_coherence.get('temporal_accuracy', 0.0):.4f}\n")
    f.write(f"Chronological Violations: {temp_coherence.get('chronological_violations', 0)}\n\n")
    
    # Content coverage
    content_coverage = results.get('content_coverage', {})
    f.write("CONTENT COVERAGE (PLACEHOLDER VALUES)\n")
    f.write("-" * 36 + "\n")
    f.write(f"Event Coverage: {content_coverage.get('event_coverage', 0.0):.4f}\n")
    f.write(f"Gospel Representation: {content_coverage.get('gospel_representation', 0.0):.4f}\n\n")
    
    # Conflict handling (real analysis)
    conflicts = results.get('conflict_handling', {})
    f.write("CONFLICT HANDLING (REAL ANALYSIS)\n")
    f.write("-" * 34 + "\n")
    f.write(f"Conflicts Mentioned (Real): {conflicts.get('conflicts_mentioned', 0)}\n")
    f.write(f"Conflicts Resolved (Real): {conflicts.get('conflicts_resolved', 0)}\n")
    f.write(f"Handling Rate (Real): {conflicts.get('conflict_handling_rate', 0.0):.4f}\n")
    f.write(f"  - Fuzzy Conflicts: {conflicts.get('fuzzy_conflicts_detected', 0)}\n")
    f.write(f"  - Enhanced Conflicts: {conflicts.get('enhanced_conflicts_detected', 0)}\n")
    f.write(f"  - Known Test Cases: {conflicts.get('known_test_cases', 0)}\n\n")
    
    f.write("CONFLICT HANDLING (PLACEHOLDER COMPARISON)\n")
    f.write("-" * 42 + "\n")
    f.write(f"Placeholder Mentioned: {conflicts.get('placeholder_mentioned', 0)}\n")
    f.write(f"Placeholder Resolved: {conflicts.get('placeholder_resolved', 0)}\n")
    f.write(f"Placeholder Rate: {conflicts.get('placeholder_rate', 0.0):.4f}\n\n")
    
    f.write("TECHNICAL DETAILS\n")
    f.write("-" * 16 + "\n")
    f.write(f"Max Fuzzy Conflict Score: {conflicts.get('max_fuzzy_conflict_score', 0.0):.4f}\n")
    f.write(f"Fuzzy Threshold Used: {conflicts.get('conflict_threshold_used', 0.0)}\n\n")
    
    f.write("CONFLICT DETECTION METHODS\n")
    f.write("-" * 25 + "\n")
    f.write("1. FUZZY RELATIONS: Uses semantic similarity and text analysis\n")
    f.write("   to detect conflicts between Gospel accounts automatically.\n")
    f.write("2. ENHANCED DETECTION: Uses dictionaries of participants,\n")
    f.write("   locations, numbers, and temporal indicators to identify\n")
    f.write("   specific types of conflicts (e.g., Peter's denial details).\n")
    f.write("3. KNOWN TEST CASES: Analyzes predefined conflict scenarios\n")
    f.write("   from theological literature (config.yaml test_cases).\n\n")
    
    f.write("IMPORTANT NOTE\n")
    f.write("-" * 14 + "\n")
    f.write("The system now uses REAL conflict detection methods alongside\n")
    f.write("placeholder values for comparison. The 'Real Analysis' results\n")
    f.write("are based on actual data processing, while placeholder values\n")
    f.write("remain for historical comparison. Current implementation shows\n")
    f.write("that the fuzzy relation method may need threshold adjustment\n")
    f.write("to detect more subtle theological conflicts.\n\n")

print("Relatório corrigido gerado em: results/evaluation_report_corrected.txt")