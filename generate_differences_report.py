#!/usr/bin/env python3
"""
Script para gerar relatório com nova abordagem - diferenças documentadas, não resolvidas
"""
import json
from datetime import datetime

# Carrega os resultados JSON
with open('results/evaluation_results_20250925_150440.json', 'r') as f:
    results = json.load(f)

# Gera o relatório com nova abordagem
with open('results/evaluation_report_differences_focused.txt', 'w', encoding='utf-8') as f:
    f.write("FUZZY GOSPEL CONSOLIDATION - ANÁLISE DE DIFERENÇAS\n")
    f.write("=" * 55 + "\n\n")
    f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    f.write("ABORDAGEM: IDENTIFICAÇÃO E DOCUMENTAÇÃO DE DIFERENÇAS\n")
    f.write("-" * 50 + "\n")
    f.write("Este sistema identifica e documenta diferenças entre os evangelhos,\n")
    f.write("sem tentar 'resolvê-las'. O objetivo é análise comparativa,\n") 
    f.write("preservando as perspectivas distintas de cada evangelho.\n\n")
    
    # Summary statistics
    f.write("ESTATÍSTICAS DO SUMÁRIO\n")
    f.write("-" * 24 + "\n")
    f.write(f"Tamanho: {results.get('summary_length', 0)} caracteres\n")
    f.write(f"Palavras: {results.get('summary_word_count', 0)} palavras\n")
    f.write(f"Score geral: {results.get('overall_score', 0.0):.4f}\n\n")
    
    # Automatic metrics
    auto_metrics = results.get('automatic_metrics', {})
    f.write("MÉTRICAS AUTOMÁTICAS DE QUALIDADE\n")
    f.write("-" * 33 + "\n")
    f.write(f"ROUGE-1: {auto_metrics.get('rouge', {}).get('rouge1', 0.0):.4f} (sobreposição de palavras)\n")
    f.write(f"ROUGE-2: {auto_metrics.get('rouge', {}).get('rouge2', 0.0):.4f} (sobreposição de bigramas)\n")
    f.write(f"ROUGE-L: {auto_metrics.get('rouge', {}).get('rougeL', 0.0):.4f} (sequência mais longa)\n")
    f.write(f"BERTScore F1: {auto_metrics.get('bertscore', {}).get('f1', 0.0):.4f} (qualidade semântica)\n")
    f.write(f"METEOR: {auto_metrics.get('meteor', 0.0):.4f} (cobertura com sinônimos)\n")
    f.write(f"BLEU: {auto_metrics.get('bleu', 0.0):.4f} (fluência do texto)\n\n")
    
    # Temporal coherence
    temp_coherence = results.get('temporal_coherence', {})
    f.write("COERÊNCIA TEMPORAL\n")
    f.write("-" * 18 + "\n")
    f.write(f"Kendall's Tau: {temp_coherence.get('kendall_tau', 0.0):.4f} (ordem cronológica)\n")
    f.write(f"Precisão Temporal: {temp_coherence.get('temporal_accuracy', 0.0):.4f}\n")
    f.write(f"Violações Cronológicas: {temp_coherence.get('chronological_violations', 0)}\n")
    f.write(f"Eventos no Sumário: {temp_coherence.get('events_in_summary', 0)}/{temp_coherence.get('events_in_reference', 0)}\n\n")
    
    # Gospel differences analysis
    conflicts = results.get('conflict_handling', {})
    f.write("ANÁLISE DE DIFERENÇAS ENTRE EVANGELHOS\n")
    f.write("-" * 38 + "\n")
    f.write(f"Diferenças Identificadas: {conflicts.get('conflicts_mentioned', 0)}\n")
    f.write(f"Diferenças Documentadas: {conflicts.get('conflicts_resolved', 0)}\n") 
    f.write(f"Taxa de Documentação: {conflicts.get('conflict_handling_rate', 0.0):.4f} (100% = todas documentadas)\n\n")
    
    f.write("MÉTODOS DE DETECÇÃO:\n")
    f.write(f"• Método Fuzzy: {conflicts.get('fuzzy_conflicts_detected', 0)} diferenças\n")
    f.write(f"  (Score máximo encontrado: {conflicts.get('max_fuzzy_conflict_score', 0.0):.4f})\n")
    f.write(f"  (Threshold usado: {conflicts.get('conflict_threshold_used', 0.0)})\n")
    f.write(f"• Detector Melhorado: {conflicts.get('enhanced_conflicts_detected', 0)} diferenças\n")
    f.write(f"• Casos Conhecidos: {conflicts.get('known_test_cases', 0)} casos de teste\n\n")
    
    f.write("CASOS DE DIFERENÇAS CONHECIDAS:\n")
    f.write("1. Pedro negação: Número de vezes que o galo canta\n")
    f.write("2. Limpeza templo: Timing da limpeza (início vs fim do ministério)\n")
    f.write("3. Entrada triunfal: Detalhes do animal usado\n\n")
    
    # Content coverage
    content_coverage = results.get('content_coverage', {})
    f.write("COBERTURA DE CONTEÚDO\n")
    f.write("-" * 21 + "\n")
    f.write(f"Cobertura de Eventos: {content_coverage.get('event_coverage', 0.0):.4f}\n")
    f.write(f"Representação dos Evangelhos: {content_coverage.get('gospel_representation', 0.0):.4f}\n")
    f.write(f"Participantes-chave Mencionados: {content_coverage.get('key_participants_mentioned', 0.0):.4f}\n\n")
    
    f.write("INTERPRETAÇÃO DOS RESULTADOS\n")
    f.write("-" * 29 + "\n")
    f.write("• ROUGE > 90%: Excelente cobertura do conteúdo original\n")
    f.write("• BERTScore > 90%: Alta qualidade semântica do sumário\n")
    f.write("• Kendall Tau = 1.0: Perfeita preservação da ordem cronológica\n")
    f.write("• Taxa documentação = 100%: Todas diferenças encontradas foram anotadas\n\n")
    
    f.write("NOTA IMPORTANTE\n")
    f.write("-" * 15 + "\n")
    f.write("Este sistema NÃO tenta resolver ou harmonizar diferenças entre evangelhos.\n")
    f.write("Seu objetivo é IDENTIFICAR e DOCUMENTAR variações, preservando a\n")
    f.write("integridade e perspectiva única de cada evangelho. As diferenças são\n")
    f.write("tratadas como características valiosas para estudo comparativo, não\n")
    f.write("como problemas a serem resolvidos.\n\n")

print("Relatório focado em diferenças gerado em: results/evaluation_report_differences_focused.txt")