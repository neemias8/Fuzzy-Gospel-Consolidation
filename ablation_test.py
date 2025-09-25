#!/usr/bin/env python3
"""
Teste de Ablação - Sistema SEM Fuzzy Relations e GNN
Apenas usando sumarização direta baseada em clustering simples
"""

import os
import sys
import logging
from typing import Dict, Any
import yaml
from datetime import datetime

# Adiciona o diretório src ao path
sys.path.append('src')

from data_processing.xml_parser import XMLParser
from summarization.consolidation_summarizer import ConsolidationSummarizer
from evaluation.evaluation_suite import EvaluationSuite

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config() -> Dict[str, Any]:
    """Carrega configuração"""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def simple_event_clustering(events, max_clusters=15):
    """
    Clustering simples baseado apenas em similaridade de texto básica
    SEM usar fuzzy relations ou GNN
    """
    from sentence_transformers import SentenceTransformer
    import numpy as np
    from sklearn.cluster import KMeans
    
    logger.info("Fazendo clustering simples sem fuzzy/GNN...")
    
    # Usa apenas sentence transformer básico
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Extrai textos dos eventos
    texts = []
    for event in events:
        text_parts = []
        if hasattr(event, 'description') and event.description:
            text_parts.append(event.description)
        if hasattr(event, 'text') and event.text:
            text_parts.append(event.text)
        if not text_parts:
            text_parts.append(f"Event {event.id}")
        texts.append(' '.join(text_parts))
    
    # Gera embeddings básicos
    embeddings = model.encode(texts)
    
    # Clustering simples com K-means
    n_clusters = min(max_clusters, len(events))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # Organiza eventos por cluster
    clusters = {}
    for i, label in enumerate(cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(events[i])
    
    logger.info(f"Criados {len(clusters)} clusters simples")
    return list(clusters.values())

def run_ablation_test():
    """Executa teste de ablação sem fuzzy/GNN"""
    
    logger.info("=== INICIANDO TESTE DE ABLAÇÃO ===")
    logger.info("Sistema SEM Fuzzy Relations e GNN")
    
    # Carrega configuração
    config = load_config()
    
    # Parse dos dados XML
    logger.info("Carregando dados dos evangelhos...")
    parser = XMLParser()
    
    # Carrega o dataset completo
    from pathlib import Path
    data_path = Path(config['data']['raw_dir'])
    corpus = parser.load_complete_dataset(data_path, config['data'])
    
    # O texto já está extraído nos eventos pelo XMLParser
    logger.info("Texto dos eventos já disponível...")
    
    logger.info(f"Total de eventos carregados: {len(corpus.events)}")
    
    # ABLAÇÃO: Clustering simples SEM fuzzy relations
    logger.info("Aplicando clustering simples (SEM fuzzy relations)...")
    event_clusters = simple_event_clustering(corpus.events)
    
    # ABLAÇÃO: Sumarização direta SEM GNN
    logger.info("Gerando sumário consolidado SEM GNN...")
    summarizer = ConsolidationSummarizer(config.get('models', {}).get('summarization_model', 'facebook/bart-large-cnn'))
    
    # Converte clusters para formato esperado pelo summarizer
    cluster_data = {}
    for i, cluster in enumerate(event_clusters):
        cluster_data[i] = {
            'events': cluster,
            'embeddings': None  # SEM embeddings do GNN
        }
    
    # Gera sumário sem usar GNN embeddings
    summary = summarizer.generate_summary_from_clusters(cluster_data, use_gnn_embeddings=False)
    
    logger.info(f"Sumário gerado: {len(summary)} caracteres, {len(summary.split())} palavras")
    
    # Avaliação
    logger.info("Avaliando resultados...")
    evaluator = EvaluationSuite(config)
    
    # Para ablação, criamos "fuzzy_relations" vazio
    dummy_fuzzy_relations = []
    
    results = evaluator.evaluate_comprehensive(summary, corpus, dummy_fuzzy_relations)
    
    # Salva resultados com timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Cria diretório de resultados se não existir
    results_dir = "results/ablation"
    os.makedirs(results_dir, exist_ok=True)
    
    # Salva arquivos de resultados
    import json
    results_file = os.path.join(results_dir, f"ablation_results_{timestamp}.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    summary_file = os.path.join(results_dir, f"ablation_summary_{timestamp}.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(summary)
    
    # Gera relatório de ablação
    report_file = os.path.join(results_dir, f"ablation_report_{timestamp}.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("TESTE DE ABLAÇÃO - SISTEMA SEM FUZZY/GNN\n")
        f.write("=" * 45 + "\n\n")
        f.write(f"Executado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("COMPONENTES REMOVIDOS:\n")
        f.write("- Fuzzy Relations (similarity calculation)\n")
        f.write("- Graph Neural Network (GNN)\n")
        f.write("- Fuzzy event graph construction\n")
        f.write("- Advanced embeddings integration\n\n")
        
        f.write("COMPONENTES MANTIDOS:\n")
        f.write("- Sentence Transformer básico\n")
        f.write("- K-means clustering simples\n")
        f.write("- Sumarização com BART\n")
        f.write("- Sistema de avaliação\n\n")
        
        # Métricas
        auto_metrics = results.get('automatic_metrics', {})
        f.write("RESULTADOS DA ABLAÇÃO:\n")
        f.write("-" * 25 + "\n")
        f.write(f"Summary length: {results.get('summary_length', 0)} characters\n")
        f.write(f"Summary words: {results.get('summary_word_count', 0)} words\n")
        f.write(f"Overall score: {results.get('overall_score', 0.0):.4f}\n\n")
        
        f.write("AUTOMATIC METRICS:\n")
        f.write(f"ROUGE-1: {auto_metrics.get('rouge', {}).get('rouge1', 0.0):.4f}\n")
        f.write(f"ROUGE-2: {auto_metrics.get('rouge', {}).get('rouge2', 0.0):.4f}\n")
        f.write(f"ROUGE-L: {auto_metrics.get('rouge', {}).get('rougeL', 0.0):.4f}\n")
        f.write(f"BERTScore F1: {auto_metrics.get('bertscore', {}).get('f1', 0.0):.4f}\n")
        f.write(f"METEOR: {auto_metrics.get('meteor', 0.0):.4f}\n")
        f.write(f"BLEU: {auto_metrics.get('bleu', 0.0):.4f}\n\n")
        
        # Temporal coherence
        temp_coherence = results.get('temporal_coherence', {})
        f.write("TEMPORAL COHERENCE:\n")
        f.write(f"Kendall's Tau: {temp_coherence.get('kendall_tau', 0.0):.4f}\n")
        f.write(f"Temporal Accuracy: {temp_coherence.get('temporal_accuracy', 0.0):.4f}\n")
        f.write(f"Chronological Violations: {temp_coherence.get('chronological_violations', 0)}\n\n")
        
        # Difference analysis
        conflicts = results.get('conflict_handling', {})
        f.write("DIFFERENCE ANALYSIS:\n")
        f.write(f"Differences Found: {conflicts.get('differences_found', conflicts.get('conflicts_mentioned', 0))}\n")
        f.write(f"Differences Documented: {conflicts.get('differences_documented', conflicts.get('conflicts_resolved', 0))}\n")
        f.write(f"Documentation Rate: {conflicts.get('documentation_rate', conflicts.get('conflict_handling_rate', 0.0)):.4f}\n\n")
        
        f.write("NOTA:\n")
        f.write("Este é um teste de ablação para medir o impacto dos\n")
        f.write("componentes fuzzy e GNN no desempenho do sistema.\n")
        f.write("Compare estes resultados com o sistema completo.\n")
    
    logger.info(f"Teste de ablação concluído!")
    logger.info(f"Resultados salvos em: {results_dir}/")
    logger.info(f"Overall Score (Ablação): {results.get('overall_score', 0.0):.4f}")
    
    return results

if __name__ == "__main__":
    try:
        results = run_ablation_test()
        print(f"\n=== TESTE DE ABLAÇÃO CONCLUÍDO ===")
        print(f"Score do sistema SEM fuzzy/GNN: {results.get('overall_score', 0.0):.4f}")
        print(f"Resultados salvos em: results/ablation/")
        
    except Exception as e:
        logger.error(f"Erro durante teste de ablação: {e}")
        raise