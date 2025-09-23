"""
Main Pipeline for Fuzzy Gospel Consolidation

This is the main entry point for the Fuzzy Gospel Consolidation project.
It orchestrates the entire pipeline from data loading to evaluation.
"""

import argparse
import logging
import yaml
from pathlib import Path
import json
import sys
from typing import Dict, Any, Optional

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from data_processing import XMLParser, TextExtractor
from fuzzy_relations import FuzzyRelationCalculator
from graph_neural_network import FuzzyEventGraph, FuzzyGNN
from summarization import ConsolidationSummarizer
from evaluation import EvaluationSuite

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/fuzzy_gospel.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class FuzzyGospelConsolidator:
    """Main class for the Fuzzy Gospel Consolidation system"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the consolidator with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.corpus = None
        self.fuzzy_relations = None
        self.graph = None
        self.model = None
        self.consolidated_summary = None
        
        # Initialize components
        self.xml_parser = XMLParser()
        self.text_extractor = None
        self.fuzzy_calculator = None
        self.summarizer = None
        self.evaluator = None
        
        logger.info("FuzzyGospelConsolidator initialized")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    def load_data(self, data_dir: Optional[str] = None) -> None:
        """
        Load Gospel data from XML files.
        
        Args:
            data_dir: Directory containing XML files (uses config default if None)
        """
        if data_dir is None:
            data_dir = self.config['data']['raw_dir']
        
        data_path = Path(data_dir)
        
        logger.info("Loading Gospel dataset...")
        
        # Load complete dataset
        self.corpus = self.xml_parser.load_complete_dataset(
            data_path, self.config['data']
        )
        
        # Initialize text extractor
        self.text_extractor = TextExtractor(self.corpus)
        
        # Validate corpus
        validation_results = self.xml_parser.validate_corpus()
        if validation_results['errors']:
            logger.error("Data validation errors found:")
            for error in validation_results['errors']:
                logger.error(f"  {error}")
        
        if validation_results['warnings']:
            logger.warning("Data validation warnings:")
            for warning in validation_results['warnings']:
                logger.warning(f"  {warning}")
        
        logger.info(f"Data loading complete: {len(self.corpus)} events loaded")
    
    def calculate_fuzzy_relations(self) -> None:
        """Calculate fuzzy relations between all event pairs"""
        if self.corpus is None:
            raise ValueError("Data must be loaded before calculating relations")
        
        logger.info(f"Calculating fuzzy relations for {len(self.corpus.events)} events...")
        
        # Initialize fuzzy calculator
        self.fuzzy_calculator = FuzzyRelationCalculator(self.config)
        
        # Calculate relations for all event pairs
        self.fuzzy_relations = self.fuzzy_calculator.calculate_relation_matrix(
            self.corpus.events, self.text_extractor
        )
        
        # Export statistics
        stats = self.fuzzy_calculator.export_relation_statistics(self.fuzzy_relations)
        logger.info("Fuzzy relation statistics:")
        logger.info(f"  Total relations calculated: {len(self.fuzzy_relations)}")
        for key, value in stats.items():
            if isinstance(value, dict):
                logger.info(f"  {key}:")
                for subkey, subvalue in value.items():
                    logger.info(f"    {subkey}: {subvalue:.4f}" if isinstance(subvalue, float) else f"    {subkey}: {subvalue}")
            else:
                logger.info(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
        
        logger.info("Fuzzy relation calculation complete")
    
    def build_graph(self) -> None:
        """Build fuzzy-enhanced event graph"""
        if self.fuzzy_relations is None:
            raise ValueError("Fuzzy relations must be calculated before building graph")
        
        logger.info(f"Building fuzzy event graph from {len(self.fuzzy_relations)} relations...")
        
        # Create graph
        self.graph = FuzzyEventGraph(
            self.corpus.events, 
            self.fuzzy_relations,
            self.config
        )
        
        # Get graph statistics
        stats = self.graph.get_statistics()
        logger.info("Graph statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
        
        # Log edge density
        num_nodes = stats['num_nodes']
        num_edges = stats['num_edges']
        max_edges = num_nodes * (num_nodes - 1)
        density = num_edges / max_edges if max_edges > 0 else 0
        logger.info(f"  edge_density: {density:.4f}")
        
        logger.info("Graph construction complete")
    
    def train_model(self) -> None:
        """Train the fuzzy GNN model"""
        if self.graph is None:
            raise ValueError("Graph must be built before training model")
        
        logger.info("Training fuzzy GNN model...")
        
        # Initialize model
        self.model = FuzzyGNN(self.config['gnn'])
        
        # Train model
        training_stats = self.model.train_model(self.graph.get_pytorch_data())
        
        logger.info("Model training complete:")
        for key, value in training_stats.items():
            logger.info(f"  {key}: {value}")
    
    def generate_summary(self) -> str:
        """Generate consolidated summary using GNN embeddings and fuzzy relations"""
        if self.model is None:
            raise ValueError("Model must be trained before generating summary")
        
        logger.info("Generating consolidated summary with fuzzy-GNN integration...")
        
        # Initialize summarizer
        self.summarizer = ConsolidationSummarizer(self.config)
        
        # Generate summary using trained model and graph
        self.consolidated_summary = self.summarizer.generate_consolidated_summary(
            self.corpus.events,
            self.graph,
            self.model
        )
        
        # Log summary statistics
        num_words = len(self.consolidated_summary.split())
        num_lines = len(self.consolidated_summary.split('\n'))
        logger.info(f"Summary generated: {len(self.consolidated_summary)} characters, {num_words} words, {num_lines} lines")
        
        return self.consolidated_summary
    
    def evaluate_results(self) -> Dict[str, Any]:
        """Evaluate the consolidated summary and save results"""
        if self.consolidated_summary is None:
            raise ValueError("Summary must be generated before evaluation")
        
        logger.info("Evaluating results...")
        
        # Initialize evaluator
        self.evaluator = EvaluationSuite(self.config)
        
        # Run comprehensive evaluation
        evaluation_results = self.evaluator.evaluate_comprehensive(
            self.consolidated_summary,
            self.corpus,
            self.fuzzy_relations
        )
        
        # Save evaluation results and consolidated summary
        results_file = self.evaluator.save_evaluation_results(
            evaluation_results, 
            self.consolidated_summary
        )
        
        logger.info("Evaluation complete and results saved")
        return evaluation_results
    
    def run_full_pipeline(self, data_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the complete pipeline from data loading to evaluation.
        
        Args:
            data_dir: Directory containing XML files
            
        Returns:
            Dictionary with all results
        """
        logger.info("Starting full pipeline execution...")
        
        try:
            # Step 1: Load data
            self.load_data(data_dir)
            
            # Step 2: Calculate fuzzy relations
            self.calculate_fuzzy_relations()
            
            # Step 3: Build graph
            self.build_graph()
            
            # Step 4: Train model
            self.train_model()
            
            # Step 5: Generate summary
            summary = self.generate_summary()
            
            # Step 6: Evaluate results
            evaluation = self.evaluate_results()
            
            # Compile results
            results = {
                'summary': summary,
                'evaluation': evaluation,
                'corpus_stats': self.corpus.get_statistics(),
                'relation_stats': self.fuzzy_calculator.export_relation_statistics(self.fuzzy_relations),
                'graph_stats': self.graph.get_statistics()
            }
            
            logger.info("Full pipeline execution complete")
            return results
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            raise
    
    def save_results(self, results: Dict[str, Any], output_dir: str = None) -> None:
        """
        Save results to files.
        
        Args:
            results: Results dictionary from run_full_pipeline
            output_dir: Output directory (uses config default if None)
        """
        if output_dir is None:
            output_dir = self.config['output']['results_dir']
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save summary
        summary_file = output_path / "consolidated_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(results['summary'])
        
        # Save evaluation results
        eval_file = output_path / "evaluation_results.json"
        with open(eval_file, 'w', encoding='utf-8') as f:
            json.dump(results['evaluation'], f, indent=2, default=str)
        
        # Save statistics
        stats_file = output_path / "statistics.json"
        stats = {
            'corpus': results['corpus_stats'],
            'relations': results['relation_stats'],
            'graph': results['graph_stats']
        }
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, default=str)
        
        logger.info(f"Results saved to {output_path}")
    
    def export_intermediate_data(self, output_dir: str = None) -> None:
        """
        Export intermediate data for analysis.
        
        Args:
            output_dir: Output directory
        """
        if output_dir is None:
            output_dir = self.config['output']['results_dir']
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Export corpus statistics
        if self.text_extractor:
            corpus_stats = self.text_extractor.export_corpus_statistics()
            with open(output_path / "corpus_analysis.json", 'w') as f:
                json.dump(corpus_stats, f, indent=2, default=str)
        
        # Export fuzzy relations
        if self.fuzzy_relations:
            relations_data = {}
            for (id1, id2), relation in self.fuzzy_relations.items():
                relations_data[f"{id1}-{id2}"] = {
                    'mu_same': relation.mu_same,
                    'mu_conflict': relation.mu_conflict,
                    'mu_before': relation.mu_before,
                    'mu_after': relation.mu_after
                }
            
            with open(output_path / "fuzzy_relations.json", 'w') as f:
                json.dump(relations_data, f, indent=2)
        
        # Export graph data
        if self.graph:
            graph_data = self.graph.export_for_analysis()
            with open(output_path / "graph_data.json", 'w') as f:
                json.dump(graph_data, f, indent=2, default=str)
        
        logger.info(f"Intermediate data exported to {output_path}")


def main():
    """Main entry point for command-line usage"""
    parser = argparse.ArgumentParser(description="Fuzzy Gospel Consolidation")
    parser.add_argument('--config', default='config.yaml', help='Configuration file path')
    parser.add_argument('--data-dir', help='Data directory (overrides config)')
    parser.add_argument('--output-dir', help='Output directory (overrides config)')
    parser.add_argument('--mode', choices=['full', 'train', 'evaluate', 'export'], 
                       default='full', help='Execution mode')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize consolidator
        consolidator = FuzzyGospelConsolidator(args.config)
        
        if args.mode == 'full':
            # Run full pipeline
            results = consolidator.run_full_pipeline(args.data_dir)
            consolidator.save_results(results, args.output_dir)
            
            print("\\nPipeline execution complete!")
            print(f"Summary length: {len(results['summary'])} characters")
            print(f"Evaluation score: {results['evaluation'].get('overall_score', 'N/A')}")
            
        elif args.mode == 'export':
            # Export intermediate data only
            consolidator.load_data(args.data_dir)
            consolidator.calculate_fuzzy_relations()
            consolidator.build_graph()
            consolidator.export_intermediate_data(args.output_dir)
            
            print("Intermediate data export complete!")
            
        else:
            print(f"Mode '{args.mode}' not fully implemented yet")
    
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
