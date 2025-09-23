"""
Automatic Evaluation Metrics

Implementation of automatic evaluation metrics for Gospel consolidation.
"""

import logging
from typing import List, Dict, Any, Tuple
import numpy as np
from scipy.stats import kendalltau
import nltk
from rouge_score import rouge_scorer
from bert_score import score as bert_score

logger = logging.getLogger(__name__)

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


class AutomaticMetrics:
    """Automatic evaluation metrics for text summarization"""
    
    def __init__(self):
        """Initialize automatic metrics"""
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        logger.info("AutomaticMetrics initialized with ROUGE, METEOR, BERTScore, and Kendall's Tau")
    
    def calculate_rouge(self, generated: str, references: List[str]) -> Dict[str, float]:
        """
        Calculate ROUGE scores.
        
        Args:
            generated: Generated summary
            references: Reference summaries
            
        Returns:
            Dictionary with ROUGE scores
        """
        if not references:
            logger.warning("No references provided for ROUGE calculation")
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
        
        # Calculate ROUGE against all references and take the best scores
        best_scores = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
        
        for reference in references:
            scores = self.rouge_scorer.score(reference, generated)
            for metric in best_scores:
                current_f1 = scores[metric].fmeasure
                if current_f1 > best_scores[metric]:
                    best_scores[metric] = current_f1
        
        logger.info(f"ROUGE scores calculated: {best_scores}")
        return best_scores
    
    def calculate_bertscore(self, generated: str, references: List[str]) -> Dict[str, float]:
        """
        Calculate BERTScore.
        
        Args:
            generated: Generated summary
            references: Reference summaries
            
        Returns:
            Dictionary with BERTScore metrics
        """
        if not references:
            logger.warning("No references provided for BERTScore calculation")
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        try:
            # Calculate BERTScore
            P, R, F1 = bert_score([generated] * len(references), references, lang='en', verbose=False)
            
            # Take the maximum scores across all references
            precision = float(P.max())
            recall = float(R.max())
            f1 = float(F1.max())
            
            result = {'precision': precision, 'recall': recall, 'f1': f1}
            logger.info(f"BERTScore calculated: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error calculating BERTScore: {e}")
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    def calculate_meteor(self, generated: str, references: List[str]) -> float:
        """
        Calculate METEOR score (simplified implementation).
        
        Args:
            generated: Generated summary
            references: Reference summaries
            
        Returns:
            METEOR score
        """
        if not references:
            logger.warning("No references provided for METEOR calculation")
            return 0.0
        
        try:
            # Simplified METEOR implementation based on token overlap
            generated_tokens = set(nltk.word_tokenize(generated.lower()))
            
            best_meteor = 0.0
            for reference in references:
                ref_tokens = set(nltk.word_tokenize(reference.lower()))
                
                # Calculate precision and recall
                matches = len(generated_tokens.intersection(ref_tokens))
                precision = matches / len(generated_tokens) if generated_tokens else 0
                recall = matches / len(ref_tokens) if ref_tokens else 0
                
                # Simplified METEOR score (harmonic mean with recall preference)
                if precision + recall > 0:
                    meteor = (10 * precision * recall) / (recall + 9 * precision)
                    best_meteor = max(best_meteor, meteor)
            
            logger.info(f"METEOR score calculated: {best_meteor:.4f}")
            return best_meteor
            
        except Exception as e:
            logger.error(f"Error calculating METEOR: {e}")
            return 0.0
    
    def calculate_kendall_tau(self, generated_order: List[str], reference_order: List[str]) -> float:
        """
        Calculate Kendall's Tau correlation for temporal ordering.
        
        Args:
            generated_order: List of event identifiers in generated order
            reference_order: List of event identifiers in reference order
            
        Returns:
            Kendall's Tau correlation coefficient
        """
        if not generated_order or not reference_order:
            logger.warning("Empty order lists provided for Kendall's Tau calculation")
            return 0.0
        
        try:
            # Find common events
            common_events = set(generated_order).intersection(set(reference_order))
            if len(common_events) < 2:
                logger.warning("Not enough common events for Kendall's Tau calculation")
                return 0.0
            
            # Create mappings to numeric indices
            gen_indices = [generated_order.index(event) for event in common_events if event in generated_order]
            ref_indices = [reference_order.index(event) for event in common_events if event in reference_order]
            
            # Calculate Kendall's Tau
            tau, p_value = kendalltau(gen_indices, ref_indices)
            
            logger.info(f"Kendall's Tau calculated: {tau:.4f} (p-value: {p_value:.4f})")
            return float(tau) if not np.isnan(tau) else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating Kendall's Tau: {e}")
            return 0.0
    
    def calculate_bleu(self, generated: str, references: List[str]) -> float:
        """
        Calculate BLEU score (simplified implementation).
        
        Args:
            generated: Generated summary
            references: Reference summaries
            
        Returns:
            BLEU score
        """
        if not references:
            logger.warning("No references provided for BLEU calculation")
            return 0.0
        
        try:
            # Simplified BLEU implementation
            generated_tokens = nltk.word_tokenize(generated.lower())
            best_bleu = 0.0
            
            for reference in references:
                ref_tokens = nltk.word_tokenize(reference.lower())
                
                # Calculate n-gram precision for n=1,2,3,4
                precisions = []
                for n in range(1, 5):
                    gen_ngrams = [tuple(generated_tokens[i:i+n]) for i in range(len(generated_tokens)-n+1)]
                    ref_ngrams = [tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens)-n+1)]
                    
                    if gen_ngrams:
                        matches = sum(1 for ngram in gen_ngrams if ngram in ref_ngrams)
                        precision = matches / len(gen_ngrams)
                        precisions.append(precision)
                
                # Calculate BLEU as geometric mean of precisions
                if precisions and all(p > 0 for p in precisions):
                    bleu = np.exp(np.mean(np.log(precisions)))
                    best_bleu = max(best_bleu, bleu)
            
            logger.info(f"BLEU score calculated: {best_bleu:.4f}")
            return best_bleu
            
        except Exception as e:
            logger.error(f"Error calculating BLEU: {e}")
            return 0.0
    
    def calculate_all_metrics(self, generated: str, references: List[str]) -> Dict[str, Any]:
        """
        Calculate all automatic metrics.
        
        Args:
            generated: Generated summary
            references: Reference summaries
            
        Returns:
            Dictionary with all metric scores
        """
        try:
            logger.info("Calculating all automatic metrics...")
            
            # Calculate ROUGE scores
            rouge_scores = self.calculate_rouge(generated, references)
            
            # Calculate BERTScore
            bertscore_scores = self.calculate_bertscore(generated, references)
            
            # Calculate METEOR
            meteor_score = self.calculate_meteor(generated, references)
            
            # Calculate BLEU
            bleu_score = self.calculate_bleu(generated, references)
            
            # Combine all metrics
            all_metrics = {
                'rouge': rouge_scores,
                'bertscore': bertscore_scores,
                'meteor': meteor_score,
                'bleu': bleu_score
            }
            
            logger.info("All automatic metrics calculated successfully")
            return all_metrics
            
        except Exception as e:
            logger.error(f"Error calculating all metrics: {e}")
            return {
                'rouge': {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0},
                'bertscore': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0},
                'meteor': 0.0,
                'bleu': 0.0
            }
