#!/usr/bin/env python3
"""
Advanced Metrics for TMM Evaluation

This module implements sophisticated evaluation metrics beyond simple accuracy,
including F1 scores, semantic similarity, memory efficiency, and domain-specific metrics.

Usage:
    from evaluation.advanced_metrics import AdvancedMetrics
    metrics = AdvancedMetrics()
    scores = metrics.compute_all_metrics(predicted, ground_truth, context)
"""

import re
import math
import logging
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from collections import Counter

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

logger = logging.getLogger(__name__)

@dataclass
class AdvancedMetricResult:
    """Comprehensive metrics result container."""
    # Core metrics
    exact_match: float
    f1_score: float
    precision: float
    recall: float
    
    # Semantic metrics
    semantic_similarity: float
    token_overlap: float
    normalized_edit_distance: float
    
    # Memory-specific metrics
    memory_efficiency: float
    context_utilization: float
    response_quality: float
    
    # Truth maintenance metrics
    truth_consistency: float
    uncertainty_handling: float
    hallucination_rate: float
    
    # Metadata
    response_length: int
    processing_time: float

class AdvancedMetrics:
    """Advanced evaluation metrics for TMM systems."""
    
    def __init__(self):
        self.stemmer = PorterStemmer()
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            self.stop_words = set()
        
    def compute_all_metrics(self, predicted: str, ground_truth: str, 
                          context: str = "", is_answerable: bool = True,
                          memory_records: List[Any] = None,
                          processing_time: float = 0.0) -> AdvancedMetricResult:
        """Compute comprehensive metrics for a prediction."""
        
        # Core metrics
        exact_match = self._exact_match(predicted, ground_truth)
        f1, precision, recall = self._f1_score(predicted, ground_truth)
        
        # Semantic metrics
        semantic_sim = self._semantic_similarity(predicted, ground_truth)
        token_overlap = self._token_overlap(predicted, ground_truth)
        edit_distance = self._normalized_edit_distance(predicted, ground_truth)
        
        # Memory-specific metrics
        memory_eff = self._memory_efficiency(memory_records or [])
        context_util = self._context_utilization(predicted, context)
        response_qual = self._response_quality(predicted, ground_truth, context)
        
        # Truth maintenance metrics
        truth_consistency = self._truth_consistency(predicted, context)
        uncertainty = self._uncertainty_handling(predicted, is_answerable)
        hallucination = self._hallucination_rate(predicted, context)
        
        return AdvancedMetricResult(
            exact_match=exact_match,
            f1_score=f1,
            precision=precision,
            recall=recall,
            semantic_similarity=semantic_sim,
            token_overlap=token_overlap,
            normalized_edit_distance=edit_distance,
            memory_efficiency=memory_eff,
            context_utilization=context_util,
            response_quality=response_qual,
            truth_consistency=truth_consistency,
            uncertainty_handling=uncertainty,
            hallucination_rate=hallucination,
            response_length=len(predicted.split()),
            processing_time=processing_time
        )
    
    def _exact_match(self, predicted: str, ground_truth: str) -> float:
        """Compute exact match score."""
        pred_norm = self._normalize_text(predicted)
        gt_norm = self._normalize_text(ground_truth)
        return 1.0 if pred_norm == gt_norm else 0.0
    
    def _f1_score(self, predicted: str, ground_truth: str) -> Tuple[float, float, float]:
        """Compute F1 score, precision, and recall."""
        pred_tokens = self._get_tokens(predicted)
        gt_tokens = self._get_tokens(ground_truth)
        
        if not gt_tokens:
            return 1.0 if not pred_tokens else 0.0, 0.0, 0.0
        
        if not pred_tokens:
            return 0.0, 0.0, 0.0
        
        # Token-level F1
        common_tokens = Counter(pred_tokens) & Counter(gt_tokens)
        num_common = sum(common_tokens.values())
        
        precision = num_common / len(pred_tokens)
        recall = num_common / len(gt_tokens)
        
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return f1, precision, recall
    
    def _semantic_similarity(self, predicted: str, ground_truth: str) -> float:
        """Compute semantic similarity using simple word embeddings approximation."""
        # Simplified semantic similarity based on shared meaningful words
        pred_words = self._get_meaningful_words(predicted)
        gt_words = self._get_meaningful_words(ground_truth)
        
        if not pred_words or not gt_words:
            return 0.0
        
        # Jaccard similarity of meaningful words
        intersection = len(pred_words & gt_words)
        union = len(pred_words | gt_words)
        
        return intersection / union if union > 0 else 0.0
    
    def _token_overlap(self, predicted: str, ground_truth: str) -> float:
        """Compute token overlap ratio."""
        pred_tokens = set(self._get_tokens(predicted))
        gt_tokens = set(self._get_tokens(ground_truth))
        
        if not gt_tokens:
            return 1.0 if not pred_tokens else 0.0
        
        overlap = len(pred_tokens & gt_tokens)
        return overlap / len(gt_tokens)
    
    def _normalized_edit_distance(self, predicted: str, ground_truth: str) -> float:
        """Compute normalized Levenshtein distance."""
        def levenshtein(s1, s2):
            if len(s1) < len(s2):
                return levenshtein(s2, s1)
            
            if len(s2) == 0:
                return len(s1)
            
            previous_row = list(range(len(s2) + 1))
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            
            return previous_row[-1]
        
        distance = levenshtein(predicted.lower(), ground_truth.lower())
        max_len = max(len(predicted), len(ground_truth))
        
        return 1.0 - (distance / max_len) if max_len > 0 else 1.0
    
    def _memory_efficiency(self, memory_records: List[Any]) -> float:
        """Compute memory efficiency score."""
        if not memory_records:
            return 1.0  # Perfect efficiency with no memory usage
        
        # Simple efficiency metric: inverse of memory usage
        # In practice, this would consider memory utilization vs. performance gain
        memory_size = len(memory_records)
        
        # Normalize to 0-1 scale (assuming 100 records is "reasonable" usage)
        efficiency = max(0.0, 1.0 - (memory_size / 100.0))
        return efficiency
    
    def _context_utilization(self, predicted: str, context: str) -> float:
        """Measure how well the prediction utilizes the provided context."""
        if not context:
            return 0.0
        
        pred_words = self._get_meaningful_words(predicted)
        context_words = self._get_meaningful_words(context)
        
        if not context_words:
            return 0.0
        
        # Calculate ratio of prediction words that come from context
        utilized_words = pred_words & context_words
        utilization = len(utilized_words) / len(context_words) if context_words else 0.0
        
        return min(1.0, utilization)  # Cap at 1.0
    
    def _response_quality(self, predicted: str, ground_truth: str, context: str) -> float:
        """Assess overall response quality."""
        # Combine multiple quality indicators
        
        # Completeness: Does response address the question?
        completeness = self._f1_score(predicted, ground_truth)[0]
        
        # Relevance: Is response relevant to context?
        relevance = self._context_utilization(predicted, context)
        
        # Coherence: Is response well-formed?
        coherence = self._coherence_score(predicted)
        
        # Weighted combination
        quality = 0.5 * completeness + 0.3 * relevance + 0.2 * coherence
        return quality
    
    def _truth_consistency(self, predicted: str, context: str) -> float:
        """Measure consistency between prediction and context."""
        if not context:
            return 0.5  # Neutral when no context available
        
        # Look for contradictions or inconsistencies
        pred_words = self._get_meaningful_words(predicted)
        context_words = self._get_meaningful_words(context)
        
        # Simple consistency check: shared meaningful words
        if not pred_words:
            return 0.5
        
        consistent_words = pred_words & context_words
        consistency = len(consistent_words) / len(pred_words)
        
        return consistency
    
    def _uncertainty_handling(self, predicted: str, is_answerable: bool) -> float:
        """Evaluate how well uncertainty is handled."""
        # Check for uncertainty indicators
        uncertainty_phrases = [
            "i don't know", "not sure", "uncertain", "unclear",
            "cannot determine", "not provided", "no information",
            "not mentioned", "unable to answer"
        ]
        
        pred_lower = predicted.lower()
        has_uncertainty = any(phrase in pred_lower for phrase in uncertainty_phrases)
        
        if not is_answerable:
            # For unanswerable questions, uncertainty handling is good
            return 1.0 if has_uncertainty else 0.0
        else:
            # For answerable questions, uncertainty handling should be minimal
            return 0.0 if has_uncertainty else 1.0
    
    def _hallucination_rate(self, predicted: str, context: str) -> float:
        """Estimate hallucination rate based on context consistency."""
        if not context:
            return 0.5  # Cannot assess without context
        
        pred_words = self._get_meaningful_words(predicted)
        context_words = self._get_meaningful_words(context)
        
        if not pred_words:
            return 0.0
        
        # Words in prediction but not in context might be hallucinations
        hallucinated_words = pred_words - context_words
        hallucination_rate = len(hallucinated_words) / len(pred_words)
        
        # Some hallucination is normal (common words, inference)
        # Penalize only excessive hallucination
        return max(0.0, hallucination_rate - 0.3)  # Allow 30% "normal" hallucination
    
    def _coherence_score(self, text: str) -> float:
        """Assess text coherence and well-formedness."""
        if not text.strip():
            return 0.0
        
        # Simple coherence indicators
        sentences = text.split('.')
        
        # Check for reasonable sentence length
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
        length_score = 1.0 if 3 <= avg_sentence_length <= 30 else 0.5
        
        # Check for capitalization
        cap_score = 1.0 if text[0].isupper() else 0.5
        
        # Check for ending punctuation
        punct_score = 1.0 if text.strip()[-1] in '.!?' else 0.5
        
        coherence = (length_score + cap_score + punct_score) / 3
        return coherence
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        # Remove extra whitespace and convert to lowercase
        text = re.sub(r'\s+', ' ', text.strip().lower())
        
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        
        return text
    
    def _get_tokens(self, text: str) -> List[str]:
        """Get word tokens from text."""
        try:
            tokens = word_tokenize(self._normalize_text(text))
        except:
            # Fallback if NLTK fails
            tokens = self._normalize_text(text).split()
        
        return [self.stemmer.stem(token) for token in tokens if token.isalnum()]
    
    def _get_meaningful_words(self, text: str) -> set:
        """Extract meaningful words (excluding stop words)."""
        tokens = self._get_tokens(text)
        return set(token for token in tokens if token not in self.stop_words and len(token) > 2)

class MetricAggregator:
    """Aggregates metrics across multiple examples."""
    
    def __init__(self):
        self.results = []
    
    def add_result(self, result: AdvancedMetricResult):
        """Add a metric result."""
        self.results.append(result)
    
    def compute_aggregated_metrics(self) -> Dict[str, float]:
        """Compute aggregated metrics across all results."""
        if not self.results:
            return {}
        
        aggregated = {}
        
        # Get all numeric fields from the dataclass
        fields = [field.name for field in AdvancedMetricResult.__dataclass_fields__.values()
                 if field.type in [float, int]]
        
        for field in fields:
            values = [getattr(result, field) for result in self.results]
            aggregated[f"avg_{field}"] = sum(values) / len(values)
            aggregated[f"std_{field}"] = self._std_dev(values)
            aggregated[f"min_{field}"] = min(values)
            aggregated[f"max_{field}"] = max(values)
        
        # Composite scores
        aggregated["composite_accuracy"] = self._composite_accuracy()
        aggregated["composite_quality"] = self._composite_quality()
        aggregated["composite_efficiency"] = self._composite_efficiency()
        
        return aggregated
    
    def _std_dev(self, values: List[float]) -> float:
        """Compute standard deviation."""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return math.sqrt(variance)
    
    def _composite_accuracy(self) -> float:
        """Compute composite accuracy score."""
        if not self.results:
            return 0.0
        
        scores = []
        for result in self.results:
            # Weighted combination of accuracy metrics
            composite = (
                0.4 * result.exact_match +
                0.4 * result.f1_score +
                0.2 * result.semantic_similarity
            )
            scores.append(composite)
        
        return sum(scores) / len(scores)
    
    def _composite_quality(self) -> float:
        """Compute composite quality score."""
        if not self.results:
            return 0.0
        
        scores = []
        for result in self.results:
            # Weighted combination of quality metrics
            composite = (
                0.3 * result.response_quality +
                0.2 * result.truth_consistency +
                0.2 * result.uncertainty_handling +
                0.2 * (1.0 - result.hallucination_rate) +  # Inverse hallucination
                0.1 * result.context_utilization
            )
            scores.append(composite)
        
        return sum(scores) / len(scores)
    
    def _composite_efficiency(self) -> float:
        """Compute composite efficiency score."""
        if not self.results:
            return 0.0
        
        scores = []
        for result in self.results:
            # Weighted combination of efficiency metrics
            composite = (
                0.4 * result.memory_efficiency +
                0.3 * (1.0 / max(1.0, result.processing_time / 1.0)) +  # Inverse time
                0.3 * result.context_utilization
            )
            scores.append(composite)
        
        return sum(scores) / len(scores)
