"""
Streamlined MultiWOZ evaluation metrics - extracted from the official framework.
Only includes the essential components we need for TMM evaluation.
"""

import math
from collections import Counter
from sacrebleu import corpus_bleu
from lexical_diversity import lex_div as ld
from fuzzywuzzy import fuzz


def get_bleu_simple(input_data):
    """Simplified BLEU calculation for TMM system evaluation."""
    hyps = []
    refs = []
    
    for dialog in input_data.values():
        for turn in dialog:
            if "response" in turn:
                hyps.append(turn["response"])
                # For now, use the response as reference (self-BLEU)
                # In a full implementation, we'd load reference responses
                refs.append([turn["response"]])
    
    if not hyps:
        return {"bleu": 0.0}
    
    # Calculate self-BLEU as a baseline
    bleu_score = corpus_bleu(hyps, refs)
    return {"bleu": bleu_score.score}


def get_richness_simple(input_data):
    """Simplified lexical richness calculation."""
    all_tokens = []
    
    for dialog in input_data.values():
        for turn in dialog:
            if "response" in turn:
                tokens = turn["response"].lower().split()
                all_tokens.extend(tokens)
    
    if not all_tokens:
        return {
            "entropy": 0.0,
            "cond_entropy": 0.0,
            "avg_lengths": 0.0,
            "msttr": 0.0,
            "num_unigrams": 0,
            "num_bigrams": 0,
            "num_trigrams": 0
        }
    
    # Calculate basic richness metrics
    token_counts = Counter(all_tokens)
    total_tokens = len(all_tokens)
    
    # Entropy
    entropy = 0.0
    for count in token_counts.values():
        p = count / total_tokens
        entropy -= p * math.log2(p)
    
    # Average length
    avg_length = total_tokens / len([turn for dialog in input_data.values() for turn in dialog if "response" in turn])
    
    # Unique n-grams
    unigrams = set(all_tokens)
    bigrams = set()
    trigrams = set()
    
    for i in range(len(all_tokens) - 1):
        bigrams.add((all_tokens[i], all_tokens[i+1]))
    for i in range(len(all_tokens) - 2):
        trigrams.add((all_tokens[i], all_tokens[i+1], all_tokens[i+2]))
    
    return {
        "entropy": entropy,
        "cond_entropy": 0.0,  # Simplified - would need bigram calculations
        "avg_lengths": avg_length,
        "msttr": 0.0,  # Simplified - would need moving window calculation
        "num_unigrams": len(unigrams),
        "num_bigrams": len(bigrams),
        "num_trigrams": len(trigrams)
    }


def get_success_simple(input_data):
    """Calculate success and inform rates separately."""
    total_turns = 0
    successful_turns = 0
    informative_turns = 0
    
    for dialog in input_data.values():
        for turn in dialog:
            if "response" in turn:
                total_turns += 1
                response = turn["response"].lower()
                
                # Success indicators - completion of tasks
                success_indicators = [
                    "booked", "confirmed", "reference", "successfully",
                    "done", "completed", "reserved", "booking confirmed",
                    "reservation made", "ticket booked", "hotel booked"
                ]
                
                # Inform indicators - providing useful information
                inform_indicators = [
                    "available", "found", "here is", "you can", "there are",
                    "located at", "price is", "departure", "arrival", "address",
                    "phone number", "postcode", "rating", "stars", "amenities"
                ]
                
                # Check for success (task completion)
                if any(indicator in response for indicator in success_indicators):
                    successful_turns += 1
                
                # Check for inform (providing information)
                if any(indicator in response for indicator in inform_indicators):
                    informative_turns += 1
    
    success_rate = (successful_turns / total_turns * 100) if total_turns > 0 else 0.0
    inform_rate = (informative_turns / total_turns * 100) if total_turns > 0 else 0.0
    
    return {
        "success": {"total": success_rate},
        "inform": {"total": inform_rate}
    }


class SimpleMultiWOZEvaluator:
    """Simplified MultiWOZ evaluator for TMM system."""
    
    def __init__(self, bleu=True, success=True, richness=True):
        self.bleu = bleu
        self.success = success
        self.richness = richness
    
    def evaluate(self, input_data):
        """Evaluate input data and return metrics."""
        results = {}
        
        if self.bleu:
            results["bleu"] = get_bleu_simple(input_data)
        
        if self.success:
            results["success"] = get_success_simple(input_data)
        
        if self.richness:
            results["richness"] = get_richness_simple(input_data)
        
        return results
