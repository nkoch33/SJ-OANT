#!/usr/bin/env python3
"""
Evaluation Framework Definitions
Clear definitions of metrics and evaluation criteria for all 4 benchmarks
"""

import json
import os
from typing import Dict, List, Any

class EvaluationFrameworkDefinitions:
    """Comprehensive evaluation framework definitions for all benchmarks."""
    
    def __init__(self):
        """Initialize the evaluation framework definitions."""
        self.frameworks = {}
        self._define_frameworks()
    
    def _define_frameworks(self):
        """Define evaluation frameworks for all benchmarks."""
        
        # MultiWOZ Framework
        self.frameworks['multiwoz'] = {
            'name': 'MultiWOZ Evaluation Framework',
            'description': 'Official MultiWOZ benchmark evaluation metrics',
            'source': 'https://github.com/budzianowski/multiwoz',
            'metrics': {
                'bleu_score': {
                    'name': 'BLEU Score',
                    'description': 'Bilingual Evaluation Understudy - measures n-gram overlap with reference responses',
                    'range': '0-100 (higher is better)',
                    'interpretation': {
                        '0-5': 'Poor - minimal overlap with reference',
                        '5-10': 'Fair - some overlap with reference',
                        '10-20': 'Good - substantial overlap with reference',
                        '20+': 'Excellent - high overlap with reference'
                    },
                    'calculation': 'sacrebleu corpus_bleu with 4-gram precision',
                    'research_context': 'Standard metric for dialogue generation evaluation'
                },
                'rouge_score': {
                    'name': 'ROUGE Score',
                    'description': 'Recall-Oriented Understudy for Gisting Evaluation - measures recall of n-grams',
                    'range': '0-100 (higher is better)',
                    'interpretation': {
                        '0-10': 'Poor - low recall of reference content',
                        '10-25': 'Fair - moderate recall of reference content',
                        '25-50': 'Good - good recall of reference content',
                        '50+': 'Excellent - high recall of reference content'
                    },
                    'calculation': 'rouge_score ROUGE with F1 measure',
                    'research_context': 'Standard metric for text summarization and dialogue evaluation'
                },
                'semantic_similarity': {
                    'name': 'Semantic Similarity',
                    'description': 'Cosine similarity between generated and reference response embeddings',
                    'range': '0-1 (higher is better)',
                    'interpretation': {
                        '0-0.3': 'Poor - low semantic similarity',
                        '0.3-0.5': 'Fair - moderate semantic similarity',
                        '0.5-0.7': 'Good - good semantic similarity',
                        '0.7+': 'Excellent - high semantic similarity'
                    },
                    'calculation': 'sentence-transformers all-MiniLM-L6-v2 cosine similarity',
                    'research_context': 'Measures semantic meaning preservation in responses'
                },
                'task_completion': {
                    'name': 'Task Completion Rate',
                    'description': 'Percentage of conversations where the user\'s goal was successfully achieved',
                    'range': '0-100% (higher is better)',
                    'interpretation': {
                        '0-30%': 'Poor - most tasks not completed',
                        '30-60%': 'Fair - some tasks completed',
                        '60-80%': 'Good - most tasks completed',
                        '80%+': 'Excellent - nearly all tasks completed'
                    },
                    'calculation': 'Success indicator detection in final responses',
                    'research_context': 'Primary metric for task-oriented dialogue systems'
                }
            },
            'evaluation_criteria': {
                'conversation_flow': 'Natural progression from task initiation to completion',
                'information_accuracy': 'Correctness of provided information',
                'user_satisfaction': 'Appropriate responses to user requests',
                'domain_expertise': 'Knowledge of travel and booking domains'
            }
        }
        
        # SGD Framework
        self.frameworks['sgd'] = {
            'name': 'Schema-Guided Dialogue Evaluation Framework',
            'description': 'Official Google SGD benchmark evaluation metrics',
            'source': 'https://github.com/google-research-datasets/dstc8-schema-guided-dialogue',
            'metrics': {
                'intent_accuracy': {
                    'name': 'Intent Classification Accuracy',
                    'description': 'Percentage of user intents correctly identified',
                    'range': '0-100% (higher is better)',
                    'interpretation': {
                        '0-50%': 'Poor - many intents misclassified',
                        '50-75%': 'Fair - some intents correctly identified',
                        '75-90%': 'Good - most intents correctly identified',
                        '90%+': 'Excellent - nearly all intents correctly identified'
                    },
                    'calculation': 'Exact match between predicted and ground truth intents',
                    'research_context': 'Critical for understanding user goals in dialogue systems'
                },
                'slot_f1_score': {
                    'name': 'Slot Filling F1 Score',
                    'description': 'F1 score for entity extraction and slot filling',
                    'range': '0-1 (higher is better)',
                    'interpretation': {
                        '0-0.3': 'Poor - many entities missed or incorrectly extracted',
                        '0.3-0.6': 'Fair - some entities correctly extracted',
                        '0.6-0.8': 'Good - most entities correctly extracted',
                        '0.8+': 'Excellent - nearly all entities correctly extracted'
                    },
                    'calculation': 'Precision, Recall, F1 for entity extraction',
                    'research_context': 'Essential for information extraction in dialogue systems'
                },
                'success_rate': {
                    'name': 'Success Rate',
                    'description': 'Percentage of dialogues where the system successfully fulfills user requests',
                    'range': '0-100% (higher is better)',
                    'interpretation': {
                        '0-40%': 'Poor - most requests not fulfilled',
                        '40-70%': 'Fair - some requests fulfilled',
                        '70-90%': 'Good - most requests fulfilled',
                        '90%+': 'Excellent - nearly all requests fulfilled'
                    },
                    'calculation': 'Success indicator detection in system responses',
                    'research_context': 'Primary metric for task completion in schema-guided dialogue'
                },
                'bleu_score': {
                    'name': 'BLEU Score',
                    'description': 'Bilingual Evaluation Understudy for response quality',
                    'range': '0-100 (higher is better)',
                    'interpretation': {
                        '0-5': 'Poor - responses don\'t match reference patterns',
                        '5-15': 'Fair - some overlap with reference patterns',
                        '15-30': 'Good - good overlap with reference patterns',
                        '30+': 'Excellent - high overlap with reference patterns'
                    },
                    'calculation': 'sacrebleu corpus_bleu with 4-gram precision',
                    'research_context': 'Standard metric for response generation quality'
                }
            },
            'evaluation_criteria': {
                'schema_compliance': 'Adherence to predefined dialogue schemas',
                'entity_extraction': 'Accuracy of slot filling and entity recognition',
                'intent_understanding': 'Correct interpretation of user intents',
                'response_appropriateness': 'Suitability of responses to user requests'
            }
        }
        
        # Taskmaster Framework
        self.frameworks['taskmaster'] = {
            'name': 'Taskmaster Evaluation Framework',
            'description': 'Google Taskmaster benchmark evaluation metrics',
            'source': 'https://github.com/google-research-datasets/Taskmaster',
            'metrics': {
                'bleu_score': {
                    'name': 'BLEU Score',
                    'description': 'Bilingual Evaluation Understudy for response quality',
                    'range': '0-100 (higher is better)',
                    'interpretation': {
                        '0-10': 'Poor - responses don\'t match reference patterns',
                        '10-25': 'Fair - some overlap with reference patterns',
                        '25-50': 'Good - good overlap with reference patterns',
                        '50+': 'Excellent - high overlap with reference patterns'
                    },
                    'calculation': 'sacrebleu corpus_bleu with 4-gram precision',
                    'research_context': 'Standard metric for response generation quality'
                },
                'rouge_score': {
                    'name': 'ROUGE Score',
                    'description': 'Recall-Oriented Understudy for Gisting Evaluation',
                    'range': '0-100 (higher is better)',
                    'interpretation': {
                        '0-20': 'Poor - low recall of reference content',
                        '20-40': 'Fair - moderate recall of reference content',
                        '40-70': 'Good - good recall of reference content',
                        '70+': 'Excellent - high recall of reference content'
                    },
                    'calculation': 'rouge_score ROUGE with F1 measure',
                    'research_context': 'Standard metric for text summarization and dialogue evaluation'
                },
                'semantic_similarity': {
                    'name': 'Semantic Similarity',
                    'description': 'Cosine similarity between generated and reference response embeddings',
                    'range': '0-1 (higher is better)',
                    'interpretation': {
                        '0-0.2': 'Poor - low semantic similarity',
                        '0.2-0.4': 'Fair - moderate semantic similarity',
                        '0.4-0.6': 'Good - good semantic similarity',
                        '0.6+': 'Excellent - high semantic similarity'
                    },
                    'calculation': 'sentence-transformers all-MiniLM-L6-v2 cosine similarity',
                    'research_context': 'Measures semantic meaning preservation in responses'
                },
                'task_completion': {
                    'name': 'Task Completion Rate',
                    'description': 'Percentage of conversations where the user\'s goal was successfully achieved',
                    'range': '0-100% (higher is better)',
                    'interpretation': {
                        '0-25%': 'Poor - most tasks not completed',
                        '25-50%': 'Fair - some tasks completed',
                        '50-75%': 'Good - most tasks completed',
                        '75%+': 'Excellent - nearly all tasks completed'
                    },
                    'calculation': 'Success indicator detection in final responses',
                    'research_context': 'Primary metric for task-oriented dialogue systems'
                }
            },
            'evaluation_criteria': {
                'conversation_naturalness': 'Natural flow and progression of conversations',
                'task_understanding': 'Correct interpretation of user tasks and goals',
                'response_quality': 'Appropriate and helpful responses',
                'domain_adaptation': 'Ability to handle diverse task domains'
            }
        }
        
        # MultiDoGO Framework
        self.frameworks['multidogo'] = {
            'name': 'MultiDoGO Evaluation Framework',
            'description': 'MultiDoGO benchmark evaluation metrics',
            'source': 'Custom evaluation framework for MultiDoGO dataset',
            'metrics': {
                'intent_classification_accuracy': {
                    'name': 'Intent Classification Accuracy',
                    'description': 'Percentage of user intents correctly identified',
                    'range': '0-100% (higher is better)',
                    'interpretation': {
                        '0-30%': 'Poor - many intents misclassified',
                        '30-60%': 'Fair - some intents correctly identified',
                        '60-80%': 'Good - most intents correctly identified',
                        '80%+': 'Excellent - nearly all intents correctly identified'
                    },
                    'calculation': 'Pattern matching against intent keywords',
                    'research_context': 'Critical for understanding user goals in dialogue systems'
                },
                'slot_filling_f1_score': {
                    'name': 'Slot Filling F1 Score',
                    'description': 'F1 score for entity extraction and slot filling',
                    'range': '0-1 (higher is better)',
                    'interpretation': {
                        '0-0.2': 'Poor - many entities missed or incorrectly extracted',
                        '0.2-0.5': 'Fair - some entities correctly extracted',
                        '0.5-0.7': 'Good - most entities correctly extracted',
                        '0.7+': 'Excellent - nearly all entities correctly extracted'
                    },
                    'calculation': 'Precision, Recall, F1 for entity extraction',
                    'research_context': 'Essential for information extraction in dialogue systems'
                },
                'domain_adaptation_score': {
                    'name': 'Domain Adaptation Score',
                    'description': 'Ability to adapt to different domains and contexts',
                    'range': '0-100% (higher is better)',
                    'interpretation': {
                        '0-40%': 'Poor - limited domain adaptation',
                        '40-70%': 'Fair - moderate domain adaptation',
                        '70-90%': 'Good - good domain adaptation',
                        '90%+': 'Excellent - excellent domain adaptation'
                    },
                    'calculation': 'Cross-domain performance consistency',
                    'research_context': 'Important for general-purpose dialogue systems'
                },
                'response_quality_score': {
                    'name': 'Response Quality Score',
                    'description': 'Overall quality of generated responses',
                    'range': '0-100% (higher is better)',
                    'interpretation': {
                        '0-30%': 'Poor - low quality responses',
                        '30-60%': 'Fair - moderate quality responses',
                        '60-80%': 'Good - good quality responses',
                        '80%+': 'Excellent - high quality responses'
                    },
                    'calculation': 'Combined BLEU and quality indicators',
                    'research_context': 'Comprehensive measure of response quality'
                }
            },
            'evaluation_criteria': {
                'multi_domain_handling': 'Ability to handle diverse domains and contexts',
                'intent_recognition': 'Accuracy of intent classification across domains',
                'entity_extraction': 'Quality of slot filling and entity recognition',
                'response_appropriateness': 'Suitability of responses to user requests'
            }
        }
    
    def get_framework(self, benchmark: str) -> Dict[str, Any]:
        """Get evaluation framework for a specific benchmark."""
        return self.frameworks.get(benchmark, {})
    
    def get_all_frameworks(self) -> Dict[str, Any]:
        """Get all evaluation frameworks."""
        return self.frameworks
    
    def print_framework_summary(self, benchmark: str):
        """Print a summary of the evaluation framework for a benchmark."""
        framework = self.get_framework(benchmark)
        if not framework:
            print(f"❌ No framework found for benchmark: {benchmark}")
            return
        
        print(f"\n📊 {framework['name']}")
        print("=" * 60)
        print(f"📝 Description: {framework['description']}")
        print(f"🔗 Source: {framework['source']}")
        
        print(f"\n📈 Metrics:")
        for metric_name, metric_info in framework['metrics'].items():
            print(f"  • {metric_info['name']}")
            print(f"    Range: {metric_info['range']}")
            print(f"    Description: {metric_info['description']}")
            print(f"    Research Context: {metric_info['research_context']}")
            print()
        
        print(f"🎯 Evaluation Criteria:")
        for criterion, description in framework['evaluation_criteria'].items():
            print(f"  • {criterion.replace('_', ' ').title()}: {description}")
    
    def print_all_frameworks(self):
        """Print summaries of all evaluation frameworks."""
        print("📊 COMPREHENSIVE EVALUATION FRAMEWORKS")
        print("=" * 80)
        
        for benchmark in ['multiwoz', 'sgd', 'taskmaster', 'multidogo']:
            self.print_framework_summary(benchmark)
            print("\n" + "=" * 80)
    
    def save_frameworks(self, filepath: str = "results/evaluation_frameworks.json"):
        """Save evaluation frameworks to JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.frameworks, f, indent=2)
        
        print(f"💾 Evaluation frameworks saved to: {filepath}")

def main():
    """Main function to display evaluation frameworks."""
    frameworks = EvaluationFrameworkDefinitions()
    
    # Print all frameworks
    frameworks.print_all_frameworks()
    
    # Save frameworks
    frameworks.save_frameworks()
    
    return frameworks

if __name__ == "__main__":
    main()
