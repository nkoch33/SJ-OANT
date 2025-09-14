#!/usr/bin/env python3
"""
Deep Model Analysis and Performance Understanding
Analyze patterns, identify bottlenecks, and understand model behavior
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from tmm_pipeline import TMMPipelineFixed
from memory.typed_store import InMemoryStore
import json
import random
import logging
from collections import defaultdict, Counter
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeepModelAnalyzer:
    """Deep analysis of TMM model performance and behavior patterns."""
    
    def __init__(self):
        """Initialize the analyzer."""
        self.api_key = os.getenv('GOOGLE_API_KEY')
        if not self.api_key:
            logger.error("GOOGLE_API_KEY environment variable not set")
            sys.exit(1)
        
        self.memory_store = InMemoryStore()
        self.tmm_pipeline = TMMPipelineFixed(self.api_key)
        
        # Analysis data structures
        self.performance_patterns = defaultdict(list)
        self.memory_usage_patterns = defaultdict(list)
        self.response_quality_patterns = defaultdict(list)
        self.error_patterns = defaultdict(int)
        
    def analyze_response_patterns(self, responses: list) -> dict:
        """Analyze response patterns and quality indicators."""
        patterns = {
            'length_distribution': [],
            'success_indicators': [],
            'question_patterns': [],
            'booking_patterns': [],
            'information_patterns': [],
            'error_patterns': []
        }
        
        for response in responses:
            # Length analysis
            word_count = len(response.split())
            patterns['length_distribution'].append(word_count)
            
            # Success indicator analysis
            success_words = ['successfully', 'confirmed', 'booked', 'reserved', 'done', 'accomplished', 'achieved', 'ready', 'available', 'found', 'located', 'identified']
            success_count = sum(1 for word in success_words if word in response.lower())
            patterns['success_indicators'].append(success_count)
            
            # Question pattern analysis
            question_count = response.count('?')
            patterns['question_patterns'].append(question_count)
            
            # Booking pattern analysis
            booking_words = ['book', 'reserve', 'schedule', 'arrange', 'confirm']
            booking_count = sum(1 for word in booking_words if word in response.lower())
            patterns['booking_patterns'].append(booking_count)
            
            # Information pattern analysis
            info_words = ['information', 'details', 'address', 'phone', 'price', 'time', 'location']
            info_count = sum(1 for word in info_words if word in response.lower())
            patterns['information_patterns'].append(info_count)
            
            # Error pattern analysis
            error_words = ['sorry', 'apologize', 'unable', 'cannot', 'error', 'problem', 'issue']
            error_count = sum(1 for word in error_words if word in response.lower())
            patterns['error_patterns'].append(error_count)
        
        return patterns
    
    def analyze_memory_usage(self, memory_summaries: list) -> dict:
        """Analyze memory usage patterns."""
        patterns = {
            'l1_usage': [],
            'l2_usage': [],
            'l3_usage': [],
            'flagged_usage': [],
            'total_usage': [],
            'memory_efficiency': []
        }
        
        for summary in memory_summaries:
            l1_count = summary.get('l1_count', 0)
            l2_count = summary.get('l2_count', 0)
            l3_count = summary.get('l3_count', 0)
            flagged_count = summary.get('flagged_count', 0)
            total = l1_count + l2_count + l3_count + flagged_count
            
            patterns['l1_usage'].append(l1_count)
            patterns['l2_usage'].append(l2_count)
            patterns['l3_usage'].append(l3_count)
            patterns['flagged_usage'].append(flagged_count)
            patterns['total_usage'].append(total)
            
            # Memory efficiency (L2+L3 vs L1)
            if total > 0:
                efficiency = (l2_count + l3_count) / total
                patterns['memory_efficiency'].append(efficiency)
            else:
                patterns['memory_efficiency'].append(0)
        
        return patterns
    
    def analyze_conversation_flow(self, conversations: list) -> dict:
        """Analyze conversation flow and turn patterns."""
        patterns = {
            'conversation_length': [],
            'turn_patterns': [],
            'context_retention': [],
            'task_progression': []
        }
        
        for conv in conversations:
            turns = conv.get('turns', [])
            patterns['conversation_length'].append(len(turns))
            
            # Analyze turn patterns
            user_turns = [t for t in turns if t.get('speaker') == 'user']
            system_turns = [t for t in turns if t.get('speaker') == 'system']
            patterns['turn_patterns'].append({
                'user_turns': len(user_turns),
                'system_turns': len(system_turns),
                'ratio': len(system_turns) / max(len(user_turns), 1)
            })
            
            # Context retention analysis
            if len(turns) > 1:
                context_retention = self._analyze_context_retention(turns)
                patterns['context_retention'].append(context_retention)
            
            # Task progression analysis
            task_progression = self._analyze_task_progression(turns)
            patterns['task_progression'].append(task_progression)
        
        return patterns
    
    def _analyze_context_retention(self, turns: list) -> float:
        """Analyze how well context is retained across turns."""
        if len(turns) < 2:
            return 0.0
        
        # Simple analysis: check if later turns reference earlier information
        context_retention_score = 0.0
        total_comparisons = 0
        
        for i in range(1, len(turns)):
            current_turn = turns[i].get('text', '').lower()
            previous_turns = [turns[j].get('text', '').lower() for j in range(i)]
            
            # Check for reference to previous information
            for prev_turn in previous_turns:
                if prev_turn and current_turn:
                    # Simple word overlap analysis
                    prev_words = set(prev_turn.split())
                    current_words = set(current_turn.split())
                    overlap = len(prev_words & current_words)
                    total_words = len(prev_words | current_words)
                    
                    if total_words > 0:
                        overlap_ratio = overlap / total_words
                        context_retention_score += overlap_ratio
                        total_comparisons += 1
        
        return context_retention_score / max(total_comparisons, 1)
    
    def _analyze_task_progression(self, turns: list) -> dict:
        """Analyze task progression patterns."""
        progression = {
            'task_initiation': False,
            'information_gathering': False,
            'task_completion': False,
            'confirmation': False
        }
        
        all_text = ' '.join([turn.get('text', '') for turn in turns]).lower()
        
        # Task initiation patterns
        initiation_words = ['need', 'want', 'looking for', 'help', 'book', 'find', 'search']
        progression['task_initiation'] = any(word in all_text for word in initiation_words)
        
        # Information gathering patterns
        info_words = ['what', 'where', 'when', 'how', 'which', 'details', 'information']
        progression['information_gathering'] = any(word in all_text for word in info_words)
        
        # Task completion patterns
        completion_words = ['successfully', 'confirmed', 'booked', 'reserved', 'done', 'completed']
        progression['task_completion'] = any(word in all_text for word in completion_words)
        
        # Confirmation patterns
        confirmation_words = ['confirm', 'yes', 'correct', 'right', 'thank you', 'thanks']
        progression['confirmation'] = any(word in all_text for word in confirmation_words)
        
        return progression
    
    def identify_optimization_opportunities(self, analysis_results: dict) -> list:
        """Identify specific optimization opportunities."""
        opportunities = []
        
        # Response quality opportunities
        response_patterns = analysis_results.get('response_patterns', {})
        
        # Length optimization
        avg_length = sum(response_patterns.get('length_distribution', [0])) / max(len(response_patterns.get('length_distribution', [1])), 1)
        if avg_length < 20:
            opportunities.append("Increase response length for better information delivery")
        elif avg_length > 150:
            opportunities.append("Reduce response length for better user experience")
        
        # Success indicator optimization
        avg_success = sum(response_patterns.get('success_indicators', [0])) / max(len(response_patterns.get('success_indicators', [1])), 1)
        if avg_success < 1:
            opportunities.append("Increase success indicator usage for better task completion detection")
        
        # Memory optimization
        memory_patterns = analysis_results.get('memory_patterns', {})
        avg_efficiency = sum(memory_patterns.get('memory_efficiency', [0])) / max(len(memory_patterns.get('memory_efficiency', [1])), 1)
        if avg_efficiency < 0.3:
            opportunities.append("Improve memory efficiency - more L2/L3 usage needed")
        
        # Context retention optimization
        conversation_patterns = analysis_results.get('conversation_patterns', {})
        avg_retention = sum(conversation_patterns.get('context_retention', [0])) / max(len(conversation_patterns.get('context_retention', [1])), 1)
        if avg_retention < 0.2:
            opportunities.append("Improve context retention across conversation turns")
        
        return opportunities
    
    def run_comprehensive_analysis(self, sample_size: int = 10) -> dict:
        """Run comprehensive analysis on sample conversations."""
        print("🔍 Running Deep Model Analysis...")
        print("=" * 60)
        
        # Sample test conversations from different domains
        test_conversations = [
            # MultiWOZ-style conversations
            {
                'domain': 'multiwoz',
                'turns': [
                    {'speaker': 'user', 'text': 'I need a hotel in Cambridge'},
                    {'speaker': 'system', 'text': 'I can help you find a hotel in Cambridge. What are your preferences?'},
                    {'speaker': 'user', 'text': 'Something cheap with free wifi'},
                    {'speaker': 'system', 'text': 'I found several budget hotels with free wifi in Cambridge.'}
                ]
            },
            # SGD-style conversations
            {
                'domain': 'sgd',
                'turns': [
                    {'speaker': 'user', 'text': 'I want to book a flight from New York to London'},
                    {'speaker': 'system', 'text': 'I can help you book a flight. When would you like to travel?'},
                    {'speaker': 'user', 'text': 'Next Friday'},
                    {'speaker': 'system', 'text': 'I found several flights for next Friday from New York to London.'}
                ]
            },
            # Taskmaster-style conversations
            {
                'domain': 'taskmaster',
                'turns': [
                    {'speaker': 'user', 'text': 'I need a restaurant recommendation'},
                    {'speaker': 'system', 'text': 'I can help you find a restaurant. What type of cuisine do you prefer?'},
                    {'speaker': 'user', 'text': 'Italian food'},
                    {'speaker': 'system', 'text': 'I found several Italian restaurants in your area.'}
                ]
            }
        ]
        
        analysis_results = {
            'response_patterns': {},
            'memory_patterns': {},
            'conversation_patterns': {},
            'optimization_opportunities': []
        }
        
        all_responses = []
        all_memory_summaries = []
        all_conversations = []
        
        for i, conv in enumerate(test_conversations):
            print(f"📊 Analyzing conversation {i+1}/{len(test_conversations)}: {conv['domain']}")
            
            # Process conversation through TMM
            responses = []
            memory_summaries = []
            
            for turn in conv['turns']:
                if turn['speaker'] == 'user':
                    try:
                        response = self.tmm_pipeline.process(turn['text'])
                        responses.append(response)
                        
                        # Get memory summary
                        memory_summary = self.tmm_pipeline.get_memory_summary()
                        memory_summaries.append(memory_summary)
                        
                    except Exception as e:
                        logger.error(f"Error processing turn: {e}")
                        responses.append("Error in processing")
                        memory_summaries.append({})
            
            all_responses.extend(responses)
            all_memory_summaries.extend(memory_summaries)
            all_conversations.append(conv)
        
        # Analyze patterns
        print("📈 Analyzing response patterns...")
        analysis_results['response_patterns'] = self.analyze_response_patterns(all_responses)
        
        print("🧠 Analyzing memory patterns...")
        analysis_results['memory_patterns'] = self.analyze_memory_usage(all_memory_summaries)
        
        print("💬 Analyzing conversation patterns...")
        analysis_results['conversation_patterns'] = self.analyze_conversation_flow(all_conversations)
        
        print("🎯 Identifying optimization opportunities...")
        analysis_results['optimization_opportunities'] = self.identify_optimization_opportunities(analysis_results)
        
        return analysis_results
    
    def print_analysis_report(self, results: dict):
        """Print comprehensive analysis report."""
        print("\n📊 DEEP MODEL ANALYSIS REPORT")
        print("=" * 60)
        
        # Response patterns
        response_patterns = results.get('response_patterns', {})
        print(f"\n📝 RESPONSE PATTERNS:")
        print(f"  Average length: {sum(response_patterns.get('length_distribution', [0])) / max(len(response_patterns.get('length_distribution', [1])), 1):.1f} words")
        print(f"  Average success indicators: {sum(response_patterns.get('success_indicators', [0])) / max(len(response_patterns.get('success_indicators', [1])), 1):.1f}")
        print(f"  Average questions: {sum(response_patterns.get('question_patterns', [0])) / max(len(response_patterns.get('question_patterns', [1])), 1):.1f}")
        print(f"  Average booking words: {sum(response_patterns.get('booking_patterns', [0])) / max(len(response_patterns.get('booking_patterns', [1])), 1):.1f}")
        print(f"  Average error words: {sum(response_patterns.get('error_patterns', [0])) / max(len(response_patterns.get('error_patterns', [1])), 1):.1f}")
        
        # Memory patterns
        memory_patterns = results.get('memory_patterns', {})
        print(f"\n🧠 MEMORY PATTERNS:")
        print(f"  Average L1 usage: {sum(memory_patterns.get('l1_usage', [0])) / max(len(memory_patterns.get('l1_usage', [1])), 1):.1f}")
        print(f"  Average L2 usage: {sum(memory_patterns.get('l2_usage', [0])) / max(len(memory_patterns.get('l2_usage', [1])), 1):.1f}")
        print(f"  Average L3 usage: {sum(memory_patterns.get('l3_usage', [0])) / max(len(memory_patterns.get('l3_usage', [1])), 1):.1f}")
        print(f"  Average memory efficiency: {sum(memory_patterns.get('memory_efficiency', [0])) / max(len(memory_patterns.get('memory_efficiency', [1])), 1):.2f}")
        
        # Conversation patterns
        conversation_patterns = results.get('conversation_patterns', {})
        print(f"\n💬 CONVERSATION PATTERNS:")
        print(f"  Average conversation length: {sum(conversation_patterns.get('conversation_length', [0])) / max(len(conversation_patterns.get('conversation_length', [1])), 1):.1f} turns")
        
        # Optimization opportunities
        opportunities = results.get('optimization_opportunities', [])
        print(f"\n🎯 OPTIMIZATION OPPORTUNITIES:")
        for i, opp in enumerate(opportunities, 1):
            print(f"  {i}. {opp}")
        
        if not opportunities:
            print("  ✅ No major optimization opportunities identified!")

def main():
    """Main function to run deep analysis."""
    analyzer = DeepModelAnalyzer()
    results = analyzer.run_comprehensive_analysis()
    analyzer.print_analysis_report(results)
    
    # Save results
    with open("results/deep_model_analysis.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Analysis results saved to: results/deep_model_analysis.json")
    return results

if __name__ == "__main__":
    main()
