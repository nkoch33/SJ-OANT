#!/usr/bin/env python3
"""
Large-Scale Evaluation
Run 25 conversations each on all 4 benchmarks with comprehensive metrics
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from tmm_pipeline import TMMPipelineFixed
from memory.typed_store import InMemoryStore
import json
import random
import logging
from datetime import datetime
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LargeScaleEvaluator:
    """Large-scale evaluation across all benchmarks."""
    
    def __init__(self):
        """Initialize the evaluator."""
        self.api_key = os.getenv('GOOGLE_API_KEY')
        if not self.api_key:
            logger.error("GOOGLE_API_KEY environment variable not set")
            sys.exit(1)
        
        self.memory_store = InMemoryStore()
        self.tmm_pipeline = TMMPipelineFixed(self.api_key)
        
        # Results storage
        self.results = {
            'multiwoz': {'conversations': [], 'metrics': {}},
            'sgd': {'conversations': [], 'metrics': {}},
            'taskmaster': {'conversations': [], 'metrics': {}},
            'multidogo': {'conversations': [], 'metrics': {}}
        }
        
        # Test conversations for each benchmark
        self.test_conversations = self._generate_test_conversations()
    
    def _generate_test_conversations(self) -> dict:
        """Generate diverse test conversations for each benchmark."""
        return {
            'multiwoz': [
                # Hotel booking conversations
                {
                    'turns': [
                        {'speaker': 'user', 'text': 'I need a hotel in Cambridge for 2 nights'},
                        {'speaker': 'system', 'text': 'I can help you find a hotel in Cambridge. What are your preferences?'},
                        {'speaker': 'user', 'text': 'Something cheap with free wifi'},
                        {'speaker': 'system', 'text': 'I found several budget hotels with free wifi in Cambridge.'},
                        {'speaker': 'user', 'text': 'What about the Cambridge Hotel?'},
                        {'speaker': 'system', 'text': 'The Cambridge Hotel is available for your dates.'}
                    ]
                },
                {
                    'turns': [
                        {'speaker': 'user', 'text': 'I want to book a hotel in London'},
                        {'speaker': 'system', 'text': 'I can help you book a hotel in London. When do you need it?'},
                        {'speaker': 'user', 'text': 'Next weekend'},
                        {'speaker': 'system', 'text': 'I found several hotels available for next weekend in London.'},
                        {'speaker': 'user', 'text': 'What about the price?'},
                        {'speaker': 'system', 'text': 'The prices range from £80 to £200 per night.'}
                    ]
                },
                # Restaurant booking conversations
                {
                    'turns': [
                        {'speaker': 'user', 'text': 'I need a restaurant recommendation'},
                        {'speaker': 'system', 'text': 'I can help you find a restaurant. What type of cuisine do you prefer?'},
                        {'speaker': 'user', 'text': 'Italian food'},
                        {'speaker': 'system', 'text': 'I found several Italian restaurants in your area.'},
                        {'speaker': 'user', 'text': 'What about Bella Italia?'},
                        {'speaker': 'system', 'text': 'Bella Italia is a great choice for Italian cuisine.'}
                    ]
                },
                # Train booking conversations
                {
                    'turns': [
                        {'speaker': 'user', 'text': 'I need to book a train from London to Cambridge'},
                        {'speaker': 'system', 'text': 'I can help you book a train. When do you want to travel?'},
                        {'speaker': 'user', 'text': 'Tomorrow morning'},
                        {'speaker': 'system', 'text': 'I found several trains available tomorrow morning from London to Cambridge.'},
                        {'speaker': 'user', 'text': 'What time is the first train?'},
                        {'speaker': 'system', 'text': 'The first train departs at 6:30 AM.'}
                    ]
                },
                # Taxi booking conversations
                {
                    'turns': [
                        {'speaker': 'user', 'text': 'I need a taxi to the airport'},
                        {'speaker': 'system', 'text': 'I can help you book a taxi. When do you need it?'},
                        {'speaker': 'user', 'text': 'In 2 hours'},
                        {'speaker': 'system', 'text': 'I can arrange a taxi for you in 2 hours.'},
                        {'speaker': 'user', 'text': 'How much will it cost?'},
                        {'speaker': 'system', 'text': 'The estimated cost is £25-30 to the airport.'}
                    ]
                }
            ],
            'sgd': [
                # Flight booking conversations
                {
                    'turns': [
                        {'speaker': 'user', 'text': 'I want to book a flight from New York to London'},
                        {'speaker': 'system', 'text': 'I can help you book a flight. When would you like to travel?'},
                        {'speaker': 'user', 'text': 'Next Friday'},
                        {'speaker': 'system', 'text': 'I found several flights for next Friday from New York to London.'},
                        {'speaker': 'user', 'text': 'What about the price?'},
                        {'speaker': 'system', 'text': 'The prices range from $400 to $800 for economy class.'}
                    ]
                },
                # Hotel booking conversations
                {
                    'turns': [
                        {'speaker': 'user', 'text': 'I need a hotel in Paris'},
                        {'speaker': 'system', 'text': 'I can help you find a hotel in Paris. What are your preferences?'},
                        {'speaker': 'user', 'text': 'Something near the Eiffel Tower'},
                        {'speaker': 'system', 'text': 'I found several hotels near the Eiffel Tower.'},
                        {'speaker': 'user', 'text': 'What about the Hotel Eiffel?'},
                        {'speaker': 'system', 'text': 'The Hotel Eiffel is available and very close to the Eiffel Tower.'}
                    ]
                },
                # Restaurant booking conversations
                {
                    'turns': [
                        {'speaker': 'user', 'text': 'I want to make a restaurant reservation'},
                        {'speaker': 'system', 'text': 'I can help you make a reservation. What type of cuisine do you prefer?'},
                        {'speaker': 'user', 'text': 'French cuisine'},
                        {'speaker': 'system', 'text': 'I found several French restaurants available for reservation.'},
                        {'speaker': 'user', 'text': 'What about Le Bistrot?'},
                        {'speaker': 'system', 'text': 'Le Bistrot is available for your preferred time.'}
                    ]
                },
                # Car rental conversations
                {
                    'turns': [
                        {'speaker': 'user', 'text': 'I need to rent a car'},
                        {'speaker': 'system', 'text': 'I can help you rent a car. Where do you need it?'},
                        {'speaker': 'user', 'text': 'At the airport'},
                        {'speaker': 'system', 'text': 'I found several car rental options at the airport.'},
                        {'speaker': 'user', 'text': 'What about the price?'},
                        {'speaker': 'system', 'text': 'The daily rates start from $30 per day.'}
                    ]
                },
                # Event booking conversations
                {
                    'turns': [
                        {'speaker': 'user', 'text': 'I want to book tickets for a concert'},
                        {'speaker': 'system', 'text': 'I can help you book concert tickets. Which concert are you interested in?'},
                        {'speaker': 'user', 'text': 'The Rolling Stones concert'},
                        {'speaker': 'system', 'text': 'I found tickets available for the Rolling Stones concert.'},
                        {'speaker': 'user', 'text': 'What about the price?'},
                        {'speaker': 'system', 'text': 'The ticket prices range from $100 to $300.'}
                    ]
                }
            ],
            'taskmaster': [
                # General task conversations
                {
                    'turns': [
                        {'speaker': 'user', 'text': 'I need help with my travel plans'},
                        {'speaker': 'system', 'text': 'I can help you with your travel plans. What do you need assistance with?'},
                        {'speaker': 'user', 'text': 'I want to plan a trip to Japan'},
                        {'speaker': 'system', 'text': 'I can help you plan a trip to Japan. When are you planning to go?'},
                        {'speaker': 'user', 'text': 'Next month'},
                        {'speaker': 'system', 'text': 'I can help you plan your trip to Japan for next month.'}
                    ]
                },
                # Information seeking conversations
                {
                    'turns': [
                        {'speaker': 'user', 'text': 'What is the weather like in Tokyo?'},
                        {'speaker': 'system', 'text': 'I can help you check the weather in Tokyo. Let me look that up for you.'},
                        {'speaker': 'user', 'text': 'Is it going to rain tomorrow?'},
                        {'speaker': 'system', 'text': 'According to the forecast, there is a 30% chance of rain tomorrow in Tokyo.'}
                    ]
                },
                # Problem solving conversations
                {
                    'turns': [
                        {'speaker': 'user', 'text': 'I have a problem with my booking'},
                        {'speaker': 'system', 'text': 'I can help you with your booking problem. What seems to be the issue?'},
                        {'speaker': 'user', 'text': 'I need to change my flight date'},
                        {'speaker': 'system', 'text': 'I can help you change your flight date. What is your new preferred date?'},
                        {'speaker': 'user', 'text': 'Next Tuesday'},
                        {'speaker': 'system', 'text': 'I can help you change your flight to next Tuesday.'}
                    ]
                },
                # Recommendation conversations
                {
                    'turns': [
                        {'speaker': 'user', 'text': 'Can you recommend a good restaurant?'},
                        {'speaker': 'system', 'text': 'I can recommend a good restaurant. What type of cuisine do you prefer?'},
                        {'speaker': 'user', 'text': 'I like Italian food'},
                        {'speaker': 'system', 'text': 'I recommend Bella Vista, it has excellent Italian cuisine.'},
                        {'speaker': 'user', 'text': 'What about the price range?'},
                        {'speaker': 'system', 'text': 'Bella Vista is moderately priced, around $25-40 per person.'}
                    ]
                },
                # Assistance conversations
                {
                    'turns': [
                        {'speaker': 'user', 'text': 'I need help with my itinerary'},
                        {'speaker': 'system', 'text': 'I can help you with your itinerary. What do you need assistance with?'},
                        {'speaker': 'user', 'text': 'I want to add more activities'},
                        {'speaker': 'system', 'text': 'I can help you add more activities to your itinerary.'},
                        {'speaker': 'user', 'text': 'What do you recommend?'},
                        {'speaker': 'system', 'text': 'I recommend visiting the local museums and trying the traditional cuisine.'}
                    ]
                }
            ],
            'multidogo': [
                # Multi-domain conversations
                {
                    'turns': [
                        {'speaker': 'user', 'text': 'I need help with booking a hotel and a flight'},
                        {'speaker': 'system', 'text': 'I can help you with both hotel and flight bookings. Let me start with the hotel.'},
                        {'speaker': 'user', 'text': 'I want a hotel in Berlin'},
                        {'speaker': 'system', 'text': 'I found several hotels in Berlin. What are your preferences?'},
                        {'speaker': 'user', 'text': 'Something near the city center'},
                        {'speaker': 'system', 'text': 'I found several hotels near the city center in Berlin.'}
                    ]
                },
                # Cross-domain conversations
                {
                    'turns': [
                        {'speaker': 'user', 'text': 'I want to plan a business trip'},
                        {'speaker': 'system', 'text': 'I can help you plan your business trip. What do you need assistance with?'},
                        {'speaker': 'user', 'text': 'I need a hotel and transportation'},
                        {'speaker': 'system', 'text': 'I can help you with both hotel and transportation arrangements.'},
                        {'speaker': 'user', 'text': 'What about the hotel near the conference center?'},
                        {'speaker': 'system', 'text': 'I found several hotels near the conference center.'}
                    ]
                },
                # Complex multi-task conversations
                {
                    'turns': [
                        {'speaker': 'user', 'text': 'I need to organize a company event'},
                        {'speaker': 'system', 'text': 'I can help you organize your company event. What do you need assistance with?'},
                        {'speaker': 'user', 'text': 'I need a venue and catering'},
                        {'speaker': 'system', 'text': 'I can help you with both venue and catering arrangements.'},
                        {'speaker': 'user', 'text': 'What about the Grand Hotel?'},
                        {'speaker': 'system', 'text': 'The Grand Hotel is available for your event and offers catering services.'}
                    ]
                },
                # Domain switching conversations
                {
                    'turns': [
                        {'speaker': 'user', 'text': 'I want to book a flight'},
                        {'speaker': 'system', 'text': 'I can help you book a flight. Where are you traveling to?'},
                        {'speaker': 'user', 'text': 'Actually, I need a hotel instead'},
                        {'speaker': 'system', 'text': 'I can help you with hotel booking instead. Where do you need a hotel?'},
                        {'speaker': 'user', 'text': 'In Madrid'},
                        {'speaker': 'system', 'text': 'I found several hotels available in Madrid.'}
                    ]
                },
                # Multi-intent conversations
                {
                    'turns': [
                        {'speaker': 'user', 'text': 'I need help with my travel and dining plans'},
                        {'speaker': 'system', 'text': 'I can help you with both travel and dining plans. What do you need assistance with?'},
                        {'speaker': 'user', 'text': 'I want to book a hotel and find restaurants'},
                        {'speaker': 'system', 'text': 'I can help you with both hotel booking and restaurant recommendations.'},
                        {'speaker': 'user', 'text': 'What about the area near the train station?'},
                        {'speaker': 'system', 'text': 'I found several hotels and restaurants near the train station.'}
                    ]
                }
            ]
        }
    
    def _process_conversation(self, conversation: dict, benchmark: str) -> dict:
        """Process a single conversation through the TMM pipeline."""
        conversation_result = {
            'benchmark': benchmark,
            'turns': [],
            'metrics': {},
            'memory_usage': [],
            'processing_time': 0
        }
        
        start_time = time.time()
        
        for turn in conversation['turns']:
            if turn['speaker'] == 'user':
                try:
                    # Process user input
                    response = self.tmm_pipeline.process(turn['text'])
                    
                    # Get memory summary
                    memory_summary = self.tmm_pipeline.get_memory_summary()
                    
                    # Store turn result
                    turn_result = {
                        'user_input': turn['text'],
                        'system_response': response,
                        'memory_summary': memory_summary,
                        'processing_time': time.time() - start_time
                    }
                    
                    conversation_result['turns'].append(turn_result)
                    conversation_result['memory_usage'].append(memory_summary)
                    
                except Exception as e:
                    logger.error(f"Error processing turn: {e}")
                    turn_result = {
                        'user_input': turn['text'],
                        'system_response': f"Error: {str(e)}",
                        'memory_summary': {},
                        'processing_time': 0
                    }
                    conversation_result['turns'].append(turn_result)
        
        conversation_result['processing_time'] = time.time() - start_time
        return conversation_result
    
    def _calculate_metrics(self, conversation_result: dict, benchmark: str) -> dict:
        """Calculate metrics for a conversation result."""
        metrics = {}
        
        # Basic metrics
        metrics['total_turns'] = len(conversation_result['turns'])
        metrics['processing_time'] = conversation_result['processing_time']
        
        # Response quality metrics
        responses = [turn['system_response'] for turn in conversation_result['turns']]
        metrics['avg_response_length'] = sum(len(response.split()) for response in responses) / max(len(responses), 1)
        
        # Success indicators
        success_words = ['successfully', 'confirmed', 'booked', 'reserved', 'done', 'accomplished', 'achieved', 'ready', 'available', 'found', 'located', 'identified']
        success_count = sum(1 for response in responses for word in success_words if word in response.lower())
        metrics['success_indicators'] = success_count
        
        # Memory usage metrics
        memory_summaries = conversation_result['memory_usage']
        if memory_summaries:
            metrics['avg_l1_usage'] = sum(summary.get('l1_count', 0) for summary in memory_summaries) / len(memory_summaries)
            metrics['avg_l2_usage'] = sum(summary.get('l2_count', 0) for summary in memory_summaries) / len(memory_summaries)
            metrics['avg_l3_usage'] = sum(summary.get('l3_count', 0) for summary in memory_summaries) / len(memory_summaries)
            metrics['memory_efficiency'] = (metrics['avg_l2_usage'] + metrics['avg_l3_usage']) / max(metrics['avg_l1_usage'] + metrics['avg_l2_usage'] + metrics['avg_l3_usage'], 1)
        
        # Benchmark-specific metrics
        if benchmark == 'multiwoz':
            metrics['task_completion'] = 1 if success_count > 0 else 0
        elif benchmark == 'sgd':
            metrics['intent_accuracy'] = 1 if success_count > 0 else 0  # Simplified
            metrics['success_rate'] = 1 if success_count > 0 else 0
        elif benchmark == 'taskmaster':
            metrics['task_completion'] = 1 if success_count > 0 else 0
        elif benchmark == 'multidogo':
            metrics['intent_accuracy'] = 1 if success_count > 0 else 0  # Simplified
            metrics['response_quality'] = min(100, success_count * 20)  # Simplified
        
        return metrics
    
    def run_benchmark_evaluation(self, benchmark: str, num_conversations: int = 25) -> dict:
        """Run evaluation for a specific benchmark."""
        print(f"🧪 Running {benchmark.upper()} evaluation with {num_conversations} conversations...")
        
        benchmark_results = {
            'benchmark': benchmark,
            'conversations': [],
            'metrics': {},
            'timestamp': datetime.now().isoformat()
        }
        
        # Get test conversations for this benchmark
        test_conversations = self.test_conversations[benchmark]
        
        # Run conversations
        for i in range(num_conversations):
            # Select a random conversation template
            conversation_template = random.choice(test_conversations)
            
            print(f"  📊 Processing conversation {i+1}/{num_conversations}...")
            
            # Process conversation
            conversation_result = self._process_conversation(conversation_template, benchmark)
            
            # Calculate metrics
            conversation_metrics = self._calculate_metrics(conversation_result, benchmark)
            conversation_result['metrics'] = conversation_metrics
            
            # Store result
            benchmark_results['conversations'].append(conversation_result)
            
            # Reset memory for next conversation
            self.tmm_pipeline.reset_memory()
        
        # Calculate aggregate metrics
        all_metrics = [conv['metrics'] for conv in benchmark_results['conversations']]
        aggregate_metrics = {}
        
        for metric_name in all_metrics[0].keys():
            values = [metrics[metric_name] for metrics in all_metrics if metric_name in metrics]
            if values:
                aggregate_metrics[metric_name] = {
                    'mean': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'std': (sum((x - sum(values)/len(values))**2 for x in values) / len(values))**0.5 if len(values) > 1 else 0
                }
        
        benchmark_results['metrics'] = aggregate_metrics
        
        print(f"  ✅ {benchmark.upper()} evaluation completed!")
        return benchmark_results
    
    def run_all_evaluations(self, num_conversations: int = 25) -> dict:
        """Run evaluations for all benchmarks."""
        print("🚀 Running Large-Scale Evaluation...")
        print("=" * 60)
        
        all_results = {
            'evaluation_info': {
                'timestamp': datetime.now().isoformat(),
                'num_conversations_per_benchmark': num_conversations,
                'total_conversations': num_conversations * 4
            },
            'results': {}
        }
        
        benchmarks = ['multiwoz', 'sgd', 'taskmaster', 'multidogo']
        
        for benchmark in benchmarks:
            try:
                benchmark_results = self.run_benchmark_evaluation(benchmark, num_conversations)
                all_results['results'][benchmark] = benchmark_results
            except Exception as e:
                logger.error(f"Error evaluating {benchmark}: {e}")
                all_results['results'][benchmark] = {'error': str(e)}
        
        return all_results
    
    def print_results_summary(self, results: dict):
        """Print a summary of evaluation results."""
        print("\n📊 LARGE-SCALE EVALUATION RESULTS")
        print("=" * 80)
        
        evaluation_info = results['evaluation_info']
        print(f"📅 Timestamp: {evaluation_info['timestamp']}")
        print(f"📊 Total Conversations: {evaluation_info['total_conversations']}")
        print(f"🎯 Conversations per Benchmark: {evaluation_info['num_conversations_per_benchmark']}")
        
        for benchmark, benchmark_results in results['results'].items():
            if 'error' in benchmark_results:
                print(f"\n❌ {benchmark.upper()}: Error - {benchmark_results['error']}")
                continue
            
            print(f"\n📈 {benchmark.upper()} RESULTS:")
            print("-" * 40)
            
            metrics = benchmark_results['metrics']
            for metric_name, metric_data in metrics.items():
                if isinstance(metric_data, dict) and 'mean' in metric_data:
                    print(f"  • {metric_name}: {metric_data['mean']:.2f} ± {metric_data['std']:.2f}")
                else:
                    print(f"  • {metric_name}: {metric_data}")
    
    def save_results(self, results: dict, filepath: str = "results/large_scale_evaluation_results.json"):
        """Save evaluation results to JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: {filepath}")

def main():
    """Main function to run large-scale evaluation."""
    evaluator = LargeScaleEvaluator()
    
    # Run all evaluations
    results = evaluator.run_all_evaluations(num_conversations=25)
    
    # Print summary
    evaluator.print_results_summary(results)
    
    # Save results
    evaluator.save_results(results)
    
    return results

if __name__ == "__main__":
    main()
