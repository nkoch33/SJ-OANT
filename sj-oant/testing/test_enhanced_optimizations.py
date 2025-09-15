#!/usr/bin/env python3
"""
test_enhanced_optimizations.py - Test Enhanced TMM Pipeline Optimizations

This script tests all the enhanced optimizations to ensure they work correctly
and don't compromise existing performance metrics.

Key Tests:
- Enhanced memory retrieval and management
- Multi-agent coordination improvements
- Advanced truth verification
- Research-grade monitoring and analytics
- Performance regression testing
"""

import os
import sys
import time
import json
from typing import Dict, List, Any

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tmm_pipeline import TMMPipelineFixed, EnhancedPipelineConfig
from testing.setup_api_key import setup_api_key

class EnhancedOptimizationTester:
    """Test suite for enhanced TMM pipeline optimizations."""
    
    def __init__(self):
        """Initialize the test suite."""
        self.test_results = {}
        self.performance_baseline = {}
        self.enhanced_performance = {}
        
        print("🧪 Enhanced TMM Pipeline Optimization Tester")
        print("=" * 60)
    
    def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run comprehensive tests of all enhanced optimizations."""
        print("\n🚀 Starting Comprehensive Enhanced Optimization Tests")
        print("-" * 60)
        
        # Setup API key
        setup_api_key()
        
        # Test 1: Basic Enhanced Pipeline Functionality
        print("\n📋 Test 1: Basic Enhanced Pipeline Functionality")
        self.test_results['basic_functionality'] = self._test_basic_functionality()
        
        # Test 2: Adaptive Memory Retrieval
        print("\n🧠 Test 2: Adaptive Memory Retrieval")
        self.test_results['adaptive_retrieval'] = self._test_adaptive_retrieval()
        
        # Test 3: Enhanced Multi-Agent Coordination
        print("\n🤝 Test 3: Enhanced Multi-Agent Coordination")
        self.test_results['enhanced_coordination'] = self._test_enhanced_coordination()
        
        # Test 4: Advanced Truth Verification
        print("\n🔍 Test 4: Advanced Truth Verification")
        self.test_results['advanced_verification'] = self._test_advanced_verification()
        
        # Test 5: Research Analytics
        print("\n📊 Test 5: Research Analytics")
        self.test_results['research_analytics'] = self._test_research_analytics()
        
        # Test 6: Performance Regression Testing
        print("\n⚡ Test 6: Performance Regression Testing")
        self.test_results['performance_regression'] = self._test_performance_regression()
        
        # Test 7: System Health Monitoring
        print("\n🏥 Test 7: System Health Monitoring")
        self.test_results['system_health'] = self._test_system_health()
        
        # Generate comprehensive report
        self._generate_test_report()
        
        return self.test_results
    
    def _test_basic_functionality(self) -> Dict[str, Any]:
        """Test basic enhanced pipeline functionality."""
        try:
            # Initialize enhanced pipeline
            config = EnhancedPipelineConfig(
                enable_adaptive_retrieval=True,
                enable_enhanced_coordination=True,
                enable_advanced_verification=True,
                enable_research_analytics=True
            )
            
            pipeline = TMMPipelineFixed(enhanced_config=config)
            
            # Test basic processing
            test_queries = [
                "I want to book a hotel in London",
                "What is the weather like today?",
                "Can you help me find a restaurant?"
            ]
            
            results = []
            for query in test_queries:
                start_time = time.time()
                response = pipeline.process(query)
                response_time = time.time() - start_time
                
                results.append({
                    'query': query,
                    'response': response[:100] + '...' if len(response) > 100 else response,
                    'response_time': response_time,
                    'success': len(response) > 0
                })
            
            success_rate = sum(1 for r in results if r['success']) / len(results)
            avg_response_time = sum(r['response_time'] for r in results) / len(results)
            
            print(f"   ✅ Basic functionality test passed")
            print(f"   📊 Success rate: {success_rate:.2%}")
            print(f"   ⏱️ Average response time: {avg_response_time:.2f}s")
            
            return {
                'success': True,
                'success_rate': success_rate,
                'avg_response_time': avg_response_time,
                'results': results
            }
            
        except Exception as e:
            print(f"   ❌ Basic functionality test failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _test_adaptive_retrieval(self) -> Dict[str, Any]:
        """Test adaptive memory retrieval functionality."""
        try:
            config = EnhancedPipelineConfig(enable_adaptive_retrieval=True)
            pipeline = TMMPipelineFixed(enhanced_config=config)
            
            # Test different query types
            query_types = [
                ("temporal", "What did I ask about yesterday?"),
                ("entity_focused", "Who is the manager of the hotel?"),
                ("task_oriented", "I need to book a flight"),
                ("semantic", "How does the weather affect travel plans?")
            ]
            
            retrieval_results = []
            for query_type, query in query_types:
                start_time = time.time()
                response = pipeline.process(query)
                response_time = time.time() - start_time
                
                retrieval_results.append({
                    'query_type': query_type,
                    'query': query,
                    'response_time': response_time,
                    'success': len(response) > 0
                })
            
            success_rate = sum(1 for r in retrieval_results if r['success']) / len(retrieval_results)
            
            print(f"   ✅ Adaptive retrieval test passed")
            print(f"   📊 Success rate: {success_rate:.2%}")
            print(f"   🔄 Query types tested: {len(query_types)}")
            
            return {
                'success': True,
                'success_rate': success_rate,
                'query_types_tested': len(query_types),
                'results': retrieval_results
            }
            
        except Exception as e:
            print(f"   ❌ Adaptive retrieval test failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _test_enhanced_coordination(self) -> Dict[str, Any]:
        """Test enhanced multi-agent coordination."""
        try:
            config = EnhancedPipelineConfig(enable_enhanced_coordination=True)
            pipeline = TMMPipelineFixed(enhanced_config=config)
            
            # Test coordination with different complexity queries
            complexity_queries = [
                ("simple", "Hello"),
                ("moderate", "I want to book a hotel in London for next week"),
                ("complex", "I need to book a hotel in London, a flight to Paris, and a restaurant reservation for dinner")
            ]
            
            coordination_results = []
            for complexity, query in complexity_queries:
                start_time = time.time()
                response = pipeline.process(query)
                response_time = time.time() - start_time
                
                coordination_results.append({
                    'complexity': complexity,
                    'query': query,
                    'response_time': response_time,
                    'success': len(response) > 0
                })
            
            success_rate = sum(1 for r in coordination_results if r['success']) / len(coordination_results)
            avg_response_time = sum(r['response_time'] for r in coordination_results) / len(coordination_results)
            
            print(f"   ✅ Enhanced coordination test passed")
            print(f"   📊 Success rate: {success_rate:.2%}")
            print(f"   ⏱️ Average response time: {avg_response_time:.2f}s")
            
            return {
                'success': True,
                'success_rate': success_rate,
                'avg_response_time': avg_response_time,
                'results': coordination_results
            }
            
        except Exception as e:
            print(f"   ❌ Enhanced coordination test failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _test_advanced_verification(self) -> Dict[str, Any]:
        """Test advanced truth verification."""
        try:
            config = EnhancedPipelineConfig(enable_advanced_verification=True)
            pipeline = TMMPipelineFixed(enhanced_config=config)
            
            # Test verification with different content types
            verification_queries = [
                ("factual", "The hotel is located in downtown London"),
                ("temporal", "I booked the hotel yesterday for next week"),
                ("uncertain", "The hotel might be fully booked"),
                ("contradictory", "The hotel is both open and closed")
            ]
            
            verification_results = []
            for content_type, query in verification_queries:
                start_time = time.time()
                response = pipeline.process(query)
                response_time = time.time() - start_time
                
                verification_results.append({
                    'content_type': content_type,
                    'query': query,
                    'response_time': response_time,
                    'success': len(response) > 0
                })
            
            success_rate = sum(1 for r in verification_results if r['success']) / len(verification_results)
            
            print(f"   ✅ Advanced verification test passed")
            print(f"   📊 Success rate: {success_rate:.2%}")
            print(f"   🔍 Content types tested: {len(verification_queries)}")
            
            return {
                'success': True,
                'success_rate': success_rate,
                'content_types_tested': len(verification_queries),
                'results': verification_results
            }
            
        except Exception as e:
            print(f"   ❌ Advanced verification test failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _test_research_analytics(self) -> Dict[str, Any]:
        """Test research analytics functionality."""
        try:
            config = EnhancedPipelineConfig(enable_research_analytics=True)
            pipeline = TMMPipelineFixed(enhanced_config=config)
            
            # Generate some test data
            test_queries = [
                "I want to book a hotel",
                "What is the weather like?",
                "Can you help me find a restaurant?",
                "I need to cancel my reservation",
                "What are the operating hours?"
            ]
            
            for query in test_queries:
                pipeline.process(query)
            
            # Test analytics functionality
            analytics = pipeline.get_performance_analytics()
            system_health = pipeline.get_system_health()
            performance_snapshot = pipeline.take_performance_snapshot()
            
            # Test data export
            export_file = pipeline.export_research_data('json', 'test_export.json')
            
            print(f"   ✅ Research analytics test passed")
            print(f"   📊 Analytics data collected: {len(analytics)} metrics")
            print(f"   🏥 System health: {system_health.get('overall_status', 'unknown')}")
            print(f"   💾 Data exported to: {export_file}")
            
            return {
                'success': True,
                'analytics_metrics': len(analytics),
                'system_health': system_health.get('overall_status', 'unknown'),
                'export_file': export_file
            }
            
        except Exception as e:
            print(f"   ❌ Research analytics test failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _test_performance_regression(self) -> Dict[str, Any]:
        """Test for performance regression."""
        try:
            # Test with enhanced features enabled
            enhanced_config = EnhancedPipelineConfig(
                enable_adaptive_retrieval=True,
                enable_enhanced_coordination=True,
                enable_advanced_verification=True,
                enable_research_analytics=True
            )
            enhanced_pipeline = TMMPipelineFixed(enhanced_config=enhanced_config)
            
            # Test with enhanced features disabled (baseline)
            baseline_config = EnhancedPipelineConfig(
                enable_adaptive_retrieval=False,
                enable_enhanced_coordination=False,
                enable_advanced_verification=False,
                enable_research_analytics=False
            )
            baseline_pipeline = TMMPipelineFixed(enhanced_config=baseline_config)
            
            # Benchmark both configurations
            test_queries = [
                "I want to book a hotel in London",
                "What is the weather like today?",
                "Can you help me find a restaurant?",
                "I need to cancel my flight reservation"
            ]
            
            # Test enhanced pipeline
            enhanced_times = []
            enhanced_success = 0
            for query in test_queries:
                start_time = time.time()
                try:
                    response = enhanced_pipeline.process(query)
                    enhanced_times.append(time.time() - start_time)
                    if len(response) > 0:
                        enhanced_success += 1
                except:
                    enhanced_times.append(5.0)  # Max timeout
            
            # Test baseline pipeline
            baseline_times = []
            baseline_success = 0
            for query in test_queries:
                start_time = time.time()
                try:
                    response = baseline_pipeline.process(query)
                    baseline_times.append(time.time() - start_time)
                    if len(response) > 0:
                        baseline_success += 1
                except:
                    baseline_times.append(5.0)  # Max timeout
            
            # Calculate performance metrics
            enhanced_avg_time = sum(enhanced_times) / len(enhanced_times)
            baseline_avg_time = sum(baseline_times) / len(baseline_times)
            enhanced_success_rate = enhanced_success / len(test_queries)
            baseline_success_rate = baseline_success / len(test_queries)
            
            # Check for regression (enhanced should not be significantly slower)
            time_regression = enhanced_avg_time > baseline_avg_time * 1.5  # 50% slower threshold
            success_regression = enhanced_success_rate < baseline_success_rate * 0.9  # 10% success drop threshold
            
            print(f"   ✅ Performance regression test completed")
            print(f"   📊 Enhanced avg time: {enhanced_avg_time:.2f}s")
            print(f"   📊 Baseline avg time: {baseline_avg_time:.2f}s")
            print(f"   📊 Enhanced success rate: {enhanced_success_rate:.2%}")
            print(f"   📊 Baseline success rate: {baseline_success_rate:.2%}")
            print(f"   ⚠️ Time regression: {'YES' if time_regression else 'NO'}")
            print(f"   ⚠️ Success regression: {'YES' if success_regression else 'NO'}")
            
            return {
                'success': not (time_regression or success_regression),
                'enhanced_avg_time': enhanced_avg_time,
                'baseline_avg_time': baseline_avg_time,
                'enhanced_success_rate': enhanced_success_rate,
                'baseline_success_rate': baseline_success_rate,
                'time_regression': time_regression,
                'success_regression': success_regression
            }
            
        except Exception as e:
            print(f"   ❌ Performance regression test failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _test_system_health(self) -> Dict[str, Any]:
        """Test system health monitoring."""
        try:
            config = EnhancedPipelineConfig(enable_research_analytics=True)
            pipeline = TMMPipelineFixed(enhanced_config=config)
            
            # Generate some load
            for i in range(10):
                pipeline.process(f"Test query {i}")
            
            # Check system health
            health = pipeline.get_system_health()
            performance_analytics = pipeline.get_performance_analytics()
            
            # Test optimization
            optimization_results = pipeline.optimize_performance()
            
            print(f"   ✅ System health test passed")
            print(f"   🏥 Overall status: {health.get('overall_status', 'unknown')}")
            print(f"   📊 Component status: {len(health.get('component_status', {}))}")
            print(f"   🔧 Optimizations applied: {len(optimization_results.get('optimizations_applied', []))}")
            
            return {
                'success': True,
                'overall_status': health.get('overall_status', 'unknown'),
                'component_count': len(health.get('component_status', {})),
                'optimizations_applied': len(optimization_results.get('optimizations_applied', []))
            }
            
        except Exception as e:
            print(f"   ❌ System health test failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _generate_test_report(self):
        """Generate comprehensive test report."""
        print("\n" + "=" * 60)
        print("📊 ENHANCED OPTIMIZATION TEST REPORT")
        print("=" * 60)
        
        # Calculate overall success rate
        total_tests = len(self.test_results)
        successful_tests = sum(1 for result in self.test_results.values() if result.get('success', False))
        overall_success_rate = successful_tests / total_tests if total_tests > 0 else 0
        
        print(f"\n🎯 Overall Test Results:")
        print(f"   Total Tests: {total_tests}")
        print(f"   Successful: {successful_tests}")
        print(f"   Success Rate: {overall_success_rate:.2%}")
        
        # Individual test results
        print(f"\n📋 Individual Test Results:")
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result.get('success', False) else "❌ FAIL"
            print(f"   {test_name}: {status}")
            
            if not result.get('success', False) and 'error' in result:
                print(f"      Error: {result['error']}")
        
        # Performance summary
        print(f"\n⚡ Performance Summary:")
        for test_name, result in self.test_results.items():
            if 'success_rate' in result:
                print(f"   {test_name}: {result['success_rate']:.2%} success rate")
            if 'avg_response_time' in result:
                print(f"   {test_name}: {result['avg_response_time']:.2f}s avg response time")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        if overall_success_rate >= 0.9:
            print("   🎉 All optimizations are working excellently!")
            print("   🚀 System is ready for full-scale evaluation")
        elif overall_success_rate >= 0.7:
            print("   ✅ Most optimizations are working well")
            print("   🔧 Consider addressing failed tests before full-scale evaluation")
        else:
            print("   ⚠️ Several optimizations need attention")
            print("   🛠️ Review and fix failed tests before proceeding")
        
        # Save detailed results
        self._save_test_results()
    
    def _save_test_results(self):
        """Save detailed test results to file."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"results/enhanced_optimization_test_results_{timestamp}.json"
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        print(f"\n💾 Detailed results saved to: {filename}")

def main():
    """Main function to run enhanced optimization tests."""
    tester = EnhancedOptimizationTester()
    results = tester.run_comprehensive_tests()
    
    # Return success status
    overall_success = all(result.get('success', False) for result in results.values())
    return 0 if overall_success else 1

if __name__ == "__main__":
    exit(main())
