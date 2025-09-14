#!/usr/bin/env python3
"""
Comprehensive Results Analysis
Interpret and document results for each benchmark with detailed analysis
"""

import json
import os
from typing import Dict, List, Any
from datetime import datetime

class ComprehensiveResultsAnalyzer:
    """Comprehensive analysis of TMM model results across all benchmarks."""
    
    def __init__(self):
        """Initialize the analyzer."""
        self.results_dir = "results"
        self.analysis_results = {}
    
    def load_results(self) -> Dict[str, Any]:
        """Load all benchmark results."""
        results = {}
        
        # Load individual benchmark results
        benchmark_files = {
            'multiwoz': 'multiwoz_standard_evaluation_results.json',
            'sgd': 'sgd_optimization_test_results.json',
            'taskmaster': 'taskmaster_optimization_test_results.json',
            'multidogo': 'multidogo_optimization_test_results.json'
        }
        
        for benchmark, filename in benchmark_files.items():
            filepath = os.path.join(self.results_dir, filename)
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    results[benchmark] = json.load(f)
            else:
                print(f"⚠️ Warning: {filename} not found")
        
        return results
    
    def analyze_benchmark_performance(self, benchmark: str, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance for a specific benchmark."""
        analysis = {
            'benchmark': benchmark,
            'metrics': {},
            'interpretation': {},
            'strengths': [],
            'weaknesses': [],
            'recommendations': []
        }
        
        if benchmark == 'multiwoz':
            analysis = self._analyze_multiwoz(results)
        elif benchmark == 'sgd':
            analysis = self._analyze_sgd(results)
        elif benchmark == 'taskmaster':
            analysis = self._analyze_taskmaster(results)
        elif benchmark == 'multidogo':
            analysis = self._analyze_multidogo(results)
        
        return analysis
    
    def _analyze_multiwoz(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze MultiWOZ results."""
        # Extract nested results
        nested_results = results.get('results', {})
        analysis = {
            'benchmark': 'multiwoz',
            'metrics': {
                'bleu_score': nested_results.get('bleu_score', {}).get('bleu', 0),
                'rouge_score': nested_results.get('rouge_score', {}).get('rouge', 0),
                'semantic_similarity': nested_results.get('semantic_similarity', {}).get('semantic_similarity', 0),
                'task_completion': nested_results.get('task_completion', {}).get('task_completion', 0)
            },
            'interpretation': {},
            'strengths': [],
            'weaknesses': [],
            'recommendations': []
        }
        
        # Interpret metrics
        bleu = analysis['metrics']['bleu_score']
        rouge = analysis['metrics']['rouge_score']
        semantic = analysis['metrics']['semantic_similarity']
        task_completion = analysis['metrics']['task_completion']
        
        analysis['interpretation'] = {
            'bleu_score': {
                'value': f"{bleu:.2f}%",
                'assessment': 'Fair' if 5 <= bleu <= 10 else 'Poor' if bleu < 5 else 'Good',
                'context': 'Standard for dialogue systems (5-15% is typical)',
                'research_grade': 'Acceptable for dialogue research'
            },
            'rouge_score': {
                'value': f"{rouge:.2f}%",
                'assessment': 'Fair' if 5 <= rouge <= 10 else 'Poor' if rouge < 5 else 'Good',
                'context': 'Measures recall of reference content',
                'research_grade': 'Within expected range for dialogue systems'
            },
            'semantic_similarity': {
                'value': f"{semantic:.2f}%",
                'assessment': 'Poor' if semantic < 10 else 'Fair' if semantic < 20 else 'Good',
                'context': 'Measures semantic meaning preservation',
                'research_grade': 'Needs improvement for better semantic understanding'
            },
            'task_completion': {
                'value': f"{task_completion:.2f}%",
                'assessment': 'Critical Issue' if task_completion == 0 else 'Poor' if task_completion < 30 else 'Fair',
                'context': 'Primary metric for task-oriented dialogue systems',
                'research_grade': 'Major concern - needs immediate attention'
            }
        }
        
        # Strengths and weaknesses
        if bleu >= 5:
            analysis['strengths'].append("BLEU score within acceptable range for dialogue systems")
        if rouge >= 5:
            analysis['strengths'].append("ROUGE score shows some content recall capability")
        
        if task_completion == 0:
            analysis['weaknesses'].append("CRITICAL: Zero task completion - model not fulfilling user goals")
        if semantic < 10:
            analysis['weaknesses'].append("Low semantic similarity - poor meaning preservation")
        if bleu < 5:
            analysis['weaknesses'].append("BLEU score below typical dialogue system range")
        
        # Recommendations
        if task_completion == 0:
            analysis['recommendations'].append("URGENT: Implement task completion detection and success indicators")
            analysis['recommendations'].append("Enhance response generation to include booking confirmations")
        if semantic < 10:
            analysis['recommendations'].append("Improve semantic understanding through better context processing")
        analysis['recommendations'].append("Focus on domain-specific knowledge integration for travel/booking")
        
        return analysis
    
    def _analyze_sgd(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze SGD results."""
        # Extract nested results
        nested_results = results.get('results', {})
        analysis = {
            'benchmark': 'sgd',
            'metrics': {
                'intent_accuracy': nested_results.get('intent_accuracy', {}).get('total', 0),
                'slot_f1_score': nested_results.get('slot_f1', {}).get('total', 0),
                'success_rate': nested_results.get('success_rate', {}).get('total', 0),
                'bleu_score': nested_results.get('bleu', {}).get('bleu', 0)
            },
            'interpretation': {},
            'strengths': [],
            'weaknesses': [],
            'recommendations': []
        }
        
        # Interpret metrics
        intent_acc = analysis['metrics']['intent_accuracy']
        slot_f1 = analysis['metrics']['slot_f1_score']
        success_rate = analysis['metrics']['success_rate']
        bleu = analysis['metrics']['bleu_score']
        
        analysis['interpretation'] = {
            'intent_accuracy': {
                'value': f"{intent_acc:.2f}%",
                'assessment': 'Excellent' if intent_acc >= 90 else 'Good' if intent_acc >= 75 else 'Fair' if intent_acc >= 50 else 'Poor',
                'context': 'Critical for understanding user goals in schema-guided dialogue',
                'research_grade': 'Outstanding performance'
            },
            'slot_f1_score': {
                'value': f"{slot_f1:.2f}%",
                'assessment': 'Critical Issue' if slot_f1 == 0 else 'Poor' if slot_f1 < 0.3 else 'Fair' if slot_f1 < 0.6 else 'Good',
                'context': 'Essential for entity extraction in dialogue systems',
                'research_grade': 'Major concern - needs immediate attention'
            },
            'success_rate': {
                'value': f"{success_rate:.2f}%",
                'assessment': 'Critical Issue' if success_rate == 0 else 'Poor' if success_rate < 40 else 'Fair' if success_rate < 70 else 'Good',
                'context': 'Primary metric for task completion in schema-guided dialogue',
                'research_grade': 'Major concern - needs immediate attention'
            },
            'bleu_score': {
                'value': f"{bleu:.2f}%",
                'assessment': 'Fair' if 5 <= bleu <= 15 else 'Poor' if bleu < 5 else 'Good',
                'context': 'Standard metric for response generation quality',
                'research_grade': 'Within acceptable range for dialogue systems'
            }
        }
        
        # Strengths and weaknesses
        if intent_acc >= 90:
            analysis['strengths'].append("Excellent intent classification accuracy - model understands user goals well")
        if bleu >= 5:
            analysis['strengths'].append("BLEU score within acceptable range for dialogue systems")
        
        if slot_f1 == 0:
            analysis['weaknesses'].append("CRITICAL: Zero slot F1 score - no entity extraction capability")
        if success_rate == 0:
            analysis['weaknesses'].append("CRITICAL: Zero success rate - model not completing user tasks")
        
        # Recommendations
        if slot_f1 == 0:
            analysis['recommendations'].append("URGENT: Implement robust entity extraction and slot filling")
            analysis['recommendations'].append("Enhance TACS filter slot extraction patterns")
        if success_rate == 0:
            analysis['recommendations'].append("URGENT: Implement success detection and task completion logic")
        analysis['recommendations'].append("Leverage strong intent classification to improve other metrics")
        
        return analysis
    
    def _analyze_taskmaster(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze Taskmaster results."""
        # Extract nested results
        nested_results = results.get('results', {})
        analysis = {
            'benchmark': 'taskmaster',
            'metrics': {
                'bleu_score': nested_results.get('bleu', {}).get('bleu', 0),
                'rouge_score': nested_results.get('rouge', {}).get('rouge', 0),
                'semantic_similarity': nested_results.get('semantic_similarity', {}).get('semantic_similarity', 0),
                'task_completion': nested_results.get('task_completion', {}).get('task_completion', 0)
            },
            'interpretation': {},
            'strengths': [],
            'weaknesses': [],
            'recommendations': []
        }
        
        # Interpret metrics
        bleu = analysis['metrics']['bleu_score']
        rouge = analysis['metrics']['rouge_score']
        semantic = analysis['metrics']['semantic_similarity']
        task_completion = analysis['metrics']['task_completion']
        
        analysis['interpretation'] = {
            'bleu_score': {
                'value': f"{bleu:.2f}%",
                'assessment': 'Good' if bleu >= 10 else 'Fair' if bleu >= 5 else 'Poor',
                'context': 'Standard metric for response generation quality',
                'research_grade': 'Good performance for dialogue systems'
            },
            'rouge_score': {
                'value': f"{rouge:.2f}%",
                'assessment': 'Fair' if 5 <= rouge <= 10 else 'Poor' if rouge < 5 else 'Good',
                'context': 'Measures recall of reference content',
                'research_grade': 'Within expected range for dialogue systems'
            },
            'semantic_similarity': {
                'value': f"{semantic:.2f}%",
                'assessment': 'Good' if semantic >= 10 else 'Fair' if semantic >= 5 else 'Poor',
                'context': 'Measures semantic meaning preservation',
                'research_grade': 'Good semantic understanding'
            },
            'task_completion': {
                'value': f"{task_completion:.2f}%",
                'assessment': 'Critical Issue' if task_completion == 0 else 'Poor' if task_completion < 25 else 'Fair',
                'context': 'Primary metric for task-oriented dialogue systems',
                'research_grade': 'Major concern - needs immediate attention'
            }
        }
        
        # Strengths and weaknesses
        if bleu >= 10:
            analysis['strengths'].append("Good BLEU score - strong response generation quality")
        if semantic >= 10:
            analysis['strengths'].append("Good semantic similarity - effective meaning preservation")
        if rouge >= 5:
            analysis['strengths'].append("ROUGE score shows content recall capability")
        
        if task_completion == 0:
            analysis['weaknesses'].append("CRITICAL: Zero task completion - model not fulfilling user goals")
        
        # Recommendations
        if task_completion == 0:
            analysis['recommendations'].append("URGENT: Implement task completion detection and success indicators")
            analysis['recommendations'].append("Enhance response generation to include task completion confirmations")
        analysis['recommendations'].append("Leverage good BLEU and semantic scores to improve task completion")
        analysis['recommendations'].append("Focus on domain-specific task understanding")
        
        return analysis
    
    def _analyze_multidogo(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze MultiDoGO results."""
        # Extract nested results
        nested_results = results.get('results', {})
        analysis = {
            'benchmark': 'multidogo',
            'metrics': {
                'intent_classification_accuracy': nested_results.get('intent_accuracy', {}).get('total', 0),
                'slot_filling_f1_score': nested_results.get('slot_f1', {}).get('total', 0),
                'domain_adaptation_score': nested_results.get('domain_adaptation', {}).get('total', 0),
                'response_quality_score': nested_results.get('response_quality', {}).get('total', 0)
            },
            'interpretation': {},
            'strengths': [],
            'weaknesses': [],
            'recommendations': []
        }
        
        # Interpret metrics
        intent_acc = analysis['metrics']['intent_classification_accuracy']
        slot_f1 = analysis['metrics']['slot_filling_f1_score']
        domain_adapt = analysis['metrics']['domain_adaptation_score']
        response_quality = analysis['metrics']['response_quality_score']
        
        analysis['interpretation'] = {
            'intent_classification_accuracy': {
                'value': f"{intent_acc:.2f}%",
                'assessment': 'Poor' if intent_acc < 30 else 'Fair' if intent_acc < 60 else 'Good' if intent_acc < 80 else 'Excellent',
                'context': 'Critical for understanding user goals across domains',
                'research_grade': 'Needs significant improvement'
            },
            'slot_filling_f1_score': {
                'value': f"{slot_f1:.2f}%",
                'assessment': 'Critical Issue' if slot_f1 == 0 else 'Poor' if slot_f1 < 0.2 else 'Fair' if slot_f1 < 0.5 else 'Good',
                'context': 'Essential for entity extraction in multi-domain dialogue',
                'research_grade': 'Major concern - needs immediate attention'
            },
            'domain_adaptation_score': {
                'value': f"{domain_adapt:.2f}%",
                'assessment': 'Critical Issue' if domain_adapt == 0 else 'Poor' if domain_adapt < 40 else 'Fair' if domain_adapt < 70 else 'Good',
                'context': 'Ability to adapt to different domains and contexts',
                'research_grade': 'Major concern - needs immediate attention'
            },
            'response_quality_score': {
                'value': f"{response_quality:.2f}%",
                'assessment': 'Fair' if 30 <= response_quality <= 60 else 'Poor' if response_quality < 30 else 'Good',
                'context': 'Overall quality of generated responses',
                'research_grade': 'Acceptable but needs improvement'
            }
        }
        
        # Strengths and weaknesses
        if response_quality >= 30:
            analysis['strengths'].append("Response quality score shows some capability")
        
        if intent_acc < 30:
            analysis['weaknesses'].append("CRITICAL: Very low intent classification accuracy")
        if slot_f1 == 0:
            analysis['weaknesses'].append("CRITICAL: Zero slot F1 score - no entity extraction capability")
        if domain_adapt == 0:
            analysis['weaknesses'].append("CRITICAL: Zero domain adaptation - cannot handle multi-domain tasks")
        
        # Recommendations
        if intent_acc < 30:
            analysis['recommendations'].append("URGENT: Enhance intent classification patterns and training")
        if slot_f1 == 0:
            analysis['recommendations'].append("URGENT: Implement robust multi-domain entity extraction")
        if domain_adapt == 0:
            analysis['recommendations'].append("URGENT: Implement domain adaptation mechanisms")
        analysis['recommendations'].append("Focus on multi-domain knowledge integration")
        analysis['recommendations'].append("Improve cross-domain context understanding")
        
        return analysis
    
    def generate_comprehensive_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive analysis report."""
        report = []
        report.append("# 📊 COMPREHENSIVE TMM MODEL EVALUATION REPORT")
        report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        report.append("## 🎯 Executive Summary")
        report.append("")
        
        # Overall assessment
        critical_issues = 0
        total_benchmarks = len(results)
        
        for benchmark, data in results.items():
            if 'task_completion' in data and data['task_completion'] == 0:
                critical_issues += 1
            elif 'success_rate' in data and data['success_rate'] == 0:
                critical_issues += 1
            elif 'slot_f1_score' in data and data['slot_f1_score'] == 0:
                critical_issues += 1
        
        if critical_issues == total_benchmarks:
            report.append("🚨 **CRITICAL STATUS**: All benchmarks show major issues requiring immediate attention.")
        elif critical_issues > 0:
            report.append(f"⚠️ **ATTENTION REQUIRED**: {critical_issues}/{total_benchmarks} benchmarks have critical issues.")
        else:
            report.append("✅ **GOOD STATUS**: No critical issues detected across benchmarks.")
        
        report.append("")
        report.append("## 📈 Benchmark-by-Benchmark Analysis")
        report.append("")
        
        # Analyze each benchmark
        for benchmark, data in results.items():
            analysis = self.analyze_benchmark_performance(benchmark, data)
            
            report.append(f"### 🔍 {benchmark.upper()} Analysis")
            report.append("")
            
            # Metrics table
            report.append("| Metric | Value | Assessment | Research Grade |")
            report.append("|--------|-------|------------|----------------|")
            
            for metric_name, metric_data in analysis['interpretation'].items():
                metric_display = metric_name.replace('_', ' ').title()
                report.append(f"| {metric_display} | {metric_data['value']} | {metric_data['assessment']} | {metric_data['research_grade']} |")
            
            report.append("")
            
            # Strengths
            if analysis['strengths']:
                report.append("**✅ Strengths:**")
                for strength in analysis['strengths']:
                    report.append(f"- {strength}")
                report.append("")
            
            # Weaknesses
            if analysis['weaknesses']:
                report.append("**❌ Weaknesses:**")
                for weakness in analysis['weaknesses']:
                    report.append(f"- {weakness}")
                report.append("")
            
            # Recommendations
            if analysis['recommendations']:
                report.append("**🎯 Recommendations:**")
                for rec in analysis['recommendations']:
                    report.append(f"- {rec}")
                report.append("")
        
        # Overall recommendations
        report.append("## 🚀 Overall Recommendations")
        report.append("")
        report.append("### Immediate Actions (Critical)")
        report.append("1. **Implement Task Completion Detection**: All benchmarks show 0% task completion/success rate")
        report.append("2. **Enhance Entity Extraction**: SGD and MultiDoGO show 0% slot F1 scores")
        report.append("3. **Fix Response Quality Enhancement**: Remove broken `_enhance_response_quality` method")
        report.append("")
        report.append("### Short-term Improvements")
        report.append("1. **Domain-Specific Knowledge Integration**: Improve travel/booking domain expertise")
        report.append("2. **Multi-Domain Adaptation**: Enhance cross-domain capability for MultiDoGO")
        report.append("3. **Semantic Understanding**: Improve context and meaning preservation")
        report.append("")
        report.append("### Research Publication Readiness")
        report.append("- **Current Status**: Not ready for publication due to critical issues")
        report.append("- **Required**: Fix all critical issues before submission")
        report.append("- **Timeline**: 2-3 weeks of focused development needed")
        report.append("")
        
        return "\n".join(report)
    
    def save_analysis(self, report: str, filepath: str = "results/comprehensive_analysis_report.md"):
        """Save analysis report."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            f.write(report)
        
        print(f"💾 Comprehensive analysis report saved to: {filepath}")
    
    def run_comprehensive_analysis(self):
        """Run comprehensive analysis."""
        print("🔍 Running Comprehensive Results Analysis...")
        print("=" * 60)
        
        # Load results
        results = self.load_results()
        
        if not results:
            print("❌ No results found to analyze")
            return
        
        print(f"📊 Loaded results for {len(results)} benchmarks: {list(results.keys())}")
        
        # Generate report
        report = self.generate_comprehensive_report(results)
        
        # Save report
        self.save_analysis(report)
        
        # Print summary
        print("\n📋 ANALYSIS SUMMARY:")
        print("-" * 40)
        
        for benchmark, data in results.items():
            print(f"\n🔍 {benchmark.upper()}:")
            
            if 'task_completion' in data:
                tc = data['task_completion']
                print(f"  Task Completion: {tc:.2f}% {'🚨 CRITICAL' if tc == 0 else '✅'}")
            
            if 'success_rate' in data:
                sr = data['success_rate']
                print(f"  Success Rate: {sr:.2f}% {'🚨 CRITICAL' if sr == 0 else '✅'}")
            
            if 'slot_f1_score' in data:
                sf1 = data['slot_f1_score']
                print(f"  Slot F1: {sf1:.2f}% {'🚨 CRITICAL' if sf1 == 0 else '✅'}")
            
            if 'bleu_score' in data:
                bleu = data['bleu_score']
                print(f"  BLEU: {bleu:.2f}% {'✅' if bleu >= 5 else '⚠️'}")
        
        print(f"\n💾 Full analysis report saved to: results/comprehensive_analysis_report.md")
        
        return report

def main():
    """Main function to run comprehensive analysis."""
    analyzer = ComprehensiveResultsAnalyzer()
    report = analyzer.run_comprehensive_analysis()
    return report

if __name__ == "__main__":
    main()
