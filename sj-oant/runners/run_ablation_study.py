#!/usr/bin/env python3
"""
Ablation Study Runner

This script runs comprehensive ablation studies to understand TMM component contributions.

Usage:
    python runners/run_ablation_study.py --api-key YOUR_API_KEY --limit 25
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from evaluation.ablation_study import AblationStudy
from evaluation.squad_eval import SQuADEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def print_ablation_results(results):
    """Print comprehensive ablation study results."""
    print("\n" + "=" * 80)
    print("🧪 TMM ABLATION STUDY RESULTS")
    print("=" * 80)
    
    # Print individual variant results
    print("\n📊 VARIANT PERFORMANCE:")
    print("-" * 60)
    
    for variant_name, result in results["results"].items():
        print(f"\n{variant_name}:")
        print(f"  Accuracy: {result.accuracy:.2%}")
        print(f"  Correct: {result.correct_answers}/{result.total_examples}")
        print(f"  Avg Response Time: {result.avg_response_time:.3f}s")
        if result.errors:
            print(f"  Errors: {len(result.errors)}")
    
    # Print component analysis
    if "analysis" in results and "component_analysis" in results["analysis"]:
        print("\n🔍 COMPONENT IMPACT ANALYSIS:")
        print("-" * 60)
        
        for component, analysis in results["analysis"]["component_analysis"].items():
            print(f"\n{component.upper()} Component:")
            print(f"  Avg Accuracy Drop: {analysis['avg_accuracy_drop']:+.2%}")
            print(f"  Avg Time Change: {analysis['avg_time_change']:+.3f}s")
            print(f"  Importance Score: {analysis['importance_score']:.3f}")
            print(f"  Tested in: {', '.join(analysis['variants_tested'])}")
    
    # Print component rankings
    if "analysis" in results and "ranked_components" in results["analysis"]:
        print("\n🏆 COMPONENT IMPORTANCE RANKING:")
        print("-" * 60)
        
        for i, (component, data) in enumerate(results["analysis"]["ranked_components"], 1):
            print(f"{i}. {component.upper()}")
            print(f"   Impact: {data['importance_score']:+.2%} accuracy when removed")
            print(f"   Time: {data['avg_time_change']:+.3f}s change")
    
    # Print summary insights
    if "summary" in results:
        summary = results["summary"]
        print("\n💡 KEY INSIGHTS:")
        print("-" * 60)
        
        print(f"Best Performing: {summary['best_variant']['name']}")
        print(f"  └─ {summary['best_variant']['accuracy']:.2%} accuracy, {summary['best_variant']['response_time']:.3f}s")
        
        print(f"Fastest System: {summary['fastest_variant']['name']}")
        print(f"  └─ {summary['fastest_variant']['response_time']:.3f}s, {summary['fastest_variant']['accuracy']:.2%} accuracy")
        
        if "most_important_component" in summary:
            comp = summary["most_important_component"]
            print(f"Most Critical Component: {comp['name'].upper()}")
            print(f"  └─ {comp['importance_score']:+.2%} accuracy impact when removed")
        
        if "least_important_component" in summary:
            comp = summary["least_important_component"]
            print(f"Least Critical Component: {comp['name'].upper()}")
            print(f"  └─ {comp['importance_score']:+.2%} accuracy impact when removed")
        
        print(f"Performance Range: {summary['performance_range']['accuracy_range']:.2%} accuracy spread")
        print(f"Speed Range: {summary['performance_range']['time_range']:.3f}s time spread")

def print_optimization_recommendations(results):
    """Print optimization recommendations based on ablation results."""
    print("\n" + "=" * 80)
    print("🎯 OPTIMIZATION RECOMMENDATIONS")
    print("=" * 80)
    
    if "analysis" not in results or "ranked_components" not in results["analysis"]:
        print("❌ Insufficient data for recommendations")
        return
    
    ranked = results["analysis"]["ranked_components"]
    
    print("\n🔥 HIGH PRIORITY OPTIMIZATIONS:")
    print("-" * 50)
    
    # Focus on most impactful components
    for i, (component, data) in enumerate(ranked[:2], 1):
        impact = data["importance_score"]
        time_change = data["avg_time_change"]
        
        print(f"{i}. OPTIMIZE {component.upper()} COMPONENT")
        print(f"   Current Impact: {impact:+.2%} accuracy when removed")
        
        if component == "memory":
            print("   💡 Recommendations:")
            print("      - Implement memory compression algorithms")
            print("      - Add smarter tier promotion/demotion policies")
            print("      - Optimize memory retrieval indexing")
            
        elif component == "filtering":
            print("   💡 Recommendations:")
            print("      - Enhance TACS relevance scoring")
            print("      - Implement learned filtering policies")
            print("      - Add contextual noise detection")
            
        elif component == "verification":
            print("   💡 Recommendations:")
            print("      - Strengthen truth verification rules")
            print("      - Add confidence-based verification thresholds")
            print("      - Implement contradiction detection")
        
        if time_change > 0:
            print(f"   ⚡ Speed Impact: +{time_change:.3f}s overhead")
            print("      - Consider component optimization or caching")
        
        print()
    
    print("🔧 COMPONENT INTEGRATION:")
    print("-" * 50)
    
    # Look for components that work well together
    minimal_variant = None
    for variant_name, result in results["results"].items():
        if "Minimal" in variant_name:
            minimal_variant = result
            break
    
    if minimal_variant:
        baseline_acc = results["analysis"].get("baseline_accuracy", 0)
        degradation = baseline_acc - minimal_variant.accuracy
        
        print(f"Removing ALL components: {degradation:+.2%} accuracy loss")
        if degradation < 0.1:  # Less than 10% loss
            print("💡 Consider simplified TMM architecture for production")
        else:
            print("💡 Multiple components provide synergistic benefits")
    
    print("\n⚖️  SPEED vs ACCURACY TRADE-OFFS:")
    print("-" * 50)
    
    summary = results.get("summary", {})
    if "best_variant" in summary and "fastest_variant" in summary:
        best = summary["best_variant"]
        fastest = summary["fastest_variant"]
        
        if best["name"] != fastest["name"]:
            acc_diff = best["accuracy"] - fastest["accuracy"]
            time_diff = best["response_time"] - fastest["response_time"]
            
            print(f"Best accuracy: {best['name']} ({best['accuracy']:.2%})")
            print(f"Fastest system: {fastest['name']} ({fastest['response_time']:.3f}s)")
            print(f"Trade-off: {acc_diff:+.2%} accuracy for {-time_diff:.3f}s speedup")
            
            if acc_diff < 0.05:  # Less than 5% accuracy difference
                print("💡 Recommend faster variant for production")
            else:
                print("💡 Accuracy gain justifies slower response time")

def main():
    parser = argparse.ArgumentParser(description="Run TMM ablation study")
    parser.add_argument("--api-key", required=True, help="Google API key")
    parser.add_argument("--limit", type=int, default=25, help="Number of examples to test per variant")
    parser.add_argument("--output", default="results/ablation_study_results.json", 
                       help="Output file path")
    
    args = parser.parse_args()
    
    print("🚀 STARTING TMM ABLATION STUDY")
    print("=" * 80)
    print(f"Testing {args.limit} examples per variant")
    print(f"Results will be saved to: {args.output}")
    
    try:
        # Initialize ablation study
        study = AblationStudy(args.api_key)
        
        # Load test examples
        evaluator = SQuADEvaluator()
        examples = evaluator.load_dataset("validation")
        print(f"Loaded {len(examples)} total examples")
        
        # Run ablation study
        print(f"\n🧪 Running ablation study with {len(study.variants)} variants...")
        results = study.run_ablation_study(examples, limit=args.limit)
        
        # Save results
        output_file = study.save_results(results, args.output)
        
        # Print comprehensive results
        print_ablation_results(results)
        print_optimization_recommendations(results)
        
        print("\n" + "=" * 80)
        print("✅ ABLATION STUDY COMPLETE")
        print(f"📁 Results saved to: {output_file}")
        print("=" * 80)
        
    except Exception as e:
        logger.error(f"Ablation study failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
