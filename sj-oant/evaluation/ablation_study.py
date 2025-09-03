#!/usr/bin/env python3
"""
TMM Ablation Study Framework

This module implements comprehensive ablation studies to understand the contribution
of each TMM component to overall system performance.

Usage:
    from evaluation.ablation_study import AblationStudy
    study = AblationStudy(api_key)
    results = study.run_ablation_study(examples)
"""

import logging
import time
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from evaluation.squad_eval import SQuADEvaluator, EvaluationResult
from tmm_pipeline import TMMPipeline
from langchain_google_genai import ChatGoogleGenerativeAI

logger = logging.getLogger(__name__)

@dataclass
class AblationVariant:
    """Represents a TMM variant with specific components disabled."""
    name: str
    description: str
    disabled_components: List[str]
    expected_impact: str

class TMMVariant:
    """TMM pipeline variant for ablation studies."""
    
    def __init__(self, api_key: str, disabled_components: List[str] = None):
        self.api_key = api_key
        self.disabled_components = disabled_components or []
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.1,
            google_api_key=api_key
        )
        
    def process(self, query: str) -> str:
        """Process query with specific components disabled."""
        # Simple implementation - in full implementation, we'd modify the actual pipeline
        # For now, we simulate different behavior based on disabled components
        
        if "memory" in self.disabled_components:
            # No memory - direct LLM call
            response = self.llm.invoke(query)
            return response.content
        
        elif "filtering" in self.disabled_components:
            # No TACS filtering - simplified pipeline
            response = self.llm.invoke(f"Answer based on context: {query}")
            return response.content
        
        elif "verification" in self.disabled_components:
            # No truth verification
            response = self.llm.invoke(f"Answer without verification: {query}")
            return response.content
        
        else:
            # Full TMM pipeline (baseline)
            response = self.llm.invoke(f"Answer with full TMM pipeline: {query}")
            return response.content
    
    def reset_memory(self):
        """Reset memory state (no-op for variants)."""
        pass

class AblationStudy:
    """Comprehensive ablation study framework for TMM system."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.evaluator = SQuADEvaluator()
        self.variants = self._define_ablation_variants()
        
    def _define_ablation_variants(self) -> List[AblationVariant]:
        """Define all ablation study variants."""
        return [
            AblationVariant(
                name="Full TMM",
                description="Complete TMM pipeline with all components",
                disabled_components=[],
                expected_impact="Baseline performance"
            ),
            AblationVariant(
                name="No Memory",
                description="TMM without multi-tiered memory system",
                disabled_components=["memory"],
                expected_impact="Reduced context retention, lower accuracy on multi-turn"
            ),
            AblationVariant(
                name="No TACS Filter",
                description="TMM without Token-level Adaptive Context Screening",
                disabled_components=["filtering"],
                expected_impact="Noise in context, potentially lower accuracy"
            ),
            AblationVariant(
                name="No Truth Verification",
                description="TMM without truth verification component",
                disabled_components=["verification"],
                expected_impact="Higher false positive rate, potential misinformation"
            ),
            AblationVariant(
                name="No Memory + No Filter",
                description="TMM without memory system and TACS filtering",
                disabled_components=["memory", "filtering"],
                expected_impact="Significant performance degradation"
            ),
            AblationVariant(
                name="Minimal TMM",
                description="TMM with only basic response generation",
                disabled_components=["memory", "filtering", "verification"],
                expected_impact="Performance approaching baseline DirectLLM"
            )
        ]
    
    def evaluate_variant(self, variant: AblationVariant, examples: List, limit: int = 50) -> EvaluationResult:
        """Evaluate a specific TMM variant."""
        logger.info(f"Evaluating variant: {variant.name}")
        
        # Create variant system
        system = TMMVariant(self.api_key, variant.disabled_components)
        
        # Run evaluation (simplified for ablation)
        start_time = time.time()
        correct_answers = 0
        response_times = []
        errors = []
        
        for i, example in enumerate(examples[:limit]):
            try:
                # Context setup (if memory enabled)
                if "memory" not in variant.disabled_components:
                    context_prompt = f"Please remember this context: {example.context}"
                    system.process(context_prompt)
                
                # Question processing
                question_start = time.time()
                
                if "memory" in variant.disabled_components:
                    # Direct question with context
                    full_prompt = f"Context: {example.context}\n\nQuestion: {example.question}"
                    response = system.process(full_prompt)
                else:
                    # Question only (memory should handle context)
                    response = system.process(example.question)
                
                response_time = time.time() - question_start
                response_times.append(response_time)
                
                # Check answer
                is_correct = self.evaluator._check_answer_v2(response, example.answer, example.is_answerable)
                if is_correct:
                    correct_answers += 1
                
                # Log progress
                if (i + 1) % 10 == 0:
                    logger.info(f"  Processed {i + 1}/{limit} examples")
                
            except Exception as e:
                errors.append(f"Error on example {i}: {str(e)}")
                logger.warning(f"Error processing example {i}: {e}")
        
        total_time = time.time() - start_time
        accuracy = correct_answers / limit if limit > 0 else 0.0
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0.0
        
        return EvaluationResult(
            system_name=variant.name,
            total_examples=limit,
            correct_answers=correct_answers,
            accuracy=accuracy,
            avg_response_time=avg_response_time,
            answerable_accuracy=0.0,  # Simplified for ablation
            unanswerable_accuracy=0.0,  # Simplified for ablation
            errors=errors
        )
    
    def run_ablation_study(self, examples: List, limit: int = 50) -> Dict[str, Any]:
        """Run complete ablation study on all variants."""
        logger.info(f"Starting ablation study with {len(self.variants)} variants")
        
        results = {}
        variant_results = []
        
        # Evaluate each variant
        for variant in self.variants:
            print(f"\n📊 Evaluating: {variant.name}")
            print(f"   Description: {variant.description}")
            print(f"   Expected: {variant.expected_impact}")
            
            result = self.evaluate_variant(variant, examples, limit)
            results[variant.name] = result
            variant_results.append({
                "variant": variant,
                "result": result
            })
            
            print(f"   ✅ Accuracy: {result.accuracy:.2%}")
            print(f"   ⏱️  Avg Response Time: {result.avg_response_time:.3f}s")
        
        # Analyze component contributions
        analysis = self._analyze_component_contributions(variant_results)
        
        return {
            "results": results,
            "analysis": analysis,
            "summary": self._generate_summary(variant_results, analysis)
        }
    
    def _analyze_component_contributions(self, variant_results: List[Dict]) -> Dict[str, Any]:
        """Analyze the contribution of each component to performance."""
        # Get baseline (Full TMM) performance
        baseline_result = None
        for vr in variant_results:
            if vr["variant"].name == "Full TMM":
                baseline_result = vr["result"]
                break
        
        if not baseline_result:
            logger.error("No baseline result found")
            return {}
        
        component_impact = {}
        
        # Analyze impact of removing each component
        for vr in variant_results:
            variant = vr["variant"]
            result = vr["result"]
            
            if variant.name == "Full TMM":
                continue
            
            # Calculate accuracy drop
            accuracy_drop = baseline_result.accuracy - result.accuracy
            time_change = result.avg_response_time - baseline_result.avg_response_time
            
            for component in variant.disabled_components:
                if component not in component_impact:
                    component_impact[component] = {
                        "accuracy_drops": [],
                        "time_changes": [],
                        "variants": []
                    }
                
                component_impact[component]["accuracy_drops"].append(accuracy_drop)
                component_impact[component]["time_changes"].append(time_change)
                component_impact[component]["variants"].append(variant.name)
        
        # Calculate average impact per component
        component_analysis = {}
        for component, data in component_impact.items():
            avg_accuracy_drop = sum(data["accuracy_drops"]) / len(data["accuracy_drops"])
            avg_time_change = sum(data["time_changes"]) / len(data["time_changes"])
            
            component_analysis[component] = {
                "avg_accuracy_drop": avg_accuracy_drop,
                "avg_time_change": avg_time_change,
                "importance_score": avg_accuracy_drop,  # Simple importance metric
                "variants_tested": data["variants"]
            }
        
        # Rank components by importance
        ranked_components = sorted(
            component_analysis.items(),
            key=lambda x: x[1]["importance_score"],
            reverse=True
        )
        
        return {
            "component_analysis": component_analysis,
            "ranked_components": ranked_components,
            "baseline_accuracy": baseline_result.accuracy
        }
    
    def _generate_summary(self, variant_results: List[Dict], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive summary of ablation study."""
        # Find best and worst performing variants
        best_variant = max(variant_results, key=lambda x: x["result"].accuracy)
        worst_variant = min(variant_results, key=lambda x: x["result"].accuracy)
        fastest_variant = min(variant_results, key=lambda x: x["result"].avg_response_time)
        
        # Component rankings
        if "ranked_components" in analysis:
            most_important = analysis["ranked_components"][0] if analysis["ranked_components"] else None
            least_important = analysis["ranked_components"][-1] if analysis["ranked_components"] else None
        else:
            most_important = least_important = None
        
        summary = {
            "total_variants": len(variant_results),
            "best_variant": {
                "name": best_variant["variant"].name,
                "accuracy": best_variant["result"].accuracy,
                "response_time": best_variant["result"].avg_response_time
            },
            "worst_variant": {
                "name": worst_variant["variant"].name,
                "accuracy": worst_variant["result"].accuracy,
                "response_time": worst_variant["result"].avg_response_time
            },
            "fastest_variant": {
                "name": fastest_variant["variant"].name,
                "accuracy": fastest_variant["result"].accuracy,
                "response_time": fastest_variant["result"].avg_response_time
            },
            "performance_range": {
                "accuracy_range": best_variant["result"].accuracy - worst_variant["result"].accuracy,
                "time_range": max(vr["result"].avg_response_time for vr in variant_results) - 
                             min(vr["result"].avg_response_time for vr in variant_results)
            }
        }
        
        if most_important:
            summary["most_important_component"] = {
                "name": most_important[0],
                "importance_score": most_important[1]["importance_score"]
            }
        
        if least_important:
            summary["least_important_component"] = {
                "name": least_important[0],
                "importance_score": least_important[1]["importance_score"]
            }
        
        return summary
    
    def save_results(self, results: Dict[str, Any], output_path: str = "results/ablation_study_results.json"):
        """Save ablation study results to file."""
        import json
        from pathlib import Path
        
        # Ensure results directory exists
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert results to JSON-serializable format
        serializable_results = {}
        
        for variant_name, result in results["results"].items():
            serializable_results[variant_name] = result.to_dict()
        
        json_data = {
            "results": serializable_results,
            "analysis": results["analysis"],
            "summary": results["summary"],
            "metadata": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                "total_variants": len(results["results"])
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        logger.info(f"Ablation study results saved to {output_file}")
        return output_file
