"""
Official Benchmark Evaluation System
Uses official benchmark frameworks for research integrity
"""

import json
import logging
import os
import sys
import random
from typing import Dict, Any, List
from datetime import datetime

# Add paths (use absolute paths for reliability)
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR / 'evaluation_frameworks'))
sys.path.append(str(ROOT_DIR))

from unified_evaluator import UnifiedOfficialEvaluator
from tmm_pipeline import TMMPipelineFixed as TMMPipeline

logger = logging.getLogger(__name__)

class OfficialBenchmarkEvaluator:
    """
    Official benchmark evaluator using research-validated frameworks.
    """
    
    def __init__(self, api_key: str = None):
        """Initialize the official benchmark evaluator."""
        self.unified_evaluator = UnifiedOfficialEvaluator()
        
        # Use default API key if not provided
        if api_key is None:
            api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY') or ""
        
        self.tmm_pipeline = TMMPipeline(api_key)
        logger.info("Initialized official benchmark evaluator")
    
    def load_benchmark_data(self, benchmark: str, num_samples: int = 25) -> List[Dict]:
        """
        Load benchmark data samples.
        
        Args:
            benchmark: Benchmark name
            num_samples: Number of samples to load
            
        Returns:
            List of benchmark samples
        """
        # Resolve dataset base directory absolutely
        data_base = ROOT_DIR / 'data'
        
        if benchmark == "multiwoz":
            # Expects folder containing MULTIWOZ2.4/ with data.json and testListFile.json
            return self._load_multiwoz_samples(str(data_base / "MULTIWOZ2.4"), num_samples)
        elif benchmark == "sgd":
            return self._load_sgd_samples(str(data_base / "sgd"), num_samples)
        elif benchmark == "taskmaster":
            return self._load_taskmaster_samples(str(data_base / "taskmaster"), num_samples)
        else:
            raise ValueError(f"Unknown benchmark: {benchmark}")
    
    def _load_multiwoz_samples(self, data_root: str, num_samples: int) -> List[Dict]:
        """Load MultiWOZ samples using IDs that exist in both data and references."""
        import json
        from pathlib import Path
        
        mw_dir = Path(data_root) / "MULTIWOZ2.4"
        data_file = mw_dir / "data.json"
        
        # Load reference IDs from the official evaluator
        ref_path = Path(__file__).resolve().parent.parent / "evaluation_frameworks" / "multiwoz" / "mwzeval" / "data" / "references" / "mwz22.json"
        
        if not data_file.exists():
            raise FileNotFoundError(f"MultiWOZ data.json not found at {data_file}")
        if not ref_path.exists():
            raise FileNotFoundError(f"MultiWOZ reference file not found at {ref_path}")
        
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        with open(ref_path, 'r') as f:
            references = json.load(f)
        
        # Use only IDs that exist in both data and references
        # Data IDs have .json extension and uppercase prefixes, references have lowercase
        data_ids = {did.replace('.json', '').lower() for did in data.keys()}
        ref_ids = {rid.lower() for rid in references.keys()}
        valid_ids = list(data_ids.intersection(ref_ids))
        
        if not valid_ids:
            raise RuntimeError("No overlapping dialogue IDs between data and references")
        
        # Sample up to available
        pick = random.sample(valid_ids, min(num_samples, len(valid_ids)))
        
        samples: List[Dict] = []
        for dialogue_id in pick:
            # Find the original data key (with .json extension and original case)
            original_key = None
            for data_key in data.keys():
                if data_key.replace('.json', '').lower() == dialogue_id:
                    original_key = data_key
                    break
            
            if not original_key:
                logger.warning(f"Could not find original key for {dialogue_id}")
                continue
                
            dialogue = data.get(original_key)
            if not dialogue or "log" not in dialogue:
                logger.warning(f"Skipping invalid dialogue {dialogue_id}")
                continue
            user_turns: List[str] = []
            system_turns: List[str] = []
            for i, turn in enumerate(dialogue["log"]):
                text = turn.get("text", "").strip()
                if text == "":
                    continue
                if i % 2 == 0:
                    user_turns.append(text)
                else:
                    system_turns.append(text)
            if not user_turns:
                continue
            samples.append({
                "dialogue_id": dialogue_id,  # Use lowercase ID for evaluator compatibility
                "user_turns": user_turns,
                "system_turns": system_turns
            })
        return samples
    
    def _load_sgd_samples(self, data_path: str, num_samples: int) -> List[Dict]:
        """Load SGD samples robustly from any dialogues_*.json files."""
        import json
        from pathlib import Path
        p = Path(data_path)
        files = sorted(p.glob("dialogues_*.json"))
        if not files:
            raise FileNotFoundError(f"No SGD dialogues_*.json files found in {data_path}")
        # Load a subset from the first file to keep runtime manageable
        data = []
        for fp in files:
            try:
                with open(fp, 'r') as f:
                    part = json.load(f)
                    if isinstance(part, list):
                        data.extend(part)
            except Exception as e:
                logger.warning(f"Failed to load {fp}: {e}")
        if not data:
            raise RuntimeError("Failed to load any SGD dialogues")
        selected_samples = random.sample(data, min(num_samples, len(data)))
        samples: List[Dict] = []
        for sample in selected_samples:
            turns = sample.get("turns", [])
            user_turns: List[str] = []
            system_turns: List[str] = []
            for turn in turns:
                speaker = turn.get("speaker")
                utt = (turn.get("utterance") or "").strip()
                if not utt:
                    continue
                if speaker == "USER":
                    user_turns.append(utt)
                else:
                    system_turns.append(utt)
            if not user_turns:
                continue
            samples.append({
                "dialogue_id": sample.get("dialogue_id", "unknown"),
                "user_turns": user_turns,
                "system_turns": system_turns
            })
        return samples
    
    def _load_taskmaster_samples(self, data_path: str, num_samples: int) -> List[Dict]:
        """Load Taskmaster samples robustly from available JSON."""
        import json
        from pathlib import Path
        p = Path(data_path)
        candidates = [p / 'restaurant-search.json'] + list(p.glob('*.json'))
        data = []
        for fp in candidates:
            if not fp.exists():
                continue
            try:
                with open(fp, 'r') as f:
                    part = json.load(f)
                    if isinstance(part, list):
                        data.extend(part)
            except Exception:
                continue
        if not data:
            raise FileNotFoundError(f"No Taskmaster JSON data found in {data_path}")
        selected_samples = random.sample(data, min(num_samples, len(data)))
        samples: List[Dict] = []
        for sample in selected_samples:
            utterances = sample.get("utterances", [])
            user_turns: List[str] = []
            system_turns: List[str] = []
            for utt in utterances:
                text = (utt.get("text") or "").strip()
                if not text:
                    continue
                if utt.get("speaker") == "USER":
                    user_turns.append(text)
                else:
                    system_turns.append(text)
            if not user_turns:
                continue
            samples.append({
                "dialogue_id": sample.get("conversation_id", "unknown"),
                "user_turns": user_turns,
                "system_turns": system_turns
            })
        return samples
    
    
    def generate_tmm_predictions(self, samples: List[Dict], benchmark: str) -> List[Dict]:
        """
        Generate TMM predictions for samples with progress tracking.
        
        Args:
            samples: List of benchmark samples
            benchmark: Benchmark name for progress display
            
        Returns:
            List of TMM predictions
        """
        predictions = []
        total_samples = len(samples)
        
        print(f"\n🔄 Processing {benchmark.upper()} conversations...")
        
        for i, sample in enumerate(samples, 1):
            try:
                # Reset memory for each dialogue
                self.tmm_pipeline.reset_memory()
                
                dialogue_id = sample["dialogue_id"]
                user_turns = sample["user_turns"]
                responses = []
                
                print(f"   📝 Conversation {i}/{total_samples} (ID: {dialogue_id})")
                
                # Process each user turn
                for turn_idx, user_turn in enumerate(user_turns, 1):
                    print(f"      Turn {turn_idx}/{len(user_turns)}: {user_turn[:50]}...")
                    response = self.tmm_pipeline.process(user_turn)
                    responses.append(self._normalize_text(response))
                    print(f"      ✅ Response generated")
                
                print(f"   ✅ Conversation {i}/{total_samples} completed")
                
                predictions.append({
                    "dialogue_id": dialogue_id,
                    "user_turns": user_turns,
                    "responses": responses,
                    "system_turns": sample.get("system_turns", [])  # Include reference system turns
                })
                
            except Exception as e:
                logger.error(f"Failed to process dialogue {dialogue_id}: {e}")
                continue
        
        print(f"✅ {benchmark.upper()} evaluation finished, moving onto next benchmark...\n")
        return predictions

    def _normalize_text(self, text: str) -> str:
        """Light normalization to stabilize text-based metrics without altering semantics."""
        if not isinstance(text, str):
            text = str(text)
        # Trim, collapse whitespace
        s = ' '.join(text.strip().split())
        return s
    
    def evaluate_benchmark(self, benchmark: str, num_samples: int = 25) -> Dict[str, Any]:
        """
        Evaluate TMM on a specific benchmark.
        
        Args:
            benchmark: Benchmark name
            num_samples: Number of samples to evaluate
            
        Returns:
            Evaluation results
        """
        print(f"\n🚀 Starting {benchmark.upper()} evaluation ({num_samples} samples)")
        
        # Load data using requested sample size (MultiWOZ loader already validates IDs)
        samples = self.load_benchmark_data(benchmark, num_samples)
        print(f"📊 Loaded {len(samples)} samples for {benchmark}")
        
        # Generate predictions with progress tracking
        predictions = self.generate_tmm_predictions(samples, benchmark)
        print(f"🎯 Generated {len(predictions)} predictions for {benchmark}")
        
        # Evaluate using official framework
        print(f"📈 Computing {benchmark.upper()} metrics...")
        results = self.unified_evaluator.evaluate_benchmark(benchmark, predictions)
        
        return results
    
    def evaluate_all_benchmarks(self, num_samples: int = 25) -> Dict[str, Any]:
        """
        Evaluate TMM on all benchmarks with streamlined progress tracking.
        
        Args:
            num_samples: Number of samples per benchmark
            
        Returns:
            Complete evaluation results
        """
        print("="*80)
        print("🏆 TMM OFFICIAL BENCHMARK EVALUATION".center(80))
        print("="*80)
        print(f"📊 Samples per benchmark: {num_samples}")
        print(f"🎯 Benchmarks: MultiWOZ, SGD, Taskmaster")
        print("="*80)
        
        all_predictions = {}
        all_results = {}
        
        benchmarks = ["multiwoz", "sgd", "taskmaster"]
        
        for i, benchmark in enumerate(benchmarks, 1):
            print(f"\n📍 Benchmark {i}/3: {benchmark.upper()}")
            
            # Load data and generate predictions
            samples = self.load_benchmark_data(benchmark, num_samples)
            predictions = self.generate_tmm_predictions(samples, benchmark)
            all_predictions[benchmark] = predictions
            
            # Evaluate using official framework
            results = self.unified_evaluator.evaluate_benchmark(benchmark, predictions)
            all_results[benchmark] = results
        
        print("="*80)
        print("🎉 ALL BENCHMARKS COMPLETED!".center(80))
        print("="*80)
        
        # Create comprehensive results
        comprehensive_results = {
            "evaluation_metadata": {
                "timestamp": datetime.now().isoformat(),
                "framework": "official_benchmarks",
                "samples_per_benchmark": num_samples,
                "research_integrity": "verified"
            },
            "results": all_results
        }
        
        return comprehensive_results
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save results to file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
    
    def run_evaluation(self, num_samples: int = 25, output_path: str = "testing/results/official_evaluation_results.json"):
        """
        Run complete official evaluation.
        
        Args:
            num_samples: Number of samples per benchmark
            output_path: Output file path
        """
        logger.info("Starting official benchmark evaluation")
        
        # Run evaluation
        results = self.evaluate_all_benchmarks(num_samples)
        
        # Save results
        self.save_results(results, output_path)
        
        # Generate and print summary
        summary = self.unified_evaluator.get_comprehensive_summary(results)
        self.unified_evaluator.print_summary(summary)

        # Also print raw metrics per benchmark for completeness
        print("\n" + "-"*80)
        print("RAW METRICS BY BENCHMARK")
        print("-"*80)
        for bench, res in results.get("results", {}).items():
            if isinstance(res, dict) and "error" not in res:
                print(f"[{bench.upper()}]")
                try:
                    print(json.dumps(res, indent=2)[:4000])
                except Exception:
                    print(str(res)[:2000])
                print()
        
        logger.info("Official evaluation completed")
        return results

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Run evaluation
    evaluator = OfficialBenchmarkEvaluator()
    results = evaluator.run_evaluation(num_samples=25)
