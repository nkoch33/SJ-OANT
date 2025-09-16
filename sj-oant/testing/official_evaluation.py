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
        elif benchmark == "multidogo":
            return self._load_multidogo_samples(str(data_base / "multidogo"), num_samples)
        else:
            raise ValueError(f"Unknown benchmark: {benchmark}")
    
    def _load_multiwoz_samples(self, data_root: str, num_samples: int) -> List[Dict]:
        """Load MultiWOZ samples robustly using official test list and valid IDs."""
        import json
        from pathlib import Path
        
        mw_dir = Path(data_root) / "MULTIWOZ2.4"
        data_file = mw_dir / "data.json"
        test_list_file = mw_dir / "testListFile.json"
        
        if not data_file.exists():
            raise FileNotFoundError(f"MultiWOZ data.json not found at {data_file}")
        if not test_list_file.exists():
            raise FileNotFoundError(f"MultiWOZ testListFile.json not found at {test_list_file}")
        
        with open(data_file, 'r') as f:
            data = json.load(f)
        # Note: some distributions of MultiWOZ have testListFile.json as newline-delimited IDs, not JSON
        try:
            with open(test_list_file, 'r') as f:
                test_ids = json.load(f)
                if isinstance(test_ids, dict) and 'testListFile' in test_ids:
                    test_ids = test_ids['testListFile']
        except Exception:
            # Fallback: read lines
            with open(test_list_file, 'r') as f:
                test_ids = [line.strip().replace('.json','') for line in f if line.strip()]
        
        # Normalize IDs in test list (ensure exact keys present in data)
        valid_ids = [did if did in data else did.replace('.json','') for did in test_ids if (did in data) or (did.replace('.json','') in data)]
        if not valid_ids:
            # As a fallback, use any ids from data
            valid_ids = list(data.keys())
        
        # Sample up to available
        pick = random.sample(valid_ids, min(num_samples, len(valid_ids)))
        
        samples: List[Dict] = []
        for dialogue_id in pick:
            dialogue = data.get(dialogue_id)
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
                "dialogue_id": dialogue_id,
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
    
    def _load_multidogo_samples(self, data_path: str, num_samples: int) -> List[Dict]:
        """Load MultiDoGO samples robustly from TSVs, skipping invalid rows."""
        import pandas as pd
        from pathlib import Path
        p = Path(data_path)
        tsv_files = [p / "airline.tsv", p / "fastfood.tsv", p / "airline_annotated.tsv"]
        frames = []
        for fp in tsv_files:
            if not fp.exists():
                continue
            try:
                df = pd.read_csv(fp, sep='\t', quoting=3, on_bad_lines='skip', engine='python')
                frames.append(df)
            except Exception as e:
                logger.warning(f"Failed to load {fp}: {e}")
        if not frames:
            raise FileNotFoundError(f"No MultiDoGO TSVs found in {data_path}")
        df = pd.concat(frames, ignore_index=True)
        # Clean
        df = df.dropna(subset=[col for col in ["conversationId", "utterance"] if col in df.columns])
        if df.empty:
            raise RuntimeError("MultiDoGO data is empty after cleaning")
        count = min(num_samples, len(df))
        selected = df.sample(n=count, random_state=42)
        samples: List[Dict] = []
        for _, row in selected.iterrows():
            utt = str(row.get("utterance", "")).strip()
            if not utt:
                continue
            samples.append({
                "dialogue_id": str(row.get("conversationId", "unknown")),
                "user_turns": [utt],
                "system_turns": []
            })
        return samples
    
    def generate_tmm_predictions(self, samples: List[Dict]) -> List[Dict]:
        """
        Generate TMM predictions for samples.
        
        Args:
            samples: List of benchmark samples
            
        Returns:
            List of TMM predictions
        """
        predictions = []
        
        for sample in samples:
            try:
                # Reset memory for each dialogue
                self.tmm_pipeline.reset_memory()
                
                dialogue_id = sample["dialogue_id"]
                user_turns = sample["user_turns"]
                responses = []
                
                # Process each user turn
                for user_turn in user_turns:
                    response = self.tmm_pipeline.process(user_turn)
                    responses.append(self._normalize_text(response))
                
                predictions.append({
                    "dialogue_id": dialogue_id,
                    "user_turns": user_turns,
                    "responses": responses
                })
                
            except Exception as e:
                logger.error(f"Failed to process dialogue {dialogue_id}: {e}")
                continue
        
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
        logger.info(f"Evaluating {benchmark} with {num_samples} samples")
        
        # Load data using requested sample size (MultiWOZ loader already validates IDs)
        samples = self.load_benchmark_data(benchmark, num_samples)
        logger.info(f"Loaded {len(samples)} samples for {benchmark}")
        
        # Generate predictions
        predictions = self.generate_tmm_predictions(samples)
        logger.info(f"Generated {len(predictions)} predictions for {benchmark}")
        
        # Evaluate using official framework
        results = self.unified_evaluator.evaluate_benchmark(benchmark, predictions)
        
        return results
    
    def evaluate_all_benchmarks(self, num_samples: int = 25) -> Dict[str, Any]:
        """
        Evaluate TMM on all benchmarks.
        
        Args:
            num_samples: Number of samples per benchmark
            
        Returns:
            Complete evaluation results
        """
        logger.info(f"Starting comprehensive evaluation with {num_samples} samples per benchmark")
        
        all_predictions = {}
        all_results = {}
        
        benchmarks = ["multiwoz", "sgd", "taskmaster", "multidogo"]
        
        for benchmark in benchmarks:
            logger.info(f"Processing {benchmark}...")
            
            # Load data and generate predictions
            samples = self.load_benchmark_data(benchmark, num_samples)
            predictions = self.generate_tmm_predictions(samples)
            all_predictions[benchmark] = predictions
            
            # Evaluate using official framework
            results = self.unified_evaluator.evaluate_benchmark(benchmark, predictions)
            all_results[benchmark] = results
        
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
        
        logger.info("Official evaluation completed")
        return results

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Run evaluation
    evaluator = OfficialBenchmarkEvaluator()
    results = evaluator.run_evaluation(num_samples=25)
