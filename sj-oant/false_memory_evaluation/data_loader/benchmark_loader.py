"""
Benchmark Data Loader for Baseline Testing

Loads benchmark data (MultiWOZ, SGD, Taskmaster) for false memory testing.
Reuses the same data loading logic as the main TMM evaluation system.
"""

import json
import random
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import sys
import os

# Add parent directory to path to import from main system
sys.path.append(str(Path(__file__).parent.parent.parent))

logger = logging.getLogger(__name__)

class BenchmarkLoader:
    """
    Loads benchmark data for false memory testing.
    
    Reuses the same data loading logic as the main TMM evaluation system
    to ensure identical test data across all models.
    """
    
    def __init__(self, data_root: str = None):
        """
        Initialize the benchmark loader.
        
        Args:
            data_root: Root directory containing benchmark data
        """
        if data_root is None:
            # Default to parent directory's data folder
            self.data_root = Path(__file__).parent.parent.parent / "data"
        else:
            self.data_root = Path(data_root)
        
        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root directory not found: {self.data_root}")
        
        logger.info(f"Initialized BenchmarkLoader with data root: {self.data_root}")
    
    def load_benchmark_data(self, benchmark: str, num_samples: int = 100) -> List[Dict[str, Any]]:
        """
        Load benchmark data samples.
        
        Args:
            benchmark: Benchmark name (multiwoz, sgd, taskmaster)
            num_samples: Number of samples to load
            
        Returns:
            List of benchmark samples
        """
        if benchmark.lower() == "multiwoz":
            return self._load_multiwoz_samples(num_samples)
        elif benchmark.lower() == "sgd":
            return self._load_sgd_samples(num_samples)
        elif benchmark.lower() == "taskmaster":
            return self._load_taskmaster_samples(num_samples)
        else:
            raise ValueError(f"Unknown benchmark: {benchmark}")
    
    def _load_multiwoz_samples(self, num_samples: int) -> List[Dict[str, Any]]:
        """Load MultiWOZ samples from the dataset."""
        mw_dir = self.data_root / "MULTIWOZ2.4" / "MULTIWOZ2.4"
        data_file = mw_dir / "data.json"
        
        if not data_file.exists():
            raise FileNotFoundError(f"MultiWOZ data.json not found at {data_file}")
        
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        # Use all available dialogue IDs from the data
        data_ids = list(data.keys())
        
        if not data_ids:
            raise RuntimeError("No dialogue IDs found in MultiWOZ data")
        
        # Sample up to available
        pick = random.sample(data_ids, min(num_samples, len(data_ids)))
        
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
                "system_turns": system_turns,
                "benchmark": "multiwoz"
            })
        
        logger.info(f"Loaded {len(samples)} MultiWOZ samples")
        return samples
    
    def _load_sgd_samples(self, num_samples: int) -> List[Dict[str, Any]]:
        """Load SGD samples from dialogues_*.json files."""
        sgd_dir = self.data_root / "sgd"
        if not sgd_dir.exists():
            raise FileNotFoundError(f"SGD directory not found: {sgd_dir}")
        
        files = sorted(sgd_dir.glob("dialogues_*.json"))
        if not files:
            raise FileNotFoundError(f"No SGD dialogues_*.json files found in {sgd_dir}")
        
        # Load data from all files
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
        
        # Sample data
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
                "system_turns": system_turns,
                "benchmark": "sgd"
            })
        
        logger.info(f"Loaded {len(samples)} SGD samples")
        return samples
    
    def _load_taskmaster_samples(self, num_samples: int) -> List[Dict[str, Any]]:
        """Load Taskmaster samples from available JSON files."""
        taskmaster_dir = self.data_root / "taskmaster"
        if not taskmaster_dir.exists():
            raise FileNotFoundError(f"Taskmaster directory not found: {taskmaster_dir}")
        
        # Look for JSON files
        candidates = [taskmaster_dir / 'restaurant-search.json'] + list(taskmaster_dir.glob('*.json'))
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
            raise FileNotFoundError(f"No Taskmaster JSON data found in {taskmaster_dir}")
        
        # Sample data
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
                "system_turns": system_turns,
                "benchmark": "taskmaster"
            })
        
        logger.info(f"Loaded {len(samples)} Taskmaster samples")
        return samples
    
    def load_all_benchmarks(self, num_samples_per_benchmark: int = 100) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load data from all benchmarks.
        
        Args:
            num_samples_per_benchmark: Number of samples to load per benchmark
            
        Returns:
            Dictionary mapping benchmark names to sample lists
        """
        benchmarks = ["multiwoz", "sgd", "taskmaster"]
        all_data = {}
        
        for benchmark in benchmarks:
            try:
                logger.info(f"Loading {benchmark} data...")
                samples = self.load_benchmark_data(benchmark, num_samples_per_benchmark)
                all_data[benchmark] = samples
                logger.info(f"Loaded {len(samples)} samples for {benchmark}")
            except Exception as e:
                logger.error(f"Failed to load {benchmark} data: {e}")
                all_data[benchmark] = []
        
        return all_data
    
    def get_benchmark_statistics(self, data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Get statistics about loaded benchmark data."""
        stats = {}
        
        for benchmark, samples in data.items():
            if not samples:
                stats[benchmark] = {"count": 0, "avg_turns": 0}
                continue
            
            total_turns = sum(len(sample["user_turns"]) for sample in samples)
            avg_turns = total_turns / len(samples) if samples else 0
            
            stats[benchmark] = {
                "count": len(samples),
                "total_turns": total_turns,
                "avg_turns": avg_turns,
                "min_turns": min(len(sample["user_turns"]) for sample in samples) if samples else 0,
                "max_turns": max(len(sample["user_turns"]) for sample in samples) if samples else 0
            }
        
        return stats
    
    def save_loaded_data(self, data: Dict[str, List[Dict[str, Any]]], output_path: str):
        """Save loaded benchmark data to file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved loaded benchmark data to {output_file}")
    
    def load_saved_data(self, input_path: str) -> Dict[str, List[Dict[str, Any]]]:
        """Load previously saved benchmark data."""
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        logger.info(f"Loaded saved benchmark data from {input_path}")
        return data
