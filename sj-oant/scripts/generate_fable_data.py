#!/usr/bin/env python3
"""
Generate FABLE benchmark data from FictionalQA dataset.

This script downloads the FictionalQA dataset and converts it into
FABLE-compatible scenarios for truth-maintained memory evaluation.

Usage:
    python scripts/generate_fable_data.py --limit 100 --output benchmarks/fable
"""
import argparse
import logging
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.fable.data_loader import FictionalQALoader

def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    parser = argparse.ArgumentParser(description="Generate FABLE data from FictionalQA")
    parser.add_argument("--limit", type=int, default=50, 
                       help="Limit number of scenarios to generate (default: 50)")
    parser.add_argument("--output", type=str, default="benchmarks/fable",
                       help="Output directory for FABLE scenarios")
    parser.add_argument("--split", type=str, default="train",
                       choices=["train", "validation", "test"],
                       help="Dataset split to use")
    parser.add_argument("--no-misinformation", action="store_true",
                       help="Don't inject misinformation (pure accuracy testing)")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting FABLE data generation from FictionalQA")
    
    try:
        # Initialize loader
        loader = FictionalQALoader()
        
        # Load records
        logger.info(f"Loading {args.limit} records from {args.split} split...")
        records = loader.get_records(split=args.split, limit=args.limit)
        logger.info(f"Loaded {len(records)} FictionalQA records")
        
        # Convert to FABLE scenarios
        scenarios = []
        inject_misinformation = not args.no_misinformation
        
        for i, record in enumerate(records):
            logger.debug(f"Converting record {i+1}/{len(records)}: {record.title}")
            scenario = loader.convert_to_fable_scenario(record, 
                                                       inject_misinformation=inject_misinformation)
            scenarios.append(scenario)
        
        logger.info(f"Generated {len(scenarios)} FABLE scenarios")
        
        # Save scenarios
        logger.info(f"Saving scenarios to {args.output}")
        loader.save_fable_scenarios(scenarios, output_dir=args.output)
        
        # Print summary
        difficulty_counts = {}
        for scenario in scenarios:
            difficulty_counts[scenario.difficulty] = difficulty_counts.get(scenario.difficulty, 0) + 1
        
        print("\n" + "="*50)
        print("FABLE Data Generation Complete!")
        print("="*50)
        print(f"Total scenarios generated: {len(scenarios)}")
        print(f"Misinformation injection: {'Yes' if inject_misinformation else 'No'}")
        print("\nDifficulty distribution:")
        for difficulty, count in sorted(difficulty_counts.items()):
            print(f"  {difficulty.title()}: {count} scenarios")
        print(f"\nOutput directory: {args.output}")
        print("  - Scenarios: benchmarks/fable/scenarios/fictional/")
        print("  - Knowledge bases: benchmarks/fable/kb/")
        
        logger.info("FABLE data generation completed successfully!")
        
    except Exception as e:
        logger.error(f"Failed to generate FABLE data: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
