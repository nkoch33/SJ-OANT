#!/usr/bin/env python3
"""
📊 TMM Results Analysis Script

This script analyzes and visualizes evaluation results from the TMM system
and baseline comparisons.
"""
import json
import sys
from pathlib import Path
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import pandas as pd

def load_results(results_file: str) -> Dict[str, Any]:
    """Load evaluation results from JSON file."""
    results_path = Path(results_file)
    if not results_path.exists():
        print(f"❌ Results file not found: {results_file}")
        return {}
    
    with open(results_path) as f:
        return json.load(f)

def analyze_performance(results: Dict[str, Any]) -> pd.DataFrame:
    """Analyze system performance metrics."""
    if not results.get("results"):
        print("❌ No results data found")
        return pd.DataFrame()
    
    # Extract metrics for each system
    data = []
    for result in results["results"]:
        data.append({
            "System": result["system_name"],
            "Accuracy": result["accuracy"],
            "Correct": result["correct_answers"],
            "Total": result["total_questions"],
            "Avg_Time": result["avg_response_time"],
            "Errors": len(result.get("errors", []))
        })
    
    df = pd.DataFrame(data)
    return df.sort_values("Accuracy", ascending=False)

def print_summary(df: pd.DataFrame):
    """Print detailed results summary."""
    print("\n📊 EVALUATION RESULTS SUMMARY")
    print("="*60)
    
    for _, row in df.iterrows():
        print(f"\n🤖 {row['System']}:")
        print(f"   Accuracy: {row['Accuracy']:.2%}")
        print(f"   Correct: {row['Correct']}/{row['Total']}")
        print(f"   Avg Time: {row['Avg_Time']:.2f}s")
        if row['Errors'] > 0:
            print(f"   Errors: {row['Errors']}")
    
    # Best performer
    best = df.iloc[0]
    print(f"\n🏆 BEST PERFORMER: {best['System']} ({best['Accuracy']:.2%})")
    
    # TMM vs others
    tmm_row = df[df['System'].str.contains('TMM', case=False, na=False)]
    if not tmm_row.empty:
        tmm_acc = tmm_row.iloc[0]['Accuracy']
        baseline_acc = df[~df['System'].str.contains('TMM', case=False, na=False)]['Accuracy'].max()
        improvement = ((tmm_acc - baseline_acc) / baseline_acc) * 100
        print(f"📈 TMM vs Best Baseline: {improvement:+.1f}% improvement")

def plot_results(df: pd.DataFrame, save_path: str = None):
    """Create visualization of results."""
    if df.empty:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Accuracy comparison
    bars1 = ax1.bar(df['System'], df['Accuracy'])
    ax1.set_title('System Accuracy Comparison')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1)
    ax1.tick_params(axis='x', rotation=45)
    
    # Color TMM bar differently
    for i, bar in enumerate(bars1):
        if 'TMM' in df.iloc[i]['System']:
            bar.set_color('orange')
        else:
            bar.set_color('skyblue')
    
    # Response time comparison
    bars2 = ax2.bar(df['System'], df['Avg_Time'])
    ax2.set_title('Average Response Time')
    ax2.set_ylabel('Time (seconds)')
    ax2.tick_params(axis='x', rotation=45)
    
    # Color bars
    for i, bar in enumerate(bars2):
        if 'TMM' in df.iloc[i]['System']:
            bar.set_color('orange')
        else:
            bar.set_color('lightcoral')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📈 Plot saved to: {save_path}")
    else:
        plt.show()

def main():
    """Main analysis function."""
    print("📊 TMM RESULTS ANALYZER")
    print("="*40)
    
    # Look for results files
    results_dir = Path("results")
    if not results_dir.exists():
        print("❌ Results directory not found")
        return
    
    result_files = list(results_dir.glob("*.json"))
    if not result_files:
        print("❌ No result files found in results/")
        print("   Run evaluation first: python runners/eval_baselines.py")
        return
    
    # Analyze latest results
    latest_file = max(result_files, key=lambda x: x.stat().st_mtime)
    print(f"📄 Analyzing: {latest_file}")
    
    results = load_results(latest_file)
    if not results:
        return
    
    df = analyze_performance(results)
    if df.empty:
        return
    
    print_summary(df)
    
    # Create visualization
    plot_path = results_dir / "performance_comparison.png"
    plot_results(df, str(plot_path))
    
    # Save CSV summary
    csv_path = results_dir / "results_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"💾 Summary saved to: {csv_path}")

if __name__ == "__main__":
    main()
