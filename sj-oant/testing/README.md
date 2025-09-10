# TMM Testing Framework

This folder contains testing scripts to evaluate the TMM model against 4 dialogue benchmarks.

## 🎯 What This Tests

**TMM Model Only:**
- Truth-Maintained Memory (TMM) System
- Multi-Agent Pipeline (Strategic Planner, TACS Filter, Truth Verifier, Memory Curator, Responder)
- Multi-Tiered Memory System (L1, L2, L3, Flagged bin)
- False Memory Prevention and Truth Verification

**Across 4 Benchmarks (25 examples each):**
- **MultiWOZ** - Multi-turn dialogue with state tracking
- **SGD** - Schema-guided dialogue with intent/slot filling
- **Taskmaster** - Human-human conversations with task completion
- **MultiDoGO** - Multi-domain goal-oriented dialogues

## 🚀 Quick Start

### 1. Set up API Key
```bash
export GOOGLE_API_KEY='your-api-key-here'
```

### 2. Run TMM Evaluation
```bash
python testing/run_tmm_evaluation.py
```

### 3. Analyze Results
```bash
python testing/analyze_tmm_results.py
```

## 📁 Files

- `tmm_evaluation.py` - Main TMM evaluation script
- `run_tmm_evaluation.py` - Runner script
- `analyze_tmm_results.py` - Results analysis and reporting
- `results/` - Output directory for results

## 📊 Output

The evaluation generates:
- `tmm_evaluation_results.json` - Raw TMM results data
- `tmm_summary.txt` - Human-readable summary report

## ⏱️ Runtime

- **Total time**: ~10-15 minutes
- **Per benchmark**: ~2-3 minutes
- **TMM system**: ~30-45 seconds per benchmark

## 🔧 Requirements

- Google API key (Gemini 1.5 Flash)
- All dependencies from main project
- ~25 examples per benchmark (100 total)

## 📈 Metrics Evaluated

**MultiWOZ:**
- BLEU Score, Success Rate, Inform Rate

**SGD:**
- Intent Accuracy, Slot F1, Success Rate

**Taskmaster:**
- BLEU Score, ROUGE Score, Task Completion

**MultiDoGO:**
- Intent Classification, Slot Filling, Domain Adaptation

## 🎯 Expected Results

The evaluation will show:
- TMM system performance across all benchmarks
- Benchmark-specific strengths/weaknesses
- Memory operations and truth verification metrics
- Overall TMM system performance summary
