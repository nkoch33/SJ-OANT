# 🚀 TMM System Pipeline Overview

This document shows the complete pipeline flow from input to results.

## 🔐 Security Setup

### API Key Management
```bash
# Your API key will be stored in .env (protected by .gitignore)
GOOGLE_API_KEY=your_actual_api_key_here

# To setup:
python scripts/setup_api_key.py "YOUR_GOOGLE_API_KEY"

# NEVER commit .env to GitHub!
# Use env.example as template for others
```

## 📊 Data Pipeline

### Input: SQuAD Dataset
```
📥 Input: SQuAD reading comprehension
├── Context: "Super Bowl 50 was an American football game..."
├── Question: "Which NFL team represented the AFC?"
└── Answer: "Denver Broncos"
```

### Processing Flow
```
📊 SQuAD Examples
    ↓
🤖 5 Systems Process Each Example:
    ├── TMM Pipeline (your system)
    ├── DirectLLM (baseline)
    ├── LongContext (baseline)
    ├── SimpleRAG (baseline)
    └── BasicMemory (baseline)
    ↓
📈 Results Collection
    ↓
💾 Storage in results/
    ↓
📊 Analysis & Visualization
```

## 🧠 TMM Pipeline (Your System)

### Agent Flow
```
🎯 User Input
    ↓
🧠 Strategic Planner
    ├── Query analysis
    ├── Context refinement
    └── Execution planning
    ↓
🔍 TACS Filter
    ├── Relevance scoring
    ├── Context filtering
    └── Noise reduction
    ↓
✅ Truth Verifier
    ├── Confidence scoring
    ├── Evidence assessment
    └── Contradiction detection
    ↓
📝 Writer/Editor
    ├── Memory curation
    ├── Selective addition
    └── Combined deletion
    ↓
💬 Responder
    ├── Context assembly
    ├── Response generation
    └── Quality control
    ↓
📤 Final Response
```

### Memory System
```
🧠 Multi-Tier Memory Store
├── L1 Working Memory (recent contexts)
├── L2 Summarized Memory (processed facts)
├── L3 Archival Memory (verified knowledge)
└── Flagged Memory (questionable content)
```

## 📊 Baseline Systems

### 1. DirectLLM
- **Process**: Raw question → LLM → Answer
- **Memory**: None
- **Purpose**: Control baseline

### 2. LongContext  
- **Process**: Full conversation history → LLM → Answer
- **Memory**: Simple concatenation
- **Purpose**: Long context comparison

### 3. SimpleRAG
- **Process**: Store context → Keyword retrieval → LLM → Answer
- **Memory**: Basic keyword matching
- **Purpose**: Standard RAG baseline

### 4. BasicMemory
- **Process**: Store everything → Full recall → LLM → Answer
- **Memory**: No filtering
- **Purpose**: Unfiltered memory comparison

## 📁 Results Pipeline

### File Structure
```
results/
├── squad2_evaluation_results.json      # SQuAD 2.0 evaluation data
├── multiturn_evaluation_results.json   # Multi-turn conversation data
├── results_summary.csv                 # Processed metrics
├── performance_comparison.png          # Visualization
└── evaluation_logs/                    # Detailed logs
    ├── tmm_pipeline_log.txt
    ├── baseline_logs/
    └── error_reports/
```

### JSON Output Format
```json
{
  "evaluation_timestamp": 1234567890,
  "results": [
    {
      "system_name": "TMM_Pipeline",
      "total_questions": 10,
      "correct_answers": 8,
      "accuracy": 0.8,
      "avg_response_time": 3.2,
      "memory_metrics": {...},
      "errors": []
    },
    // ... other systems
  ],
  "comparison": {
    "best_accuracy": {...},
    "fastest_system": {...}
  }
}
```

## 🎯 Evaluation Metrics

### Primary Metrics
- **Accuracy**: Correct answers / Total questions
- **Response Time**: Average processing time per question
- **Error Rate**: Failed responses / Total attempts

### TMM-Specific Metrics
- **Memory Operations**: Add/Update/Delete counts
- **Verification Calls**: Truth checking frequency
- **Memory Consistency**: Cross-reference accuracy

## 🚀 Running the Pipeline

### 1. Pre-flight Check
```bash
python scripts/preflight_check.py
```

### 2. Small Test Run
```bash
python runners/eval_squad2.py --api-key "YOUR_API_KEY" --limit 5
```

### 3. Full Evaluation
```bash
python runners/eval_squad2.py --api-key "YOUR_API_KEY" --limit 100
```

### 4. Analyze Results
```bash
python scripts/analyze_results.py
```

## 📈 Expected Outputs

### Console Output
```
============================================================
SQUAD EVALUATION RESULTS
============================================================
[Results will appear here after running evaluation]

python runners/eval_squad2.py --api-key "YOUR_API_KEY" --limit 50
```

### Generated Files
- 📊 **performance_comparison.png**: Bar charts comparing systems
- 📈 **results_summary.csv**: Spreadsheet-ready metrics
- 🔍 **detailed_logs/**: Per-system execution logs

## 🔒 GitHub Safety

### Protected Files (.gitignore)
```
.env                    # Your API keys
__pycache__/           # Python cache
*.pyc                  # Compiled Python
results/logs/          # Sensitive logs
```

### Safe to Commit
```
env.example            # Template for others
results/*.json         # Evaluation results (no keys)
results/*.csv          # Summary data
results/*.png          # Visualizations
```

## 🎯 Research Questions This Answers

1. **Does TMM improve accuracy over baselines?**
   - Compare TMM vs DirectLLM/SimpleRAG accuracy

2. **What's the cost of truth maintenance?**
   - Compare response times across systems

3. **How does memory filtering help?**
   - Compare TMM vs BasicMemory (unfiltered)

4. **Is context length the key factor?**
   - Compare TMM vs LongContext (same info, different processing)

**Ready to prove your hypothesis! 🧠✨**
