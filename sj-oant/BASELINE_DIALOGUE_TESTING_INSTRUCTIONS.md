# Baseline Dialogue Testing Instructions

## 🎯 Overview

This document provides complete instructions for setting up baseline testing of open source LLMs against our TMM model on dialogue performance metrics. The goal is to compare how different models perform on standard dialogue tasks across 3 benchmarks.

## 📁 Current Codebase State

### File Structure
```
sj-oant/
├── multi_agent_pipeline.py          # Our TMM model
├── evaluation_frameworks/           # Official benchmark evaluation
│   ├── multiwoz/                   # MultiWOZ 2.4 evaluation
│   ├── sgd/                        # Schema-Guided Dialogue evaluation
│   └── taskmaster/                 # Taskmaster evaluation
├── testing/
│   └── official_evaluation.py      # Main evaluation script
├── data/                           # Benchmark datasets
│   ├── MULTIWOZ2.4/               # MultiWOZ data
│   ├── sgd/                       # SGD data
│   └── taskmaster/                # Taskmaster data
└── false_memory_evaluation/        # False memory testing (separate)
```

### Our 3 Benchmarks

#### 1. MultiWOZ 2.4
- **Data Location**: `data/MULTIWOZ2.4/MULTIWOZ2.4/`
- **Key Files**: `data.json`, `testListFile.json`
- **Evaluation Script**: `evaluation_frameworks/multiwoz/official_evaluator.py`
- **Metrics**:
  - **Response Diversity**: Measures lexical richness and variety in responses
  - **Response Relevance**: How well responses address user requests
  - **Information Accuracy**: Correctness of factual information provided
  - **Task Understanding**: System's comprehension of user goals

#### 2. Schema-Guided Dialogue (SGD)
- **Data Location**: `data/sgd/`
- **Key Files**: `dialogues_001.json`, `schema.json`
- **Evaluation Script**: `evaluation_frameworks/sgd/official_evaluator.py`
- **Metrics**:
  - **BLEU**: Measures response quality against reference responses
  - **Slot F1**: Accuracy of extracting and filling required information slots
  - **Semantic Similarity**: Semantic closeness to reference responses
  - **Intent Accuracy**: Correct identification of user intent

#### 3. Taskmaster
- **Data Location**: `data/taskmaster/`
- **Key Files**: `restaurant-search.json`, `ontology.json`
- **Evaluation Script**: `evaluation_frameworks/taskmaster/official_evaluator.py`
- **Metrics**:
  - **BLEU**: Response quality measurement
  - **ROUGE**: Overlap-based response quality metric
  - **Semantic Similarity**: Semantic alignment with references
  - **Slot Extraction F1**: Accuracy of extracting task-specific information

## 🚀 How to Test Our TMM Model

### Quick Test (5 conversations each)
```bash
cd sj-oant
export GEMINI_API_KEY="your_api_key_here"
python testing/official_evaluation.py
```

### Large-Scale Test (100 conversations each)
```bash
# Edit testing/official_evaluation.py
# Change num_samples = 5 to num_samples = 100
python testing/official_evaluation.py
```

## 🏗️ Setting Up Baseline Testing Environment

### Step 1: Create Baseline Testing Folder
```bash
mkdir baseline_dialogue_testing
cd baseline_dialogue_testing
```

### Step 2: Cursor Prompt for Baseline Setup
Copy this prompt to Cursor:

```
I need to create a baseline testing environment for comparing open source LLMs against our TMM model on dialogue benchmarks. 

REQUIREMENTS:
1. Create a folder structure for testing 3 open source LLMs (suggest: Llama-2, Mistral, GPT-3.5-turbo)
2. Set up evaluation scripts that can run the same 3 benchmarks (MultiWOZ, SGD, Taskmaster) 
3. Each LLM should be tested on 100 conversations per benchmark
4. Output the same metrics as our TMM model for fair comparison
5. Use the same data loading logic as our existing evaluation_frameworks
6. Create a unified results comparison script

FOLDER STRUCTURE NEEDED:
baseline_dialogue_testing/
├── models/
│   ├── llama2_baseline.py
│   ├── mistral_baseline.py
│   └── gpt35_baseline.py
├── evaluation/
│   ├── baseline_evaluator.py
│   └── results_comparator.py
├── data_loader/
│   └── benchmark_loader.py
├── results/
└── run_baseline_evaluation.py

The evaluation should:
- Load the same 100 conversations per benchmark that our TMM model uses
- Run each baseline LLM on these conversations
- Calculate the same metrics (BLEU, ROUGE, Semantic Similarity, Slot F1, Intent Accuracy, etc.)
- Save results in a format that can be compared with our TMM results
- Generate a comparison report showing TMM vs baselines performance

Please create this baseline testing environment with proper error handling, logging, and result formatting.
```

### Step 3: Implementation Steps

1. **Copy Data Loading Logic**: Use the same data loading from `testing/official_evaluation.py`
2. **Create LLM Wrappers**: Simple wrappers for each baseline LLM
3. **Reuse Evaluation Metrics**: Use the same metric calculations from `evaluation_frameworks/`
4. **Run 100 Conversations**: Scale up from 5 to 100 conversations per benchmark
5. **Generate Comparison Report**: Side-by-side comparison of all models

### Step 4: Expected Output

The baseline testing should produce:
- **Individual Results**: Performance of each baseline LLM on each benchmark
- **Comparison Table**: TMM vs Baseline 1 vs Baseline 2 vs Baseline 3
- **Statistical Analysis**: Significance testing between models
- **Visualization**: Charts showing performance differences

## 📊 Expected Results Format

```json
{
  "multiwoz": {
    "tmm": {
      "response_diversity": 0.85,
      "response_relevance": 0.78,
      "information_accuracy": 0.82,
      "task_understanding": 0.79
    },
    "llama2": {
      "response_diversity": 0.72,
      "response_relevance": 0.71,
      "information_accuracy": 0.75,
      "task_understanding": 0.73
    },
    "mistral": {
      "response_diversity": 0.78,
      "response_relevance": 0.74,
      "information_accuracy": 0.77,
      "task_understanding": 0.76
    },
    "gpt35": {
      "response_diversity": 0.81,
      "response_relevance": 0.76,
      "information_accuracy": 0.79,
      "task_understanding": 0.78
    }
  }
}
```

## ⚠️ Important Notes

1. **Don't Modify Main Model**: These baselines should NOT affect our TMM model performance
2. **Use Same Data**: Ensure all models are tested on identical conversation sets
3. **Same Metrics**: Use identical metric calculations for fair comparison
4. **Reproducible**: Set random seeds for consistent results
5. **Documentation**: Document all baseline model configurations and versions

## 🎯 Success Criteria

The baseline testing is successful when:
- [ ] All 3 baseline LLMs are tested on 100 conversations per benchmark
- [ ] Same metrics are calculated as our TMM model
- [ ] Results are saved in comparable format
- [ ] Comparison report shows TMM vs baselines performance
- [ ] Statistical significance is tested
- [ ] No impact on main TMM model performance

---

**Goal**: Prove that TMM performs competitively on standard dialogue tasks while maintaining its false memory prevention capabilities.
