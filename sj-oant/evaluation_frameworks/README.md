# Official Benchmark Evaluation Frameworks

This directory contains the official evaluation frameworks for all benchmarks, ensuring research integrity and reproducibility. These frameworks evaluate TMM's dialogue performance across 3 major benchmarks with objective, research-validated metrics.

## 🏗️ Architecture

```
evaluation_frameworks/
├── multiwoz/                    # Official MultiWOZ 2.4 evaluation
│   ├── official_evaluator.py   # TMM integration wrapper (4 objective metrics)
│   └── [official framework]    # Tomiinek/MultiWOZ_Evaluation
├── sgd/                        # Official Schema-Guided Dialogue evaluation
│   ├── official_evaluator.py   # TMM integration wrapper (4 objective metrics)
│   └── [official framework]    # google-research-datasets/dstc8-schema-guided-dialogue
├── taskmaster/                 # Official Taskmaster evaluation
│   └── official_evaluator.py   # TMM integration wrapper (4 objective metrics)
└── unified_evaluator.py        # Unified evaluation system for all benchmarks
```

## 🔬 Research Integrity

### MultiWOZ 2.4 Evaluation
- **Framework**: [Tomiinek/MultiWOZ_Evaluation](https://github.com/Tomiinek/MultiWOZ_Evaluation)
- **Paper**: [Shades of BLEU, Flavours of Success: The Case of MultiWOZ](https://arxiv.org/abs/2106.05555)
- **Metrics**: 
  - **Response Diversity**: Lexical richness and variety in responses
  - **Response Relevance**: How well responses address user requests
  - **Information Accuracy**: Correctness of factual information provided
  - **Task Understanding**: System's comprehension of user goals
- **Status**: ✅ 4 objective metrics implemented (BLEU removed due to compatibility issues)

### Schema-Guided Dialogue (SGD) Evaluation
- **Framework**: [google-research-datasets/dstc8-schema-guided-dialogue](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue)
- **Paper**: [Schema-Guided Dialogue Dataset](https://arxiv.org/abs/1909.05855)
- **Metrics**:
  - **BLEU**: Response quality against reference responses
  - **Slot F1**: Accuracy of extracting and filling required information slots
  - **Semantic Similarity**: Semantic closeness to reference responses
  - **Intent Accuracy**: Correct identification of user intent
- **Status**: ✅ 4 objective metrics implemented

### Taskmaster Evaluation
- **Framework**: General evaluation toolkit approach with official libraries
- **Metrics**:
  - **BLEU**: Response quality measurement using sacrebleu
  - **ROUGE**: Overlap-based response quality metric
  - **Semantic Similarity**: Semantic alignment with references (0-100% scale)
  - **Slot Extraction F1**: Accuracy of extracting task-specific information
- **Status**: ✅ 4 objective metrics implemented


## 🚀 Usage

### Unified Evaluation
```python
from evaluation_frameworks.unified_evaluator import UnifiedOfficialEvaluator

# Initialize evaluator
evaluator = UnifiedOfficialEvaluator()

# Evaluate on specific benchmark
results = evaluator.evaluate_benchmark("multiwoz", tmm_predictions)

# Evaluate on all benchmarks
all_results = evaluator.evaluate_all_benchmarks(tmm_predictions)
```

### Individual Benchmark Evaluation
```python
from evaluation_frameworks.multiwoz.official_evaluator import OfficialMultiWOZEvaluator

# Initialize specific evaluator
evaluator = OfficialMultiWOZEvaluator(bleu=True, success=True, richness=True)

# Evaluate predictions
results = evaluator.evaluate(tmm_predictions)
```

## 📊 Metrics Overview

| Benchmark | Primary Metrics | Evaluation Framework | Research Paper |
|-----------|----------------|---------------------|----------------|
| MultiWOZ | Response Diversity, Response Relevance, Information Accuracy, Task Understanding | Tomiinek/MultiWOZ_Evaluation | [arXiv:2106.05555](https://arxiv.org/abs/2106.05555) |
| SGD | BLEU, Slot F1, Semantic Similarity, Intent Accuracy | DSTC8 Challenge | [arXiv:1909.05855](https://arxiv.org/abs/1909.05855) |
| Taskmaster | BLEU, ROUGE, Semantic Similarity, Slot Extraction F1 | General Evaluation Toolkit | Multi-turn dialogue standards |

## 🎯 Current Status (September 2024)

### ✅ Completed
- **MultiWOZ**: 4 objective metrics implemented and tested
- **SGD**: 4 objective metrics implemented and tested  
- **Taskmaster**: 4 objective metrics implemented and tested
- **Unified Evaluator**: Single interface for all benchmarks
- **Research Integrity**: All metrics use official libraries and frameworks
- **Baseline Testing Ready**: Framework prepared for comparison with open source LLMs

### 🔄 Recent Updates
- **MultiDoGO Removed**: Eliminated problematic benchmark, focused on 3 core benchmarks
- **BLEU Optimization**: Removed from MultiWOZ due to evaluation compatibility issues
- **Metric Refinement**: Streamlined to 4 objective metrics per benchmark
- **Codebase Cleanup**: Removed outdated files, organized structure
- **Documentation**: Updated READMEs and created baseline testing instructions

## 🔧 Installation

The official frameworks are automatically installed when setting up the evaluation system:

```bash
# MultiWOZ framework (already installed)
cd evaluation_frameworks/multiwoz
pip install -e .

# SGD framework (already cloned)
# No additional installation required

# Other frameworks use standard Python libraries
```

## 📝 Input Format

All evaluators expect TMM predictions in the following format:

```python
[
    {
        "dialogue_id": "sng0073",
        "user_turns": ["I need a restaurant", "Book it for 2 people"],
        "responses": ["I can help you find a restaurant", "I've booked the table"]
    },
    # ... more dialogues
]
```

## 🎯 Output Format

Each evaluator returns results in a standardized format:

```python
{
    "benchmark": "multiwoz",
    "evaluation_framework": "official",
    "timestamp": "2024-01-01T00:00:00",
    "response_diversity": {
        "avg_lengths": 12.5,
        "entropy": 8.2
    },
    "response_relevance": {
        "total": 75.3
    },
    "information_accuracy": {
        "total": 80.0
    },
    "task_understanding": {
        "total": 93.3
    },
    "evaluation_note": "All 4 metrics are objective and response-focused. BLEU removed due to evaluation framework compatibility issues."
}
```

## ⚠️ Important Notes

1. **Research Integrity**: All metrics are based on official benchmark definitions
2. **No Custom Metrics**: No keyword-based or custom approximations are used
3. **Reproducible**: All evaluations can be reproduced using the official frameworks
4. **Validated**: Metrics match those used in the original research papers

## 🚀 Next Steps: Baseline Testing

The evaluation frameworks are now ready for baseline testing against open source LLMs:

### Baseline Testing Setup
1. **Use Instructions**: Follow `BASELINE_DIALOGUE_TESTING_INSTRUCTIONS.md` in project root
2. **Test 3 LLMs**: Llama-2, Mistral, GPT-3.5-turbo (or similar)
3. **100 Conversations**: Per benchmark for statistical significance
4. **Same Metrics**: Use identical evaluation for fair comparison
5. **Results Comparison**: Generate side-by-side performance analysis

### Expected Research Impact
- **Prove TMM Competitiveness**: Show TMM performs well on standard dialogue tasks
- **Validate Framework**: Demonstrate evaluation framework works across different models
- **Research Integrity**: Provide fair, reproducible comparison methodology

## 🔄 Migration from Custom Frameworks

The old custom evaluation frameworks have been removed and replaced with these official ones. The new system provides:

- ✅ Research-validated metrics
- ✅ Official benchmark compliance
- ✅ Reproducible results
- ✅ Proper citation and attribution
- ✅ No overfitting to custom keyword patterns
- ✅ Ready for baseline comparison studies

## 📚 References

1. **MultiWOZ**: Nekvinda, T., & Dušek, O. (2021). Shades of BLEU, Flavours of Success: The Case of MultiWOZ. *Proceedings of the 1st Workshop on Natural Language Generation, Evaluation, and Metrics (GEM 2021)*.

2. **SGD**: Rastogi, A., et al. (2020). Schema-Guided Dialogue Dataset. *Proceedings of the 8th Dialogue System Technology Challenge (DSTC8)*.

3. **Taskmaster**: Byrne, B., et al. (2019). Taskmaster-1: Toward a Realistic and Diverse Dialog Dataset. *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing*.

