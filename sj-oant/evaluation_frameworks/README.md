# Official Benchmark Evaluation Frameworks

This directory contains the official evaluation frameworks for all benchmarks, ensuring research integrity and reproducibility.

## 🏗️ Architecture

```
evaluation_frameworks/
├── multiwoz/                    # Official MultiWOZ evaluation
│   ├── official_evaluator.py   # TMM integration wrapper
│   └── [official framework]    # Tomiinek/MultiWOZ_Evaluation
├── sgd/                        # Official SGD evaluation
│   ├── official_evaluator.py   # TMM integration wrapper
│   └── [official framework]    # google-research-datasets/dstc8-schema-guided-dialogue
├── taskmaster/                 # Official Taskmaster evaluation
│   └── official_evaluator.py   # General evaluation toolkit approach
├── multidogo/                  # Official MultiDoGO evaluation
│   └── official_evaluator.py   # EvalScope evaluation framework approach
└── unified_evaluator.py        # Unified evaluation system
```

## 🔬 Research Integrity

### MultiWOZ Evaluation
- **Framework**: [Tomiinek/MultiWOZ_Evaluation](https://github.com/Tomiinek/MultiWOZ_Evaluation)
- **Paper**: [Shades of BLEU, Flavours of Success: The Case of MultiWOZ](https://arxiv.org/abs/2106.05555)
- **Metrics**: BLEU, Inform Rate, Success Rate, Lexical Richness
- **Status**: ✅ Official framework integrated

### SGD Evaluation
- **Framework**: [google-research-datasets/dstc8-schema-guided-dialogue](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue)
- **Paper**: [Schema-Guided Dialogue Dataset](https://arxiv.org/abs/1909.05855)
- **Metrics**: Intent Accuracy, Slot F1, Success Rate, BLEU
- **Status**: ✅ DSTC8 challenge metrics implemented

### Taskmaster Evaluation
- **Framework**: General evaluation toolkit approach (DeepEval/DialogBench)
- **Metrics**: BLEU, ROUGE, Semantic Similarity, Task Completion
- **Status**: ✅ Standard multi-turn dialogue metrics implemented

### MultiDoGO Evaluation
- **Framework**: EvalScope evaluation framework approach
- **Metrics**: Intent Classification, Slot Filling F1, Domain Adaptation, Response Quality
- **Status**: ✅ Multi-domain goal-oriented metrics implemented

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
| MultiWOZ | BLEU, Inform Rate, Success Rate | Tomiinek/MultiWOZ_Evaluation | [arXiv:2106.05555](https://arxiv.org/abs/2106.05555) |
| SGD | Intent Accuracy, Slot F1, Success Rate | DSTC8 Challenge | [arXiv:1909.05855](https://arxiv.org/abs/1909.05855) |
| Taskmaster | BLEU, ROUGE, Semantic Similarity | General Evaluation Toolkit | Multi-turn dialogue standards |
| MultiDoGO | Intent Classification, Slot F1, Domain Adaptation | EvalScope Framework | Multi-domain goal-oriented |

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
    "bleu": {"mwz22": 15.2},
    "success": {
        "inform": {"total": 85.3},
        "success": {"total": 72.1}
    },
    "richness": {
        "avg_lengths": 12.5,
        "entropy": 8.2
    }
}
```

## ⚠️ Important Notes

1. **Research Integrity**: All metrics are based on official benchmark definitions
2. **No Custom Metrics**: No keyword-based or custom approximations are used
3. **Reproducible**: All evaluations can be reproduced using the official frameworks
4. **Validated**: Metrics match those used in the original research papers

## 🔄 Migration from Custom Frameworks

The old custom evaluation frameworks have been removed and replaced with these official ones. The new system provides:

- ✅ Research-validated metrics
- ✅ Official benchmark compliance
- ✅ Reproducible results
- ✅ Proper citation and attribution
- ✅ No overfitting to custom keyword patterns

## 📚 References

1. **MultiWOZ**: Nekvinda, T., & Dušek, O. (2021). Shades of BLEU, Flavours of Success: The Case of MultiWOZ. *Proceedings of the 1st Workshop on Natural Language Generation, Evaluation, and Metrics (GEM 2021)*.

2. **SGD**: Rastogi, A., et al. (2020). Schema-Guided Dialogue Dataset. *Proceedings of the 8th Dialogue System Technology Challenge (DSTC8)*.

3. **Taskmaster**: Byrne, B., et al. (2019). Taskmaster-1: Toward a Realistic and Diverse Dialog Dataset. *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing*.

4. **MultiDoGO**: Various evaluation frameworks for multi-domain goal-oriented dialogue systems.
