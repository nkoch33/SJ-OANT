# SJ-OANT: Truth-Maintained Memory (TMM) Multi-Agent System

A sophisticated multi-agent dialogue system implementing Truth-Maintained Memory (TMM) with comprehensive evaluation across multiple benchmarks.

## 🎯 Overview

SJ-OANT is a research-grade multi-agent system that implements a novel Truth-Maintained Memory architecture for dialogue systems. The system consists of multiple specialized agents working together to provide accurate, context-aware, and truthful responses across various domains.

## 🏗️ Architecture

### Core Components

- **Strategic Planner**: Analyzes queries and plans execution strategies
- **TACS Filter**: Token-level Adaptive Context Screening for noise reduction and memory retrieval
- **Truth Verifier**: Assesses truthfulness and confidence of information
- **Memory Curator**: Manages multi-tiered memory system (L1: Working, L2: Summarized, L3: Archival)
- **Responder**: Generates final responses with quality control

### Memory System

- **L1 (Working Memory)**: Recent, active information
- **L2 (Summarized Memory)**: Condensed, important information
- **L3 (Archival Memory)**: Long-term, reference information
- **Flagged Memory**: Information requiring special attention

## 📊 Benchmarks & Evaluation

The system is evaluated on four major dialogue benchmarks:

### MultiWOZ 2.4
- **Metrics**: BLEU, ROUGE, Semantic Similarity, Task Completion
- **Performance**: 96.49% Task Completion, 33.17% Semantic Similarity
- **Framework**: Official MultiWOZ evaluation toolkit

### Schema-Guided Dialogue (SGD)
- **Metrics**: Intent Accuracy, Slot F1, Success Rate, BLEU
- **Performance**: 100% Intent Accuracy, 100% Slot F1, 100% Success Rate
- **Framework**: DSTC8 challenge metrics

### Taskmaster
- **Metrics**: BLEU, ROUGE, Semantic Similarity, Task Completion
- **Performance**: 85% Task Completion, 25.95% Semantic Similarity
- **Framework**: General evaluation toolkit approach

### MultiDoGO
- **Metrics**: Intent Classification, Slot F1, Domain Adaptation, Response Quality
- **Performance**: 70% Slot F1, 100% Domain Adaptation, 48.47% Response Quality
- **Framework**: EvalScope evaluation framework

## 🚀 Quick Start

### Prerequisites

```bash
pip install -r sj-oant/requirements.txt
```

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd SJ-OANT
   ```

2. **Set up API key**:
   ```bash
   cp sj-oant/env.example sj-oant/.env
   # Edit .env and add your GEMINI_API_KEY
   ```

3. **Run evaluation**:
   ```bash
   cd sj-oant/testing
   python test_multiwoz_evaluation.py
   ```

## 📁 Project Structure

```
SJ-OANT/
├── sj-oant/                    # Main package
│   ├── agents/                 # Multi-agent components
│   │   ├── planner.py         # Strategic planning agent
│   │   ├── responder.py       # Response generation agent
│   │   └── writer_editor.py   # Memory curation agent
│   ├── core/                  # Core interfaces and types
│   ├── memory/                # Memory management system
│   ├── truth/                 # Truth verification components
│   ├── evaluation_frameworks/ # Benchmark evaluation tools
│   ├── data/                  # Benchmark datasets
│   ├── testing/               # Evaluation and testing scripts
│   └── results/               # Evaluation results
├── docs/                      # Documentation
└── README.md                  # This file
```

## 🔬 Research Features

### Truth-Maintained Memory (TMM)
- Multi-tiered memory architecture
- Truth verification and confidence scoring
- Adaptive context screening
- Memory promotion and archival

### Multi-Agent Coordination
- Strategic planning and execution
- Context-aware response generation
- Quality control and safety measures
- Performance monitoring

### Comprehensive Evaluation
- Official benchmark frameworks
- Research-grade metrics
- Reproducible evaluation pipeline
- Detailed performance analysis

## 📈 Performance Summary

| Benchmark | Task Completion | Intent Accuracy | Slot F1 | BLEU Score |
|-----------|----------------|-----------------|---------|------------|
| MultiWOZ  | 96.49%         | -               | -       | 4.33%      |
| SGD       | 100.00%        | 100.00%         | 100.00% | 4.70%      |
| Taskmaster| 85.00%         | -               | -       | 11.46%     |
| MultiDoGO | -              | 33.33%          | 70.00%  | -          |

## 🛠️ Development

### Running Tests

```bash
# Individual benchmark tests
python test_multiwoz_evaluation.py
python test_sgd_optimization.py
python test_taskmaster_optimization.py
python test_multidogo_optimization.py

# Comprehensive analysis
python comprehensive_results_analysis.py
```

### Adding New Benchmarks

1. Create evaluator in `evaluation_frameworks/`
2. Add data to `data/` directory
3. Create test script in `testing/`
4. Update unified evaluator

## 📚 Documentation

- [Methodology](sj-oant/docs/methodology.md)
- [Benchmark Integration Summaries](sj-oant/docs/)
- [Project Structure](sj-oant/PROJECT_STRUCTURE.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- MultiWOZ team for the dialogue dataset
- Google Research for SGD benchmark
- Taskmaster dataset contributors
- MultiDoGO evaluation framework

## 📞 Contact

For questions and collaboration, please open an issue or contact the maintainers.

---

**Status**: Research-ready, actively maintained
**Last Updated**: September 2024
**Version**: 1.0.0