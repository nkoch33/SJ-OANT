# SJ-OANT Project Structure

## 📁 Directory Overview

```
SJ-OANT/sj-oant/
├── agents/                        # Multi-agent system components
│   ├── planner.py                # Strategic planning agent
│   ├── responder.py              # Response generation agent
│   └── writer_editor.py          # Memory curation agent
├── core/                         # Core system types and utilities
│   ├── types.py                  # Type definitions
│   └── config.py                 # Configuration management
├── data/                         # Benchmark datasets
│   ├── MULTIWOZ2.4/             # MultiWOZ 2.4 dataset
│   ├── sgd/                     # Schema-Guided Dialogue dataset
│   ├── taskmaster/              # Taskmaster dataset
│   └── multidogo/               # MultiDoGO dataset
├── evaluation/                   # TMM-specific evaluation logic
│   ├── multiwoz_streamlined_eval.py
│   ├── iterative_testing.py
│   └── methodology_metrics.py
├── evaluation_frameworks/        # Official benchmark evaluation frameworks
│   ├── multiwoz/                # Official MultiWOZ evaluation
│   ├── sgd/                     # Official SGD evaluation
│   ├── taskmaster/              # Official Taskmaster evaluation
│   ├── multidogo/               # Official MultiDoGO evaluation
│   └── unified_evaluator.py     # Unified evaluation system
├── memory/                       # Memory system components
│   ├── typed_store.py           # Multi-tier memory store
│   └── retrieval.py             # Memory retrieval mechanisms
├── runners/                      # Execution scripts
│   ├── eval_streamlined_multiwoz.py
│   └── run_iterative_testing.py
├── testing/                      # Testing and evaluation
│   ├── official_evaluation.py   # Official benchmark evaluation
│   ├── run_official_evaluation.py
│   ├── tmm_evaluation.py        # TMM-specific evaluation
│   ├── run_tmm_evaluation.py
│   └── results/                 # Evaluation results
├── truth/                        # Truth verification components
│   ├── verifier.py              # Truth verification agent
│   └── tacs_filter.py           # TACS filtering system
├── multi_agent_pipeline.py      # Main multi-agent pipeline
├── tmm_pipeline.py              # TMM pipeline wrapper
└── requirements.txt             # Dependencies
```

## 🔬 Research Integrity

### Official Benchmark Frameworks
- **MultiWOZ**: [Tomiinek/MultiWOZ_Evaluation](https://github.com/Tomiinek/MultiWOZ_Evaluation)
- **SGD**: [google-research-datasets/dstc8-schema-guided-dialogue](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue)
- **Taskmaster**: General evaluation toolkit approach
- **MultiDoGO**: EvalScope evaluation framework approach

### Key Features
- ✅ All metrics based on official benchmark definitions
- ✅ No custom keyword-based approximations
- ✅ Reproducible evaluation results
- ✅ Proper citation and attribution
- ✅ Research integrity compliance

## 🚀 Usage

### Run Official Evaluation
```bash
cd testing
python run_official_evaluation.py
```

### Run TMM-Specific Evaluation
```bash
cd testing
python run_tmm_evaluation.py
```

### Run Iterative Testing
```bash
cd runners
python run_iterative_testing.py
```

## 📊 Evaluation Results

Results are saved in `testing/results/` directory with comprehensive metrics for each benchmark.
