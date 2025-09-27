# Testing Framework

This directory contains the comprehensive testing framework for TMMA evaluation across both dialogue performance and false memory prevention tasks.

## Testing Components

### `official_evaluation.py`
Standard dialogue performance testing that:
- **Benchmark Evaluation**: Tests TMMA on MultiWOZ, SGD, and Taskmaster
- **Metric Computation**: Calculates BLEU, ROUGE, semantic similarity, slot F1
- **Baseline Comparison**: Compares against standard LLMs and RAG systems
- **Performance Analysis**: Generates comprehensive performance reports

### `false_memory_testing.py`
False memory prevention testing that:
- **Injection Testing**: Controlled false memory injection across benchmarks
- **Prevention Assessment**: Evaluates FMR, MEL, DAR, and CDR metrics
- **Resilience Testing**: Tests system robustness under adversarial conditions
- **Recovery Analysis**: Measures error detection and correction capabilities

## Testing Protocol

### Dual-Level Evaluation
1. **Level 1**: Standard dialogue performance across established benchmarks
2. **Level 2**: False memory prevention through controlled injection tests

### Configuration
- **Test Conversations**: 100 per benchmark (300 total)
- **Random Seed**: Fixed at 42 for reproducibility
- **Model Parameters**: Identical configuration across all systems
- **Evaluation Metrics**: Standardized computation across all tests

## Usage

### Running Complete Evaluation
```bash
# Run standard dialogue evaluation
python official_evaluation.py

# Run false memory prevention testing
python false_memory_testing.py

# Run both evaluations
python official_evaluation.py && python false_memory_testing.py
```

### Custom Testing
```python
from sj_oant.testing import official_evaluation, false_memory_testing

# Custom benchmark evaluation
results = official_evaluation.evaluate_benchmark('multiwoz', num_conversations=50)

# Custom false memory testing
fm_results = false_memory_testing.test_false_memory_prevention('sgd', injection_rate=0.1)
```

## Test Results

### Output Files
- **Dialogue Results**: Performance metrics across all benchmarks
- **False Memory Results**: Prevention and recovery metrics
- **Comparison Reports**: Baseline vs TMMA performance analysis
- **Statistical Analysis**: Comprehensive evaluation summaries

### Result Interpretation
- **Dialogue Quality**: Response fluency, relevance, and task completion
- **False Memory Prevention**: Error avoidance and detection capabilities
- **System Robustness**: Performance under adversarial conditions
- **Recovery Mechanisms**: Error correction and system stability

## Integration

The testing framework integrates with:
- **Evaluation Frameworks**: Uses official benchmark evaluators
- **False Memory Evaluation**: Leverages injection and detection systems
- **Baseline Models**: Compares against standard LLMs and RAG systems
- **Metrics Calculation**: Standardized evaluation across all components

## Validation

### Test Validation
- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end system validation
- **Regression Tests**: Performance consistency checks
- **Stress Tests**: System behavior under extreme conditions

### Quality Assurance
- **Reproducibility**: Fixed seeds and deterministic evaluation
- **Completeness**: Comprehensive coverage of all system components
- **Accuracy**: Validated metric computation and result interpretation
- **Documentation**: Clear testing procedures and result analysis

## Extensibility

The testing framework supports:
- **New Benchmarks**: Easy addition of new evaluation datasets
- **Custom Metrics**: Extensible metric computation framework
- **Additional Baselines**: Flexible baseline model integration
- **Advanced Testing**: Sophisticated evaluation scenarios

## Best Practices

### Testing Guidelines
- **Consistent Configuration**: Identical parameters across all evaluations
- **Statistical Rigor**: Multiple runs and significance testing
- **Comprehensive Coverage**: All system components and edge cases
- **Clear Documentation**: Detailed testing procedures and results

### Result Reporting
- **Transparent Metrics**: Clear explanation of all evaluation measures
- **Comparative Analysis**: Fair comparison with baseline systems
- **Statistical Validation**: Confidence intervals and significance testing
- **Reproducible Results**: Complete configuration and seed information
