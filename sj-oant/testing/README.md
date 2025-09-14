# Testing & Evaluation Framework

This directory contains comprehensive testing and evaluation scripts for the SJ-OANT Truth-Maintained Memory (TMM) system.

## 🎯 Overview

The testing framework provides:
- Individual benchmark evaluation scripts
- Comprehensive analysis tools
- Performance optimization scripts
- Large-scale evaluation capabilities

## 📊 Available Tests

### Individual Benchmark Tests

#### MultiWOZ Evaluation
```bash
python test_multiwoz_evaluation.py
```
- **Purpose**: Evaluate TMM system on MultiWOZ 2.4 benchmark
- **Metrics**: BLEU, ROUGE, Semantic Similarity, Task Completion
- **Sample Size**: 10 conversations (configurable)
- **Output**: `results/multiwoz_standard_evaluation_results.json`

#### SGD Optimization Test
```bash
python test_sgd_optimization.py
```
- **Purpose**: Evaluate TMM system on Schema-Guided Dialogue benchmark
- **Metrics**: Intent Accuracy, Slot F1, Success Rate, BLEU
- **Sample Size**: 10 conversations (configurable)
- **Output**: `results/sgd_optimization_test_results.json`

#### Taskmaster Optimization Test
```bash
python test_taskmaster_optimization.py
```
- **Purpose**: Evaluate TMM system on Taskmaster benchmark
- **Metrics**: BLEU, ROUGE, Semantic Similarity, Task Completion
- **Sample Size**: 10 conversations (configurable)
- **Output**: `results/taskmaster_optimization_test_results.json`

#### MultiDoGO Optimization Test
```bash
python test_multidogo_optimization.py
```
- **Purpose**: Evaluate TMM system on MultiDoGO benchmark
- **Metrics**: Intent Classification, Slot F1, Domain Adaptation, Response Quality
- **Sample Size**: 10 conversations (configurable)
- **Output**: `results/multidogo_optimization_test_results.json`

### Analysis & Optimization Scripts

#### Comprehensive Results Analysis
```bash
python comprehensive_results_analysis.py
```
- **Purpose**: Analyze results from all benchmarks
- **Features**: Performance interpretation, strengths/weaknesses analysis, recommendations
- **Output**: `results/comprehensive_analysis_report.md`

#### Deep Model Analysis
```bash
python deep_model_analysis.py
```
- **Purpose**: Deep analysis of model performance patterns
- **Features**: Memory usage analysis, agent activity monitoring, optimization opportunities
- **Output**: Console logs and analysis reports

#### General Optimizations
```bash
python general_optimizations.py
```
- **Purpose**: Apply general model improvements
- **Features**: Context retention, response quality, slot extraction, truth verification enhancements
- **Output**: Optimization recommendations and implementations

#### Evaluation Framework Definitions
```bash
python evaluation_framework_definitions.py
```
- **Purpose**: Define and document evaluation metrics
- **Features**: Metric explanations, evaluation criteria, research integrity documentation
- **Output**: Framework documentation and metric definitions

### Large-Scale Evaluation

#### Large-Scale Evaluation Script
```bash
python large_scale_evaluation.py
```
- **Purpose**: Run comprehensive evaluation across all benchmarks
- **Features**: 25 conversations per benchmark, full metric analysis
- **Output**: Complete evaluation results for all benchmarks

#### Optimization Improvements Test
```bash
python test_optimization_improvements.py
```
- **Purpose**: Test specific optimization improvements
- **Features**: BLEU score improvements, intent classification accuracy, response quality
- **Output**: Optimization effectiveness analysis

## 🔧 Configuration

### API Key Setup
```bash
python setup_api_key.py
```
- Sets up Google Gemini API key for LLM access
- Creates environment configuration
- Validates API connectivity

### Environment Variables
```bash
export GEMINI_API_KEY="your-api-key-here"
```

## 📈 Results Structure

### Individual Results
Each test generates JSON results with:
- **Metrics**: Specific benchmark metrics
- **Conversations**: Sample conversations processed
- **Performance**: Detailed performance breakdown
- **Metadata**: Test configuration and timestamps

### Comprehensive Analysis
The comprehensive analysis provides:
- **Executive Summary**: Overall performance assessment
- **Benchmark Analysis**: Individual benchmark performance
- **Strengths/Weaknesses**: Detailed analysis
- **Recommendations**: Improvement suggestions
- **Research Readiness**: Publication readiness assessment

## 🚀 Running Full Evaluation

### Quick Test (10 conversations each)
```bash
# Run all individual tests
python test_multiwoz_evaluation.py
python test_sgd_optimization.py
python test_taskmaster_optimization.py
python test_multidogo_optimization.py

# Generate comprehensive analysis
python comprehensive_results_analysis.py
```

### Large-Scale Test (25 conversations each)
```bash
python large_scale_evaluation.py
```

### Full-Scale Test (200+ conversations each)
```bash
# Modify sample sizes in scripts and run
python large_scale_evaluation.py  # Update to 200+ samples
```

## 📊 Performance Monitoring

### Real-time Monitoring
- Console logs show agent activity
- Memory usage tracking
- Response generation monitoring
- Error detection and reporting

### Performance Metrics
- **Response Time**: Average processing time per conversation
- **Memory Efficiency**: Memory usage and retrieval effectiveness
- **Quality Scores**: Response quality and truthfulness scores
- **Success Rates**: Task completion and success rates

## 🔍 Debugging

### Common Issues
1. **API Key Issues**: Ensure GEMINI_API_KEY is set correctly
2. **Memory Issues**: Check memory store initialization
3. **Data Loading**: Verify benchmark data is available
4. **Dependencies**: Ensure all requirements are installed

### Debug Scripts
- `debug_sgd_responses.py`: Debug SGD-specific issues
- `quick_response_test.py`: Quick response generation test
- `test_slot_extraction.py`: Test slot extraction functionality

## 📝 Output Files

### Results Directory Structure
```
results/
├── multiwoz_standard_evaluation_results.json
├── sgd_optimization_test_results.json
├── taskmaster_optimization_test_results.json
├── multidogo_optimization_test_results.json
├── comprehensive_analysis_report.md
└── benchmark_results/
    └── [individual benchmark results]
```

### Log Files
- Console output with detailed logging
- Error logs for debugging
- Performance metrics logs

## 🎯 Best Practices

### Testing Workflow
1. **Setup**: Configure API keys and environment
2. **Quick Test**: Run individual benchmark tests
3. **Analysis**: Generate comprehensive analysis
4. **Optimization**: Apply improvements based on analysis
5. **Validation**: Re-test to verify improvements
6. **Large-Scale**: Run full-scale evaluation

### Performance Optimization
1. **Monitor**: Track performance metrics
2. **Analyze**: Identify bottlenecks and issues
3. **Optimize**: Apply targeted improvements
4. **Validate**: Test improvements thoroughly
5. **Scale**: Run large-scale evaluations

## 📚 Documentation

- [Main README](../README.md)
- [Project Structure](../PROJECT_STRUCTURE.md)
- [Methodology](../docs/methodology.md)
- [Benchmark Integration Summaries](../docs/)

## 🤝 Contributing

When adding new tests:
1. Follow naming convention: `test_[benchmark]_[purpose].py`
2. Include comprehensive documentation
3. Add proper error handling
4. Generate structured output
5. Update this README

---

**Last Updated**: September 2024
**Version**: 1.0.0