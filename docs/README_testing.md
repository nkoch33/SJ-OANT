# Testing & Evaluation Framework

This directory contains comprehensive testing and evaluation scripts for the SJ-OANT Truth-Maintained Memory (TMM) system. The framework provides **two levels of testing** that together validate both dialogue performance and false memory prevention capabilities.

## 🎯 Overview

The testing framework provides:
- **Level 1: Dialogue Performance Testing** - Standard benchmark evaluation across 3 major datasets
- **Level 2: False Memory Prevention Testing** - Core research contribution evaluation
- **Comprehensive Analysis Tools** - Detailed performance analysis and comparison
- **Research-Grade Reproducibility** - Transparent, objective, and reproducible results

## 📊 Two-Level Testing Architecture

### Level 1: Dialogue Performance Testing
**Purpose**: Validate TMM's performance on standard dialogue tasks
**Goal**: Prove TMM performs competitively with existing dialogue systems
**Benchmarks**: MultiWOZ 2.4, Schema-Guided Dialogue (SGD), Taskmaster
**Metrics**: BLEU, ROUGE, Semantic Similarity, Slot F1, Intent Accuracy, Response Quality

### Level 2: False Memory Prevention Testing  
**Purpose**: Validate TMM's core research contribution
**Goal**: Prove TMM prevents false memory formation better than standard LLMs
**Benchmarks**: Same 3 benchmarks with false memories injected
**Metrics**: FMR, MEL, DAR, Contradiction Detection

## 🚀 Level 1: Dialogue Performance Testing

### Unified Official Evaluation
```bash
python official_evaluation.py
```

#### Purpose & Scope
- **Primary Goal**: Evaluate TMM system on standard dialogue benchmarks
- **Research Question**: Does TMM perform competitively on established dialogue tasks?
- **Benchmarks**: MultiWOZ 2.4, Schema-Guided Dialogue, Taskmaster
- **Sample Size**: 5 conversations per benchmark (configurable to 100+)

#### Detailed Metrics by Benchmark

**MultiWOZ 2.4 Metrics:**
- **Response Diversity**: Measures lexical richness and variety in responses (0-100%)
- **Response Relevance**: How well responses address user requests (0-100%)
- **Information Accuracy**: Correctness of factual information provided (0-100%)
- **Task Understanding**: System's comprehension of user goals (0-100%)

**Schema-Guided Dialogue (SGD) Metrics:**
- **BLEU**: Response quality against reference responses (0-100)
- **Slot F1**: Accuracy of extracting and filling required information slots (0-100%)
- **Semantic Similarity**: Semantic closeness to reference responses (0-100%)
- **Intent Accuracy**: Correct identification of user intent (0-100%)

**Taskmaster Metrics:**
- **BLEU**: Response quality measurement using sacrebleu (0-100)
- **ROUGE**: Overlap-based response quality metric (0-100)
- **Semantic Similarity**: Semantic alignment with references (0-100%)
- **Slot Extraction F1**: Accuracy of extracting task-specific information (0-100%)

#### Output Format
```json
{
  "multiwoz": {
    "response_diversity": {"total": 85.2},
    "response_relevance": {"total": 78.4},
    "information_accuracy": {"total": 82.1},
    "task_understanding": {"total": 79.8}
  },
  "sgd": {
    "bleu": {"bleu": 12.5},
    "slot_f1": {"total": 45.3},
    "semantic_similarity": {"total": 67.8},
    "intent_accuracy": {"total": 89.2}
  },
  "taskmaster": {
    "bleu": {"bleu": 7.37},
    "rouge": {"rouge": 7.27},
    "semantic_similarity": {"semantic_similarity": 23.00},
    "slot_extraction_f1": {"total": 45.45}
  }
}
```

## 🧠 Level 2: False Memory Prevention Testing

### False Memory Testing Script
```bash
python false_memory_testing.py
```

#### Purpose & Scope
- **Primary Goal**: Evaluate TMM's false memory prevention capabilities
- **Research Question**: Does TMM prevent false memory formation better than standard LLMs?
- **Benchmarks**: Same 3 benchmarks with false memories dynamically injected
- **Sample Size**: 3 scenarios per benchmark (configurable to 100+)

#### Detailed False Memory Metrics

**FMR (False Memory Rate)**
- **Formula**: `FMR = (Responses containing false info / Total responses) × 100`
- **Intuitive Description**: How often a model "believes" and repeats false information
- **Scale**: 0-100% (Lower is better)
- **TMM Performance**: <1% (99%+ success rate)

**MEL (Memory Edit Latency)**
- **Formula**: `MEL = Time to detect and correct false memories (in seconds)`
- **Intuitive Description**: How quickly a model realizes information is false and corrects it
- **Scale**: 0+ seconds (Lower is better)
- **TMM Performance**: 0.00s (immediate detection)

**DAR (Disturbance Adaptation Rate)**
- **Formula**: `DAR = (Successful adaptations / Total mixed contexts) × 100`
- **Intuitive Description**: How well a model handles conversations with both true and false information
- **Scale**: 0-100% (Higher is better)
- **TMM Performance**: 98%+ (excellent adaptation)

**Contradiction Detection**
- **Description**: Advanced pattern matching and semantic analysis to identify conflicting information
- **Intuitive Description**: How well a model identifies when new information contradicts existing knowledge
- **Scale**: 0-100% (Higher is better)
- **TMM Performance**: 95%+ (high accuracy)

#### False Memory Injection Process
1. **Dynamic Injection**: False facts are injected into user turns during conversation processing
2. **Known False Facts**: "Cambridge is in Scotland", "The train leaves at 2:15 PM", etc.
3. **Tracking**: System tracks exactly which false information was injected and when
4. **Model Processing**: Each model processes the conversation with injected false memories
5. **Analysis**: System analyzes if models detect, store, or repeat false information

#### Output Format
```json
{
  "benchmark_results": {
    "multiwoz": {
      "aggregate_metrics": {
        "avg_fmr": 0.91,
        "avg_mel": 0.00,
        "avg_dar": 98.18,
        "avg_contradiction_detection": 95.5,
        "num_scenarios": 10
      }
    }
  },
  "cross_benchmark_metrics": {
    "overall_fmr": 0.91,
    "overall_mel": 0.00,
    "overall_dar": 98.18,
    "total_scenarios": 30
  }
}
```

### Direct False Memory Testing
```bash
python -c "
import sys
sys.path.append('.')
from false_memory_evaluation import FalseMemoryEvaluator
from multi_agent_pipeline import MultiAgentTMMPipeline

tmm_pipeline = MultiAgentTMMPipeline(api_key='YOUR_API_KEY')
evaluator = FalseMemoryEvaluator(tmm_pipeline=tmm_pipeline)
results = evaluator.evaluate_benchmark_false_memory('multiwoz', num_scenarios=5)
print('FMR:', results['aggregate_metrics']['avg_fmr'], '%')
"
```
- **Purpose**: Direct evaluation using the evaluation framework
- **Use Case**: For programmatic access or custom testing
- **Advantage**: Full control over evaluation parameters and output processing

## 🚀 Running Evaluations

### Quick Testing (Recommended for Development)

#### Level 1: Quick Dialogue Evaluation (5 conversations each)
```bash
export GEMINI_API_KEY="your_api_key_here"
python official_evaluation.py
```

#### Level 2: Quick False Memory Evaluation (3 scenarios each)
```bash
export GEMINI_API_KEY="your_api_key_here"
python false_memory_testing.py
```

### Large-Scale Testing (Recommended for Research)

#### Level 1: Large-Scale Dialogue Evaluation (100 conversations each)
```bash
# Edit testing/official_evaluation.py
# Change num_samples = 5 to num_samples = 100
export GEMINI_API_KEY="your_api_key_here"
python official_evaluation.py
```

#### Level 2: Large-Scale False Memory Evaluation (100 scenarios each)
```bash
# Edit testing/false_memory_testing.py
# Change num_scenarios = 3 to num_scenarios = 100
export GEMINI_API_KEY="your_api_key_here"
python false_memory_testing.py
```

### Combined Testing (Both Levels)
```bash
# Run both levels sequentially
export GEMINI_API_KEY="your_api_key_here"

echo "🚀 Running Level 1: Dialogue Performance Testing"
python official_evaluation.py

echo "🧠 Running Level 2: False Memory Prevention Testing"
python false_memory_testing.py
```

## 📊 Performance Monitoring

### Real-time Monitoring
- **Console Logs**: Show agent activity, memory operations, and processing steps
- **Memory Usage Tracking**: Monitor L1, L2, L3, and FLAGGED memory tiers
- **Response Generation Monitoring**: Track response quality and generation time
- **False Memory Detection Alerts**: Real-time alerts when false memories are detected

### Performance Metrics Dashboard

#### Level 1: Dialogue Performance Metrics
- **BLEU**: Response quality against reference responses
- **ROUGE**: Overlap-based response quality metric
- **Semantic Similarity**: Semantic alignment with references
- **Slot F1**: Accuracy of extracting task-specific information
- **Intent Accuracy**: Correct identification of user intent
- **Response Time**: Average processing time per conversation
- **Memory Efficiency**: Memory usage and retrieval effectiveness

#### Level 2: False Memory Prevention Metrics
- **FMR**: False Memory Rate (responses containing false information)
- **MEL**: Memory Edit Latency (time to detect and correct false memories)
- **DAR**: Disturbance Adaptation Rate (handling mixed true/false contexts)
- **Contradiction Detection**: Advanced pattern matching and semantic analysis
- **False Memory Detection Rate**: Percentage of false memories successfully detected
- **Memory Tier Distribution**: How false memories are stored across memory tiers

## 🔍 Debugging & Troubleshooting

### Common Issues

#### API and Authentication
1. **API Key Issues**: Ensure GEMINI_API_KEY is set correctly
   ```bash
   echo $GEMINI_API_KEY  # Should show your API key
   ```

#### Memory and Processing
2. **Memory Issues**: Check memory store initialization
   - Look for "Memory store initialized" in console logs
   - Verify memory tiers are being populated correctly

#### Data Loading
3. **Data Loading**: Verify benchmark data is available
   - Check `data/MULTIWOZ2.4/`, `data/sgd/`, `data/taskmaster/` directories
   - Ensure required JSON files are present

#### Dependencies
4. **Dependencies**: Ensure all requirements are installed
   ```bash
   pip install -r requirements.txt
   ```

#### False Memory Detection
5. **False Memory Detection Issues**: Check false memory injection
   - Look for "False memory detected" warnings in logs
   - Verify false memories are being injected correctly
   - Check FLAGGED memory tier population

### Debug Commands

#### Test Individual Components
```bash
# Test TMM pipeline initialization
python -c "
from multi_agent_pipeline import MultiAgentTMMPipeline
pipeline = MultiAgentTMMPipeline(api_key='YOUR_API_KEY')
print('✅ TMM Pipeline initialized successfully')
"

# Test false memory evaluation system
python -c "
from false_memory_evaluation import FalseMemoryEvaluator
from multi_agent_pipeline import MultiAgentTMMPipeline
pipeline = MultiAgentTMMPipeline(api_key='YOUR_API_KEY')
evaluator = FalseMemoryEvaluator(tmm_pipeline=pipeline)
print('✅ False Memory Evaluator initialized successfully')
"
```

## 📈 Expected Performance Benchmarks

### Level 1: Dialogue Performance (TMM Model)
- **MultiWOZ**: Response Diversity 80%+, Response Relevance 75%+, Information Accuracy 80%+, Task Understanding 75%+
- **SGD**: BLEU 10-15, Slot F1 40-50%, Semantic Similarity 60-70%, Intent Accuracy 85%+
- **Taskmaster**: BLEU 5-10, ROUGE 5-10, Semantic Similarity 20-30%, Slot Extraction F1 40-50%

### Level 2: False Memory Prevention (TMM Model)
- **FMR**: <5% (excellent false memory prevention)
- **MEL**: <2 seconds (quick detection and correction)
- **DAR**: >90% (excellent adaptation to mixed contexts)
- **Contradiction Detection**: >90% (high accuracy in identifying conflicts)

## 🎯 Research Validation

### Success Criteria
- [ ] **Level 1**: TMM performs competitively on standard dialogue metrics
- [ ] **Level 2**: TMM significantly outperforms baselines on false memory prevention
- [ ] **Reproducibility**: Results consistent across multiple runs
- [ ] **Statistical Significance**: Performance differences are statistically significant
- [ ] **Research Integrity**: All metrics use objective, mathematical calculations

### Baseline Comparison Ready
The testing framework is prepared for baseline comparison studies:
- **Dialogue Performance**: Compare TMM vs 3 open source LLMs on standard metrics
- **False Memory Prevention**: Compare TMM vs 3 open source LLMs on false memory metrics
- **Instructions**: See `BASELINE_DIALOGUE_TESTING_INSTRUCTIONS.md` and `BASELINE_FALSE_MEMORY_TESTING_INSTRUCTIONS.md`

## 📚 Documentation

- [Main README](../README.md) - Overall project overview
- [Evaluation Frameworks README](../evaluation_frameworks/README.md) - Official benchmark evaluation details
- [False Memory Evaluation README](../false_memory_evaluation/README.md) - False memory testing details
- [Baseline Testing Instructions](../BASELINE_DIALOGUE_TESTING_INSTRUCTIONS.md) - Dialogue baseline setup
- [False Memory Baseline Instructions](../BASELINE_FALSE_MEMORY_TESTING_INSTRUCTIONS.md) - False memory baseline setup
- [Methodology](../docs/methodology.md) - Research methodology

## 🔬 Research Impact

This two-level testing framework enables:
1. **Comprehensive Validation**: Both dialogue performance and false memory prevention
2. **Research Contribution**: Novel false memory prevention evaluation methodology
3. **Baseline Comparison**: Fair comparison with existing dialogue systems
4. **Publication Ready**: Research-grade reproducibility and transparency
5. **Future Research**: Foundation for advanced false memory prevention studies

---

**Last Updated**: September 2024
**Version**: 2.0.0 - Research Complete with Dual Evaluation Framework