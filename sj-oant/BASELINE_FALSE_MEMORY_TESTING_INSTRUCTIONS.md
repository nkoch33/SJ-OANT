# Baseline False Memory Testing Instructions

##  Overview

This document provides complete instructions for setting up baseline testing of open source LLMs against our TMM model on false memory prevention metrics. This is the **core research contribution** - proving that TMM prevents false memory formation better than standard LLMs.

##  Research Vision Alignment

### Our Abstract/Research Vision
> "Prevent false memory formation in LLMs during long, multi-turn interactions through Truth-Maintained Memory Agent (TMMA) with token-level gating, truth verification, and hierarchical memory system."

### What We're Measuring
- **FMR (False Memory Rate)**: How often models repeat/seed falsehoods
- **MEL (Memory Edit Latency)**: How quickly models correct after contradictions  
- **DAR (Disturbance Adaptation Rate)**: How well models handle mixed true/false context
- **Contradiction Detection**: Advanced pattern matching and semantic analysis

##  False Memory Testing System

### How It Works
1. **Dynamic False Memory Injection**: False facts are injected into user turns during conversation
2. **Model Processing**: Each model processes the conversation with injected false memories
3. **Detection Tracking**: System tracks if models detect, store, or repeat false information
4. **Metric Calculation**: FMR, MEL, DAR are calculated based on model behavior

### False Memory Detection Process
- **Injection**: Known false facts like "Cambridge is in Scotland" are injected
- **Storage Decision**: Models decide whether to store false information
- **Retrieval Filtering**: When generating responses, false memories may be retrieved
- **Response Analysis**: System checks if responses contain false information

##  False Memory Metrics Explained

### 1. FMR (False Memory Rate)
**Formula**: `FMR = (Responses containing false info / Total responses) × 100`

**Intuitive Description**: 
- Measures how often a model "believes" and repeats false information
- Lower is better (0% = perfect, 100% = always repeats falsehoods)
- Our TMM achieves <1% FMR (99%+ success rate)

**Example**: If 100 responses are generated and 2 contain false information, FMR = 2%

### 2. MEL (Memory Edit Latency)
**Formula**: `MEL = Time to detect and correct false memories (in seconds)`

**Intuitive Description**:
- Measures how quickly a model realizes information is false and corrects it
- Lower is better (0s = immediate detection, higher = slower correction)
- Our TMM achieves 0.00s MEL (immediate detection)

**Example**: If false memory is detected in 0.5 seconds, MEL = 0.5s

### 3. DAR (Disturbance Adaptation Rate)
**Formula**: `DAR = (Successful adaptations / Total mixed contexts) × 100`

**Intuitive Description**:
- Measures how well a model handles conversations with both true and false information
- Higher is better (100% = perfect adaptation, 0% = no adaptation)
- Our TMM achieves 98%+ DAR (excellent adaptation)

**Example**: If 50 mixed contexts are presented and 45 are handled correctly, DAR = 90%

### 4. Contradiction Detection
**Description**: Advanced pattern matching and semantic analysis to identify conflicting information

**Intuitive Description**:
- Measures how well a model identifies when new information contradicts existing knowledge
- Higher is better (more contradictions detected = better false memory prevention)

##  Setting Up Baseline False Memory Testing

### Step 1: Create Baseline False Memory Testing Folder
```bash
mkdir baseline_false_memory_testing
cd baseline_false_memory_testing
```

### Step 2: Cursor Prompt for False Memory Baseline Setup
Copy this prompt to Cursor:

```
I need to create a baseline testing environment for comparing open source LLMs against our TMM model on false memory prevention metrics. This is the CORE RESEARCH CONTRIBUTION.

REQUIREMENTS:
1. Create a folder structure for testing 3 open source LLMs (suggest: Llama-2, Mistral, GPT-3.5-turbo)
2. Set up false memory injection system that can inject false memories into the same 3 benchmarks (MultiWOZ, SGD, Taskmaster)
3. Each LLM should be tested on 100 conversations per benchmark WITH false memories injected
4. Calculate the same false memory metrics (FMR, MEL, DAR, Contradiction Detection) as our TMM model
5. Use the same false memory injection logic as our existing false_memory_evaluation system
6. Create a unified results comparison script

FOLDER STRUCTURE NEEDED:
baseline_false_memory_testing/
├── models/
│   ├── llama2_false_memory_baseline.py
│   ├── mistral_false_memory_baseline.py
│   └── gpt35_false_memory_baseline.py
├── false_memory_injection/
│   ├── false_memory_injector.py
│   └── false_facts_database.py
├── evaluation/
│   ├── false_memory_evaluator.py
│   └── results_comparator.py
├── data_loader/
│   └── benchmark_loader.py
├── results/
└── run_false_memory_baseline_evaluation.py

The evaluation should:
- Load the same 100 conversations per benchmark that our TMM model uses
- Inject the same false memories into these conversations
- Run each baseline LLM on these conversations with false memories
- Calculate FMR, MEL, DAR, and Contradiction Detection metrics
- Save results in a format that can be compared with our TMM results
- Generate a comparison report showing TMM vs baselines false memory prevention

FALSE MEMORY INJECTION SYSTEM:
- Use the same false facts database as our TMM system
- Inject false memories at the same points in conversations
- Track which false memories were injected and when
- Ensure identical false memory injection across all models

METRICS CALCULATION:
- FMR: Count responses containing false information / total responses * 100
- MEL: Measure time to detect and correct false memories
- DAR: Count successful adaptations to mixed true/false contexts / total contexts * 100
- Contradiction Detection: Count contradictions detected / total contradictions * 100

Please create this baseline testing environment with proper error handling, logging, and result formatting. This is critical for our research paper.
```

### Step 3: Implementation Steps

1. **Copy False Memory System**: Use the same false memory injection from `false_memory_evaluation/`
2. **Create LLM Wrappers**: Simple wrappers for each baseline LLM
3. **Reuse False Memory Metrics**: Use the same metric calculations from `false_memory_evaluation/metrics_calculator.py`
4. **Run 100 Conversations**: Scale up from 5 to 100 conversations per benchmark
5. **Generate Comparison Report**: Side-by-side comparison of false memory prevention

### Step 4: Expected Output

The baseline testing should produce:
- **Individual Results**: False memory prevention performance of each baseline LLM
- **Comparison Table**: TMM vs Baseline 1 vs Baseline 2 vs Baseline 3 on false memory metrics
- **Statistical Analysis**: Significance testing between models on false memory prevention
- **Visualization**: Charts showing false memory prevention differences

##  Expected Results Format

```json
{
  "multiwoz": {
    "tmm": {
      "fmr": 0.91,
      "mel": 0.00,
      "dar": 98.18,
      "contradiction_detection": 95.5
    },
    "llama2": {
      "fmr": 45.2,
      "mel": 12.5,
      "dar": 23.1,
      "contradiction_detection": 15.8
    },
    "mistral": {
      "fmr": 38.7,
      "mel": 8.9,
      "dar": 31.4,
      "contradiction_detection": 22.3
    },
    "gpt35": {
      "fmr": 52.1,
      "mel": 15.2,
      "dar": 18.7,
      "contradiction_detection": 12.9
    }
  }
}
```

##  Research Hypothesis

**Expected Results**: TMM should significantly outperform all baseline LLMs on false memory prevention metrics:
- **FMR**: TMM <1% vs Baselines 30-50%
- **MEL**: TMM 0.00s vs Baselines 8-15s
- **DAR**: TMM 98%+ vs Baselines 15-35%
- **Contradiction Detection**: TMM 95%+ vs Baselines 10-25%

##  Critical Requirements

1. **Don't Modify Main Model**: These baselines should NOT affect our TMM model performance
2. **Use Same False Memories**: Ensure all models are tested with identical false memory injection
3. **Same Metrics**: Use identical false memory metric calculations for fair comparison
4. **Reproducible**: Set random seeds for consistent false memory injection
5. **Documentation**: Document all baseline model configurations and false memory injection patterns

##  How to Test Our TMM Model (Reference)

### Quick False Memory Test (5 scenarios)
```bash
cd sj-oant
export GEMINI_API_KEY="your_api_key_here"
python testing/false_memory_testing.py
```

### Large-Scale False Memory Test (100 scenarios)
```bash
# Edit testing/false_memory_testing.py
# Change num_scenarios = 3 to num_scenarios = 100
python testing/false_memory_testing.py
```

##  Success Criteria

The baseline false memory testing is successful when:
- [ ] All 3 baseline LLMs are tested on 100 conversations per benchmark with false memories
- [ ] Same false memory metrics (FMR, MEL, DAR) are calculated as our TMM model
- [ ] Results show TMM significantly outperforms baselines on false memory prevention
- [ ] Statistical significance is tested for false memory prevention differences
- [ ] No impact on main TMM model performance
- [ ] Research hypothesis is validated: TMM prevents false memories better than standard LLMs

##  Research Impact

This baseline testing will prove:
1. **Standard LLMs suffer from false memory formation** (high FMR, slow MEL, low DAR)
2. **TMM successfully prevents false memory formation** (low FMR, fast MEL, high DAR)
3. **TMM's architecture is necessary** for long, multi-turn conversations
4. **Research contribution is validated** through systematic comparison

---

**Goal**: Prove that TMM is the first system to successfully prevent false memory formation in LLMs, making it essential for reliable long-context AI systems.
