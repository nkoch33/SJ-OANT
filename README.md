# Truth-Maintained Memory Agent (TMMA)

A multi-agent architecture for preventing false memory formation in Large Language Models through proactive write-time quality control and hierarchical memory management.

## Abstract

Large Language Models (LLMs) are prone to false memory formation during long, multi-turn interactions, incorporating incorrect, irrelevant, or contradictory information. Traditional methods such as enlarging context windows, summarizing memory, or selective retrieval, are often computationally expensive and reactive, which allows errors to accumulate. We propose the Truth-Maintained Memory Agent (TMMA), a proactive multi-agent framework that enforces write-time quality control through a four-tier hierarchical memory system and token-level adaptive context screening.

## Architecture

### Multi-Agent Pipeline

TMMA implements a five-stage control stack:

```
User Input → Strategic Planner → TACS Filter → Truth Verifier → Memory Curator → Responder
```

### Core Components

#### Multi-Agent Orchestration
- **`multi_agent_pipeline.py`** - Main orchestrator using LangGraph
- **`agents/`** - Specialized agents (Planner, Coordinator, Responder, Writer-Editor, Arbiter)

#### Memory System
- **`memory/`** - Four-tier hierarchical storage:
  - **L1 (Working)**: Recent, active information with relaxed admission criteria
  - **L2 (Summarized)**: Condensed, important information with normalized entities
  - **L3 (Archival)**: Long-term, verified facts with high confidence requirements
  - **FLAGGED**: Contradicted or low-confidence items in quarantine

#### Truth Verification
- **`truth/`** - Verification and filtering systems:
  - **TACS Filter**: Token-level Adaptive Context Screening
  - **Enhanced Verifier**: Advanced contradiction detection
  - **False Memory Gate**: Proactive identification of false information

#### Core Infrastructure
- **`core/`** - Protocol interfaces and data structures
- **`monitoring/`** - Research analytics and performance tracking

## Evaluation Framework

### Dual-Level Assessment

**Level 1: Dialogue Performance**
- MultiWOZ 2.4: Multi-domain task-oriented dialogue
- Schema-Guided Dialogue (SGD): Service-oriented conversations
- Taskmaster: Realistic conversational interactions

**Level 2: False Memory Prevention**
- Controlled false memory injection tests
- Contradiction detection and resolution
- Memory resilience under adversarial conditions

### Metrics

**Dialogue Quality**: BLEU, ROUGE, semantic similarity, slot extraction F1, intent accuracy

**Memory Robustness**: False Memory Rate (FMR), Memory Edit Latency (MEL), Disturbance Adaptation Rate (DAR), Contradiction Detection Rate (CDR)

## Installation

```bash
git clone https://github.com/nkoch33/SJ-OANT.git
cd SJ-OANT
pip install -r sj-oant/requirements.txt
export GEMINI_API_KEY="your_api_key_here"
```

## Usage

### Basic Evaluation
```bash
cd sj-oant
python testing/official_evaluation.py
```

### False Memory Testing
```bash
cd sj-oant
python testing/false_memory_testing.py
```

## Repository Structure

```
SJ-OANT/
├── sj-oant/                          # Main codebase
│   ├── agents/                       # Multi-agent system components (Planner, Coordinator, Responder, etc.)
│   ├── core/                         # Protocol interfaces and data structures
│   ├── memory/                       # Four-tier hierarchical memory system (L1, L2, L3, FLAGGED)
│   ├── truth/                        # Truth verification and TACS filtering
│   ├── evaluation_frameworks/        # Benchmark evaluation (MultiWOZ, SGD, Taskmaster)
│   ├── false_memory_evaluation/      # False memory prevention evaluation and testing
│   ├── testing/                      # Comprehensive testing framework
│   ├── data/                         # Benchmark datasets (MultiWOZ, SGD, Taskmaster, MultiDoGO)
│   ├── monitoring/                   # Research analytics and performance monitoring
│   ├── multi_agent_pipeline.py       # Main orchestrator using LangGraph
│   ├── tmm_pipeline.py              # TMMA pipeline implementation
│   └── requirements.txt              # Python dependencies
├── docs/                             # Comprehensive documentation
└── README.md                         # This file
```

## Documentation

### Central Documentation (`docs/`)
- **Methodology**: `docs/METHODOLOGY.md` - Detailed research methodology
- **Evaluation**: `docs/EVALUATION.md` - Evaluation framework overview
- **Results**: `docs/RESULTS.md` - Experimental results and analysis
- **Discussion**: `docs/DISCUSSION.md` - Research discussion and insights

### Component Documentation
Each major component includes comprehensive documentation:
- **`sj-oant/agents/README.md`** - Multi-agent system components
- **`sj-oant/memory/README.md`** - Hierarchical memory system
- **`sj-oant/truth/README.md`** - Truth verification and filtering
- **`sj-oant/evaluation_frameworks/README.md`** - Benchmark evaluation
- **`sj-oant/false_memory_evaluation/README.md`** - False memory prevention
- **`sj-oant/testing/README.md`** - Testing framework
- **`sj-oant/data/README.md`** - Benchmark datasets
- **`sj-oant/core/README.md`** - Core infrastructure
- **`sj-oant/monitoring/README.md`** - Research analytics

## Citation

If you use this work in your research, please cite:

```bibtex
@inproceedings{tmma2024,
  title={Truth-Maintained Memory Agent: Preventing False Memory Formation in Large Language Models},
  author={[Authors]},
  booktitle={Proceedings of AACL},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

We thank the creators of MultiWOZ, SGD, and Taskmaster datasets for providing evaluation benchmarks. We acknowledge the contributions of the LangChain and Hugging Face communities for open-source frameworks.

---

**Status**: Research complete, submitted to AACL 2024