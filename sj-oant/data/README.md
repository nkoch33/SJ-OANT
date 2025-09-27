# Benchmark Datasets

This directory contains the benchmark datasets used for evaluating TMMA across dialogue performance and false memory prevention tasks.

## Datasets

### MultiWOZ 2.4
**Location**: `MULTIWOZ2.4/`
**Description**: Large-scale multi-domain task-oriented dialogue corpus
**Size**: 10,438 dialogues across 7 domains
**Domains**: Hotel, restaurant, taxi, train, attraction, hospital, police
**Features**:
- Multi-turn goal-driven conversations
- Detailed slot annotations
- Database queries and responses
- Domain-specific ontologies

**Files**:
- `data.json`: Main dialogue data with train/dev/test splits
- `dialogue_acts.json`: Dialogue act annotations
- `ontology.json`: Domain ontologies and slot definitions
- `*_db.json`: Domain-specific databases
- `*ListFile.json`: Train/validation/test split definitions

**Citation**: Budzianowski et al., "MultiWOZ - A Large-Scale Multi-Domain Wizard-of-Oz Dataset for Task-Oriented Dialogue Modelling", EMNLP 2018

### Schema-Guided Dialogue (SGD)
**Location**: `sgd/`
**Description**: Schema-driven dialogue dataset from DSTC8
**Size**: 22,825 dialogues across 20+ services
**Features**:
- Multi-service, multi-intent conversations
- Structured schemas for each service
- API call annotations
- Intent and slot annotations

**Files**:
- `dialogues_001.json`: Dialogue data with annotations
- `schema.json`: Service schemas and API definitions

**Citation**: Rastogi et al., "Schema-Guided Dialogue Dataset", DSTC8 2019

### Taskmaster
**Location**: `taskmaster/`
**Description**: Realistic conversational dataset from Google Research
**Size**: 13,215 dialogues across 6 domains
**Features**:
- Human-human and human-assistant dialogue styles
- Natural conversation flows
- Task-oriented interactions
- Domain-specific ontologies

**Files**:
- `restaurant-search.json`: Restaurant booking dialogues
- `sample.json`: Sample dialogue data
- `ontology.json`: Domain ontologies

**Citation**: Byrne et al., "Taskmaster-1: Toward a Realistic and Diverse Dialog Dataset", EMNLP 2019

### MultiDoGO
**Location**: `multidogo/`
**Description**: Multi-domain goal-oriented dialogue dataset
**Features**:
- Annotated goal-oriented conversations
- Multi-domain coverage
- Structured dialogue annotations

**Files**:
- `airline.tsv`: Airline booking dialogues
- `airline_annotated.tsv`: Annotated airline dialogues
- `fastfood.tsv`: Fast food ordering dialogues

## Data Usage

### Loading Datasets
```python
from sj_oant.data_loader import benchmark_loader

# Load MultiWOZ dataset
multiwoz_data = benchmark_loader.load_multiwoz()

# Load SGD dataset
sgd_data = benchmark_loader.load_sgd()

# Load Taskmaster dataset
taskmaster_data = benchmark_loader.load_taskmaster()
```

### Data Preprocessing
- **Train/Dev/Test Splits**: Preserved from original datasets
- **Dialogue Formatting**: Standardized across all benchmarks
- **Annotation Processing**: Unified format for slots, intents, and dialogue acts
- **Domain Mapping**: Consistent domain representation

### Evaluation Protocol
- **Test Set Usage**: 100 randomly selected conversations per benchmark
- **Random Seed**: Fixed at 42 for reproducibility
- **Domain Balance**: Maintained across evaluation subsets
- **Dialogue Diversity**: Preserved conversation variety

## Dataset Statistics

### MultiWOZ 2.4
- **Total Dialogues**: 10,438
- **Domains**: 7 (hotel, restaurant, taxi, train, attraction, hospital, police)
- **Average Turns**: 13.5
- **Total Slots**: 24

### Schema-Guided Dialogue
- **Total Dialogues**: 22,825
- **Services**: 20+
- **Intents**: 100+
- **Slots**: 200+

### Taskmaster
- **Total Dialogues**: 13,215
- **Domains**: 6
- **Average Turns**: 8.2
- **Dialogue Styles**: Human-human, human-assistant

## License Information

- **MultiWOZ 2.4**: MIT License
- **Schema-Guided Dialogue**: Apache 2.0 License
- **Taskmaster**: Apache 2.0 License
- **MultiDoGO**: Available for research use

## Data Integrity

All datasets are used in their original form with:
- **No Modifications**: Original data preserved
- **Proper Attribution**: Citations maintained
- **License Compliance**: Usage within license terms
- **Research Purpose**: Academic and research use only

## Updates and Maintenance

- **Version Control**: Dataset versions tracked
- **Integrity Checks**: Regular validation of data completeness
- **Documentation**: Comprehensive usage documentation
- **Support**: Contact information for dataset-related issues
