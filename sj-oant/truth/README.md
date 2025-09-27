# Truth Verification and Filtering System

This directory contains the truth verification and filtering components that implement proactive quality control in TMMA.

## Core Components

### `tacs_filter.py` - Typed Adaptive Context Selector
Token-level filtering system that:
- **Lexical Filtering**: Removes distractors using lexical similarity
- **Embedding Filtering**: Semantic similarity-based content selection
- **Recency Weighting**: Prioritizes recent, relevant information
- **Utility Learning**: Adapts to learned content utility patterns
- **Slot Extraction**: Entity and slot-aware context assembly
- **Output**: Compact, focused context windows for downstream processing

### `verifier.py` - Truth Verification Engine
Core verification system providing:
- **Truth Scoring**: Assigns confidence scores s_truth ∈ [0,1]
- **Contradiction Detection**: Identifies conflicting information
- **Evidentiality Assessment**: Measures supporting evidence quality
- **Calibration Labeling**: Provides reliability indicators
- **Model-Agnostic Interface**: Supports rules, LLMs, or hybrid approaches

### `enhanced_verifier.py` - Advanced Verification
Extended verification capabilities including:
- **Pattern Recognition**: Identifies risky phrasing and uncertainty markers
- **Semantic Analysis**: Deep contradiction detection via embeddings
- **Temporal Consistency**: Checks for time-based conflicts
- **Logical Validation**: Ensures logical consistency across statements
- **Multi-Modal Verification**: Combines multiple verification signals

## False Memory Gate

### Proactive Detection System
The False Memory Gate operates prior to memory commitment and includes:

#### Layer 1: Dictionary Matching
- **Known False Facts**: High-precision rejection of verified false information
- **Pattern Database**: Curated collection of common falsehoods
- **Weight**: ~0.95 (highest confidence in rejections)

#### Layer 2: Pattern Detection
- **Risk Indicators**: Identifies hedging, uncertainty markers, adversarial formulations
- **Historical Correlation**: Patterns that correlate with low reliability
- **Weight**: 0.7-0.8 (moderate confidence signals)
- **Action**: Routes content with stricter scrutiny rather than outright rejection

#### Layer 3: Semantic Contradiction
- **Embedding Retrieval**: Semantic similarity-based conflict detection
- **Tuple Extraction**: Identifies entities, dates, numbers for direct comparison
- **Temporal Logic**: Checks for time-based inconsistencies
- **Weight**: Scales with similarity and evidence strength

### Risk Scoring and Routing
- **Fused Score**: Combines all layer outputs into single risk assessment
- **Threshold Routing**: Content exceeding risk threshold → FLAGGED tier
- **Confidence Suppression**: FLAGGED content receives confidence = 0.05
- **Rationale Logging**: Full audit trail of detection decisions

## Integration with Memory System

### Write-Time Control
- **Pre-Commitment Screening**: All content evaluated before storage
- **Tier Routing**: Risk scores inform memory tier placement
- **Quality Assurance**: Only verified content enters active memory

### Retrieval-Time Filtering
- **FLAGGED Exclusion**: Quarantined content excluded from retrieval
- **Credibility Weighting**: Verification status influences retrieval ranking
- **Conflict Resolution**: Explicit handling of contradictory information

## Configuration

### Verification Parameters
- Risk thresholds for tier routing
- Confidence scoring weights
- Pattern detection sensitivity
- Semantic similarity thresholds

### False Memory Detection
- Dictionary update frequency
- Pattern learning parameters
- Contradiction detection sensitivity
- Audit trail retention policies

## Usage

The truth verification system integrates seamlessly with the TMMA pipeline:

1. **Input Processing**: Content enters through TACS filter
2. **Token-Level Screening**: Distractor removal and focus enhancement
3. **Truth Verification**: Confidence scoring and contradiction detection
4. **Risk Assessment**: False memory gate evaluation
5. **Memory Routing**: Tier placement based on quality and risk scores
6. **Retrieval Filtering**: Credibility-aware context assembly

## Extensibility

The system is designed for extensibility:
- **Plugin Architecture**: Easy addition of new verification methods
- **Model Agnostic**: Supports various verification backends
- **Configurable Weights**: Tunable scoring and routing parameters
- **Audit Framework**: Comprehensive logging for system transparency
