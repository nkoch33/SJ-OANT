# Hierarchical Memory System

This directory contains the four-tier hierarchical memory system that forms the core storage architecture of TMMA.

## Memory Architecture

### Four-Tier System

#### L1 (Working Memory)
- **Purpose**: Short-term buffer for recent turns and momentary notes
- **Characteristics**: Rapid access, relaxed admission criteria, faster expiry
- **Capacity**: 100 records
- **Use Case**: Active conversation context and temporary information

#### L2 (Summarized Memory)
- **Purpose**: Condensed, important information with normalized entities
- **Characteristics**: Abstractive summaries, canonical tuples, stable representations
- **Capacity**: 500 records
- **Use Case**: Compressed volatile details into reusable representations

#### L3 (Archival Memory)
- **Purpose**: Authoritative repository for long-term, verified facts
- **Characteristics**: High confidence requirements, evidential support
- **Capacity**: 1000 records
- **Use Case**: Permanent storage of verified, high-quality information

#### FLAGGED (Quarantine)
- **Purpose**: Isolation zone for contradicted or uncertain content
- **Characteristics**: Preserved for transparency, suppressed confidence
- **Capacity**: 200 records
- **Use Case**: Audit trails and potential future reinstatement

## Core Components

### `typed_store.py`
Main memory store implementation featuring:
- Immutable, typed memory records with rich metadata
- Tier-aware indices for efficient operations
- Capacity management with LRU eviction
- Confidence-based routing and promotion
- Contradiction detection and quarantine logic

### `adaptive_retrieval.py`
Intelligent retrieval system providing:
- Composite relevance scoring combining tier signals
- Verification status and confidence weighting
- Semantic similarity with credibility bias
- FLAGGED content exclusion (unless explicitly requested)
- Maintenance routines and audit logging

## Key Features

### Write-Time Quality Control
- Content evaluation before storage commitment
- Confidence scoring across multiple dimensions
- Automatic routing based on quality thresholds
- False memory detection and quarantine

### Retrieval-Time Filtering
- Credibility-weighted candidate ranking
- Tier-aware relevance scoring
- Contradiction-aware context assembly
- Transparent audit trails for all operations

### Maintenance Operations
- Capacity enforcement with LRU eviction
- Soft deletion with rationale logging
- Tier movement tracking and cross-linking
- Background contradiction detection sweeps

## Usage

The memory system is designed to work seamlessly with the TMMA pipeline:

1. **Input Processing**: Content enters through the Truth Verifier
2. **Quality Assessment**: Confidence scoring and contradiction detection
3. **Tier Routing**: Automatic placement based on quality thresholds
4. **Retrieval**: Adaptive retrieval with credibility weighting
5. **Maintenance**: Ongoing cleanup and optimization

## Configuration

Memory tiers can be configured through:
- Capacity limits per tier
- Confidence thresholds for promotion
- Eviction policies and strategies
- Retrieval scoring weights
- Maintenance schedule and parameters
