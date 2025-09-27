# Core Infrastructure

This directory contains the fundamental data structures, interfaces, and type definitions that form the foundation of the TMMA architecture.

## Core Components

### `types.py`
Core data type definitions including:
- **Memory Records**: Immutable, typed memory objects with rich metadata
- **Memory Tiers**: L1, L2, L3, and FLAGGED tier definitions
- **Confidence Scores**: Truth scores, evidentiality measures, and calibration labels
- **Verification Results**: Contradiction flags and risk assessments
- **Agent States**: State management for multi-agent coordination

### `ports.py`
Protocol interfaces and abstractions:
- **Memory Interface**: Standardized memory operations (store, retrieve, update)
- **Verification Interface**: Truth verification and contradiction detection
- **Agent Interface**: Common agent communication protocols
- **Evaluation Interface**: Standardized evaluation and metrics computation

## Design Principles

### Type Safety
- **Immutable Records**: All memory records are immutable with rich metadata
- **Strong Typing**: Clear type definitions for all system components
- **Validation**: Built-in validation for all data structures
- **Consistency**: Unified type system across all modules

### Interface Segregation
- **Clear Contracts**: Well-defined interfaces for all system components
- **Loose Coupling**: Components interact through standardized interfaces
- **Extensibility**: Easy addition of new implementations
- **Testability**: Mockable interfaces for comprehensive testing

### Data Integrity
- **Audit Trails**: Complete logging of all memory operations
- **Version Control**: Tracked changes and state transitions
- **Consistency Checks**: Built-in validation and integrity verification
- **Recovery Mechanisms**: Error handling and state recovery

## Usage

### Memory Operations
```python
from sj_oant.core.types import MemoryRecord, MemoryTier
from sj_oant.core.ports import MemoryInterface

# Create memory record
record = MemoryRecord(
    content="Example content",
    confidence=0.95,
    tier=MemoryTier.L3,
    metadata={"source": "user_input", "timestamp": "2024-01-01"}
)

# Store in memory
memory_interface.store(record)
```

### Verification Interface
```python
from sj_oant.core.ports import VerificationInterface

# Verify content
verification_result = verification_interface.verify(content)
if verification_result.truth_score > 0.8:
    # High confidence content
    pass
```

## Integration

The core infrastructure integrates with:
- **Memory System**: Provides type definitions and interfaces
- **Truth Verification**: Defines verification protocols and results
- **Multi-Agent System**: Manages agent state and communication
- **Evaluation Framework**: Standardizes evaluation interfaces

## Extensibility

### Adding New Types
```python
from sj_oant.core.types import BaseMemoryRecord

class CustomMemoryRecord(BaseMemoryRecord):
    def __init__(self, custom_field, **kwargs):
        super().__init__(**kwargs)
        self.custom_field = custom_field
```

### Implementing Interfaces
```python
from sj_oant.core.ports import MemoryInterface

class CustomMemoryImplementation(MemoryInterface):
    def store(self, record):
        # Custom storage implementation
        pass
    
    def retrieve(self, query):
        # Custom retrieval implementation
        pass
```

## Best Practices

### Type Design
- **Immutability**: Prefer immutable data structures
- **Rich Metadata**: Include comprehensive metadata for all records
- **Validation**: Built-in validation for all data types
- **Documentation**: Clear documentation for all type definitions

### Interface Design
- **Single Responsibility**: Each interface has a clear, focused purpose
- **Consistent Naming**: Standardized naming conventions across interfaces
- **Error Handling**: Comprehensive error handling and recovery
- **Performance**: Efficient implementations for all operations

## Testing

### Unit Testing
- **Type Validation**: Test all type definitions and constraints
- **Interface Compliance**: Verify all implementations meet interface contracts
- **Edge Cases**: Test boundary conditions and error scenarios
- **Performance**: Validate performance characteristics

### Integration Testing
- **Cross-Component**: Test interactions between core components
- **State Management**: Verify state consistency across operations
- **Error Recovery**: Test error handling and recovery mechanisms
- **End-to-End**: Complete system integration testing
