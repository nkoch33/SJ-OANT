# FABLE Knowledge Base Directory

This directory contains the ground-truth knowledge bases for FABLE scenarios.

## File Format

Knowledge bases are stored as JSON files with the following structure:

```json
{
  "scenario_id": "scenario_001",
  "domain": "personal_assistant", 
  "entities": {
    "person_1": {
      "name": "Alice Johnson",
      "attributes": {
        "age": 32,
        "occupation": "Software Engineer",
        "location": "San Francisco"
      }
    }
  },
  "facts": {
    "fact_1": {
      "statement": "Alice works at TechCorp",
      "confidence": 1.0,
      "source": "profile",
      "timestamp": "2024-01-01T00:00:00Z"
    }
  },
  "relationships": {
    "rel_1": {
      "subject": "person_1",
      "predicate": "works_at", 
      "object": "company_1",
      "confidence": 1.0
    }
  },
  "temporal_events": {
    "event_1": {
      "description": "Alice started working at TechCorp",
      "timestamp": "2023-06-01T00:00:00Z",
      "participants": ["person_1", "company_1"]
    }
  },
  "contradictions": {
    "contradiction_1": {
      "original_fact": "fact_1",
      "contradicting_statement": "Alice works at StartupXYZ",
      "introduced_at_turn": 15,
      "corrected_at_turn": 23
    }
  }
}
```

## Purpose

These knowledge bases serve as the ground truth for:
- Fact verification during evaluation
- Contradiction detection testing  
- Truth maintenance assessment
- False memory propagation measurement

Each KB is scenario-specific and contains all factual information needed to evaluate system performance without relying on external knowledge sources.
