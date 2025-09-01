# FABLE Scenarios Directory

This directory contains the scenario files for the FABLE (False-memory-Aware Benchmark for Long-term Engagements) benchmark.

## File Format

Scenarios are stored as JSONL files with the following structure:

```json
{
  "scenario_id": "scenario_001",
  "difficulty": "medium",
  "domain": "personal_assistant",
  "turns": 45,
  "scenario_kb": "kb/scenario_001.json",
  "conversation": [
    {
      "turn": 1,
      "speaker": "user",
      "text": "...",
      "facts_referenced": ["fact_1", "fact_2"],
      "planted_misinformation": null,
      "is_correction": false
    },
    {
      "turn": 2,
      "speaker": "assistant",
      "text": "...",
      "should_contain_facts": ["fact_1"],
      "should_not_contain": ["misinformation_1"]
    }
  ]
}
```

## Directory Structure

- `easy/` - Single correction, low distractors (50 scenarios)
- `medium/` - Multiple corrections, intertwined entities (100 scenarios)  
- `hard/` - Overlapping entities, delayed corrections, adversarial distractors (50 scenarios)
- `forensics/` - Suggestibility and false-memory pressure scenarios (25 scenarios)

## Generation

Scenarios are generated using templates with human quality control passes to ensure realistic and challenging evaluation conditions.
