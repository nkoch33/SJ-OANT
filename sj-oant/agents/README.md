# Multi-Agent System Components

This directory contains the specialized agents that form the core of the Truth-Maintained Memory Agent (TMMA) architecture.

## Agent Components

### `planner.py`
Strategic planning agent responsible for:
- Conversation flow management
- Task decomposition and planning
- Resource allocation and budget management
- Triggering retrieve → verify → refine loops when confidence is low

### `coordinator.py`
Orchestration agent that:
- Coordinates between different agents
- Manages inter-agent communication
- Handles workflow sequencing
- Ensures proper handoffs between pipeline stages

### `responder.py`
Response generation agent that:
- Produces final responses from curated memory
- Uses deterministic templates for factual lookups
- Implements grounded generation for complex queries
- Ensures responses are strictly based on verified information

### `writer_editor.py`
Memory curation agent responsible for:
- Content admission and rejection decisions
- Tier promotion and demotion (L1 → L2 → L3)
- Memory consolidation and summarization
- Quality control and content validation

### `arbiter.py`
Conflict resolution agent that:
- Handles contradiction detection and resolution
- Manages arbitration processes for contested content
- Implements weighted retention scoring
- Coordinates with micro-agents for decision making

## Architecture Integration

All agents are integrated through the main `multi_agent_pipeline.py` orchestrator using LangGraph for:
- State management across agents
- Workflow coordination
- Error handling and recovery
- Performance monitoring

## Usage

Agents are designed to work together in the TMMA pipeline:
```
User Input → Planner → TACS Filter → Truth Verifier → Writer-Editor → Responder
```

Each agent maintains its own state while contributing to the overall system goal of preventing false memory formation through proactive quality control.
