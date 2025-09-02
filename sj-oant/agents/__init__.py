"""
TMM Agents Package

This package contains the agent implementations for the Truth-Maintained Memory system.

Key Components:
- planner: Strategic planning and intent analysis
- arbiter: Context assembly and decision arbitration
- responder: Response generation and quality control
- writer_editor: Memory writing and editing operations

Example Usage:
    from agents.planner import create_strategic_planner
    from agents.arbiter import create_context_arbiter
    from agents.responder import create_responder
    
    # Create agents with dependency injection
    planner = create_strategic_planner("default")
    arbiter = create_context_arbiter("weighted_voting")
    responder = create_responder(["template_based"])
"""

from agents.planner import (
    StrategicPlanner,
    PromptRefinementAgent,
    create_strategic_planner
)

from agents.arbiter import (
    ContextArbiter,
    CandidateOutput,
    ArbitrationDecision,
    create_context_arbiter
)

from agents.responder import (
    Responder,
    ResponseCandidate,
    ResponseDecision,
    create_responder
)

# Legacy compatibility
from agents.writer_editor import (
    MemoryCurationAgent,
    WriterEditor
)

__version__ = "0.1.0"

__all__ = [
    # Main agent classes
    "StrategicPlanner",
    "ContextArbiter", 
    "Responder",
    "WriterEditor",
    
    # Sub-components
    "PromptRefinementAgent",
    "MemoryCurationAgent",
    
    # Data structures
    "CandidateOutput",
    "ArbitrationDecision",
    "ResponseCandidate", 
    "ResponseDecision",
    
    # Factory functions
    "create_strategic_planner",
    "create_context_arbiter",
    "create_responder"
]
