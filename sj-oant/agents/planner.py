"""
planner.py - Strategic Planning Agent

This module contains the Planner agent responsible for high-level strategic planning
in the Truth-Maintained Memory (TMM) system. The planner:

1. Analyzes incoming user queries and determines the appropriate response strategy
2. Coordinates the overall flow of information through the TMM pipeline
3. Decides when to trigger memory operations (retrieval, verification, curation)
4. Plans multi-step reasoning sequences when complex queries require it
5. Interfaces with the Arbiter to make final decisions about context assembly

Key responsibilities:
- Query analysis and intent detection
- Pipeline orchestration and flow control
- Strategic decision making for memory operations
- Multi-hop reasoning planning
- Context prioritization and resource allocation

The planner operates at the highest level of the agent hierarchy and ensures
that all downstream components work together effectively to maintain truth
and prevent false memory formation.
"""

from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from memory.typed_store import MemoryState


class PromptRefinementAgent:
    """
    Agent responsible for cleaning and refining user input before processing.
    
    This agent removes conversational noise, clarifies ambiguous references,
    and prepares the input for downstream processing in the TMM pipeline.
    """
    
    def __init__(self, llm):
        """
        Initialize the prompt refinement agent.
        
        Args:
            llm: Language model instance for processing
        """
        self.llm = llm
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("human", """You are the Prompt Refinement Agent. Your role is to clean and refine user input,
            removing conversational noise and ambiguity.

            Guidelines:
            - Remove filler words and conversational noise
            - Clarify ambiguous references
            - Maintain the core intent and information
            - Return only the refined input, no explanations

            Raw user input: {input}""")
        ])
    
    def refine_input(self, user_input: str) -> str:
        """
        Clean and refine user input for downstream processing.
        
        Args:
            user_input: Raw user input string
            
        Returns:
            Refined and cleaned input string
        """
        # Create the prompt with user input
        formatted_prompt = self.prompt_template.format(input=user_input)
        
        # TODO: Invoke LLM to refine the input
        # For now, return basic cleaning
        refined = user_input.strip()
        
        # Basic noise removal
        noise_words = ["um", "uh", "like", "you know", "actually"]
        words = refined.split()
        cleaned_words = [word for word in words if word.lower() not in noise_words]
        
        return " ".join(cleaned_words)
    
    def analyze_intent(self, refined_input: str) -> Dict[str, Any]:
        """
        Analyze the intent and complexity of the refined input.
        
        Args:
            refined_input: Cleaned user input
            
        Returns:
            Dict containing intent analysis and processing recommendations
        """
        intent_analysis = {
            "query_type": "information_request",  # question, command, statement, etc.
            "complexity": "simple",  # simple, complex, multi_hop
            "requires_memory": True,
            "requires_verification": False,
            "priority": "normal"
        }
        
        # TODO: Implement sophisticated intent analysis
        # For now, basic heuristics
        if "?" in refined_input:
            intent_analysis["query_type"] = "question"
        elif any(word in refined_input.lower() for word in ["remember", "save", "store"]):
            intent_analysis["query_type"] = "memory_store"
            intent_analysis["requires_verification"] = True
        elif any(word in refined_input.lower() for word in ["update", "correct", "change"]):
            intent_analysis["query_type"] = "memory_update"
            intent_analysis["requires_verification"] = True
            intent_analysis["priority"] = "high"
        
        return intent_analysis
    
    def execute(self, state: MemoryState) -> MemoryState:
        """
        Execute prompt refinement as part of the LangGraph pipeline.
        
        Args:
            state: Current memory state
            
        Returns:
            Updated memory state with refined input
        """
        print("🧠 Prompt Refinement Agent: Processing user input...")
        
        # Refine the user input
        refined_input = self.refine_input(state["user_input"])
        
        # Analyze intent for downstream planning
        intent = self.analyze_intent(refined_input)
        
        # Update state with refined input
        updated_state = state.copy()
        updated_state["user_input"] = refined_input
        
        # Store intent analysis in metadata (extend MemoryState if needed)
        # For now, just log the intent
        print(f"   Refined input: {refined_input}")
        print(f"   Detected intent: {intent}")
        
        return updated_state


class StrategicPlanner:
    """
    High-level strategic planner that coordinates the overall TMM pipeline.
    
    The strategic planner makes decisions about which components to invoke,
    how to sequence operations, and how to handle complex multi-step queries.
    """
    
    def __init__(self):
        """Initialize the strategic planner."""
        self.refinement_agent = None  # Will be set when LLM is available
    
    def set_llm(self, llm):
        """
        Set the language model for the planner and initialize sub-agents.
        
        Args:
            llm: Language model instance
        """
        self.refinement_agent = PromptRefinementAgent(llm)
    
    def plan_pipeline_execution(self, state: MemoryState) -> Dict[str, Any]:
        """
        Plan the execution strategy for processing the current state.
        
        Args:
            state: Current memory state
            
        Returns:
            Execution plan with pipeline configuration
        """
        execution_plan = {
            "skip_redundancy_check": False,
            "enable_contradiction_detection": True,
            "memory_update_strategy": "selective_addition",
            "response_priority": "accuracy_over_speed",
            "additional_verification": False
        }
        
        # TODO: Implement sophisticated planning logic based on:
        # - Input complexity
        # - Memory state
        # - Recent interaction patterns
        # - System load and performance
        
        return execution_plan
    
    def execute_refinement(self, state: MemoryState) -> MemoryState:
        """
        Execute the prompt refinement step of the pipeline.
        
        This is the entry point for the TMM pipeline, called by LangGraph.
        
        Args:
            state: Current memory state
            
        Returns:
            Updated memory state with refined input
        """
        if self.refinement_agent is None:
            print("⚠️  Warning: LLM not set for refinement agent")
            return state
            
        return self.refinement_agent.execute(state)
