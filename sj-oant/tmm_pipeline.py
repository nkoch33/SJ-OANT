"""
tmm_pipeline.py - Truth-Maintained Memory Pipeline Orchestrator

This module contains the main TMM pipeline that orchestrates all components
extracted and enhanced from the collaborator's original agent.py file.

The pipeline coordinates:
1. Prompt refinement (planning)
2. Context filtering (TACS filter)
3. Truth verification 
4. Memory curation (writer/editor)
5. Response generation

This implementation uses LangGraph to manage the multi-agent workflow
and integrates all the distributed TMM components into a cohesive system.
"""

from langgraph.graph import StateGraph, END
from typing import Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

# Import TMM components
from memory.typed_store import MemoryState, InMemoryStore
from agents.planner import StrategicPlanner
from truth.tacs_filter import TACSFilter
from truth.verifier import create_verifier
from agents.writer_editor import WriterEditor
from agents.responder import Responder


class TMMPipeline:
    """
    Main Truth-Maintained Memory pipeline that orchestrates all TMM components.
    
    This class recreates and enhances the original AgentPipeline from the
    collaborator's agent.py file, but with distributed components and
    enhanced functionality for truth maintenance.
    """
    
    def __init__(self, llm, config: Dict[str, Any] = None):
        """
        Initialize the TMM pipeline with all components.
        
        Args:
            llm: Language model instance
            config: Configuration dictionary for pipeline settings
        """
        self.llm = llm
        self.config = config or self._default_config()
        
        # Initialize memory store
        self.memory_store = InMemoryStore()
        
        # Initialize all TMM components
        self.strategic_planner = StrategicPlanner()
        self.strategic_planner.set_llm(llm)
        
        self.tacs_filter = TACSFilter(
            llm, 
            relevance_threshold=self.config.get("relevance_threshold", 0.5)
        )
        
        self.truth_verifier = create_verifier(
            verifier_type="rule_based"
        )
        
        self.writer_editor = WriterEditor(llm, self.memory_store)
        self.responder = Responder()
        
        # Build the LangGraph pipeline
        self.graph = self._build_graph()
    
    def _default_config(self) -> Dict[str, Any]:
        """
        Provide default configuration for the TMM pipeline.
        
        Returns:
            Default configuration dictionary
        """
        return {
            "l1_limit": 10,
            "l2_limit": 20, 
            "l3_limit": 100,
            "relevance_threshold": 0.5,
            "confidence_threshold": 0.8,
            "enable_redundancy_check": True,
            "enable_contradiction_detection": True,
            "enable_quality_control": True
        }
    
    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph state graph for the TMM pipeline.
        
        This recreates the graph structure from the original agent.py
        but with enhanced components and error handling.
        
        Returns:
            Compiled LangGraph StateGraph
        """
        graph = StateGraph(MemoryState)
        
        # Add nodes for each pipeline stage
        graph.add_node("refine", self._prompt_refinement_node)
        graph.add_node("filter", self._context_filtering_node)
        graph.add_node("verify", self._truth_verification_node)
        graph.add_node("curate", self._memory_curation_node)
        graph.add_node("respond", self._response_generation_node)
        
        # Define the pipeline flow (same as original but with new names)
        graph.add_edge("refine", "filter")
        graph.add_edge("filter", "verify")
        graph.add_edge("verify", "curate")
        graph.add_edge("curate", "respond")
        graph.add_edge("respond", END)
        
        # Set entry point
        graph.set_entry_point("refine")
        
        # Compile and return the graph
        return graph.compile()
    
    def _prompt_refinement_node(self, state: MemoryState) -> MemoryState:
        """
        Pipeline node for prompt refinement (planning stage).
        
        Args:
            state: Current memory state
            
        Returns:
            Updated memory state with refined input
        """
        try:
            return self.strategic_planner.execute_refinement(state)
        except Exception as e:
            print(f"Error in prompt refinement: {e}")
            return state
    
    def _context_filtering_node(self, state: MemoryState) -> MemoryState:
        """
        Pipeline node for context filtering (TACS filter stage).
        
        Args:
            state: Current memory state
            
        Returns:
            Updated memory state with filtered context
        """
        try:
            # Update state with current memory store
            updated_state = self.memory_store.get_state()
            updated_state["user_input"] = state["user_input"]
            
            return self.tacs_filter.execute(updated_state)
        except Exception as e:
            print(f"Error in context filtering: {e}")
            return state
    
    def _truth_verification_node(self, state: MemoryState) -> MemoryState:
        """
        Pipeline node for truth verification.
        
        Args:
            state: Current memory state
            
        Returns:
            Updated memory state with verification results
        """
        try:
            return self.truth_verifier.execute(state)
        except Exception as e:
            print(f"Error in truth verification: {e}")
            return state
    
    def _memory_curation_node(self, state: MemoryState) -> MemoryState:
        """
        Pipeline node for memory curation (writer/editor stage).
        
        Args:
            state: Current memory state
            
        Returns:
            Updated memory state after memory operations
        """
        try:
            # TODO: Pass verification results from previous stage
            verification_result = {
                "confidence": 0.7,
                "truth_score": 0.7,
                "contradiction_detected": False
            }
            return self.writer_editor.execute(state, verification_result)
        except Exception as e:
            print(f"Error in memory curation: {e}")
            return state
    
    def _response_generation_node(self, state: MemoryState) -> MemoryState:
        """
        Pipeline node for response generation with memory retrieval.
        
        Args:
            state: Current memory state
            
        Returns:
            Final memory state with LLM-generated response based on retrieved memory
        """
        try:
            # Get stored memory from memory store
            stored_memory = self.memory_store.search("", limit=10)  # Get all recent memory
            
            # Convert memory records to context strings
            memory_context = []
            for record in stored_memory:
                memory_context.append(record.payload)
            
            # Use LLM with retrieved memory context  
            from langchain_core.prompts import ChatPromptTemplate
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant. Use the provided context to answer the user's question. Answer directly and concisely."),
                ("user", "Context: {context}\n\nQuestion: {question}")
            ])
            
            chain = prompt | self.llm
            context_text = "\n".join(memory_context) if memory_context else "No context available"
            
            print(f"🔍 Context being used: {context_text[:200]}...")
            print(f"❓ Question: {state.get('user_input', '')}")
            
            try:
                result = chain.invoke({
                    "context": context_text,
                    "question": state.get("user_input", "")
                })
                response = result.content if hasattr(result, 'content') else str(result)
                print(f"🤖 LLM Response: {response}")
            except Exception as llm_error:
                print(f"❌ LLM Error: {llm_error}")
                response = f"LLM Error: {str(llm_error)}"
            
            state["final_response"] = response
            print(f"💬 Generated LLM response based on {len(memory_context)} memory records")
            return state
        except Exception as e:
            print(f"Error in response generation: {e}")
            # Fallback to simple response
            state["final_response"] = "I apologize, but I encountered an error processing your request."
            return state
    
    def process(self, user_input: str) -> str:
        """
        Process a user input through the complete TMM pipeline.
        
        This is the main interface for the FictionalQA evaluation.
        
        Args:
            user_input: Raw user input string
            
        Returns:
            Final response string
        """
        print("=" * 60)
        print("🧠 Truth-Maintained Memory Pipeline")
        print("=" * 60)
        
        # Create initial state
        initial_state = {
            "user_input": user_input,
            "L1": [],
            "L2": [],
            "L3": [],
            "flagged": []
        }
        
        print(f"Processing input: {user_input}")
        print(f"Initial memory state: L1={len(initial_state['L1'])}, "
              f"L2={len(initial_state['L2'])}, L3={len(initial_state['L3'])}, "
              f"Flagged={len(initial_state['flagged'])}")
        
        # Execute the pipeline
        try:
            final_state = self.graph.invoke(initial_state)
            
            print("=" * 60)
            print("✅ Pipeline completed successfully!")
            print("=" * 60)
            
            # Return the final response
            return final_state.get("final_response", "No response generated")
            
        except Exception as e:
            print(f"❌ Pipeline error: {e}")
            # For evaluation, return an error response that can be evaluated
            return f"Error: {str(e)}"
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """
        Get current memory state summary.
        
        Returns:
            Memory summary with tier sizes and statistics
        """
        try:
            return self.memory_store.get_metrics()
        except AttributeError:
            return {"total_records": 0, "tier_sizes": {"L1": 0, "L2": 0, "L3": 0, "flagged": 0}}
    
    def reset_memory(self) -> None:
        """
        Reset the memory store to empty state.
        """
        self.memory_store = InMemoryStore()
        # Reinitialize writer_editor with new memory store
        self.writer_editor = WriterEditor(self.llm, self.memory_store)
        
        print("🔄 Memory store reset to empty state")


def create_tmm_pipeline(google_api_key: str = None, config: Dict[str, Any] = None) -> TMMPipeline:
    """
    Factory function to create a TMM pipeline with Google Gemini LLM.
    
    This recreates the setup from the original agent.py main() function.
    
    Args:
        google_api_key: Google API key (if not provided, will load from env)
        config: Pipeline configuration dictionary
        
    Returns:
        Initialized TMMPipeline instance
    """
    # Load environment variables
    load_dotenv()
    
    # Get API key from parameter or environment
    api_key = google_api_key or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Google API key must be provided or set in GOOGLE_API_KEY environment variable")
    
    # Initialize LLM with current model
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.1,
        google_api_key=api_key
    )
    
    # Create and return pipeline
    return TMMPipeline(llm, config)


def main():
    """
    Main function that recreates the original agent.py demonstration.
    
    This provides the same interface as the original agent.py for testing
    and demonstration purposes.
    """
    try:
        # Create the TMM pipeline
        pipeline = create_tmm_pipeline()
        
        # Example test case (matching original agent.py)
        test_input = "Barack Obama was born in Hawaii."
        
        print("Starting TMM pipeline demonstration...")
        result = pipeline.process(test_input)
        
        if result["success"]:
            print("\n🎉 Demonstration completed successfully!")
            print(f"Memory summary: {result['memory_summary']}")
        else:
            print(f"\n❌ Demonstration failed: {result['error']}")
            
    except Exception as e:
        print(f"❌ Failed to initialize TMM pipeline: {e}")
        print("Make sure GOOGLE_API_KEY is set in your environment")


if __name__ == "__main__":
    main()
