"""
responder.py - Response Generation Agent

This module contains the Responder agent responsible for generating final responses
to user queries in the Truth-Maintained Memory (TMM) system.

The responder:

1. Takes the curated context from the Arbiter
2. Generates natural language responses using the verified information
3. Ensures responses are coherent, helpful, and grounded in truth
4. Handles cases where insufficient information is available
5. Formats responses appropriately for the user interface

Key responsibilities:
- Natural language response generation
- Context grounding and factual accuracy
- Response quality control and coherence
- Handling of incomplete information gracefully
- User-facing communication and formatting

The responder is the final component in the TMM pipeline and the only one
that directly interacts with users. It must balance being helpful while
maintaining strict adherence to verified information only.
"""

from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from memory.typed_store import MemoryState


class ResponseGenerator:
    """
    Main response generation component that creates natural language responses
    using curated memory context from the TMM pipeline.
    
    This component implements the original LLM generation logic from the
    collaborator's agent.py file, enhanced with additional safety and
    grounding mechanisms.
    """
    
    def __init__(self, llm):
        """
        Initialize the response generator.
        
        Args:
            llm: Language model instance for response generation
        """
        self.llm = llm
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("human", """You are the main LLM. Generate responses using the curated memory context.
            Prioritize recent information from L1 and L2, and ground facts in L3.

            L1 Cache (Recent): {l1_cache}
            L2 Cache (Summarized): {l2_cache}
            L3 Cache (Facts): {l3_cache}

            Generate a helpful, accurate response based on this curated context.

            User input: {user_input}""")
        ])
    
    def prepare_context(self, state: MemoryState) -> Dict[str, str]:
        """
        Prepare and format memory context for response generation.
        
        Args:
            state: Current memory state
            
        Returns:
            Dict with formatted context strings for each memory tier
        """
        context = {
            "l1_cache": "\n".join(state["L1"]) if state["L1"] else "No recent information",
            "l2_cache": "\n".join(state["L2"]) if state["L2"] else "No summarized information", 
            "l3_cache": "\n".join(state["L3"]) if state["L3"] else "No established facts",
            "user_input": state["user_input"]
        }
        
        return context
    
    def generate_response(self, state: MemoryState) -> str:
        """
        Generate a response using the curated memory context.
        
        Args:
            state: Current memory state with curated context
            
        Returns:
            Generated response string
        """
        # Prepare context for the prompt
        context = self.prepare_context(state)
        
        # TODO: Invoke LLM to generate response
        # For now, create a simple template-based response
        
        response_parts = []
        
        # Acknowledge the user input
        response_parts.append(f"I understand you said: {context['user_input']}")
        
        # Include relevant information from memory tiers
        if state["L3"]:
            response_parts.append(f"Based on established facts: {'; '.join(state['L3'][:2])}")
        
        if state["L2"]:
            response_parts.append(f"Recent context suggests: {'; '.join(state['L2'][:2])}")
        
        if state["L1"]:
            response_parts.append(f"From our recent conversation: {'; '.join(state['L1'][:1])}")
        
        # Default helpful response
        if not any([state["L1"], state["L2"], state["L3"]]):
            response_parts.append("I don't have specific information about this topic in my memory yet.")
        
        return " ".join(response_parts)
    
    def validate_response(self, response: str, state: MemoryState) -> Dict[str, Any]:
        """
        Validate the generated response for safety and grounding.
        
        Args:
            response: Generated response
            state: Memory state used for generation
            
        Returns:
            Validation results with safety and grounding scores
        """
        validation = {
            "is_safe": True,
            "is_grounded": True,
            "confidence": 0.8,
            "issues": []
        }
        
        # Check for potential hallucination (content not in memory)
        response_lower = response.lower()
        memory_content = " ".join(state["L1"] + state["L2"] + state["L3"]).lower()
        
        # Simple grounding check - response should relate to memory content
        if memory_content and not any(word in memory_content for word in response_lower.split()[-10:]):
            validation["is_grounded"] = False
            validation["issues"].append("Response may not be grounded in memory")
            validation["confidence"] *= 0.7
        
        # Check for harmful content patterns (basic safety check)
        harmful_patterns = ["ignore previous", "forget everything", "override"]
        for pattern in harmful_patterns:
            if pattern in response_lower:
                validation["is_safe"] = False
                validation["issues"].append(f"Potential harmful pattern: {pattern}")
        
        return validation
    
    def execute(self, state: MemoryState) -> Dict[str, Any]:
        """
        Execute response generation as part of the TMM pipeline.
        
        Args:
            state: Current memory state
            
        Returns:
            Dict with generated response and validation results
        """
        print("💬 Response Generator: Creating response...")
        
        # Generate the response
        response = self.generate_response(state)
        
        # Validate the response
        validation = self.validate_response(response, state)
        
        print(f"   Generated response: {response[:100]}...")
        print(f"   Validation: Safe={validation['is_safe']}, "
              f"Grounded={validation['is_grounded']}, "
              f"Confidence={validation['confidence']:.2f}")
        
        if validation["issues"]:
            print(f"   ⚠️  Issues found: {validation['issues']}")
        
        return {
            "response": response,
            "validation": validation,
            "context_used": {
                "l1_items": len(state["L1"]),
                "l2_items": len(state["L2"]),
                "l3_items": len(state["L3"])
            }
        }


class ResponseQualityController:
    """
    Quality control component that ensures response safety and accuracy.
    
    This component provides additional checks and fallbacks to ensure
    that generated responses meet TMM system standards for truthfulness
    and safety.
    """
    
    def __init__(self, min_confidence: float = 0.6):
        """
        Initialize the quality controller.
        
        Args:
            min_confidence: Minimum confidence threshold for responses
        """
        self.min_confidence = min_confidence
    
    def should_fallback(self, validation: Dict[str, Any]) -> bool:
        """
        Determine if response should use a fallback instead.
        
        Args:
            validation: Response validation results
            
        Returns:
            True if fallback should be used, False otherwise
        """
        return (
            not validation["is_safe"] or
            not validation["is_grounded"] or
            validation["confidence"] < self.min_confidence
        )
    
    def generate_fallback_response(self, state: MemoryState, issues: List[str]) -> str:
        """
        Generate a safe fallback response when quality checks fail.
        
        Args:
            state: Current memory state
            issues: List of validation issues
            
        Returns:
            Safe fallback response
        """
        fallback_templates = [
            "I don't have enough reliable information to answer that question fully.",
            "Let me be careful about my response since I want to ensure accuracy.",
            "I'd like to provide accurate information, but I need to verify some details first.",
            "I'm not confident in my knowledge about this topic right now."
        ]
        
        # Select fallback based on issues
        if any("grounded" in issue for issue in issues):
            return "I don't have enough information in my memory to provide a reliable answer to that question."
        elif any("harmful" in issue for issue in issues):
            return "I can't provide that type of response. Let me help you with something else."
        else:
            return fallback_templates[0]


class TMMResponder:
    """
    Main responder agent that coordinates response generation with quality control.
    
    This agent combines the response generator with quality control mechanisms
    to ensure safe, accurate, and grounded responses from the TMM system.
    """
    
    def __init__(self, llm):
        """
        Initialize the TMM responder with all components.
        
        Args:
            llm: Language model instance
        """
        self.response_generator = ResponseGenerator(llm)
        self.quality_controller = ResponseQualityController()
    
    def execute(self, state: MemoryState) -> MemoryState:
        """
        Execute the complete response generation pipeline.
        
        Args:
            state: Current memory state
            
        Returns:
            Updated memory state (unchanged, as this is final step)
        """
        print("🎯 TMM Responder: Generating final response...")
        
        # Generate response with validation
        generation_result = self.response_generator.execute(state)
        
        response = generation_result["response"]
        validation = generation_result["validation"]
        
        # Apply quality control
        if self.quality_controller.should_fallback(validation):
            print("   🛡️  Quality control triggered - using fallback response")
            response = self.quality_controller.generate_fallback_response(
                state, validation["issues"]
            )
            print(f"   Fallback response: {response}")
        else:
            print(f"   Final response: {response}")
        
        # Log response metadata
        context_used = generation_result["context_used"]
        print(f"   Context used: L1={context_used['l1_items']}, "
              f"L2={context_used['l2_items']}, L3={context_used['l3_items']}")
        
        # Return state unchanged (this is the final pipeline step)
        return state
