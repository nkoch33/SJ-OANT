from langgraph.graph import StateGraph, END # type: ignore
from typing import TypedDict, List, Dict
from langchain_google_genai import ChatGoogleGenerativeAI # type: ignore
from langchain_core.prompts import ChatPromptTemplate # type: ignore
import os
from dotenv import load_dotenv # type: ignore
import json

# ---- Define State ----
class MemoryState(TypedDict):
    L1: List[str]
    L2: List[str]
    L3: List[str]
    flagged: List[str]
    user_input: str

class AgentPipeline():
    def __init__(self, llm):
        self.llm = llm
        self.graph = self.build_graph()

        # Agent prompts
        self.prompts = {
            "prompt_refinement": ChatPromptTemplate.from_messages([
                ("human", """You are the Prompt Refinement Agent. Your role is to clean and refine user input,
                removing conversational noise and ambiguity.

                Guidelines:
                - Remove filler words and conversational noise
                - Clarify ambiguous references
                - Maintain the core intent and information
                - Return only the refined input, no explanations

                Raw user input: {input}""")
            ]),

            "redundancy_filter": ChatPromptTemplate.from_messages([
                ("human", """You are the Redundancy Filter Agent. Compare new input against recent history
                to identify and discard redundant information.

                Recent L1 Cache: {l1_cache}

                Return JSON with:
                - "is_redundant": boolean
                - "confidence": float (0-1)
                - "reasoning": string
                - "unique_elements": list of unique information if not redundant

                New input to check: {input}""")
            ]),

            "contradiction_detection": ChatPromptTemplate.from_messages([
                ("human", """You are the Contradiction Detection Agent. Check new information against
                the curated knowledge base for potential contradictions.

                L3 Cache (Ground Truth): {l3_cache}

                Return JSON with:
                - "has_contradiction": boolean
                - "confidence": float (0-1)
                - "reasoning": string
                - "conflicting_facts": list of specific contradictions found

                New information: {input}""")
            ]),

            "memory_curation": ChatPromptTemplate.from_messages([
                ("system", """You are the Memory Curation Agent (Manager).
                Your role is to orchestrate memory updates, making final storage decisions
                based on proposals and votes from other agents.
                You must ensure memory consistency, prevent contradictions, and maintain
                a clear hierarchy across the caches:

                - L1 (Working Memory): recent raw inputs.
                - L2 (Summarized Memory): concise summaries of recent turns.
                - L3 (Archival Memory): verified facts and anchors.
                - FLAGGED: contradictory or irrelevant content (do not discard permanently, just quarantine).

                Promotion Rules:
                - When L1 exceeds its limit, compress content into L2.
                - When L2 exceeds its limit, compress content into L3.
                - Never add content directly to L3 without summarization or verification.
                - If contradiction is reported, route to FLAGGED instead of caches.
                """)
            ]),

            "main_llm": ChatPromptTemplate.from_messages([
                ("human", """You are the main LLM. Generate responses using the curated memory context.
                Prioritize recent information from L1 and L2, and ground facts in L3.

                L1 Cache (Recent): {l1_cache}
                L2 Cache (Summarized): {l2_cache}
                L3 Cache (Facts): {l3_cache}

                Generate a helpful, accurate response based on this curated context.

                User input: {user_input}""")
            ]),
        }
    def build_graph(self):
        # ---- Build Graph ----
        graph = StateGraph(MemoryState)

        graph.add_node("refine", self.prompt_refinement)
        graph.add_node("redundancy", self.redundancy_filter)
        graph.add_node("contradiction", self.contradiction_detection)
        graph.add_node("curation", self.memory_curation)
        graph.add_node("generation", self.llm_generation)

        # Edges define flow
        graph.add_edge("refine", "redundancy")
        graph.add_edge("redundancy", "contradiction")
        graph.add_edge("contradiction", "curation")
        graph.add_edge("curation", "generation")
        graph.add_edge("generation", END)

        graph.set_entry_point("refine")
        # Compile the graph
        app = graph.compile()
        return app
    # ---- Define Agent Nodes ----
    def prompt_refinement(self, state: MemoryState) -> MemoryState:
        prompt = self.prompts["prompt_refinement"].format(input=state["user_input"])
        response = self.llm.invoke(prompt)

        # Update state with refined input
        state["user_input"] = response.content.strip()
        return state

    def redundancy_filter(self, state: MemoryState) -> MemoryState:
        prompt = self.prompts["redundancy_filter"].format(
            l1_cache=state["L1"],
            input = state["user_input"]
        )
        response = self.llm.invoke(prompt)

        # Parse JSON response
        result = json.loads(response.content)

        if result["is_redundant"]:
            # Don't process further
            state["user_input"] = "" # Clear input because it's redundant
        else:
            # Keep unique elements
            if "unique_elements" in result:
                state["user_input"] = " ".join(result["unique_elements"])

        return state

    def contradiction_detection(self, state: MemoryState) -> MemoryState:
        prompt = self.prompts["contradiction_detection"].format(
            l3_cache = state["L3"],
            input = state["user_input"]
        )
        response = self.llm.invoke(prompt)

        # Parse JSON response
        result = json.loads(response.content)

        if result["has_contradiction"]:
            # Route to FLAGGED instead of continuing
            state["flagged"].append({
                "content": state["user_input"],
                "reason": result["reasoning"],
                "conflicts": result["conflicting_facts"]
            })
            state["user_input"] = "" # Clear contradictory input
        return state

    def memory_curation(self, state: MemoryState) -> MemoryState:
        # Skip if input was flagged or filtered out
        if not state["user_input"]:
            return state
        
        # Add to L1 (working memory)
        state["L1"].append(state["user_input"])

        # Memory management: L1 -> L2 -> L3 promotion
        L1_LIMIT = 5 # Keep last 5 items in short term memory
        L2_LIMIT = 10 # Keep last 10 summaries

        # L1 overflow: compress into L2
        if len(state["L1"]) > L1_LIMIT:
            # Take oldest items from L1
            to_compress = state["L1"][:len(state["L1"]) - L1_LIMIT]

            # Create summary prompt
            summary_prompt = f"""
            Summarize these recent conversation items into key facts:
            {to_compress}
            Return only the essential information, one line per fact.
            """

            summary_response = self.llm.invoke(summary_prompt)
            state["L2"].append(summary_response.content)

            # Keep only recent items in L1
            state["L1"] = state["L1"][-L1_LIMIT:]
        
        # L2 overflow: compress into L3
        if len(state["L2"]) > L2_LIMIT:
            # Take oldest summaries from L2
            to_archive = state["L2"][:len(state["l1"]) - L1_LIMIT]

            # Create archival prompt
            archive_prompt = f"""
            Extract verified, factual information from these summaries for long-term storage:
            {to_archive}
            Return only concrete facts that are likely to remain true over time.
            """

            archive_response = self.llm.invoke(archive_prompt)
            state["L3"].append(archive_response.content)

            # Keep only recent summaries in L2
            state["L2"] = state["L2"][-L2_LIMIT:]

        return state

    def llm_generation(self, state: MemoryState) -> MemoryState:
        prompt = self.prompts["main_llm"].format(
            l1_cache = state["L1"],
            l2_cache = state["L2"],
            l3_cache = state["L3"],
            user_input = state["user_input"]
        )

        response = self.llm.invoke(prompt)

        # Store the final response in state
        state["final_response"] = response.content
        
        return state

def main():

    load_dotenv()
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.1,
            google_api_key=GOOGLE_API_KEY
        )
    pipeline = AgentPipeline(llm)

      # Example initial state
    state = {
        "L1": [],
        "L2": [],
        "L3": [],
        "flagged": [],
        "user_input": "Barack Obama was born in Hawaii.",
    }

    print("Starting pipeline...")
    result = pipeline.graph.invoke(state)
    print("Final result:", result)



if __name__ == "__main__":
    main()