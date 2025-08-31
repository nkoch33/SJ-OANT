from langgraph.graph import StateGraph, END # type: ignore
from typing import TypedDict, List, Dict
from langchain_google_genai import ChatGoogleGenerativeAI # type: ignore
from langchain_core.prompts import ChatPromptTemplate # type: ignore
import os
from dotenv import load_dotenv # type: ignore

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
        print("refined prompt")
        return state

    def redundancy_filter(self, state: MemoryState) -> MemoryState:
        print("Checked for redundancy")
        return state

    def contradiction_detection(self, state: MemoryState) -> MemoryState:
        print("Checked for contradictions")
        return state

    def memory_curation(self, state: MemoryState) -> MemoryState:
        print("curated memory prompt")
        return state
    def llm_generation(self, state: MemoryState) -> MemoryState:
        print("generating final message")
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