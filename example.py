from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
# --- CHANGE 1: Import the Google Gemini chat model ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from dotenv import load_dotenv
import os

load_dotenv()
GEMINI_KEY = os.getenv('GEMINI_KEY')

# Create state
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Init state machine graph
graph = StateGraph(State)

@tool
def get_weather(location: str):
    """Call to get the current weather."""
    if location.lower() in ["yorkshire"]:
        return "It's cold and wet."
    else:
        return "It's warm and sunny."

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", # You can choose your preferred Gemini model
    google_api_key=GEMINI_KEY,
)

tools = [get_weather]

llm_with_tools = llm.bind_tools(tools)

tool_node = ToolNode(tools)

graph.add_node("tool_node", tool_node)


def prompt_node(state: State) -> State:
    # LangGraph automatically handles passing the list of messages
    new_message = llm_with_tools.invoke(state["messages"])
    return {"messages": [new_message]}

graph.add_node("prompt_node", prompt_node)

def conditional_edge(state: State) -> Literal['tool_node', '__end__']:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tool_node"
    else:
        return "__end__"

graph.add_conditional_edges(
    'prompt_node',
    conditional_edge
)


graph.add_edge("tool_node", "prompt_node")
graph.set_entry_point("prompt_node")

APP = graph.compile()
new_state = APP.invoke({"messages": [HumanMessage(content="What's the weather in yorkshire?")]})
print(new_state["messages"][-1].content)