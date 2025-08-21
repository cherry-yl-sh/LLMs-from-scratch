import os
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph, END
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict
from typing import Sequence
from langchain_core.messages import trim_messages, AIMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
@tool()
def multiply(a: int, b: int) -> int:
    """Multiply a and b.
    Args:
        a: first int
        b: second int
    """
    print("tool call !!!")
    return a * b

api = os.environ.get("OPEN_AI_TOKEN")
model = init_chat_model("gpt-4o-mini", model_provider="openai"
                        , base_url="https://aihubmix.com/v1", api_key=api)
tools = [multiply]
model_with_tools = model.bind_tools(tools)
tool_node = ToolNode(tools)
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability, you can use tools multiply",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str

def call_model(state: State):
    prompt = prompt_template.invoke(
        {"messages": state["messages"]}
    )
    response = model.invoke(prompt)
    return {"messages": response}

config = {"configurable": {"thread_id": "abc123"}}
workflow = StateGraph(state_schema=State)
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)
app = workflow.compile(checkpointer=MemorySaver())
input_messages = "What is 2 multiplied by 3?"
output = app.invoke({"messages": input_messages},config)
ai_response = output["messages"][-1].content
print(ai_response)
