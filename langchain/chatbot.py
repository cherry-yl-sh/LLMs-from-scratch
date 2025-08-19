import getpass
import os
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict
from typing import Sequence

api = os.environ.get("OPEN_AI_TOKEN")
model = init_chat_model("gpt-5-mini", model_provider="openai"
                        , base_url="https://aihubmix.com/v1", api_key=api)


class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str


prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)




workflow = StateGraph(state_schema=State)
def call_model(state: MessagesState):
    prompt = prompt_template.invoke(state)
    response = model.invoke(prompt)
    return {"messages": response}



# Define the (single) node in the graph
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)
app = workflow.compile(checkpointer=MemorySaver())

config = {"configurable": {"thread_id": "abc123"}}

query1 = "Hi! I'm Bob. who are you"
language = "Spanish"
input_messages = [HumanMessage(query1)]
output = app.invoke({"messages": input_messages, "language": language}, config)
output["messages"][-1].pretty_print()  # output contains all messages in state




# restult = model.invoke(
#     [
#         HumanMessage(content="Hi! I'm Bob"),
#         AIMessage(content="Hello Bob! How can I assist you today?"),
#         HumanMessage(content="What's my name?"),
#     ]
# )
